# inference/run_inference.py
"""
Full ONNX inference pipeline for Stable Audio 3 Small.

Features:
  - Automatic EP selection via device_selector (CUDA → ROCm → OpenVINO → CPU)
  - Precision selection by hardware tier (gpu_high=fp16, gpu_low/cpu=int8)
  - Graceful FP16 fallback if quantized model is missing
  - Simplified Euler sampler over the diffusion trajectory
  - Saves output as WAV via soundfile

Tokenizer note:
  SA3 uses a T5-derived text encoder. The tokenizer is loaded from
  "google/flan-t5-large" as a safe approximation. If you know the exact
  tokenizer used by your SA3 checkpoint, set SA3_TOKENIZER_ID env var.

Usage:
    python -m inference.run_inference --prompt "Rolling thunder" \\
        --duration 3.0 --variant sfx --out thunder.wav

    python -m inference.run_inference --prompt "Soft piano melody" \\
        --duration 10.0 --variant music --out piano.wav
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import OUTPUT_ROOT, DIFFUSION_STEPS, SAMPLE_RATE, MODEL_IDS, LATENT_CHANNELS
from device_selector import create_session, get_hardware_tier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_precision(tier: str) -> str:
    """Select model precision based on hardware tier."""
    return {"gpu_high": "fp16", "gpu_low": "int8", "cpu": "int8"}[tier]


def _load_sessions(variant: str, precision: str) -> dict[str, object]:
    """
    Load all ONNX sessions for the given variant and precision.
    Falls back to fp16 for any module whose quantized file is missing.
    """
    base   = Path(OUTPUT_ROOT) / variant / precision
    fp16   = Path(OUTPUT_ROOT) / variant / "fp16"
    modules = ["text_encoder", "conditioner", "dit", "decoder"]
    sessions = {}

    for m in modules:
        path = base / f"{m}.onnx"
        if not path.exists():
            path = fp16 / f"{m}.onnx"
            if not path.exists():
                if m == "conditioner":
                    # Conditioner is optional in some architectures (like SA3 Small)
                    continue
                raise FileNotFoundError(
                    f"ONNX model not found for '{m}' ({variant}). "
                    "Run export_all.py first."
                )
            print(f"[inference] {m}: {precision} not found, using fp16 fallback.")
        sessions[m] = create_session(str(path))

    return sessions


def _get_tokenizer(variant: str = "music"):
    """
    Load the T5 tokenizer for SA3's text encoder.
    Attempts to locate the local cached tokenizer directory inside the HF Hub cache
    first to allow offline execution. Falls back to loading via transformers Hub.
    """
    from transformers import AutoTokenizer

    # Check for custom override env var first
    tokenizer_id = os.getenv("SA3_TOKENIZER_ID")
    if tokenizer_id:
        return AutoTokenizer.from_pretrained(tokenizer_id)

    # Search for local t5gemma-b-b-ul2 folder inside HF cache directory
    # Default cache path: hub/models--stabilityai--stable-audio-3-small-<variant>/snapshots
    repo_name = f"models--stabilityai--stable-audio-3-small-{variant}"
    hub_cache_root = os.getenv("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub"))
    
    # Try local cache relative path first (in the project workspace root)
    local_hub = Path(__file__).resolve().parents[1] / "hub"
    for cache_dir in [local_hub, Path(hub_cache_root)]:
        repo_dir = cache_dir / repo_name / "snapshots"
        if repo_dir.exists():
            for snapshot_dir in repo_dir.iterdir():
                tok_dir = snapshot_dir / "t5gemma-b-b-ul2"
                if tok_dir.exists():
                    print(f"[inference] Loading local tokenizer from: {tok_dir}")
                    return AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True)

    # Fallback to online loading
    tokenizer_id = "google/flan-t5-large"
    print(f"[inference] Local tokenizer not found in cache. Falling back to: {tokenizer_id}")
    return AutoTokenizer.from_pretrained(tokenizer_id)


# ---------------------------------------------------------------------------
# ONNX inference pipeline
# ---------------------------------------------------------------------------

def run_onnx_pipeline(
    prompt: str,
    duration_seconds: float,
    variant: str = "sfx",
    steps: int = None,
    cfg_scale: float = 7.0,
    seed: int = None,
) -> np.ndarray:
    """
    Run the full SA3 ONNX inference pipeline.

    Args:
        prompt:           Text prompt describing the audio.
        duration_seconds: Output clip length in seconds.
        variant:          "music" or "sfx".
        steps:            Number of diffusion steps (overrides hardware default).
        cfg_scale:        Classifier-free guidance scale (not used in simplified
                          sampler; reserved for CFG extension).
        seed:             Optional random seed for reproducibility.

    Returns:
        Waveform as float32 numpy array, shape (audio_samples,).
    """
    if seed is not None:
        np.random.seed(seed)

    tier      = get_hardware_tier()
    precision = _pick_precision(tier)
    n_steps   = steps or DIFFUSION_STEPS[tier]

    print(f"[inference] tier={tier}, precision={precision}, steps={n_steps}, "
          f"duration={duration_seconds}s, variant={variant}")

    sessions = _load_sessions(variant, precision)

    # ------------------------------------------------------------------
    # Step 1: Tokenise prompt
    # ------------------------------------------------------------------
    tokenizer = _get_tokenizer(variant)
    enc = tokenizer(
        prompt,
        return_tensors="np",
        padding="max_length",
        max_length=512,
        truncation=True,
    )

    # ------------------------------------------------------------------
    # Step 2: Text encoding
    # ------------------------------------------------------------------
    text_hidden: np.ndarray = sessions["text_encoder"].run(
        None,
        {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        },
    )[0]
    print(f"[inference] text_hidden shape: {text_hidden.shape}")

    # ------------------------------------------------------------------
    # Step 3: Duration conditioning
    # ------------------------------------------------------------------
    global_cond = None
    if "conditioner" in sessions:
        seconds_start = np.array([[0.0]],                dtype=np.float32)
        seconds_total = np.array([[duration_seconds]],   dtype=np.float32)

        cond_feeds = {
            "seconds_start": seconds_start,
            "seconds_total": seconds_total,
        }
        cond_inputs = {inp.name for inp in sessions["conditioner"].get_inputs()}
        cond_feeds = {k: v for k, v in cond_feeds.items() if k in cond_inputs}

        global_cond = sessions["conditioner"].run(None, cond_feeds)[0]
        if global_cond.ndim == 3 and global_cond.shape[1] == 1:
            global_cond = global_cond.squeeze(1)
        print(f"[inference] global_cond shape: {global_cond.shape}")

    # ------------------------------------------------------------------
    # Step 4: Initialise latent noise
    # ------------------------------------------------------------------
    # Latent time dim: sample_rate / downsampling_factor ≈ 44100 / 512 ≈ 86 frames/s
    DOWNSAMPLE = 512
    latent_t = max(1, int(duration_seconds * SAMPLE_RATE / DOWNSAMPLE))
    latents  = np.random.randn(1, LATENT_CHANNELS, latent_t).astype(np.float32)
    mask     = np.ones((1, latent_t), dtype=np.bool_)

    print(f"[inference] latent shape: {latents.shape}")

    # ------------------------------------------------------------------
    # Step 5: Euler diffusion sampling loop (simplified DDIM)
    # ------------------------------------------------------------------
    # Linear sigma schedule from 1.0 → 0.0
    sigmas = np.linspace(1.0, 0.0, n_steps + 1, dtype=np.float32)

    for i in range(n_steps):
        t_val  = np.array([[sigmas[i]]], dtype=np.float32)   # (1, 1) or (1,)
        t_flat = np.array([sigmas[i]],   dtype=np.float32)   # (B,)

        feeds: dict[str, np.ndarray] = {
            "latents":         latents,
            "timestep":        t_flat,
            "cross_attn_cond": text_hidden,
            "mask":            mask,
        }
        if global_cond is not None:
            feeds["global_cond"] = global_cond

        # Filter feeds to only include inputs expected by the DiT model
        # (PyTorch JIT tracing may optimize out unused parameters like 'mask')
        dit_inputs = {inp.name for inp in sessions["dit"].get_inputs()}
        feeds = {k: v for k, v in feeds.items() if k in dit_inputs}

        velocity: np.ndarray = sessions["dit"].run(None, feeds)[0]

        # Euler step: x_{t-1} = x_t − (sigma_t − sigma_{t-1}) * v_t
        dt      = sigmas[i + 1] - sigmas[i]
        latents = latents + dt * velocity

        if (i + 1) % 10 == 0 or i == n_steps - 1:
            print(f"[inference] step {i+1}/{n_steps}  σ={sigmas[i]:.3f}")

    # ------------------------------------------------------------------
    # Step 6: Decode latents → waveform
    # ------------------------------------------------------------------
    waveform: np.ndarray = sessions["decoder"].run(
        None, {"latents": latents}
    )[0]

    waveform = waveform.squeeze()   # (audio_samples,)
    print(f"[inference] waveform shape: {waveform.shape}, "
          f"max={waveform.max():.4f}, min={waveform.min():.4f}")
    return waveform


# ---------------------------------------------------------------------------
# PyTorch reference pipeline (validation only)
# ---------------------------------------------------------------------------

def run_pytorch_pipeline(
    prompt: str,
    duration: float,
    variant: str,
    steps: int = 20,
) -> np.ndarray:
    """
    Reference PyTorch inference pipeline for numerical comparison.
    Not used in production — only called by validate_pipeline.py.

    Args:
        prompt:   Text prompt.
        duration: Clip duration in seconds.
        variant:  "music" or "sfx".
        steps:    Diffusion steps (kept small for speed).

    Returns:
        Waveform as float32 numpy array.
    """
    import torch
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    model_id = MODEL_IDS[variant]
    print(f"[pytorch_pipeline] Loading {model_id} ...")
    model, cfg = get_pretrained_model(model_id)
    model.eval()

    sample_size = int(duration * cfg.get("sample_rate", SAMPLE_RATE))

    with torch.no_grad():
        output = generate_diffusion_cond(
            model,
            steps=steps,
            cfg_scale=7,
            conditioning=[{
                "prompt":        prompt,
                "seconds_start": 0,
                "seconds_total": duration,
            }],
            sample_size=sample_size,
            sampler_type="dpmpp-3m-sde",
            seed=42,                # fix Windows numpy.random.randint int32 overflow
        )

    return output.squeeze().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# WAV save helper
# ---------------------------------------------------------------------------

def save_wav(waveform: np.ndarray, path: str, sample_rate: int = SAMPLE_RATE) -> None:
    """Save a float32 waveform numpy array as a WAV file via soundfile."""
    import soundfile as sf
    # Normalise to [-1, 1] to avoid clipping
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform / peak * 0.95
    sf.write(path, waveform, sample_rate, subtype="PCM_16")
    print(f"[inference] WAV saved -> {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Stable Audio 3 Small ONNX inference"
    )
    parser.add_argument("--prompt",   required=True,                  help="Text prompt")
    parser.add_argument("--duration", type=float,  default=5.0,       help="Duration in seconds")
    parser.add_argument("--variant",  choices=["music", "sfx"],       default="sfx")
    parser.add_argument("--steps",    type=int,    default=None,      help="Override diffusion steps")
    parser.add_argument("--out",      default="output.wav",           help="Output WAV path")
    parser.add_argument("--seed",     type=int,    default=None,      help="Random seed")
    args = parser.parse_args()

    waveform = run_onnx_pipeline(
        prompt=args.prompt,
        duration_seconds=args.duration,
        variant=args.variant,
        steps=args.steps,
        seed=args.seed,
    )

    save_wav(waveform, args.out)


if __name__ == "__main__":
    main()
