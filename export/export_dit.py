# export/export_dit.py
"""
Exports the Diffusion Transformer (DiT) — the core denoising network and
primary inference bottleneck in Stable Audio 3 Small.

SA3 uses DiffusionTransformer from stable_audio_tools.models.diffusion.
The DiT takes:
  - latents:        (B, C, T)       — noisy latent sequence
  - t (timestep):  (B,)             — diffusion sigma value
  - cross_attn_cond: (B, S, D)      — text cross-attention context
  - mask:           (B, T) bool     — latent sequence padding mask
  - global_cond:    (B, G) optional — global conditioning (duration vectors)

Returns:
  - velocity:       (B, C, T)       — predicted velocity field

ONNX constraints enforced:
  - SDPA replaced via attention_patch (must be applied before this import)
  - No in-place tensor ops
  - Dynamic axes on batch, latent_t, and text_seq

Failure modes and fixes are documented in agent.md §Failure Recovery.

Outputs:
    models/<variant>/fp16/dit.onnx
"""

import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    MODEL_IDS,
    LOCAL_CHECKPOINT_PATHS,
    OPSET_VERSION,
    BATCH_SIZE,
    LATENT_CHANNELS,
    MAX_SEQ_LEN,
    OUTPUT_ROOT,
)
from patches.attention_patch import apply_attention_patch

# Must be first
apply_attention_patch()


def _load_model(variant: str):
    from stable_audio_tools import get_pretrained_model

    model_id = MODEL_IDS[variant]
    try:
        model, config = get_pretrained_model(model_id)
        return model, config
    except Exception as hub_err:
        local_path = LOCAL_CHECKPOINT_PATHS.get(variant, "")
        if local_path and os.path.exists(local_path):
            print(f"[WARNING] HuggingFace Hub failed ({hub_err}). "
                  f"Falling back to local checkpoint: {local_path}")
            model, config = get_pretrained_model(local_path)
            return model, config
        raise RuntimeError(
            f"Could not load model '{model_id}' from Hub and no valid local checkpoint."
        ) from hub_err


def _locate_dit(model):
    """Find the DiffusionTransformer inside the SA3 model."""
    # Primary: SA3 wraps DiT in model.model
    if hasattr(model, "model"):
        print("[export_dit] Found DiT at: model.model")
        return model.model

    # Alternate: explicit diffusion_model attribute
    if hasattr(model, "diffusion_model"):
        print("[export_dit] Found DiT at: model.diffusion_model")
        return model.diffusion_model

    # Alternate: model IS the DiT (e.g., loaded directly)
    from stable_audio_tools.models.diffusion import DiffusionTransformer
    if isinstance(model, DiffusionTransformer):
        print("[export_dit] model itself is DiffusionTransformer")
        return model

    raise AttributeError(
        "Cannot locate DiffusionTransformer. Inspect model structure and "
        "update _locate_dit() in export_dit.py."
    )


class DiTWrapper(torch.nn.Module):
    """
    Wraps the SA3 DiffusionTransformer to accept positional args for
    torch.onnx.export (which does not support **kwargs in forward()).

    The wrapper reorders arguments into keyword form for the underlying DiT.
    """

    def __init__(self, dit: torch.nn.Module, has_global_cond: bool = True):
        super().__init__()
        self.dit = dit
        self.has_global_cond = has_global_cond

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        cross_attn_cond: torch.Tensor,
        mask: torch.Tensor,
        global_cond: torch.Tensor = None,
    ) -> torch.Tensor:
        kwargs = {
            "cross_attn_cond": cross_attn_cond,
            "mask": mask,
        }
        if self.has_global_cond and global_cond is not None:
            kwargs["global_cond"] = global_cond
        return self.dit(latents, timestep, **kwargs)


def export_dit(variant: str = "music") -> str:
    """
    Export the Diffusion Transformer for the given variant to ONNX (FP16, opset 18).

    Args:
        variant: "music" or "sfx"

    Returns:
        Absolute path to the exported .onnx file.
    """
    print(f"\n[export_dit] Loading {variant} model from Hub...")
    model, model_config = _load_model(variant)
    model.eval()

    dit = _locate_dit(model)
    dit.eval()

    # -----------------------------------------------------------------
    # Resolve dimensions from model config with safe fallbacks
    # -----------------------------------------------------------------
    # Latent time dimension: at 44100 Hz with 4096x downsampling ≈ 10.76 frames/s
    # Use 512 as export length; dynamic axis handles actual lengths at runtime
    LATENT_T    = 512
    TEXT_SEQ    = int(model_config.get("conditioning", {}).get("configs", [{}])[0]
                      .get("config", {}).get("max_length", MAX_SEQ_LEN))
    COND_DIM    = int(model_config.get("conditioning", {}).get("cond_dim", 768))
    GLOBAL_DIM  = int(model_config.get("global_cond_dim", 768))
    LOCAL_DIM   = int(model_config.get("local_cond_dim", 0))

    print(f"[export_dit] Dims — latent_channels={LATENT_CHANNELS}, "
          f"latent_t={LATENT_T}, text_seq={TEXT_SEQ}, cond_dim={COND_DIM}, global_dim={GLOBAL_DIM}")

    # -----------------------------------------------------------------
    # Dummy inputs
    # -----------------------------------------------------------------
    dummy_latent    = torch.randn(BATCH_SIZE, LATENT_CHANNELS, LATENT_T)
    dummy_timestep  = torch.tensor([0.5] * BATCH_SIZE, dtype=torch.float32)
    dummy_text_cond = torch.randn(BATCH_SIZE, TEXT_SEQ, COND_DIM, dtype=torch.float32)
    dummy_mask      = torch.ones( BATCH_SIZE, LATENT_T, dtype=torch.bool)
    dummy_global    = torch.randn(BATCH_SIZE, GLOBAL_DIM) if GLOBAL_DIM > 0 else None

    has_global = dummy_global is not None
    wrapped    = DiTWrapper(dit, has_global_cond=has_global)
    wrapped.eval()

    args = (dummy_latent, dummy_timestep, dummy_text_cond, dummy_mask)
    if has_global:
        args = args + (dummy_global,)

    input_names  = ["latents", "timestep", "cross_attn_cond", "mask"]
    output_names = ["velocity"]
    dynamic_axes = {
        "latents":        {0: "batch", 2: "latent_t"},
        "timestep":       {0: "batch"},
        "cross_attn_cond":{0: "batch", 1: "text_seq"},
        "mask":           {0: "batch", 1: "latent_t"},
        "velocity":       {0: "batch", 2: "latent_t"},
    }
    if has_global:
        input_names.append("global_cond")
        dynamic_axes["global_cond"] = {0: "batch"}

    out_dir  = Path(OUTPUT_ROOT) / variant / "fp16"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "dit.onnx")

    print(f"[export_dit] Exporting to {out_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            args,
            out_path,
            dynamo=False,           # force TorchScript path (torch 2.12+ defaults to dynamo=True)
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            # Explicitly use eval mode to avoid training-only nodes (dropout, etc.)
            training=torch.onnx.TrainingMode.EVAL,
        )

    print(f"[export_dit] [OK] saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export SA3 DiT to ONNX")
    parser.add_argument("--variant", choices=["music", "sfx", "both"], default="both")
    args = parser.parse_args()

    variants = ["music", "sfx"] if args.variant == "both" else [args.variant]
    for v in variants:
        export_dit(v)
