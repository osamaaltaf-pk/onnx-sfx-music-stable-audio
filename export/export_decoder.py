# export/export_decoder.py
"""
Exports the AutoencoderOobleck decoder (latent → waveform) from SA3.

SA3 uses Oobleck — a causal audio autoencoder with Snake activation.
The decoder maps latent tensors (B, C, T_latent) → raw waveforms (B, 1, T_audio).

Snake activation note:
    Snake(x) expands to: x + (1/a) * sin(a*x)^2
    This traces through torch.onnx.export cleanly as standard elementwise ops.
    If export fails on Snake, see agent.md §F4 (FP32 export + onnxconverter_common).

Dynamic axes:
    latent_t  on input  — supports any clip length at runtime
    audio_samples on output — proportional to latent_t * downsampling_factor

Outputs:
    models/<variant>/fp16/decoder.onnx
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
    OUTPUT_ROOT,
)
from patches.attention_patch import apply_attention_patch

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


def _locate_decoder(model):
    """
    Find the audio decoder module inside the SA3 model.

    SA3 Small model structure (stable-audio-tools 0.0.20):
        model.pretransform               -> AutoencoderPretransform
        model.pretransform.model         -> AudioAutoencoder
        model.pretransform.model.decoder -> SAMEDecoder  ← primary path
    """
    # SA3 Small (0.0.20+): pretransform → AudioAutoencoder → SAMEDecoder
    try:
        dec = model.pretransform.model.decoder
        print("[export_decoder] Found decoder at: model.pretransform.model.decoder")
        return dec
    except AttributeError:
        pass

    # SA3 full / older: pretransform → decoder
    try:
        dec = model.pretransform.decoder
        print("[export_decoder] Found decoder at: model.pretransform.decoder")
        return dec
    except AttributeError:
        pass

    # Legacy: direct autoencoder
    if hasattr(model, "autoencoder") and hasattr(model.autoencoder, "decoder"):
        print("[export_decoder] Found decoder at: model.autoencoder.decoder")
        return model.autoencoder.decoder

    # Legacy: VAE naming
    if hasattr(model, "vae") and hasattr(model.vae, "decoder"):
        print("[export_decoder] Found decoder at: model.vae.decoder")
        return model.vae.decoder

    # Legacy: direct decoder attribute
    if hasattr(model, "decoder"):
        print("[export_decoder] Found decoder at: model.decoder")
        return model.decoder

    raise AttributeError(
        "Cannot locate audio decoder. Inspect model structure with "
        "`print(model)` and update _locate_decoder() in export_decoder.py.\n"
        "Hint: Run: python -c \"from stable_audio_tools import get_pretrained_model; "
        "m,_=get_pretrained_model('stabilityai/stable-audio-3-small-music'); "
        "[print(n) for n,_ in m.named_modules()]\""
    )


def export_decoder(variant: str = "music") -> str:
    """
    Export the Oobleck audio decoder for the given variant to ONNX (FP16, opset 18).

    Args:
        variant: "music" or "sfx"

    Returns:
        Absolute path to the exported .onnx file.
    """
    print(f"\n[export_decoder] Loading {variant} model from Hub...")
    model, config = _load_model(variant)
    model.eval()

    decoder = _locate_decoder(model)
    decoder.eval()

    # Export with a representative latent length; dynamic axis handles others
    LATENT_T = 512
    dummy_z  = torch.randn(BATCH_SIZE, LATENT_CHANNELS, LATENT_T, dtype=torch.float32)

    out_dir  = Path(OUTPUT_ROOT) / variant / "fp16"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "decoder.onnx")

    print(f"[export_decoder] Exporting to {out_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            decoder,
            dummy_z,
            out_path,
            dynamo=False,           # force TorchScript path (torch 2.12+ defaults to dynamo=True)
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=["latents"],
            output_names=["waveform"],
            dynamic_axes={
                "latents":  {0: "batch", 2: "latent_t"},
                "waveform": {0: "batch", 2: "audio_samples"},
            },
            training=torch.onnx.TrainingMode.EVAL,
        )

    print(f"[export_decoder] [OK] saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export SA3 decoder to ONNX")
    parser.add_argument("--variant", choices=["music", "sfx", "both"], default="both")
    args = parser.parse_args()

    variants = ["music", "sfx"] if args.variant == "both" else [args.variant]
    for v in variants:
        export_decoder(v)
