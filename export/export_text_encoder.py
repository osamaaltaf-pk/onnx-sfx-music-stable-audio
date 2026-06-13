# export/export_text_encoder.py
"""
Exports the T5/CLAP text encoder from Stable Audio 3 Small.

SA3 uses a T5-based text encoder accessed via stable_audio_tools.
The encoder is found at model.conditioner.conditioners["prompt"].model.

Dynamic axes are set on batch and seq_len so the exported model accepts
any sequence length up to MAX_SEQ_LEN at runtime.

Outputs:
    models/<variant>/fp16/text_encoder.onnx
"""

import os
import sys
import torch
from pathlib import Path

# Allow running this script directly from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MODEL_IDS, LOCAL_CHECKPOINT_PATHS, OPSET_VERSION, BATCH_SIZE, MAX_SEQ_LEN, OUTPUT_ROOT
from patches.attention_patch import apply_attention_patch

# Must be first — before any model import that may trigger SDPA
apply_attention_patch()


def _load_model(variant: str):
    """
    Load SA3 model from HuggingFace Hub or local checkpoint.
    Falls back to local path if Hub download fails (e.g., 404 / private model).
    """
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
            f"Could not load model '{model_id}' from Hub and no valid local "
            f"checkpoint set for SA3_{variant.upper()}_CKPT env var."
        ) from hub_err


def _locate_text_encoder(model):
    """
    Navigate the model hierarchy to find the text encoder module.
    SA3 stores it at model.conditioner.conditioners["prompt"].model.
    Falls back to common alternate attribute paths.
    """
    # Primary path for stable_audio_tools SA3 architecture
    try:
        encoder = model.conditioner.conditioners["prompt"].model
        print("[export_text_encoder] Found encoder at: model.conditioner.conditioners['prompt'].model")
        return encoder
    except (AttributeError, KeyError):
        pass

    # Alternate: some versions expose model.text_encoder directly
    if hasattr(model, "text_encoder"):
        print("[export_text_encoder] Found encoder at: model.text_encoder")
        return model.text_encoder

    # Alternate: conditioner wrapper without inner .model
    try:
        encoder = model.conditioner.conditioners["prompt"]
        print("[export_text_encoder] Found encoder at: model.conditioner.conditioners['prompt'] (no .model)")
        return encoder
    except (AttributeError, KeyError):
        pass

    raise AttributeError(
        "Cannot locate text encoder. Inspect model structure with "
        "`print(model)` and update _locate_text_encoder() accordingly."
    )


def export_text_encoder(variant: str = "music") -> str:
    """
    Export the text encoder for the given variant to ONNX (FP16 opset 18).

    Args:
        variant: "music" or "sfx"

    Returns:
        Absolute path to the exported .onnx file.
    """
    print(f"\n[export_text_encoder] Loading {variant} model from Hub...")
    model, config = _load_model(variant)
    model.eval()

    encoder = _locate_text_encoder(model)
    encoder.eval()
    # Cast to float32 — T5Gemma uses BFloat16 internally which produces
    # Mul(14) ONNX nodes that ORT CPU cannot execute. FP32 maps to Mul(13).
    encoder = encoder.float()

    # Dummy inputs: (batch, seq_len) integer token IDs and boolean attention mask
    dummy_ids  = torch.zeros(BATCH_SIZE, MAX_SEQ_LEN, dtype=torch.long)
    dummy_mask = torch.ones( BATCH_SIZE, MAX_SEQ_LEN, dtype=torch.long)

    out_dir  = Path(OUTPUT_ROOT) / variant / "fp16"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "text_encoder.onnx")

    print(f"[export_text_encoder] Exporting to {out_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (dummy_ids, dummy_mask),
            out_path,
            dynamo=False,           # force TorchScript path (torch 2.12+ defaults to dynamo=True)
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["hidden_states"],
            dynamic_axes={
                "input_ids":      {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "hidden_states":  {0: "batch", 1: "seq_len"},
            },
        )

    print(f"[export_text_encoder] [OK] saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export SA3 text encoder to ONNX")
    parser.add_argument("--variant", choices=["music", "sfx", "both"], default="both")
    args = parser.parse_args()

    variants = ["music", "sfx"] if args.variant == "both" else [args.variant]
    for v in variants:
        export_text_encoder(v)
