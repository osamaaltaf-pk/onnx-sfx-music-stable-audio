# export/export_conditioner.py
"""
Exports the duration/timing conditioner (NumberConditioner MLP) from SA3.

SA3 conditions on two scalar values:
  - seconds_start  (float32 scalar — when in the audio the generation starts)
  - seconds_total  (float32 scalar — total clip duration)

Both conditioners share the same NumberConditioner class and produce a
conditioning vector. We export seconds_start here; the same logic applies
to seconds_total. At inference time, both vectors are concatenated to form
the global conditioning signal for the DiT.

Outputs:
    models/<variant>/fp16/conditioner.onnx
"""

import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MODEL_IDS, LOCAL_CHECKPOINT_PATHS, OPSET_VERSION, BATCH_SIZE, OUTPUT_ROOT
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


class _DualConditionerWrapper(torch.nn.Module):
    """
    Wraps both seconds_start and seconds_total NumberConditioners into a
    single ONNX-exportable module that returns their concatenated output.

    This lets us export a single conditioner.onnx that takes two float scalars
    and returns the full global conditioning vector.
    """

    def __init__(self, cond_start, cond_total):
        super().__init__()
        self.cond_start = cond_start
        self.cond_total = cond_total

    def forward(self, seconds_start: torch.Tensor, seconds_total: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seconds_start: (B, 1) float32
            seconds_total: (B, 1) float32
        Returns:
            global_cond:   (B, cond_dim * 2) float32
        """
        vec_start = self.cond_start(seconds_start)
        vec_total = self.cond_total(seconds_total)
        return torch.cat([vec_start, vec_total], dim=-1)


def export_conditioner(variant: str = "music") -> str | None:
    """
    Export the timing conditioner for the given variant to ONNX.

    Exports a dual-input wrapper (seconds_start, seconds_total) that returns
    the concatenated conditioning vector used as global_cond in the DiT.

    Args:
        variant: "music" or "sfx"

    Returns:
        Path to exported .onnx file, or None if conditioner not found.
    """
    print(f"\n[export_conditioner] Loading {variant} model from Hub...")
    model, config = _load_model(variant)
    model.eval()

    try:
        cond_start = model.conditioner.conditioners["seconds_start"]
        cond_total = model.conditioner.conditioners["seconds_total"]
        print("[export_conditioner] Found seconds_start + seconds_total conditioners.")
    except (AttributeError, KeyError) as e:
        print(f"[export_conditioner] WARNING: Conditioners not found ({e}). "
              "Skipping conditioner export.")
        return None

    cond_start.eval()
    cond_total.eval()

    wrapper = _DualConditionerWrapper(cond_start, cond_total)
    wrapper.eval()

    # Dummy inputs: (batch, 1) float scalar values
    dummy_start = torch.tensor([[0.0]],            dtype=torch.float32)  # seconds_start = 0
    dummy_total = torch.tensor([[5.0]],            dtype=torch.float32)  # seconds_total = 5s

    out_dir  = Path(OUTPUT_ROOT) / variant / "fp16"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "conditioner.onnx")

    print(f"[export_conditioner] Exporting to {out_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_start, dummy_total),
            out_path,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=["seconds_start", "seconds_total"],
            output_names=["global_cond"],
            dynamic_axes={
                "seconds_start": {0: "batch"},
                "seconds_total": {0: "batch"},
                "global_cond":   {0: "batch"},
            },
        )

    print(f"[export_conditioner] ✓ saved → {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export SA3 conditioner to ONNX")
    parser.add_argument("--variant", choices=["music", "sfx", "both"], default="both")
    args = parser.parse_args()

    variants = ["music", "sfx"] if args.variant == "both" else [args.variant]
    for v in variants:
        export_conditioner(v)
