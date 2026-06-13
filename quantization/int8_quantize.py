# quantization/int8_quantize.py
"""
Dynamic INT8 quantization of SA3 ONNX modules.

Uses onnxruntime.quantization.quantize_dynamic:
  - Weight type:   QInt8
  - Per-channel:   True  (better accuracy than per-tensor)
  - reduce_range:  True  (required for AVX2/VNNI compatibility on Intel CPUs)
  - Quantizes:     MatMul nodes only (Linear layers in the transformer)
  - Excluded:      embedding and positional embedding nodes (kept FP32)

Recommended for ALL hardware tiers — especially Intel CPU and GPU.
For NVIDIA GTX 2000 (Turing), INT8 on CUDA is often faster than FP16 at batch=1.

Input:   models/<variant>/fp16/<name>.onnx
Output:  models/<variant>/int8/<name>.onnx

Usage:
    python -m quantization.int8_quantize --variant music
    python -m quantization.int8_quantize --variant sfx
    python -m quantization.int8_quantize --variant both
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import OUTPUT_ROOT

# Modules to quantize — ordered by size (largest last for progress clarity)
MODULES_TO_QUANTIZE = ["text_encoder", "conditioner", "decoder", "dit"]

# Embedding-like node patterns to keep in FP32
EXCLUDED_NODE_PATTERNS = ["/embeddings", "/pos_embed", "/embed_tokens", "/embed_positions"]


def quantize_int8(variant: str = "music") -> dict[str, str]:
    """
    Apply dynamic INT8 quantization to all SA3 modules for the given variant.

    Args:
        variant: "music" or "sfx"

    Returns:
        dict mapping module name → output INT8 .onnx path
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    results: dict[str, str] = {}
    base_fp16 = Path(OUTPUT_ROOT) / variant / "fp16"
    base_int8 = Path(OUTPUT_ROOT) / variant / "int8"
    base_int8.mkdir(parents=True, exist_ok=True)

    for name in MODULES_TO_QUANTIZE:
        fp16_path = base_fp16 / f"{name}.onnx"
        int8_path = base_int8  / f"{name}.onnx"

        if not fp16_path.exists():
            print(f"[int8] {name}.onnx not found at {fp16_path} — skipping.")
            continue

        print(f"[int8] Quantizing {name} ({variant}) ...")
        try:
            quantize_dynamic(
                model_input=str(fp16_path),
                model_output=str(int8_path),
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=True,       # required for AVX2/VNNI compatibility
                nodes_to_exclude=EXCLUDED_NODE_PATTERNS,
                optimize_model=True,     # run ORT graph optimization before quant
            )
            print(f"[int8] ✓ {name} → {int8_path}")
            results[name] = str(int8_path)

        except Exception as e:
            print(f"[int8] ERROR quantizing {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[int8] {variant} INT8 quantization complete. {len(results)}/{len(MODULES_TO_QUANTIZE)} modules.")
    return results


def main():
    parser = argparse.ArgumentParser(description="INT8 quantize SA3 ONNX modules")
    parser.add_argument("--variant", choices=["music", "sfx", "both"], default="both")
    args = parser.parse_args()

    variants = ["music", "sfx"] if args.variant == "both" else [args.variant]
    for v in variants:
        quantize_int8(v)


if __name__ == "__main__":
    main()
