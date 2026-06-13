# quantization/int4_quantize.py
"""
INT4 weight-only quantization for SA3 ONNX modules.

Two backends are supported, selected automatically by hardware tier:

  CUDA / AMD (gpu_high, gpu_low):
      onnxruntime MatMulNBits quantizer
      Requires onnxruntime >= 1.17

  Intel CPU (cpu tier):
      Intel Neural Compressor (INC) weight-only INT4
      Requires: pip install neural-compressor==2.5.0
      Falls back to MatMulNBits if INC is not installed.

Only the two largest modules are INT4-quantized (dit, decoder).
text_encoder and conditioner stay at INT8 — they are small enough
that INT4 accuracy loss is not worth the minimal size reduction.

Block size 32 (vs 16) is used for better accuracy across EPs.

Input:   models/<variant>/fp16/<name>.onnx
Output:  models/<variant>/int4/<name>.onnx  (or <name>_inc.onnx for INC path)

Usage:
    python -m quantization.int4_quantize --variant music
    python -m quantization.int4_quantize --variant sfx
    python -m quantization.int4_quantize --variant both
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import OUTPUT_ROOT
from device_selector import get_hardware_tier

BLOCK_SIZE = 32          # power of 2; 32 is safer than 16 across EPs
LARGE_MODULES = ["dit", "decoder"]   # only quantize heavy modules


# ---------------------------------------------------------------------------
# MatMulNBits path (CUDA + CPU fallback)
# ---------------------------------------------------------------------------

def quantize_int4_matmulnbits(variant: str, name: str) -> str | None:
    """
    INT4 via ONNX Runtime MatMulNBits — works on CUDA and CPU.

    Requires onnxruntime >= 1.17 with onnxruntime-extensions.
    """
    try:
        from onnxruntime.quantization import matmul_nbits_quantizer
    except ImportError:
        print(
            "[int4/NBits] matmul_nbits_quantizer not available. "
            "Upgrade onnxruntime to >= 1.17: pip install onnxruntime>=1.17"
        )
        return None

    fp16_path = Path(OUTPUT_ROOT) / variant / "fp16" / f"{name}.onnx"
    int4_path  = Path(OUTPUT_ROOT) / variant / "int4" / f"{name}.onnx"
    int4_path.parent.mkdir(parents=True, exist_ok=True)

    if not fp16_path.exists():
        print(f"[int4/NBits] {name}.onnx missing at {fp16_path} — run FP16 export first.")
        return None

    print(f"[int4/NBits] Quantizing {name} ({variant}, block_size={BLOCK_SIZE}) ...")
    try:
        quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            model=str(fp16_path),
            n_bits=4,
            block_size=BLOCK_SIZE,
            is_symmetric=True,
            nodes_to_exclude=[],
        )
        quantizer.process()
        quantizer.model.save_model_to_file(
            str(int4_path),
            use_external_data_format=False,
        )
        print(f"[int4/NBits] ✓ {name} → {int4_path}")
        return str(int4_path)

    except Exception as e:
        print(f"[int4/NBits] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Intel Neural Compressor path (Intel CPU / OpenVINO)
# ---------------------------------------------------------------------------

def quantize_int4_neural_compressor(variant: str, name: str) -> str | None:
    """
    INT4 via Intel Neural Compressor weight-only quantization.
    Optimal for Intel CPU / Intel Arc GPU (OpenVINO backend).

    Falls back to MatMulNBits if neural-compressor is not installed.
    """
    try:
        from neural_compressor import quantization, PostTrainingQuantConfig
    except ImportError:
        print(
            "[int4/INC] neural-compressor not installed. "
            "Falling back to MatMulNBits path."
        )
        return quantize_int4_matmulnbits(variant, name)

    fp16_path = Path(OUTPUT_ROOT) / variant / "fp16" / f"{name}.onnx"
    int4_path  = Path(OUTPUT_ROOT) / variant / "int4" / f"{name}_inc.onnx"
    int4_path.parent.mkdir(parents=True, exist_ok=True)

    if not fp16_path.exists():
        print(f"[int4/INC] {name}.onnx missing at {fp16_path} — run FP16 export first.")
        return None

    print(f"[int4/INC] Quantizing {name} ({variant}) via Intel Neural Compressor ...")
    try:
        config = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                "MatMul": {
                    "weight": {
                        "dtype":      "int4",
                        "scheme":     "sym",
                        "group_size": BLOCK_SIZE,
                    }
                }
            },
        )
        q_model = quantization.fit(model=str(fp16_path), conf=config)
        q_model.save(str(int4_path))
        print(f"[int4/INC] ✓ {name} → {int4_path}")
        return str(int4_path)

    except Exception as e:
        print(f"[int4/INC] ERROR: {e}. Retrying with MatMulNBits fallback...")
        return quantize_int4_matmulnbits(variant, name)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def quantize_int4(variant: str = "music") -> dict[str, str]:
    """
    Quantize all large SA3 modules to INT4 for the given variant.

    Automatically selects the backend based on detected hardware tier.

    Args:
        variant: "music" or "sfx"

    Returns:
        dict mapping module name → output INT4 .onnx path
    """
    tier = get_hardware_tier()
    print(f"[int4] Hardware tier: {tier}")

    # Intel CPU → INC path; everything else → MatMulNBits
    quant_fn = (
        quantize_int4_neural_compressor
        if tier == "cpu"
        else quantize_int4_matmulnbits
    )
    print(f"[int4] Using backend: {'Intel Neural Compressor' if tier == 'cpu' else 'MatMulNBits'}")

    results: dict[str, str] = {}
    for name in LARGE_MODULES:
        path = quant_fn(variant, name)
        if path:
            results[name] = path

    print(f"\n[int4] {variant} INT4 quantization complete. {len(results)}/{len(LARGE_MODULES)} modules.")
    return results


def main():
    parser = argparse.ArgumentParser(description="INT4 quantize large SA3 ONNX modules")
    parser.add_argument("--variant", choices=["music", "sfx", "both"], default="both")
    args = parser.parse_args()

    variants = ["music", "sfx"] if args.variant == "both" else [args.variant]
    for v in variants:
        quantize_int4(v)


if __name__ == "__main__":
    main()
