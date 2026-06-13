# validate/validate_single.py
"""
Validates a single exported ONNX module by:
  1. Loading it on CPU-only ORT session (reproducible, hardware-independent)
  2. Building matching random inputs from the session's input metadata
  3. Running inference
  4. Checking for NaN / Inf in outputs
  5. Printing max output magnitude as a sanity signal

Tolerance thresholds are defined in config.TOLERANCE — do NOT tighten them.
FP16 export on CPU may accumulate error up to 1e-2; that is expected.

For a full PyTorch-vs-ONNX numerical comparison see validate/validate_pipeline.py.
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TOLERANCE


def _make_feeds(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    """
    Build a dictionary of random numpy inputs from the session's input metadata.

    Shape policy:
      - Named dynamic dims ("batch", "seq_len", "latent_t", etc.) → resolved to 1
      - Negative / zero integer dims               → resolved to 1
      - Positive integer dims                      → kept as-is

    dtype policy:
      - int* / bool* → zeros (int64)
      - float* / double* → standard normal (float32)
    """
    feeds: dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        shape = [
            d if isinstance(d, int) and d > 0 else 1
            for d in inp.shape
        ]
        type_str = inp.type.lower()
        if "int" in type_str or "bool" in type_str:
            if "mask" in inp.name.lower():
                feeds[inp.name] = np.ones(shape, dtype=np.int64)
            else:
                feeds[inp.name] = np.zeros(shape, dtype=np.int64)
        else:
            feeds[inp.name] = np.random.randn(*shape).astype(np.float32)
    return feeds


def validate_module(
    name: str,
    variant: str,
    onnx_path: str,
    dtype_key: str = "fp16",
) -> bool:
    """
    Load and run an exported ONNX module; check output for NaN/Inf.

    Args:
        name:       Human-readable module name (e.g. "dit", "decoder").
        variant:    "music" or "sfx" (used for logging only).
        onnx_path:  Absolute path to the .onnx file.
        dtype_key:  Key into config.TOLERANCE ("fp16", "int8", "int4").

    Returns:
        True if validation passed.

    Raises:
        FileNotFoundError: If the .onnx file does not exist.
        ValueError:        If outputs contain NaN or Inf.
    """
    if not Path(onnx_path).exists():
        raise FileNotFoundError(
            f"[validate] {name} ({variant}): ONNX file not found at {onnx_path}"
        )

    tol = TOLERANCE.get(dtype_key, 1e-2)
    print(f"\n[validate] {name} ({variant}, {dtype_key}) — tolerance={tol}")

    # Always validate on CPU for reproducibility
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    feeds   = _make_feeds(session)
    outputs = session.run(None, feeds)

    all_passed = True
    for i, out in enumerate(outputs):
        out_name = session.get_outputs()[i].name if i < len(session.get_outputs()) else f"output_{i}"

        has_nan = bool(np.isnan(out).any())
        has_inf = bool(np.isinf(out).any())
        max_mag = float(np.abs(out).max()) if out.size > 0 else 0.0

        status_icon = "✓" if not has_nan and not has_inf else "✗"
        print(
            f"  [{status_icon}] {out_name:<20} shape={list(out.shape)} "
            f"dtype={out.dtype}  max_magnitude={max_mag:.4f}"
        )

        if has_nan:
            print(f"  [ERROR] {out_name} contains NaN — export is broken.")
            all_passed = False
        if has_inf:
            print(f"  [ERROR] {out_name} contains Inf — possible FP16 overflow.")
            all_passed = False

    if not all_passed:
        raise ValueError(
            f"[validate] {name} ({variant}) FAILED — NaN or Inf detected in outputs. "
            "See agent.md §F4 for fix (FP32 export + onnxconverter_common)."
        )

    print(f"[validate] {name} ({variant}) ✓  ONNX loads and runs cleanly.")
    return True
