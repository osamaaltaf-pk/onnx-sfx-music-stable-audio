# export/export_all.py
"""
Orchestrates all 4 ONNX exports in order for one or both SA3 variants.

Execution order per variant:
  1. text_encoder  — T5-based text encoder
  2. conditioner   — duration NumberConditioner (seconds_start + seconds_total)
  3. dit           — Diffusion Transformer (heaviest, takes longest)
  4. decoder       — Oobleck latent → waveform decoder

Each module is validated immediately after export via validate_single.
A summary of all output paths is printed at the end.

Usage:
    python -m export.export_all --variant music
    python -m export.export_all --variant sfx
    python -m export.export_all --variant both   (default)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export.export_text_encoder import export_text_encoder
from export.export_conditioner   import export_conditioner
from export.export_dit           import export_dit
from export.export_decoder       import export_decoder
from validate.validate_single    import validate_module

# Ordered export registry: (name, export_function)
EXPORT_ORDER = [
    ("text_encoder", export_text_encoder),
    ("conditioner",  export_conditioner),
    ("dit",          export_dit),
    ("decoder",      export_decoder),
]


def run(variant: str) -> dict[str, str]:
    """
    Export and validate all 4 modules for a single variant.

    Args:
        variant: "music" or "sfx"

    Returns:
        dict mapping module name → exported ONNX path
    """
    print(f"\n{'='*60}")
    print(f" Exporting variant: {variant.upper()}")
    print(f"{'='*60}")

    paths: dict[str, str] = {}
    failed: list[str]     = []

    for name, export_fn in EXPORT_ORDER:
        print(f"\n[{name}] Starting export...")
        t0 = time.time()
        try:
            path = export_fn(variant)
            elapsed = time.time() - t0

            if path is None:
                print(f"[{name}] Skipped (module not found in model).")
                continue

            print(f"[{name}] Export completed in {elapsed:.1f}s")

            # Validate immediately after export
            try:
                validate_module(name, variant, path, dtype_key="fp16")
            except Exception as val_err:
                print(f"[{name}] WARNING: Validation failed — {val_err}")
                # Don't abort the whole run for a validation failure
                failed.append(f"{name} (validation)")

            paths[name] = path

        except Exception as exp_err:
            elapsed = time.time() - t0
            print(f"[{name}] ERROR after {elapsed:.1f}s: {exp_err}")
            import traceback
            traceback.print_exc()
            failed.append(name)

    # Summary
    print(f"\n{'='*60}")
    print(f" {variant.upper()} export summary")
    print(f"{'='*60}")
    for name, path in paths.items():
        status = "[OK]" if name not in failed else "[WARN]"
        print(f"  {status:<8} {name:<16} -> {path}")
    if failed:
        print(f"\n  Failed / warnings: {', '.join(failed)}")
    else:
        print("\n  All modules exported and validated successfully.")
    print()

    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Export all Stable Audio 3 modules to ONNX"
    )
    parser.add_argument(
        "--variant",
        choices=["music", "sfx", "both"],
        default="both",
        help="Which model variant to export (default: both)",
    )
    args = parser.parse_args()

    variants = ["music", "sfx"] if args.variant == "both" else [args.variant]
    all_paths: dict[str, dict] = {}

    t_start = time.time()
    for v in variants:
        all_paths[v] = run(v)

    total = time.time() - t_start
    print(f"Total export time: {total:.1f}s")
    return all_paths


if __name__ == "__main__":
    main()
