# Stable Audio 3 Small — ONNX Export Pipeline

> **Open-source ONNX conversion pipeline for Stability AI's Stable Audio 3 Small models.**  
> Exports `stabilityai/stable-audio-3-small-music` and `stabilityai/stable-audio-3-small-sfx` to ONNX format for CPU/GPU inference without a PyTorch runtime dependency.

---

## What This Does

This repo converts the Stable Audio 3 Small diffusion pipeline into four standalone ONNX modules:

| Module | Input | Output | File |
|--------|-------|--------|------|
| **Text Encoder** | Prompt string | Text embeddings | `text_encoder.onnx` |
| **Conditioner** | Text embeddings | Cross-attention context | `conditioner.onnx` |
| **DiT (Diffusion Transformer)** | Noisy latents + context | Denoised latents | `dit.onnx` |
| **Decoder (Oobleck)** | Latents | Audio waveform | `decoder.onnx` |

Each module is exported for both `music` and `sfx` variants.

---

## Requirements

- **Python 3.10–3.12** (tested on **3.12.10 on Windows**)
- **RAM:** ≥ 16 GB recommended (model loading is CPU-heavy)
- **Disk:** ≥ 20 GB free (models + ONNX outputs)
- **HuggingFace account** with access to the SA3 Small models (free, see below)

> ⚠️ **GPU is NOT required for export.** The export traces the model graph — it runs on CPU. A GPU (CUDA/DirectML) is only needed for fast *inference* with the exported ONNX files.

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/osamaaltaf-pk/onnx-sfx-music-stable-audio.git
cd onnx-sfx-music-stable-audio
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate
```

### 3. Install PyTorch first (important — do this before requirements.txt)

**CPU only (any machine):**
```bash
pip install torch==2.12.0 torchaudio==2.11.0 torchvision==0.27.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch==2.12.0 torchaudio==2.11.0 torchvision==0.27.0 \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

### 4. Install stable-audio-tools from GitHub

> **Why GitHub and not PyPI?** PyPI version `0.0.19` does not support the `local_add_cond_dim` argument used by SA3 Small models. You must install from the GitHub `main` branch.

```bash
pip install git+https://github.com/Stability-AI/stable-audio-tools.git \
    --ignore-requires-python --no-deps
```

### 5. Install all other dependencies

```bash
pip install -r requirements.txt
```

> 💡 All conflicts are pre-resolved in `requirements.txt` — no manual debugging needed.

### 6. Get HuggingFace access to the models

The SA3 Small models are gated (you must accept the license once):

1. Log in to [huggingface.co](https://huggingface.co)
2. Visit and accept the license for **both** models:
   - [stabilityai/stable-audio-3-small-music](https://huggingface.co/stabilityai/stable-audio-3-small-music)
   - [stabilityai/stable-audio-3-small-sfx](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)
3. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Read scope is enough)
4. Log in locally:

```bash
huggingface-cli login
# paste your hf_... token when prompted
```

### 7. Run the export

```bash
# Export both music and SFX variants
python -m export.export_all --variant both

# Or export a single variant
python -m export.export_all --variant music
python -m export.export_all --variant sfx
```

The first run downloads ~4–8 GB of model weights. Subsequent runs use the local HuggingFace cache.

### 8. Validate the outputs

```bash
# Validate a single module
python -m validate.validate_single --module dit --variant music

# Full pipeline validation (runs a real denoising pass)
python -m validate.validate_pipeline --variant music
```

---

## Output Structure

After a successful export:

```
models/
├── music/
│   ├── fp16/
│   │   ├── text_encoder.onnx
│   │   ├── conditioner.onnx
│   │   ├── dit.onnx
│   │   └── decoder.onnx
│   └── int8/
│       └── ... (quantized variants)
└── sfx/
    ├── fp16/
    │   ├── text_encoder.onnx
    │   ├── conditioner.onnx
    │   ├── dit.onnx
    │   └── decoder.onnx
    └── int8/
        └── ...
```

---

## Using a Local Checkpoint (Skip Hub Download)

If you already have model weights downloaded locally, set environment variables to point to them:

```bash
# Windows PowerShell
$env:SA3_MUSIC_CKPT = "C:\path\to\music_model.ckpt"
$env:SA3_SFX_CKPT   = "C:\path\to\sfx_model.ckpt"

# Linux / Mac
export SA3_MUSIC_CKPT="/path/to/music_model.ckpt"
export SA3_SFX_CKPT="/path/to/sfx_model.ckpt"

python -m export.export_all --variant both
```

---

## Configuration

Edit [`config.py`](config.py) to change model IDs, output paths, ONNX opset version, or validation tolerances:

```python
MODEL_IDS = {
    "music": "stabilityai/stable-audio-3-small-music",
    "sfx":   "stabilityai/stable-audio-3-small-sfx",
}

OPSET_VERSION   = 18
BATCH_SIZE      = 1
MAX_SEQ_LEN     = 512
SAMPLE_RATE     = 44100
OUTPUT_ROOT     = "models"
```

---

## Dependency Conflict Reference

> **For contributors and troubleshooters.** The following conflicts were encountered and resolved during development on Python 3.12 + Windows. The pinned `requirements.txt` already bakes these in so you shouldn't hit them, but this documents *why* specific versions were chosen.

| Conflict | Root Cause | Resolution |
|----------|-----------|------------|
| `PyWavelets==1.4.1` build fails on Python 3.12 | Old `setuptools` in isolated build env uses removed `pkgutil.ImpImporter` | Pin to `PyWavelets==1.9.0` (has prebuilt wheel, backward-compatible) |
| `sentencepiece==0.1.99` build fails on Python 3.12 | Same `pkgutil.ImpImporter` issue | Upgrade to `sentencepiece==0.2.1` |
| `pandas==2.0.2` fails | No `cp312` wheel for that exact version | Upgrade to `pandas>=3.0.0` |
| `onnxruntime==1.18.0` crashes with numpy 2.x | Built against NumPy 1.x ABI | Upgrade to `onnxruntime==1.26.0` which supports numpy 2.x |
| `descript-audiotools` vs `onnx` protobuf conflict | audiotools pins `protobuf<3.20`; onnx needs `>=3.20.2` | Install audiotools with `--no-deps` to skip the pin |
| `k_diffusion` fails to import (`skimage`, `jsonmerge`, `dctorch`, `trampoline`, `ftfy`) | k-diffusion eagerly imports ALL submodules including evaluation ones | Install all k-diffusion sub-deps; patch `k_diffusion/__init__.py` to catch ImportError on `evaluation` and `external` |
| `stable-audio-tools 0.0.19` → `TransformerBlock.__init__() got unexpected keyword argument 'local_add_cond_dim'` | PyPI 0.0.19 is too old for SA3 Small architecture | Install from GitHub main (`0.0.20+`) with `--ignore-requires-python` |
| `transformers==4.40.0` → `T5GemmaEncoderModel` not found | SA3 Small uses Gemma-based text encoder added in transformers 5.x | Upgrade to `transformers==5.12.0` |
| `transformers==5.12.0` → `is_offline_mode` missing from `huggingface_hub` | `huggingface_hub 0.36.x` removed the function | Upgrade to `huggingface_hub==1.19.0` + `accelerate==1.14.0` |
| `transformers==5.12.0` → `No module named 'httpx'` / `httpcore` | transformers 5.x uses httpx; we used `--no-deps` earlier | Install `httpx`, `httpcore`, `anyio`, `sniffio`, `h11` |
| `tokenizers==0.23.1` rejected by transformers | transformers 5.12.0 requires `>=0.22.0,<=0.23.0`; 0.23.0 doesn't exist | Use `tokenizers==0.22.2` |

---

## FAQ

**Q: Do I need a GPU to export?**  
A: No. Export runs on CPU. It's slow (several minutes per module) but works fine.

**Q: Can I use the ONNX files without PyTorch?**  
A: Yes — that's the whole point. ONNX Runtime runs the exported files on any platform with zero PyTorch dependency.

**Q: Will this work on Linux / Mac?**  
A: Yes. The pinned requirements work on Linux and Mac. For Mac Apple Silicon, change the `torch` install index to `cpu` (there's no dedicated `whl/cpu` for arm64, just install from PyPI directly).

**Q: The export takes too long on CPU. How do I speed it up?**  
A: If you have a GPU, install `onnxruntime-gpu` instead of `onnxruntime`. Alternatively, run the export on Google Colab (free T4 GPU). A Colab notebook is included in this repo: [`StableAudio3_ONNX_Export.ipynb`](StableAudio3_ONNX_Export.ipynb).

**Q: The model download keeps failing. What do I do?**  
A: Make sure you accepted the license on the HuggingFace model page and are logged in with `huggingface-cli login`. Also check you have ≥ 8 GB free on the drive that hosts `~/.cache/huggingface`.

---

## Project Structure

```
stable_audio_onnx/
├── config.py                        # Model IDs, opset, output paths
├── requirements.txt                 # Pinned, conflict-resolved deps
├── export/
│   ├── export_all.py                # Main entry point (--variant music/sfx/both)
│   ├── export_text_encoder.py       # T5/Gemma text encoder → ONNX
│   ├── export_conditioner.py        # Multi-conditioner → ONNX
│   ├── export_dit.py                # Diffusion Transformer → ONNX
│   └── export_decoder.py            # Oobleck VAE decoder → ONNX
├── validate/
│   ├── validate_single.py           # Single-module output check
│   └── validate_pipeline.py         # End-to-end denoising check
├── patches/
│   └── attention_patch.py           # Replaces SDPA with ONNX-safe MatMul+Softmax
├── inference/
│   └── ort_pipeline.py              # ONNX Runtime inference pipeline
├── quantization/
│   └── quantize.py                  # INT8/INT4 post-export quantization
├── device_selector.py               # Auto-detects best ORT execution provider
└── StableAudio3_ONNX_Export.ipynb   # Google Colab notebook (GPU export)
```

---

## License

This export pipeline is released under the **MIT License**.  
The Stable Audio 3 Small model weights are subject to [Stability AI's model license](https://huggingface.co/stabilityai/stable-audio-3-small-music).

---

## Credits

- [Stability AI](https://stability.ai) — Stable Audio 3 Small models
- [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) — model loading and architecture
- [ONNX Runtime](https://onnxruntime.ai) — cross-platform inference
