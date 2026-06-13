# Stable Audio 3 Small — ONNX Export Pipeline

Convert Stability AI's **Stable Audio 3 Small** (Music + SFX) into ONNX format
with automatic multi-hardware execution provider selection at inference time.

**EP priority at runtime:**
`NVIDIA CUDA → AMD ROCm → Intel OpenVINO GPU → Intel OpenVINO CPU → DirectML → CPU`

No user action required. No crash if GPU is absent.

---

## Directory Layout

```
stable_audio_onnx/
│
├── config.py                   ← model IDs, paths, shared constants
├── device_selector.py          ← EP detection and session factory  ← CRITICAL
│
├── export/
│   ├── export_text_encoder.py  ← T5 text encoder
│   ├── export_conditioner.py   ← duration NumberConditioner
│   ├── export_dit.py           ← Diffusion Transformer (DiT)
│   ├── export_decoder.py       ← Oobleck audio decoder
│   └── export_all.py           ← orchestrates all 4 in order
│
├── patches/
│   └── attention_patch.py      ← SDPA → ONNX-safe MatMul+Softmax
│
├── quantization/
│   ├── int8_quantize.py        ← dynamic INT8 (all modules)
│   └── int4_quantize.py        ← weight-only INT4 (dit + decoder)
│
├── validate/
│   ├── validate_single.py      ← per-module NaN/shape check
│   └── validate_pipeline.py    ← end-to-end spectral similarity
│
├── inference/
│   └── run_inference.py        ← full ONNX pipeline + WAV output
│
├── models/                     ← created by export scripts
│   ├── music/fp16|int8|int4/
│   └── sfx/fp16|int8|int4/
│
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
# Core (adjust --index-url for ROCm: whl/rocm6.0)
pip install torch==2.2.2 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

pip install transformers==4.40.0 diffusers==0.27.2 accelerate==0.29.3
pip install stable-audio-tools
pip install onnx==1.16.0

# Pick ONE onnxruntime package for your hardware:
pip install onnxruntime==1.18.0          # CPU
pip install onnxruntime-gpu==1.18.0      # NVIDIA CUDA / AMD ROCm
pip install onnxruntime-directml==1.18.0 # Intel/AMD GPU on Windows

# Audio output
pip install soundfile numpy==1.26.4 scipy
```

### 2. Export models

```bash
# Export both music and sfx variants
python -m export.export_all --variant both

# Or export one at a time
python -m export.export_all --variant sfx
python -m export.export_all --variant music
```

### 3. Quantize (optional but recommended)

```bash
# INT8 — all modules, all hardware
python -m quantization.int8_quantize --variant both

# INT4 — dit + decoder only (largest modules)
python -m quantization.int4_quantize --variant both
```

### 4. Run inference

```bash
# Sound effect
python -m inference.run_inference \
    --prompt "A short thunder clap" \
    --duration 3.0 \
    --variant sfx \
    --out thunder.wav

# Music
python -m inference.run_inference \
    --prompt "Upbeat jazz piano riff" \
    --duration 10.0 \
    --variant music \
    --out jazz.wav
```

### 5. Validate

```bash
# Single module validation (fast — no model download needed after export)
python -m validate.validate_single   # called automatically by export_all

# End-to-end pipeline comparison (slow — loads PyTorch model again)
python -m validate.validate_pipeline --variant sfx
```

---

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `SA3_MUSIC_CKPT` | Local path to music model checkpoint | `""` (use Hub) |
| `SA3_SFX_CKPT`   | Local path to SFX model checkpoint   | `""` (use Hub) |
| `SA3_TOKENIZER_ID` | HuggingFace tokenizer ID for text encoder | `google/flan-t5-large` |

---

## Performance Targets

| Hardware | EP | Precision | 5s Audio (steps=30) |
|---|---|---|---|
| RTX 2060–2080 | CUDAExecutionProvider | INT8 | ~8–15 s |
| RTX 3000–4000 | CUDAExecutionProvider | FP16 | ~3–8 s |
| Cloud A10/A100 | CUDAExecutionProvider | FP16 | ~1–3 s |
| AMD RX 6000+ | ROCMExecutionProvider | INT8 | ~10–20 s |
| Intel Arc A770 | OpenVINOExecutionProvider GPU_FP16 | FP16 | ~15–25 s |
| Intel i5–i9 (no GPU) | OpenVINOExecutionProvider CPU | INT8 | ~60–120 s |
| Intel i5 7th Gen | CPUExecutionProvider | INT8 | ~120–240 s |

SFX variant (~1–2 s clips) runs ~3–4× faster than Music at the same step count.

---

## Failure Recovery

See the full playbook in `agent.md`. Quick reference:

| Code | Symptom | Fix |
|---|---|---|
| F1 | `Unsupported op: ScaledDotProductAttention` | `apply_attention_patch()` was not called before export |
| F2 | Dynamic shape inference failed on DiT | Wrap `x.shape[-1]` calls as `torch.Tensor` ops |
| F3 | `/LayerNorm` ORT crash | Add `operator_export_type=ONNX_FALLTHROUGH` to export |
| F4 | NaN in ONNX, not PyTorch | Export FP32 + convert with `onnxconverter_common.float16` |
| F5 | CUDA slower than CPU | Use INT8 on GTX 2000 (Turing) — FP16 tensor core limited |
| F6 | OpenVINO EP fails to load | Run `mo --input_model dit.onnx` to inspect unsupported ops |
| F7 | Hub 404 | Check HuggingFace for current model ID, set `SA3_MUSIC_CKPT` |

---

## Success Criteria

- [ ] All 4 modules export without graph breaks  
- [ ] `validate_single` passes (no NaN) for all 4 modules at fp16 tolerance (1e-2)  
- [ ] `validate_pipeline` spectral similarity > 0.85  
- [ ] `device_selector` silently falls to CPU when no GPU present  
- [ ] INT8 runs on CPU and CUDA EPs  
- [ ] No NaN in any ONNX output  
- [ ] Total RAM < 2 GB for INT8 pipeline on CPU  
- [ ] `run_inference.py` produces a playable WAV file  
