# config.py
import os

MODEL_IDS = {
    "music": "stabilityai/stable-audio-3-small-music",
    "sfx":   "stabilityai/stable-audio-3-small-sfx",
}

LOCAL_CHECKPOINT_PATHS = {
    "music": os.getenv("SA3_MUSIC_CKPT", ""),
    "sfx":   os.getenv("SA3_SFX_CKPT",   ""),
}

OPSET_VERSION   = 18
BATCH_SIZE      = 1
MAX_SEQ_LEN     = 512      # text encoder max tokens
LATENT_CHANNELS = 64       # SA3 Small latent dim — verify against model config
SAMPLE_RATE     = 44100    # SA3 target sample rate

OUTPUT_ROOT = "models"

# Validation tolerances — do NOT tighten these
TOLERANCE = {
    "fp16": 1e-2,
    "int8": 5e-2,
    "int4": 1e-1,
}

# Diffusion steps budget by hardware tier
DIFFUSION_STEPS = {
    "gpu_high":  50,   # RTX 3000+, cloud A10/A100
    "gpu_low":   30,   # GTX 2000 series, Intel Arc
    "cpu":       20,   # fallback
}
