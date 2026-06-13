# stable_audio_onnx/__init__.py
"""
Stable Audio 3 Small — ONNX Export Pipeline
Multi-hardware: NVIDIA CUDA · AMD ROCm · Intel OpenVINO · DirectML · CPU fallback
"""

from .device_selector import create_session, get_hardware_tier

__all__ = ["create_session", "get_hardware_tier"]
