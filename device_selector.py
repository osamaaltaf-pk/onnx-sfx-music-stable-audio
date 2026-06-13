# device_selector.py
"""
Detects available hardware and returns the best OnnxRuntime InferenceSession.
Priority: NVIDIA CUDA → AMD ROCm → Intel OpenVINO GPU → Intel OpenVINO CPU → ORT CPU

Silent fallback — never raises on missing hardware.
"""

import logging
import onnxruntime as ort

log = logging.getLogger(__name__)

# Ordered from best to worst
EP_PRIORITY = [
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "OpenVINOExecutionProvider",   # handles both Intel GPU and CPU
    "DmlExecutionProvider",        # DirectML — Intel/AMD GPU on Windows
    "CPUExecutionProvider",        # always available
]


def _get_available_eps() -> list[str]:
    return ort.get_available_providers()


def _build_ep_list(prefer_gpu: bool = True) -> list[tuple]:
    """
    Returns a prioritised EP list that ORT will try in order.
    ORT stops at the first EP that loads successfully.
    """
    available = set(_get_available_eps())
    selected  = []

    if prefer_gpu:
        if "CUDAExecutionProvider" in available:
            selected.append((
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 4 * 1024 ** 3,   # 4 GB cap for GTX 2000 safety
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": True,
                }
            ))

        if "ROCMExecutionProvider" in available:
            selected.append((
                "ROCMExecutionProvider",
                {"device_id": 0}
            ))

        if "OpenVINOExecutionProvider" in available:
            # Try Intel GPU first, CPU OpenVINO second
            selected.append((
                "OpenVINOExecutionProvider",
                {"device_type": "GPU_FP16", "cache_dir": ".ov_cache"}
            ))
            selected.append((
                "OpenVINOExecutionProvider",
                {"device_type": "CPU_FP32", "cache_dir": ".ov_cache"}
            ))

        if "DmlExecutionProvider" in available:
            selected.append(("DmlExecutionProvider", {"device_id": 0}))

    # CPU always last
    selected.append(("CPUExecutionProvider", {}))
    return selected


def create_session(model_path: str, prefer_gpu: bool = True) -> ort.InferenceSession:
    """
    Create an ORT InferenceSession on the best available hardware.
    Silently falls back down the EP chain on any failure.
    """
    ep_list = _build_ep_list(prefer_gpu)

    for i, ep_config in enumerate(ep_list):
        ep_name = ep_config[0] if isinstance(ep_config, tuple) else ep_config
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.enable_mem_pattern       = True
            opts.enable_cpu_mem_arena     = True

            session = ort.InferenceSession(
                model_path,
                sess_options=opts,
                providers=[ep_config] + [ep_list[-1]]  # chosen EP + CPU fallback
            )

            actual_ep = session.get_providers()[0]
            if actual_ep == "CPUExecutionProvider" and i == 0:
                log.warning("Requested %s but got CPU — EP not functional", ep_name)
            else:
                log.info("Session created on: %s", actual_ep)

            return session

        except Exception as exc:
            log.debug("EP %s failed (%s), trying next", ep_name, exc)
            continue

    # Should never reach here because CPUExecutionProvider always works
    raise RuntimeError("All execution providers failed — corrupted ORT install?")


def get_hardware_tier() -> str:
    """
    Returns 'gpu_high', 'gpu_low', or 'cpu'.
    Used by inference pipeline to pick diffusion step budget.
    """
    available = set(_get_available_eps())
    if "CUDAExecutionProvider" in available:
        try:
            import torch
            cc = torch.cuda.get_device_capability()
            return "gpu_high" if cc[0] >= 7 else "gpu_low"  # SM 7.0 = Turing/RTX 2000
        except Exception:
            return "gpu_low"
    if "ROCMExecutionProvider" in available or "OpenVINOExecutionProvider" in available:
        return "gpu_low"
    return "cpu"
