# patches/attention_patch.py
"""
Replaces torch.nn.functional ops with ONNX opset-18 compatible versions
before any torch.onnx.export() call.

Patches applied:
  1. scaled_dot_product_attention -> MatMul + Softmax
     (aten::sdpa has no opset-18 symbolic; FlashAttention not ONNX-compatible)
  2. rms_norm -> pow/mean/rsqrt/mul decomposition
     (aten::rms_norm has no opset-18 symbolic in TorchScript exporter)

Usage:
    from patches.attention_patch import apply_attention_patch
    apply_attention_patch()   # call ONCE before any export
"""

import torch
import torch.nn.functional as F


def _onnx_safe_sdpa(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale=None,
):
    """
    Drop-in replacement for F.scaled_dot_product_attention.
    Uses standard MatMul + Softmax -- fully ONNX opset 18 compatible.
    """
    import math

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale

    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    if is_causal:
        assert attn_mask is None, (
            "Cannot use is_causal=True together with an explicit attn_mask."
        )
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril()
        attn_bias.masked_fill_(~temp_mask, float("-inf"))
        attn_bias = attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = attn_bias.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_bias = attn_bias + attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight = attn_weight + attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    if dropout_p > 0.0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    return attn_weight @ value


def _onnx_safe_rms_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """
    Drop-in replacement for F.rms_norm.

    aten::rms_norm has no ONNX opset-18 symbolic handler in the TorchScript
    exporter. Decomposes into basic ops that trace cleanly:
        variance = mean(x^2, dims)
        output   = x * rsqrt(variance + eps) * weight

    Args:
        input:            Input tensor (..., *normalized_shape)
        normalized_shape: Shape of the trailing dims to normalize over
        weight:           Optional learnable scale (gamma)
        bias:             Optional bias (not used in RMSNorm, kept for compat)
        eps:              Numerical stability constant
    """
    dims = list(range(-len(normalized_shape), 0))
    variance = input.pow(2).mean(dims, keepdim=True)
    output = input * torch.rsqrt(variance + eps)
    if weight is not None:
        output = output * weight
    if bias is not None:
        output = output + bias
    return output


def _onnx_safe_zero_pad_modulo_sequence(x, size, dim=-2):
    """
    ONNX-safe version of _zero_pad_modulo_sequence that avoids python control flow
    based on dynamic shapes (which bakes static shapes into the ONNX graph).
    """
    if dim < 0:
        dim = x.ndim + dim

    input_len = x.shape[dim]
    pad_len = (size - input_len % size) % size

    max_pad = size - 1
    if max_pad <= 0:
        return x

    pad_shape = list(x.shape)
    pad_shape[dim] = max_pad

    zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    padded_zeros = torch.narrow(zeros, dim, 0, pad_len)
    return torch.cat([x, padded_zeros], dim=dim)


def apply_attention_patch() -> None:
    """
    Monkey-patches F.scaled_dot_product_attention, F.rms_norm, and
    autoencoders._zero_pad_modulo_sequence with ONNX-safe implementations.

    Call ONCE before any torch.onnx.export(). Idempotent -- safe to call
    multiple times.
    """
    if getattr(F, "_onnx_patch_applied", False):
        import logging
        logging.getLogger(__name__).debug("[attention_patch] Already applied -- skipping.")
        return

    # 1. SDPA -> MatMul + Softmax
    F.scaled_dot_product_attention = _onnx_safe_sdpa
    torch.nn.functional.scaled_dot_product_attention = _onnx_safe_sdpa

    # 2. rms_norm -> pow/mean/rsqrt/mul
    F.rms_norm = _onnx_safe_rms_norm
    torch.nn.functional.rms_norm = _onnx_safe_rms_norm

    # 3. autoencoders._zero_pad_modulo_sequence -> ONNX-safe slice/pad
    try:
        import stable_audio_tools.models.autoencoders as autoencoders
        autoencoders._zero_pad_modulo_sequence = _onnx_safe_zero_pad_modulo_sequence
        print("[attention_patch] _zero_pad_modulo_sequence replaced with ONNX-safe slice/pad implementation.")
    except ImportError:
        print("[attention_patch] WARNING: Could not import stable_audio_tools.models.autoencoders to patch _zero_pad_modulo_sequence.")

    F._onnx_patch_applied = True  # type: ignore[attr-defined]

    print("[attention_patch] SDPA replaced with ONNX-safe MatMul+Softmax implementation.")
    print("[attention_patch] rms_norm replaced with ONNX-safe pow/mean/rsqrt decomposition.")
