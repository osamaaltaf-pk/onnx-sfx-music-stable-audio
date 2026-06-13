# patches/attention_patch.py
"""
Replaces torch.nn.functional.scaled_dot_product_attention (SDPA) with an
ONNX opset-18 compatible implementation before any torch.onnx.export() call.

SA3 uses SDPA internally which normally exports a FlashAttention node —
not supported in ONNX opset 18. This patch replaces it with a standard
MatMul + Softmax sequence that traces cleanly.

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
    Uses standard MatMul + Softmax — fully ONNX opset 18 compatible.

    Args:
        query:      (..., L, E)
        key:        (..., S, E)
        value:      (..., S, Ev)
        attn_mask:  Optional mask; bool tensors become additive -inf masks.
        dropout_p:  Applied to attention weights during training.
        is_causal:  If True, applies causal (lower-triangular) masking.
        scale:      Optional explicit scale factor; defaults to 1/sqrt(E).

    Returns:
        Attention output (..., L, Ev)
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


def apply_attention_patch() -> None:
    """
    Monkey-patches F.scaled_dot_product_attention with the ONNX-safe version.

    Call this ONCE before any torch.onnx.export() call.
    Safe to call multiple times — subsequent calls are no-ops logged at DEBUG.
    """
    if getattr(F, "_onnx_patch_applied", False):
        # Already patched; avoid double-patching
        import logging
        logging.getLogger(__name__).debug(
            "[attention_patch] Already applied — skipping."
        )
        return

    F.scaled_dot_product_attention = _onnx_safe_sdpa
    torch.nn.functional.scaled_dot_product_attention = _onnx_safe_sdpa

    # Mark so re-entry is a no-op
    F._onnx_patch_applied = True  # type: ignore[attr-defined]

    print("[attention_patch] SDPA replaced with ONNX-safe MatMul+Softmax implementation.")
