"""Reusable scaled-dot-product attention adapters.

The implementation is intentionally model-name agnostic.  Architectures may
opt in when their query head count is an integer multiple of the KV head
count.  Unlike Transformers' generic SDPA adapter, the common CCCP path keeps
native grouped-query attention enabled when a StaticCache mask is present;
modern PyTorch CUDA and CPU backends both implement that exact operation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


_TRANSFORMERS_ATTENTION_NAME = "cccp-native-gqa-sdpa"
_DYNAMIC_SDPA_DEVICES: set[tuple[str, int | None]] = set()


def configure_dynamic_sdpa_backends(device: torch.device | str) -> str:
    """Avoid per-sequence-length cuDNN planning for autoregressive Decode.

    Flash and memory-efficient SDPA accept changing KV lengths without a new
    host-side frontend plan.  This process-wide policy is capability based,
    idempotent and shared by every Transformers-backed architecture.
    """
    resolved = torch.device(device)
    if resolved.type != "cuda" or torch.version.hip is not None:
        return "native"
    key = (resolved.type, resolved.index)
    if key not in _DYNAMIC_SDPA_DEVICES:
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        _DYNAMIC_SDPA_DEVICES.add(key)
        print(
            "[cccp-attention] dynamic-kv backend=flash-or-efficient; "
            "cudnn-sdpa=disabled",
            flush=True,
        )
    return "flash-or-efficient"


def _native_gqa_compatible(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> bool:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    query_heads = int(query.shape[1])
    key_heads = int(key.shape[1])
    return (
        query.device.type in {"cpu", "cuda"}
        and query.device == key.device == value.device
        and key_heads > 0
        and query_heads > key_heads
        and query_heads % key_heads == 0
        and int(value.shape[1]) == key_heads
        and int(key.shape[-1]) == int(value.shape[-1]) <= 256
    )


def native_gqa_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Run SDPA without materialising repeated KV heads when supported."""
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=float(dropout),
        scale=scaling,
        is_causal=bool(is_causal),
        enable_gqa=_native_gqa_compatible(query, key, value),
    )


def transformers_native_gqa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    position_bias: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Transformers AttentionInterface entry backed by the common operator."""
    from transformers.integrations.sdpa_attention import (
        create_position_bias_mask,
        sdpa_attention_forward,
    )

    if kwargs.get("output_attentions", False):
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            position_bias=position_bias,
            **kwargs,
        )
    if not _native_gqa_compatible(query, key, value):
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            position_bias=position_bias,
            **kwargs,
        )

    q_length = int(query.shape[2])
    kv_length = int(key.shape[2])
    causal = (
        bool(is_causal)
        if is_causal is not None
        else bool(getattr(module, "is_causal", True))
    )
    causal = q_length > 1 and attention_mask is None and causal
    if causal and q_length > 1 and kv_length > q_length:
        key = key[:, :, :q_length, :]
        value = value[:, :, :q_length, :]
        if position_bias is not None:
            position_bias = position_bias[..., :q_length]
    if position_bias is not None:
        attention_mask = create_position_bias_mask(
            position_bias,
            attention_mask,
            causal,
            query,
            key,
        )
        causal = False
    result = native_gqa_sdpa(
        query,
        key,
        value,
        attention_mask=attention_mask,
        dropout=dropout,
        scaling=scaling,
        is_causal=causal,
    )
    return result.transpose(1, 2).contiguous(), None


def register_transformers_native_gqa_attention() -> str:
    """Idempotently expose the common operator to Transformers models."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if _TRANSFORMERS_ATTENTION_NAME not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(
            _TRANSFORMERS_ATTENTION_NAME,
            transformers_native_gqa_attention_forward,
        )
    return _TRANSFORMERS_ATTENTION_NAME


__all__ = [
    "configure_dynamic_sdpa_backends",
    "native_gqa_sdpa",
    "register_transformers_native_gqa_attention",
    "transformers_native_gqa_attention_forward",
]
