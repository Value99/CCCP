"""Model-independent block-prefill attention implementations."""

from __future__ import annotations

import os

import torch


def _positive_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = int(default)
    return max(1, value)


def causal_latent_prefill(
    *,
    query_nope: torch.Tensor,
    query_rope: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    query_start: int,
    scale_denominator: float,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate a causal MQA latent-attention block with bounded scores.

    Queries are head-sharded while the latent and RoPE caches are shared by
    all local heads. Query rows are micro-batched, but every micro-batch sees
    its exact absolute-position prefix. This keeps the outer model block large
    without materialising an unbounded ``[H,T,S]`` score tensor.
    """
    if (
        query_nope.ndim != 3
        or query_rope.ndim != 3
        or query_nope.shape[:2] != query_rope.shape[:2]
        or latent_cache.ndim != 2
        or rope_cache.ndim != 2
        or query_nope.shape[2] != latent_cache.shape[1]
        or query_rope.shape[2] != rope_cache.shape[1]
        or latent_cache.shape[0] != rope_cache.shape[0]
    ):
        raise ValueError("causal latent prefill tensor shapes do not match")
    if query_start < 0 or scale_denominator <= 0.0:
        raise ValueError("invalid causal latent prefill position or scale")
    rows, heads, latent_dim = map(int, query_nope.shape)
    history = int(query_start) + rows
    if history > int(latent_cache.shape[0]):
        raise ValueError("causal latent prefill exceeds cache capacity")
    if output is None:
        output = torch.empty(
            rows,
            heads,
            latent_dim,
            dtype=latent_cache.dtype,
            device=latent_cache.device,
        )
    elif output.shape != query_nope.shape:
        raise ValueError("causal latent prefill output shape does not match")
    if rows == 0:
        return output

    configured_batch = _positive_env("CCCP_PREFILL_ATTN_BATCH", 256)
    workspace_bytes = (
        _positive_env("CCCP_PREFILL_ATTN_WORKSPACE_MB", 1024)
        * 1024
        * 1024
    )
    score_bytes = max(4, query_nope.element_size() * 2 + 4)
    start = 0
    while start < rows:
        visible = int(query_start) + min(rows, start + configured_batch)
        workspace_batch = max(
            1,
            workspace_bytes // max(1, heads * visible * score_bytes),
        )
        end = min(rows, start + configured_batch, start + workspace_batch)
        visible = int(query_start) + end
        q_nope = query_nope[start:end].transpose(0, 1)
        q_rope = query_rope[start:end].transpose(0, 1)
        score_nope = torch.matmul(q_nope, latent_cache[:visible].t())
        score_rope = torch.matmul(q_rope, rope_cache[:visible].t())
        scores = score_nope.add_(score_rope).div_(scale_denominator).float()
        query_positions = torch.arange(
            int(query_start) + start,
            int(query_start) + end,
            device=query_nope.device,
        )
        key_positions = torch.arange(visible, device=query_nope.device)
        scores.masked_fill_(
            key_positions.view(1, 1, -1)
            > query_positions.view(1, -1, 1),
            float("-inf"),
        )
        probabilities = scores.softmax(dim=-1).to(latent_cache.dtype)
        context = torch.matmul(probabilities, latent_cache[:visible])
        output[start:end].copy_(context.transpose(0, 1))
        start = end
    return output


__all__ = ["causal_latent_prefill"]
