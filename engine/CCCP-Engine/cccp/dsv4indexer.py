"""Pure-PyTorch reference math for DeepSeek-V4 Lightning Indexer.

Formula and state order follow DeepSeek's official ``inference/model.py``.
"""

from __future__ import annotations

import torch

from .dsv4cache import PagedKV
from .fusedext import hadamard_bf16_fused


def indexer_scores(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Official per-head positive score reduction, accumulated in FP32."""
    score = torch.einsum("bshd,btd->bsht", q.float(), kv.float())
    return (score.relu_() * weights.float().unsqueeze(-1)).sum(dim=2)


def select_index_positions(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Return all positions up to top-k, then switch to learned sparse selection."""
    count = scores.shape[-1]
    if count <= topk:
        shape = (1,) * (scores.ndim - 1) + (count,)
        return (
            torch.arange(count, device=scores.device, dtype=torch.int32)
            .view(shape)
            .expand(*scores.shape[:-1], count)
        )
    return scores.topk(topk, dim=-1, sorted=False).indices.to(torch.int32)


def hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard rotation used before Indexer FP4 simulation."""
    width = x.shape[-1]
    if width <= 0 or width & (width - 1):
        raise ValueError(f"Hadamard width must be a power of two, got {width}")
    fused = hadamard_bf16_fused(x)
    if fused is not None:
        return fused
    original_shape = x.shape
    original_dtype = x.dtype
    y = x.float().reshape(-1, width)
    block = 1
    while block < width:
        y = y.view(-1, width // (2 * block), 2, block)
        left, right = y[:, :, 0], y[:, :, 1]
        y = torch.cat((left + right, left - right), dim=-1).reshape(-1, width)
        block *= 2
    return (y * (width ** -0.5)).view(original_shape).to(original_dtype)


class IndexerState:
    """Ratio-4 Compressor state plus stable paged 128-D scoring keys."""

    ratio = 4

    def __init__(
        self,
        *,
        batch: int,
        head_dim: int = 128,
        rope_dim: int = 64,
        page_items: int = 1024,
        device="cpu",
        dtype=torch.bfloat16,
        max_items: int | None = None,
    ):
        if rope_dim > head_dim:
            raise ValueError("rope_dim cannot exceed Indexer head_dim")
        self.batch = batch
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.device = torch.device(device)
        self.dtype = dtype
        self.keys = PagedKV(
            batch=batch,
            page_items=page_items,
            dim=head_dim,
            device=self.device,
            dtype=dtype,
            max_items=max_items,
        )
        coff = 2
        self.ckv = torch.zeros(
            batch,
            coff * self.ratio,
            coff * head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.cscore = torch.full(
            self.ckv.shape,
            float("-inf"),
            device=self.device,
            dtype=torch.float32,
        )

    @property
    def compressor_state(self) -> dict[str, torch.Tensor]:
        return {"ckv": self.ckv, "cscore": self.cscore}

    def reserve_position(self, position: int) -> None:
        self.keys.reserve(position // self.ratio)

    def prefill(
        self,
        x: torch.Tensor,
        weights: dict,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        eps: float,
    ) -> torch.Tensor | None:
        from .dsv4 import compressor_prefill

        self.reset()
        pooled = compressor_prefill(
            x,
            weights,
            self.ratio,
            self.head_dim,
            self.rope_dim,
            cos,
            sin,
            eps,
            self.compressor_state,
        )
        if pooled is not None:
            pooled = hadamard_rotate(pooled)
            self.keys.write_many(0, pooled)
        return pooled

    def decode(
        self,
        x: torch.Tensor,
        weights: dict,
        position: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        eps: float,
    ) -> torch.Tensor | None:
        from .dsv4 import compressor_decode

        pooled = compressor_decode(
            x,
            weights,
            self.ratio,
            self.head_dim,
            self.rope_dim,
            cos,
            sin,
            eps,
            self.compressor_state,
            position,
        )
        if pooled is not None:
            pooled = hadamard_rotate(pooled)
            self.keys.write(position // self.ratio, pooled[:, 0])
        return pooled

    def reset(self) -> None:
        self.keys.reset()
        self.ckv.zero_()
        self.cscore.fill_(float("-inf"))
