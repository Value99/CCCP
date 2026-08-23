"""Bounded-memory packed-VQ to native-FP8 Tensor Core execution."""

from __future__ import annotations

import torch

from .api import (
    dense_vq_expand_fp8_tile,
    dense_vq_quantize_fp8_codebook,
)


class DenseVQFP8TileMMA:
    """Project through packed VQ using one reusable native-FP8 weight tile.

    The compact payload remains the authoritative weight image.  Only the
    small codebook is quantized persistently; each contiguous output-row tile
    is expanded directly to E4M3 in a fixed workspace and consumed by the
    vendor ``scaled_mm`` Tensor Core implementation.  A full BF16 or FP8
    weight matrix is never materialized.

    This is a batch/Prefill primitive.  Reconstructing tiles for a single
    token is intentionally not hidden behind this class: Decode should use a
    resident compact GEMV or resident FP8 image selected by the capacity
    planner.
    """

    DEFAULT_WORKSPACE_BYTES = 40 << 20
    DEFAULT_MINIMUM_BATCH_TOKENS = 512

    def __init__(
        self,
        payload: torch.Tensor,
        codebook: torch.Tensor,
        *,
        rows: int,
        blocks: int,
        bits: int,
        workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
        minimum_batch_tokens: int = DEFAULT_MINIMUM_BATCH_TOKENS,
    ) -> None:
        rows = int(rows)
        blocks = int(blocks)
        bits = int(bits)
        if (
            payload.device.type != "cuda"
            or codebook.device != payload.device
            or payload.dtype != torch.uint8
            or codebook.dtype != torch.float32
            or payload.ndim != 1
            or codebook.ndim != 2
            or not payload.is_contiguous()
            or not codebook.is_contiguous()
        ):
            raise ValueError(
                "VQ FP8 tile MMA requires contiguous CUDA uint8 payload "
                "and FP32 codebook"
            )
        if torch.version.hip is not None:
            raise RuntimeError(
                "VQ FP8 tile MMA currently requires native NVIDIA E4M3 "
                "Tensor Core scaled_mm"
            )
        vector = int(codebook.shape[1])
        columns = blocks * vector
        if (
            rows <= 0
            or blocks <= 0
            or bits < 8
            or bits > 16
            or rows % 16
            or columns % 16
            or int(workspace_bytes) < columns * 16
        ):
            raise ValueError(
                "VQ FP8 tile MMA requires p8-p16 weights, 16-aligned "
                "matrix dimensions, and room for at least 16 output rows"
            )
        converted = dense_vq_quantize_fp8_codebook(codebook)
        if converted is None:
            raise RuntimeError("VQ FP8 codebook operator is unavailable")
        fp8_codebook, weight_scale = converted
        # The common registry deliberately bounds one launch to 8192 rows;
        # larger output matrices advance through multiple fixed tiles.
        tile_rows = min(rows, 8192, int(workspace_bytes) // columns)
        tile_rows = max(16, tile_rows - tile_rows % 16)

        self.payload = payload
        self.fp8_codebook = fp8_codebook
        self.weight_scale = weight_scale
        self.rows = rows
        self.blocks = blocks
        self.bits = bits
        self.columns = columns
        self.tile_rows = tile_rows
        self.minimum_batch_tokens = max(1, int(minimum_batch_tokens))
        self.weight_tile = torch.empty(
            (tile_rows, columns),
            dtype=torch.float8_e4m3fn,
            device=payload.device,
        )

    @property
    def workspace_nbytes(self) -> int:
        """Persistent bytes added beyond the original packed payload."""
        return int(
            self.weight_tile.numel() * self.weight_tile.element_size()
            + self.fp8_codebook.numel()
            * self.fp8_codebook.element_size()
            + self.weight_scale.numel() * self.weight_scale.element_size()
        )

    def recommended_for(self, value: torch.Tensor) -> bool:
        """Whether the measured batch policy recommends tiled MMA."""
        return bool(
            value.device == self.payload.device
            and value.shape[-1] == self.columns
            and value.numel() // self.columns >= self.minimum_batch_tokens
        )

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        if value.device != self.payload.device or value.shape[-1] != self.columns:
            raise ValueError(
                f"VQ FP8 tile MMA expected CUDA input with {self.columns} "
                f"columns on {self.payload.device}"
            )
        original_shape = value.shape
        source = value.reshape(-1, self.columns).to(torch.bfloat16).contiguous()
        activation = torch.empty_like(source, dtype=torch.float8_e4m3fn)
        activation_scale = torch.empty(
            (1, 1), dtype=torch.float32, device=source.device
        )
        from ..fusedext import dense_fp8_quantize_rows_fused

        quantized = dense_fp8_quantize_rows_fused(
            source, activation, activation_scale
        )
        if quantized is None:
            raise RuntimeError("FP8 activation quantizer is unavailable")
        output = torch.empty(
            (source.shape[0], self.rows),
            dtype=torch.bfloat16,
            device=source.device,
        )
        for row_start in range(0, self.rows, self.tile_rows):
            row_count = min(self.tile_rows, self.rows - row_start)
            tile = dense_vq_expand_fp8_tile(
                self.payload,
                self.fp8_codebook,
                self.weight_tile,
                rows=self.rows,
                blocks=self.blocks,
                bits=self.bits,
                row_start=row_start,
                row_count=row_count,
            )
            if tile is None:
                raise RuntimeError("VQ FP8 tile expansion is unavailable")
            projected = torch._scaled_mm(
                quantized,
                tile.t(),
                scale_a=activation_scale,
                scale_b=self.weight_scale,
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
            )
            output[:, row_start:row_start + row_count].copy_(projected)
        return output.reshape(*original_shape[:-1], self.rows).to(value.dtype)
