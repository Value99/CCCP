"""Transient FP8 Tensor Core execution for compact Dense VQ Prefill.

Compact Decode is intentionally not implemented here. Batch-one Decode keeps
the packed VQ payload intact and uses the common grouped direct-dot extension
in :mod:`cccp.fusedext`. Prefill expands only the projection being consumed
into a transient E4M3 tensor and immediately calls the vendor GEMM; the
expanded tensor never becomes a persistent execution image.
"""

from __future__ import annotations

import torch


def _quantize_source(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from ..fusedext import dense_fp8_quantize_rows_fused

    contiguous = source.contiguous()
    quantized = torch.empty_like(contiguous, dtype=torch.float8_e4m3fn)
    scale = torch.empty((1, 1), dtype=torch.float32, device=source.device)
    result = dense_fp8_quantize_rows_fused(contiguous, quantized, scale)
    if result is None:
        raise RuntimeError("Dense VQ compact FP8 activation quantizer unavailable")
    return result, scale


def dense_vq_transient_fp8_gemm(
    *,
    x_rows: torch.Tensor,
    payload: torch.Tensor,
    codebook_fp8: torch.Tensor,
    codebook_scale: torch.Tensor,
    rows: int,
    blocks: int,
    bits: int,
) -> torch.Tensor:
    """Expand one current projection and consume it with vendor FP8 GEMM."""

    if (
        not x_rows.is_cuda
        or x_rows.ndim != 2
        or int(x_rows.shape[0]) <= 1
        or payload.dtype != torch.uint8
        or payload.ndim != 1
        or codebook_fp8.dtype != torch.float8_e4m3fn
        or codebook_fp8.ndim != 2
        or codebook_scale.dtype != torch.float32
        or codebook_scale.numel() != 1
    ):
        raise ValueError("invalid Dense VQ transient Prefill operands")
    if torch.cuda.get_device_capability(x_rows.device) < (8, 9):
        raise RuntimeError("Dense VQ transient FP8 GEMM requires SM89 or newer")

    from ..fusedext import dense_vq_expand_native8_fused

    rows = int(rows)
    blocks = int(blocks)
    bits = int(bits)
    columns = blocks * int(codebook_fp8.shape[1])
    if int(x_rows.shape[1]) != columns:
        raise ValueError("Dense VQ transient Prefill input width mismatch")
    weight = torch.empty(
        (rows, columns), dtype=torch.float8_e4m3fn, device=x_rows.device
    )
    expanded = dense_vq_expand_native8_fused(
        payload,
        codebook_fp8,
        weight,
        rows,
        blocks,
        bits,
    )
    if expanded is None:
        raise RuntimeError("Dense VQ native E4M3 expansion is unavailable")
    source_fp8, source_scale = _quantize_source(x_rows.to(torch.bfloat16))
    return torch._scaled_mm(
        source_fp8,
        expanded.t(),
        scale_a=source_scale,
        scale_b=codebook_scale,
        out_dtype=torch.bfloat16,
        use_fast_accum=True,
    )


def dense_vq_transient_fp8_grouped_gemm(
    *,
    x_rows: torch.Tensor,
    payloads: tuple[torch.Tensor, ...],
    codebooks_fp8: tuple[torch.Tensor, ...],
    common_codebook_scale: torch.Tensor,
    row_counts: tuple[int, ...],
    blocks: tuple[int, ...],
    bits: tuple[int, ...],
) -> torch.Tensor:
    """Expand a shared-input projection group and enter one vendor FP8 GEMM."""

    members = len(payloads)
    if not (
        x_rows.is_cuda
        and x_rows.ndim == 2
        and int(x_rows.shape[0]) > 1
        and members >= 2
        and len(codebooks_fp8) == members
        and len(row_counts) == members
        and len(blocks) == members
        and len(bits) == members
        and common_codebook_scale.is_cuda
        and common_codebook_scale.dtype == torch.float32
        and common_codebook_scale.numel() == 1
    ):
        raise ValueError("invalid grouped Dense VQ transient Prefill operands")
    if torch.cuda.get_device_capability(x_rows.device) < (8, 9):
        raise RuntimeError("grouped Dense VQ FP8 GEMM requires SM89 or newer")

    from ..fusedext import dense_vq_expand_native8_fused

    columns = int(x_rows.shape[1])
    total_rows = sum(int(value) for value in row_counts)
    weight = torch.empty(
        (total_rows, columns), dtype=torch.float8_e4m3fn, device=x_rows.device
    )
    row_offset = 0
    for payload, codebook, member_rows, member_blocks, member_bits in zip(
        payloads, codebooks_fp8, row_counts, blocks, bits
    ):
        member_rows = int(member_rows)
        member_blocks = int(member_blocks)
        if member_blocks * int(codebook.shape[1]) != columns:
            raise ValueError("grouped Dense VQ member width mismatch")
        target = weight.narrow(0, row_offset, member_rows)
        expanded = dense_vq_expand_native8_fused(
            payload,
            codebook,
            target,
            member_rows,
            member_blocks,
            int(member_bits),
        )
        if expanded is None:
            raise RuntimeError("grouped Dense VQ native E4M3 expansion unavailable")
        row_offset += member_rows

    source_fp8, source_scale = _quantize_source(x_rows.to(torch.bfloat16))
    return torch._scaled_mm(
        source_fp8,
        weight.t(),
        scale_a=source_scale,
        scale_b=common_codebook_scale,
        out_dtype=torch.bfloat16,
        use_fast_accum=True,
    )


__all__ = [
    "dense_vq_transient_fp8_gemm",
    "dense_vq_transient_fp8_grouped_gemm",
]
