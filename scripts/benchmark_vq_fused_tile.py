"""Benchmark a materialization-free VQ -> FP8 Tensor Core tile prototype.

Unlike ``benchmark_vq_tensorcore_tile.py`` this program never creates a full
``[out_features, in_features]`` execution image for its fused candidate.  A
Triton program unpacks each VQ weight tile, gathers its pre-quantized E4M3
codewords into registers, and immediately feeds that tile to ``tl.dot``.  The
expanded E4M3 + vendor GEMM path and the exact compact GEMV remain independent
baselines so the report cannot hide conversion or launch costs.

This is a proof/measurement kernel for SM89+.  It is intentionally not wired
into the public runtime until the numerical and speed gates demonstrate an
advantage over the reusable Native8 execution-image path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from cccp import fusedext
from cccp.store import CCCPStore


@triton.jit
def _vq_fp8_tile_mm_kernel(
    source_ptr,
    packed_ptr,
    codebook_ptr,
    output_ptr,
    rows: tl.constexpr,
    columns: tl.constexpr,
    blocks: tl.constexpr,
    packed_bytes: tl.constexpr,
    bits: tl.constexpr,
    vector: tl.constexpr,
    source_scale,
    codebook_scale,
    real_batch: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One CTA reconstructs only the B tile consumed by its Tensor Core dot."""

    pid_n = tl.program_id(0)
    offsets_m = tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    code_mask = (1 << bits) - 1

    for k0 in range(0, columns, BLOCK_K):
        offsets_k = k0 + tl.arange(0, BLOCK_K)
        source = tl.load(
            source_ptr + offsets_m[:, None] * columns + offsets_k[None, :],
            mask=(offsets_m[:, None] < real_batch)
            & (offsets_k[None, :] < columns),
            other=0.0,
        )

        block = offsets_k // vector
        component = offsets_k - block * vector
        linear_index = offsets_n[:, None] * blocks + block[None, :]
        bit_offset = linear_index * bits
        byte_offset = bit_offset >> 3
        shift = bit_offset & 7
        valid = (offsets_n[:, None] < rows) & (offsets_k[None, :] < columns)
        low = tl.load(
            packed_ptr + byte_offset,
            mask=valid & (byte_offset < packed_bytes),
            other=0,
        ).to(tl.uint32)
        middle = tl.load(
            packed_ptr + byte_offset + 1,
            mask=valid & (byte_offset + 1 < packed_bytes),
            other=0,
        ).to(tl.uint32)
        high = tl.load(
            packed_ptr + byte_offset + 2,
            mask=valid & (byte_offset + 2 < packed_bytes),
            other=0,
        ).to(tl.uint32)
        code = ((low | (middle << 8) | (high << 16)) >> shift) & code_mask
        weight = tl.load(
            codebook_ptr + code * vector + component[None, :],
            mask=valid,
            other=0.0,
        )
        accumulator += tl.dot(source, tl.trans(weight), out_dtype=tl.float32)

    result = accumulator * source_scale * codebook_scale
    tl.store(
        output_ptr + offsets_m[:, None] * rows + offsets_n[None, :],
        result,
        mask=(offsets_m[:, None] < real_batch) & (offsets_n[None, :] < rows),
    )


def _measure(call, repeats: int) -> tuple[torch.Tensor, float]:
    for _ in range(5):
        output = call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(max(1, repeats)):
        output = call()
    end.record()
    end.synchronize()
    return output, float(begin.elapsed_time(end) / max(1, repeats))


def _quality(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate = candidate.float()
    reference = reference.float()
    delta = candidate - reference
    denominator = reference.abs().mean().clamp_min(1.0e-12)
    return {
        "cosine": float(F.cosine_similarity(
            candidate.reshape(-1), reference.reshape(-1), dim=0
        )),
        "max_abs_error": float(delta.abs().max()),
        "relative_mae": float(delta.abs().mean() / denominator),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--expert", type=int, default=5)
    parser.add_argument(
        "--projection", choices=("gate", "up", "down"), default="gate"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 9):
        raise RuntimeError("FP8 tile prototype requires SM89+")
    if not fusedext.prebuild():
        raise RuntimeError(f"GPU extension unavailable: {fusedext.last_error()}")

    store = CCCPStore(str(args.model))
    projection_index = tuple(store.man.projection_names).index(args.projection)
    host = store.load_expert_packed(args.layer, args.expert)[projection_index]
    device = torch.device("cuda")
    payload = host.raw.to(device).contiguous().reshape(-1)
    codebook = host.cb.to(device=device, dtype=torch.float32).contiguous()
    rows = int(host.rows)
    blocks = int(host.blocks)
    vector = int(host.dim)
    columns = blocks * vector
    batch = int(args.batch)
    if not 1 <= batch <= 16:
        raise ValueError("prototype batch must be in 1..16")

    codebook_scale = max(float(codebook.abs().amax()) / 448.0, 1.0e-12)
    codebook8 = (
        codebook.div(codebook_scale).clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn).contiguous()
    )
    source = torch.randn(batch, columns, dtype=torch.bfloat16, device=device)
    source_scale = max(float(source.float().abs().amax()) / 448.0, 1.0e-12)
    padded_source8 = torch.zeros(
        16, columns, dtype=torch.float8_e4m3fn, device=device
    )
    padded_source8[:batch].copy_(
        source.float().div(source_scale).clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
    )
    fused_output = torch.empty(
        16, rows, dtype=torch.bfloat16, device=device
    )

    grid = (triton.cdiv(rows, int(args.block_n)),)

    def fused_tile() -> torch.Tensor:
        _vq_fp8_tile_mm_kernel[grid](
            padded_source8,
            payload,
            codebook8,
            fused_output,
            rows=rows,
            columns=columns,
            blocks=blocks,
            packed_bytes=int(payload.numel()),
            bits=int(host.bits),
            vector=vector,
            source_scale=source_scale,
            codebook_scale=codebook_scale,
            real_batch=batch,
            BLOCK_M=16,
            BLOCK_N=int(args.block_n),
            BLOCK_K=int(args.block_k),
            num_warps=int(args.num_warps),
        )
        return fused_output[:batch]

    row_ids = torch.empty(0, dtype=torch.long, device=device)
    expanded8 = torch.empty(
        rows, columns, dtype=torch.float8_e4m3fn, device=device
    )
    source_scale_tensor = torch.tensor(
        [[source_scale]], device=device, dtype=torch.float32
    )
    codebook_scale_tensor = torch.tensor(
        [[codebook_scale]], device=device, dtype=torch.float32
    )

    def expand() -> torch.Tensor:
        return fusedext._EXT.dense_vq_expand_native8(
            payload, codebook8, expanded8, rows, blocks, int(host.bits), row_ids
        )

    def expanded_mm() -> torch.Tensor:
        return torch._scaled_mm(
            padded_source8[:batch],
            expanded8.t(),
            scale_a=source_scale_tensor,
            scale_b=codebook_scale_tensor,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

    exact = fusedext._EXT.dense_vq_dequant_packed(
        payload, codebook, rows, blocks, int(host.bits), row_ids
    )
    expand()
    reference = F.linear(source.float(), exact.float())
    fused_result, fused_ms = _measure(fused_tile, int(args.repeats))
    _, expand_ms = _measure(expand, int(args.repeats))
    expanded_result, expanded_mm_ms = _measure(expanded_mm, int(args.repeats))
    compact_ms = None
    if batch == 1:
        _, compact_ms = _measure(
            lambda: fusedext._EXT.dense_vq_gemv_packed(
                source.float(), payload, codebook, rows, blocks, int(host.bits)
            ),
            int(args.repeats),
        )

    report = {
        "backend": "vq-register-tile-triton-tensorcore",
        "batch": batch,
        "bits": int(host.bits),
        "block_k": int(args.block_k),
        "block_n": int(args.block_n),
        "columns": columns,
        "compact_exact_ms": compact_ms,
        "expanded_bytes": int(rows * columns),
        "expanded_mm_ms": expanded_mm_ms,
        "expanded_total_ms": expand_ms + expanded_mm_ms,
        "expand_ms": expand_ms,
        "fused_tile_ms": fused_ms,
        "fused_tile_quality": _quality(fused_result, reference),
        "expanded_quality": _quality(expanded_result, reference),
        "materialized_weight_bytes": 0,
        "num_warps": int(args.num_warps),
        "projection": args.projection,
        "rows": rows,
    }
    print("[cccp-vq-fused-tile] " + json.dumps(report, sort_keys=True))
    quality = report["fused_tile_quality"]
    assert isinstance(quality, dict)
    return 0 if (
        quality["cosine"] >= 0.995 and quality["relative_mae"] <= 0.10
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
