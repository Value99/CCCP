"""Measure compact Qwen Prefill with one transient native execution image.

The model remains stored as packed VQ.  Each projection expands only its
current E4M3 matrix into a reusable allocator block, immediately executes the
vendor FP8 GEMM, and releases that temporary before the next projection.
Decode is intentionally excluded; its compact LUT/graph schedule is measured
by ``benchmark_qwen35_dense.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from cccp import fusedext
from cccp.dense_vq import DenseVQLinear
from cccp.qwen35_model import Qwen35DenseVQModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", default="128,512,4096")
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--compare-compact", action="store_true")
    args = parser.parse_args()

    lengths = tuple(int(value) for value in args.tokens.split(",") if value)
    if not lengths or min(lengths) <= 0:
        raise ValueError("tokens must contain positive integers")
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("NVIDIA CUDA is required")
    if torch.cuda.get_device_capability() < (8, 9):
        raise RuntimeError("transient E4M3 Prefill requires SM89 or newer")
    if not fusedext.prebuild():
        raise RuntimeError(f"GPU extension unavailable: {fusedext.last_error()}")

    model = Qwen35DenseVQModel(
        args.model,
        device="cuda",
        max_ctx=max(lengths) + 32,
    )
    model.cfg["mtp_layers"] = 0
    model.preload()

    original_gpu = DenseVQLinear._gpu
    phase = {"calls": 0, "expanded_bytes": 0}

    compact_reports: list[dict[str, object]] = []
    if args.compare_compact:
        warmup = max(1, int(args.warmup_tokens))
        model.forward([248000] * warmup)
        torch.cuda.synchronize()
        for tokens in lengths:
            # Triton and vendor GEMM both specialize some shapes. Compile the
            # exact batch before timing so the crossover compares execution,
            # not one route's first-use compiler latency.
            model.reset_kv()
            model.forward([248000] * tokens)
            torch.cuda.synchronize()
            model.reset_kv()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            logits = model.forward([248000] * tokens)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            compact_reports.append({
                "tokens": tokens,
                "seconds": seconds,
                "tokens_per_second": tokens / seconds,
                "finite": bool(torch.isfinite(logits).all()),
                "peak_allocated_gib": (
                    torch.cuda.max_memory_allocated() / 2**30
                ),
                "peak_reserved_gib": (
                    torch.cuda.max_memory_reserved() / 2**30
                ),
            })

    def transient_fp8(module: DenseVQLinear, rows: torch.Tensor) -> torch.Tensor:
        if module.layout == "fp8_tensor":
            return module._gpu_fp8(rows)
        if not module.compact_codebook_fp8.numel():
            raise RuntimeError(f"compact codebook missing for {module.name}")
        weight = torch.empty(
            (module.rows, module.cols),
            dtype=torch.float8_e4m3fn,
            device=rows.device,
        )
        expanded = fusedext.dense_vq_expand_native8_fused(
            module.payload,
            module.compact_codebook_fp8,
            weight,
            module.rows,
            module.blocks,
            module.bits,
        )
        if expanded is None:
            raise RuntimeError(f"native E4M3 expansion failed for {module.name}")
        source = rows.to(torch.bfloat16).contiguous()
        quantized = torch.empty_like(source, dtype=torch.float8_e4m3fn)
        source_scale = torch.empty(
            (1, 1), dtype=torch.float32, device=rows.device
        )
        if fusedext.dense_fp8_quantize_rows_fused(
            source, quantized, source_scale
        ) is None:
            raise RuntimeError("FP8 activation quantization unavailable")
        phase["calls"] += 1
        phase["expanded_bytes"] += int(weight.numel())
        return torch._scaled_mm(
            quantized,
            weight.t(),
            scale_a=source_scale,
            scale_b=module.compact_codebook_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

    DenseVQLinear._gpu = transient_fp8
    reports: list[dict[str, object]] = []
    try:
        warmup = max(1, int(args.warmup_tokens))
        model.forward([248000] * warmup)
        torch.cuda.synchronize()
        for tokens in lengths:
            model.reset_kv()
            model.forward([248000] * tokens)
            torch.cuda.synchronize()
            model.reset_kv()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            phase["calls"] = 0
            phase["expanded_bytes"] = 0
            started = time.perf_counter()
            logits = model.forward([248000] * tokens)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            reports.append({
                "tokens": tokens,
                "seconds": seconds,
                "tokens_per_second": tokens / seconds,
                "finite": bool(torch.isfinite(logits).all()),
                "projection_calls": int(phase["calls"]),
                "expanded_gib_total": phase["expanded_bytes"] / 2**30,
                "peak_allocated_gib": (
                    torch.cuda.max_memory_allocated() / 2**30
                ),
                "peak_reserved_gib": (
                    torch.cuda.max_memory_reserved() / 2**30
                ),
            })
    finally:
        DenseVQLinear._gpu = original_gpu

    print("[cccp-qwen-transient-prefill] " + json.dumps({
        "backend": "compact-vq-transient-e4m3-vendor-gemm",
        "compact_storage_gib": model.archive.packed_bytes / 2**30,
        "compact_results": compact_reports,
        "results": reports,
    }, sort_keys=True))
    return 0 if all(item["finite"] for item in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
