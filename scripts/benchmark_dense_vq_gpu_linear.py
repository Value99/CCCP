"""Benchmark both INT4-G64 decode kernels on one manifest Dense VQ Linear.

The script loads exactly one matrix, compiles its process-local GPU execution
image and compares the ordinary row kernel with the group-vector kernel.  It
does not construct the model or modify the archive.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from cccp.dense_vq import DenseVQArchive, DenseVQLinear
from cccp.fusedext import int4_gemv_fused


def _measure(function, iterations: int) -> tuple[torch.Tensor, float]:
    for _ in range(5):
        output = function()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        output = function()
    torch.cuda.synchronize()
    return output, (time.perf_counter() - started) * 1000.0 / iterations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    archive = DenseVQArchive(args.model)
    candidates = {
        name: spec
        for name, spec in archive.specs.items()
        if name.endswith(".weight") and "embed_tokens" not in name
    }
    if args.name:
        if args.name not in candidates:
            raise ValueError(f"unknown Dense VQ Linear {args.name!r}")
        name = args.name
    else:
        name = max(
            candidates,
            key=lambda item: candidates[item].rows * candidates[item].cols,
        )
    spec = candidates[name]
    linear = DenseVQLinear.from_archive(
        archive, name, torch.device("cuda")
    )
    if not linear.compile_gpu_int4():
        raise RuntimeError("INT4-G64 compilation failed")
    value = torch.randn(
        1, spec.cols, dtype=torch.bfloat16, device="cuda"
    )

    def run(group_vector: bool):
        result = int4_gemv_fused(
            value,
            linear.payload,
            linear.gpu_scales,
            spec.cols,
            64,
            group_vector=group_vector,
        )
        if result is None:
            raise RuntimeError("INT4 GEMV rejected the benchmark input")
        return result

    scalar, scalar_ms = _measure(lambda: run(False), args.iterations)
    vector, vector_ms = _measure(lambda: run(True), args.iterations)
    maximum_difference = float((vector - scalar).abs().max().item())
    torch.testing.assert_close(vector, scalar, rtol=1e-4, atol=2e-6)
    image_bytes = int(linear.payload.numel() + linear.gpu_scales.numel() * 2)
    del scalar, vector, linear
    gc.collect()
    torch.cuda.empty_cache()
    bf16_linear = DenseVQLinear.from_archive(
        archive, name, torch.device("cuda")
    )
    if not bf16_linear.compile_gpu_bf16():
        raise RuntimeError("BF16 expansion failed")
    _bf16, bf16_ms = _measure(
        lambda: bf16_linear(value), args.iterations
    )
    bf16_bytes = int(bf16_linear.payload.numel() * 2)
    report = {
        "name": name,
        "rows": spec.rows,
        "cols": spec.cols,
        "image_gib": image_bytes / 2**30,
        "row_kernel_ms": scalar_ms,
        "group_vector_ms": vector_ms,
        "speedup": scalar_ms / vector_ms,
        "group_vector_gib_s": image_bytes / 2**30 / (vector_ms / 1000.0),
        "bf16_image_gib": bf16_bytes / 2**30,
        "bf16_vendor_ms": bf16_ms,
        "bf16_vendor_gib_s": bf16_bytes / 2**30 / (bf16_ms / 1000.0),
        "maximum_absolute_difference": maximum_difference,
        "numerically_equivalent": True,
    }
    print("[cccp-dense-vq-linear] " + json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
