"""Measure one real Dense-VQ Q4 Linear without constructing the model."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--numa", choices=("off", "local"), default="local")
    parser.add_argument("--image", choices=("q4", "vq"), default="q4")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    os.environ["CCCP_CPU_THREADS"] = str(args.threads)
    os.environ["CCCP_CPU_COMPILE"] = "q4"
    os.environ["CCCP_CPU_NUMA"] = args.numa
    os.environ.setdefault("OMP_PROC_BIND", "true")
    os.environ.setdefault("OMP_PLACES", "cores")
    torch.set_num_threads(args.threads)

    from cccp.dense_vq import DenseVQArchive, DenseVQLinear

    archive = DenseVQArchive(args.model)
    candidates = {
        name: spec
        for name, spec in archive.specs.items()
        if name.endswith(".weight") and "embed_tokens" not in name
    }
    name = args.name or max(
        candidates,
        key=lambda item: candidates[item].rows * candidates[item].cols,
    )
    spec = candidates[name]
    started = time.perf_counter()
    linear = DenseVQLinear.from_archive(
        archive, name, torch.device("cpu")
    )
    if args.image == "q4" and not linear.compile_cpu():
        raise RuntimeError("CPU Q4 compilation failed")
    compile_seconds = time.perf_counter() - started
    value = torch.randn((1, spec.cols), dtype=torch.bfloat16)

    def execute():
        if args.image == "q4":
            return linear(value)
        from cccp.cpuext import vq_gemv_packed_list_cpu

        result = vq_gemv_packed_list_cpu(
            value.float(),
            [linear.payload],
            linear.codebook.float().contiguous(),
            linear.rows,
            linear.blocks,
            linear.bits,
            allow_direct=False,
        )
        if result is None:
            raise RuntimeError("CPU packed VQ GEMV unavailable")
        return result

    for _ in range(3):
        result = execute()
    samples = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        result = execute()
        samples.append((time.perf_counter() - started) * 1000.0)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("non-finite Q4 output")
    image_bytes = int(linear.payload.numel()) + int(
        linear.codebook.numel() * linear.codebook.element_size()
    )
    median_ms = statistics.median(samples)
    report = {
        "name": name,
        "rows": spec.rows,
        "cols": spec.cols,
        "threads": args.threads,
        "numa": args.numa,
        "image": args.image,
        "compile_seconds": compile_seconds,
        "image_gib": image_bytes / 2**30,
        "median_ms": median_ms,
        "minimum_ms": min(samples),
        "image_gib_s": image_bytes / 2**30 / (median_ms / 1000.0),
        "finite": True,
    }
    print("[cccp-dense-vq-cpu-linear] " + json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
