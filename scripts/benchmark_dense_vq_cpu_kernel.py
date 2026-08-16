"""Compare compact VQ and load-time Q4 CPU GEMV on real archive tensors."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from cccp.cpuext import (
    q4_0_gemv_cpu,
    vq_compile_q4_0_cpu,
    vq_gemv_packed_list_cpu,
)
from cccp.dense_vq import DenseVQArchive


def _measure(run, repeats: int) -> tuple[float, torch.Tensor]:
    result = run()
    started = time.perf_counter()
    for _ in range(repeats):
        result = run()
    elapsed = time.perf_counter() - started
    return elapsed / repeats, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name-contains", default="mlp.down_proj.weight")
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--torch-threads", type=int, default=96)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.torch_threads))

    archive = DenseVQArchive(args.model)
    matches = [
        name for name in sorted(archive.specs)
        if args.name_contains in name
    ]
    if not matches:
        raise RuntimeError(f"no tensor matches {args.name_contains!r}")
    name = matches[len(matches) // 2]
    weight = archive.load_weight(name, torch.device("cpu"))
    value = torch.randn(1, weight.cols, dtype=torch.float32)
    q4 = vq_compile_q4_0_cpu(
        weight.raw, weight.cb, weight.rows, weight.blocks, weight.bits
    )
    if q4 is None:
        raise RuntimeError("Q4 compilation unavailable")
    q4_output = torch.empty(1, weight.rows, dtype=torch.float32)

    def q4_run():
        return q4_0_gemv_cpu(
            value, q4, weight.rows, weight.cols, q4_output
        )

    def direct_run():
        return vq_gemv_packed_list_cpu(
            value,
            [weight.raw],
            weight.cb,
            weight.rows,
            weight.blocks,
            weight.bits,
            allow_direct=True,
        )

    def score_run():
        return vq_gemv_packed_list_cpu(
            value,
            [weight.raw],
            weight.cb,
            weight.rows,
            weight.blocks,
            weight.bits,
            allow_direct=False,
        )

    q4_seconds, q4_result = _measure(q4_run, args.repeats)
    direct_seconds, direct_result = _measure(direct_run, args.repeats)
    score_seconds, score_result = _measure(score_run, args.repeats)
    report = {
        "name": name,
        "shape": [weight.rows, weight.cols],
        "blocks": weight.blocks,
        "bits": weight.bits,
        "source_mib": weight.raw.numel() / 2**20,
        "q4_mib": q4.numel() / 2**20,
        "q4_ms": q4_seconds * 1000.0,
        "direct_ms": direct_seconds * 1000.0,
        "score_lookup_ms": score_seconds * 1000.0,
        "direct_finite": bool(torch.isfinite(direct_result).all()),
        "score_finite": bool(torch.isfinite(score_result).all()),
        "q4_finite": bool(torch.isfinite(q4_result).all()),
    }
    print("[cccp-dense-vq-cpu-kernel] " + json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
