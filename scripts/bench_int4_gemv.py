"""Direct microbenchmark for the int4/FP8 Dense VQ GEMV entry points.

Times one representative decode GEMV (out_rows x 5120 int4, batch=1) through
the real packed operators and reports achieved device bandwidth, so kernel
efficiency is measured rather than inferred.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cccp.fusedext import (
    dense_vq_compile_int4_g64_fused,
    dense_vq_mma_packed_m1_fused,
)


def time_loop(fn, repeats: int = 400, warmup: int = 40) -> float:
    for _ in range(warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=5120)
    args = parser.parse_args()
    rows, cols = args.rows, args.cols

    # 从真实模型取 int4 编译态 linear,按 (rows≈out, cols≈in) 选最大目标
    from cccp.qwen35_model import Qwen35DenseVQModel
    from cccp.dense_vq import DenseVQLinear
    model = Qwen35DenseVQModel(
        Path("/media/tyh20/disk22/qwen3.8-27b-cccp-l"), device="cuda", max_ctx=512
    )
    model.preload()
    candidates = []
    for module in model.network.modules():
        for _name, child in module.named_children():
            if isinstance(child, DenseVQLinear):
                candidates.append(child)
    print(f"DenseVQLinear 挂载点: {len(candidates)}")
    results = []
    for m in candidates[:60]:
        rows = int(getattr(m, 'rows', 0) or 0)
        cols = int(getattr(m, 'cols', 0) or 0)
        if rows * cols <= 0:
            continue
        try:
            x = torch.randn(1, cols, device="cuda", dtype=torch.bfloat16)
            ms = time_loop(lambda mm=m, xx=x: mm(xx), repeats=200, warmup=20)
        except Exception:  # noqa: BLE001
            continue
        nbytes = rows * cols * 0.5 + cols * 2
        results.append((ms, rows, cols, nbytes / (ms / 1e3) / 1e9))
    results.sort(key=lambda r: -r[0])
    print(f"{'ms':>9} {'rows':>7} {'cols':>7} {'GB/s':>9}")
    for ms, rows, cols, bw in results[:12]:
        print(f"{ms:9.4f} {rows:7d} {cols:7d} {bw:9.0f}")
    heavy = [r for r in results if r[1] * r[2] >= 4096 * 1024]
    if heavy:
        total_ms = sum(r[0] for r in heavy)
        total_bytes = sum(r[1] * r[2] * 0.5 for r in heavy)
        print(f"大矩阵({len(heavy)}个)合计 {total_ms:.3f} ms,"
              f"合计带宽 {total_bytes / (total_ms / 1e3) / 1e9:.0f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
