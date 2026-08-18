"""Per-kernel decode profiler for Qwen3.5 Dense VQ (all architectures).

Runs the same forward path as the benchmark under ``torch.profiler`` and
reports the top CUDA kernels by device time plus a per-token breakdown, so
operator work targets measured hot spots instead of guesses.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from cccp.qwen35_model import Qwen35DenseVQModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prefill-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model = Qwen35DenseVQModel(
        args.model,
        device="cuda",
        max_ctx=max(128, args.prefill_tokens + args.decode_tokens + 8),
    )
    model.preload()
    prompt = [248000] * args.prefill_tokens
    model.forward(prompt)
    torch.cuda.synchronize()

    # 稳态计时(不含 profiler 开销)
    started = time.perf_counter()
    for index in range(args.decode_tokens):
        model.forward([248001 + index])
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    steady_tps = args.decode_tokens / wall

    # profiler 段
    if args.eager:
        model._decode_graph = None
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for index in range(args.decode_tokens):
            model.forward([250000 + index])
        torch.cuda.synchronize()

    rows = []
    total_cuda = 0.0
    for event in prof.key_averages():
        if event.device_type == torch.autograd.DeviceType.CUDA or (
            event.self_device_time_total and event.self_device_time_total > 0
        ):
            self_us = float(event.self_device_time_total)
            if self_us <= 0:
                continue
            total_cuda += self_us
            rows.append((self_us, event.count, event.key))
    rows.sort(reverse=True)

    lines = []
    lines.append(f"steady: {steady_tps:.2f} tok/s ({wall / args.decode_tokens * 1e3:.3f} ms/tok)")
    lines.append(f"profiled device time: {total_cuda / 1e3:.2f} ms total, "
                 f"{total_cuda / args.decode_tokens / 1e3:.3f} ms/tok")
    lines.append(f"{'self-us':>12} {'count':>7} {'us/call':>9} {'%':>6}  kernel")
    for self_us, count, key in rows[: args.top]:
        lines.append(f"{self_us:>12.0f} {count:>7} {self_us / count:>9.1f} "
                     f"{self_us / total_cuda * 100:>5.1f}%  {key[:110]}")
    report = "\n".join(lines)
    print(report)
    if args.output:
        args.output.write_text(
            json.dumps({
                "steady_tokens_per_second": steady_tps,
                "ms_per_token_wall": wall / args.decode_tokens * 1e3,
                "profiled_ms_per_token": total_cuda / args.decode_tokens / 1e3,
                "kernels": [
                    {"self_us": r[0], "count": r[1], "name": r[2][:200]}
                    for r in rows[: 60]
                ],
                "cuda_memory_gib": torch.cuda.memory_reserved() / 2**30,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
