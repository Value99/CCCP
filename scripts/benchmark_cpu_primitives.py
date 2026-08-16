"""Isolated latency benchmark for CCCP's native CPU decode primitives.

This intentionally runs in a fresh Python process so two ABI-compatible
operator builds can be compared without Windows retaining a loaded ``.pyd``.
It does not load model weights and therefore separates kernel throughput from
the page-cache effects seen in the full-model benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine" / "CCCP-Engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))


def _measure(call, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        call()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1_000.0)
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": ordered[0],
        "p90_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "maximum_ms": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    os.environ["CCCP_CPU_THREADS"] = str(max(1, args.threads))
    from cccp.cpuext import (  # noqa: PLC0415
        attention_decode_cpu,
        block_fp8_compile_q4_0_cpu,
        block_fp8_gemv_cpu,
        configure_cpu_threads,
        extension_status,
        q4_0_gemv_cpu,
    )

    actual_threads = configure_cpu_threads()
    torch.manual_seed(5090)

    rows = cols = 4096
    fp8_weights = torch.randint(0, 255, (rows, cols), dtype=torch.uint8)
    fp8_scales = torch.rand((rows // 128, cols // 128), dtype=torch.float32)
    fp8_scales.mul_(0.03).add_(0.001)
    fp8_value = torch.randn((1, cols), dtype=torch.float32)
    fp8_output = torch.empty(rows, dtype=torch.float32)
    q4_weights = block_fp8_compile_q4_0_cpu(
        fp8_weights, fp8_scales, rows, cols, 128
    )
    if q4_weights is None:
        raise RuntimeError("native Q4 compilation is unavailable")
    q4_output = torch.empty((1, rows), dtype=torch.float32)

    batch, heads, dim = 1, 64, 512
    raw_count, selected_count = 128, 512
    query = torch.randn((batch, heads, dim), dtype=torch.float32)
    raw_values = torch.randn((batch, raw_count, dim), dtype=torch.float32)
    raw_positions = torch.arange(raw_count, dtype=torch.long).view(1, -1)
    selected_values = torch.randn(
        (batch, selected_count, dim), dtype=torch.float32
    )
    sink = torch.zeros((heads,), dtype=torch.float32)
    rope_cos = torch.ones((1, 1, 1, dim // 2), dtype=torch.float32)
    rope_sin = torch.zeros_like(rope_cos)

    result = {
        "label": args.label,
        "threads": actual_threads,
        "extension": extension_status(),
        "block_fp8_4096x4096": _measure(
            lambda: block_fp8_gemv_cpu(
                fp8_value,
                fp8_weights,
                fp8_scales,
                cols,
                128,
                fp8_output,
            ),
            args.warmup,
            args.repeats,
        ),
        "q4_4096x4096": _measure(
            lambda: q4_0_gemv_cpu(
                fp8_value,
                q4_weights,
                rows,
                cols,
                q4_output,
            ),
            args.warmup,
            args.repeats,
        ),
        "attention_64x640x512": _measure(
            lambda: attention_decode_cpu(
                query,
                raw_values,
                raw_positions,
                selected_values,
                sink,
                rope_cos,
                rope_sin,
                dim**-0.5,
            ),
            args.warmup,
            args.repeats,
        ),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
