"""CUDA/CPU profile for a real Qwen3.5 Dense decode or Prefill block."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cccp.qwen35_model import Qwen35DenseVQModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=4,
        help="Build this many KV/recurrent-state tokens before profiling.",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=1,
        help="Profile this many tokens in one model.forward call.",
    )
    args = parser.parse_args()

    model = Qwen35DenseVQModel(
        args.model,
        device="cuda",
        max_ctx=max(
            128,
            args.context_tokens + args.tokens + args.warmup + 8,
        ),
    )
    model.preload()
    model.forward([
        248000 + index % 64
        for index in range(max(1, args.context_tokens))
    ])
    for index in range(args.warmup):
        model.forward([248010 + index])
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profile:
        model.forward([
            248100 + index % 64
            for index in range(max(1, args.tokens))
        ])
        torch.cuda.synchronize()

    print(
        "[cccp-qwen35-profile-cuda] "
        f"context={max(1, args.context_tokens)} "
        f"tokens={max(1, args.tokens)}"
    )
    print(
        profile.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total",
            row_limit=60,
        )
    )
    print("[cccp-qwen35-profile-cpu]")
    print(
        profile.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total",
            row_limit=40,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
