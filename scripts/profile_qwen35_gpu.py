"""One-token CUDA/CPU profile for the real Qwen3.5 Dense adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cccp.qwen35_model import Qwen35DenseVQModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    model = Qwen35DenseVQModel(
        args.model,
        device="cuda",
        max_ctx=128,
    )
    model.preload()
    model.forward([248000, 248001, 248002, 248003])
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
        model.forward([248100])
        torch.cuda.synchronize()

    print("[cccp-qwen35-profile-cuda]")
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
