"""Profile real Qwen3.5 MTP generation after one warm request."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cccp.chat_adapters.base import ChatMessage, ChatOptions
from cccp.chat_adapters.qwen35 import Qwen35ChatAdapter
from cccp.engine import Engine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()

    engine = Engine(str(args.model), max_ctx=512, device="cuda")
    plan = Qwen35ChatAdapter().prepare(
        engine,
        [ChatMessage(role="user", content="请简短介绍你自己。")],
        ChatOptions(
            thinking_mode="chat",
            reasoning_effort=None,
            temperature=0.0,
            top_p=1.0,
            max_new=args.tokens,
        ),
        None,
    )
    engine.generate_speculative(plan.input_ids, max_new=8, k=args.draft)
    torch.cuda.synchronize(engine.model.device)

    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profile:
        engine.generate_speculative(
            plan.input_ids,
            max_new=args.tokens,
            k=args.draft,
        )
        torch.cuda.synchronize(engine.model.device)
    print("[cccp-qwen35-mtp-profile-cuda]")
    print(profile.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_time_total", row_limit=80
    ))
    print("[cccp-qwen35-mtp-profile-cpu]")
    print(profile.key_averages(group_by_input_shape=True).table(
        sort_by="self_cpu_time_total", row_limit=60
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
