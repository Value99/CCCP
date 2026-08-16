"""Measure real greedy and MTP generation in one loaded Qwen3.5 process."""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import torch

from cccp.chat_adapters.base import ChatMessage, ChatOptions
from cccp.chat_adapters.qwen35 import Qwen35ChatAdapter
from cccp.engine import Engine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--drafts", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--skip-greedy", action="store_true")
    parser.add_argument("--stage-timing", action="store_true")
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="benchmark-only: force max-new tokens for steady-state timing",
    )
    parser.add_argument(
        "--prompt",
        default="请用一句简短的中文介绍你自己。",
    )
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(max(1, args.torch_threads))

    engine = Engine(str(args.model), max_ctx=max(512, args.max_new + 64), device=args.device)
    if args.ignore_eos:
        engine.eos = set()
    adapter = Qwen35ChatAdapter()
    options = ChatOptions(
        thinking_mode="chat",
        reasoning_effort=None,
        temperature=0.0,
        top_p=1.0,
        max_new=args.max_new,
    )
    plan = adapter.prepare(
        engine,
        [ChatMessage(role="user", content=args.prompt)],
        options,
        None,
    )

    stage_seconds: dict[str, float] = collections.defaultdict(float)
    if args.stage_timing and args.device == "cuda":
        def wrap(owner, name: str, label: str) -> None:
            original = getattr(owner, name)

            def timed(*values, **keywords):
                torch.cuda.synchronize(engine.model.device)
                before = time.perf_counter()
                result = original(*values, **keywords)
                torch.cuda.synchronize(engine.model.device)
                stage_seconds[label] += time.perf_counter() - before
                return result

            setattr(owner, name, timed)

        wrap(engine.model.mtp, "step", "draft_step")
        wrap(engine.model.mtp, "prefill", "draft_prefill")
        wrap(engine.model.mtp, "crop", "draft_crop")
        wrap(engine.model, "snapshot_decode_state", "snapshot")
        wrap(engine.model, "restore_decode_state", "restore")
        wrap(engine.model, "forward_hidden_verify", "main_verify")
        wrap(engine.model, "forward_hidden", "main_forward_hidden")
        wrap(engine.model, "logits_of", "lm_head")

    results: list[dict[str, object]] = []
    greedy: list[int] = []
    if not args.skip_greedy:
        started = time.perf_counter()
        greedy = engine.generate(plan.input_ids, max_new=args.max_new, temp=0.0)
        if args.device == "cuda":
            torch.cuda.synchronize(engine.model.device)
        elapsed = time.perf_counter() - started
        results.append({
            "mode": "greedy",
            "seconds": elapsed,
            "tokens": len(greedy),
            "tokens_per_second": len(greedy) / max(elapsed, 1e-9),
            "output_ids": greedy,
            "output_text": engine.decode(greedy),
        })

    for draft in args.drafts:
        started = time.perf_counter()
        output = engine.generate_speculative(
            plan.input_ids,
            max_new=args.max_new,
            k=draft,
        )
        if args.device == "cuda":
            torch.cuda.synchronize(engine.model.device)
        elapsed = time.perf_counter() - started
        results.append({
            "mode": "mtp",
            "draft": draft,
            "seconds": elapsed,
            "tokens": len(output),
            "tokens_per_second": len(output) / max(elapsed, 1e-9),
            "matches_greedy": output == greedy,
            "stage_seconds": dict(stage_seconds),
            "stats": dict(engine.spec_stats or {}),
            "output_ids": output,
            "output_text": engine.decode(output),
        })

    print("[cccp-qwen35-mtp-benchmark] " + json.dumps(results, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
