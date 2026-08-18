"""Run a real two-turn Qwen3.5 Dense VQ chat regression.

The validator uses the same Engine and chat adapter as the OpenAI endpoint.
JSON is emitted with ASCII escapes so Windows, SSH, and redirected logs cannot
turn valid Chinese text into replacement characters.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from cccp.chat_adapters.base import ChatMessage, ChatOptions
from cccp.chat_adapters.qwen35 import Qwen35ChatAdapter
from cccp.engine import Engine


def _options(max_new: int) -> ChatOptions:
    return ChatOptions(
        thinking_mode="chat",
        reasoning_effort=None,
        temperature=0.0,
        top_p=1.0,
        max_new=max_new,
    )


def _run_turn(
    engine: Engine,
    adapter: Qwen35ChatAdapter,
    messages: list[ChatMessage],
    options: ChatOptions,
    ledger,
    *,
    speculative: bool,
    draft_tokens: int,
):
    plan = adapter.prepare(engine, messages, options, ledger)
    started = time.perf_counter()
    if speculative:
        output_ids = engine.generate_speculative(
            plan.input_ids,
            max_new=options.max_new,
            k=draft_tokens,
        )
    else:
        output_ids = engine.generate(
            plan.input_ids,
            max_new=options.max_new,
            temp=options.temperature,
            top_p=options.top_p,
        )
    if engine.model.device.type == "cuda":
        torch.cuda.synchronize(engine.model.device)
    elapsed = time.perf_counter() - started
    parsed = adapter.parse_complete(engine, output_ids, options)
    ledger = adapter.commit(engine, plan, output_ids, parsed)
    stats = engine.last_kv_stats
    return ledger, parsed, {
        "input_tokens": len(plan.input_ids),
        "output_tokens": len(output_ids),
        "output_ids": output_ids,
        "content": parsed.content,
        "reasoning_content": parsed.reasoning_content,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": len(output_ids) / max(elapsed, 1e-9),
        "kv": None if stats is None else {
            "mode": stats.mode,
            "reason": stats.reason,
            "baseline_tokens": stats.baseline_tokens,
            "lcp_tokens": stats.lcp_tokens,
            "suffix_tokens": stats.suffix_tokens,
            "prefill_ms": stats.prefill_ms,
        },
        "speculative": dict(getattr(engine, "spec_stats", None) or {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument("--draft-tokens", type=int, default=5)
    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(max(1, args.torch_threads))
    engine = Engine(
        str(args.model),
        max_ctx=max(512, args.max_new * 3),
        device=args.device,
    )
    adapter = Qwen35ChatAdapter()
    options = _options(args.max_new)
    messages = [ChatMessage(role="user", content="请用一句简短的中文介绍你自己。")]
    ledger, first, first_result = _run_turn(
        engine,
        adapter,
        messages,
        options,
        None,
        speculative=args.speculative,
        draft_tokens=args.draft_tokens,
    )
    messages.extend([
        ChatMessage(role="assistant", content=first.content),
        ChatMessage(role="user", content="刚才我让你做什么？请简短回答。"),
    ])
    _, _, second_result = _run_turn(
        engine,
        adapter,
        messages,
        options,
        ledger,
        speculative=args.speculative,
        draft_tokens=args.draft_tokens,
    )
    print("[cccp-qwen35-chat] " + json.dumps(
        {"turns": [first_result, second_result]},
        ensure_ascii=True,
        separators=(",", ":"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
