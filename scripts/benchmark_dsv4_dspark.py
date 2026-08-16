"""Measure one resident DSV4 model with and without its DSpark attachment."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from cccp.chat_adapters import ChatMessage, ChatOptions, adapter_for_arch
from cccp.engine import Engine
from cccp.presets import apply_preset_environment, resolve_preset


def _sync(engine: Engine) -> None:
    import torch

    if engine.model.device.type == "cuda":
        torch.cuda.synchronize(engine.model.device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--drafts", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument(
        "--prompt",
        default="Write a detailed science fiction story.",
    )
    args = parser.parse_args()

    preset = resolve_preset(args.model, profile="resident", tp=1)
    apply_preset_environment(preset)
    engine = Engine(
        str(preset.model_dir),
        device="cuda",
        max_ctx=max(512, args.max_new + args.warmup + 128),
        tp_size=1,
        dense_residency="gpu",
    )
    adapter = adapter_for_arch(preset.architecture)
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

    # Build DSpark's lazy fixed tensors and expert cache outside the measured
    # interval.  This is equivalent to service startup warmup, not hidden
    # decode work.
    engine.generate_speculative(
        plan.input_ids,
        max_new=args.warmup,
        k=args.drafts,
    )
    _sync(engine)
    engine.reset()
    engine._cache_via_spec = False
    engine._cache_ids = None
    engine._dsp.reset()

    started = time.perf_counter()
    output = engine.generate_speculative(
        plan.input_ids,
        max_new=args.max_new,
        k=args.drafts,
    )
    _sync(engine)
    elapsed = time.perf_counter() - started
    stats = dict(engine.spec_stats or {})
    result = {
        "mode": "dsv4-dspark",
        "drafts": args.drafts,
        "tokens": len(output),
        "seconds": elapsed,
        "wall_tokens_per_second": len(output) / max(elapsed, 1e-9),
        "stats": stats,
        "output_ids": output,
        "output_text": engine.decode(output),
        "dspark_file": preset.manifest.get("dspark_file"),
        "runtime_backend": os.environ.get("CCCP_RUNTIME_BACKEND", "cuda"),
    }
    print("[cccp-dsv4-dspark-benchmark] " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
