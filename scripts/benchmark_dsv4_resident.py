"""Benchmark a fully resident DSV4 model through the production Engine path."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from cccp.chat_adapters import ChatMessage, ChatOptions, adapter_for_arch
from cccp.engine import Engine
from cccp.presets import apply_preset_environment, resolve_preset


def _sync(engine: Engine) -> None:
    if engine.model.device.type == "cuda":
        torch.cuda.synchronize(engine.model.device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument(
        "--stage-profile",
        action="store_true",
        help="Disable the parent token graph and time one uncaptured token.",
    )
    parser.add_argument(
        "--prompt", default="请用中文简短介绍动态专家推理。"
    )
    args = parser.parse_args()
    if args.stage_profile:
        # A captured parent graph and child-graph event instrumentation cannot
        # safely execute in the same CUDA context.  The diagnostic process
        # therefore starts directly in the ordinary fixed-address graph path.
        os.environ["CCCP_DSV4_TOKEN_GRAPH"] = "0"
    if args.device == "cpu" and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    preset = resolve_preset(args.model, profile="resident", tp=1)
    apply_preset_environment(preset)
    engine = Engine(
        str(preset.model_dir),
        device=args.device,
        max_ctx=max(512, args.max_new + args.warmup + 128),
        tp_size=1,
        dense_residency="gpu" if args.device == "cuda" else "auto",
    )
    engine.eos = set()
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

    # Warm every lazy fixed-address graph before timing steady-state Decode.
    warm_output = engine.generate(
        plan.input_ids, max_new=max(1, args.warmup), temp=0.0
    )
    _sync(engine)
    current = warm_output[-1] if warm_output else plan.input_ids[-1]
    output: list[int] = []
    started = time.perf_counter()
    for _ in range(args.max_new):
        logits = engine.model.forward([current])
        current = int(torch.argmax(logits.float()).item())
        output.append(current)
    _sync(engine)
    elapsed = time.perf_counter() - started

    profile = {}
    finite_logits = True
    if args.stage_profile:
        # One uncaptured token exposes attention/MoE substage costs.  It is
        # outside the throughput interval and uses the same resident weights.
        probe_id = output[-1] if output else plan.input_ids[-1]
        engine.model.start_profile()
        probe_logits = engine.model.forward([probe_id])
        profile = engine.model.finish_profile()
        finite_logits = bool(torch.isfinite(probe_logits).all())
    result = {
        "mode": "dsv4-resident",
        "device": args.device,
        "tokens": len(output),
        "seconds": elapsed,
        "tokens_per_second": len(output) / max(elapsed, 1.0e-9),
        "finite_logits": finite_logits,
        "stage_profile": profile,
        "output_ids": output,
        "output_text": engine.decode(output),
    }
    print("[cccp-dsv4-resident-benchmark] " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
