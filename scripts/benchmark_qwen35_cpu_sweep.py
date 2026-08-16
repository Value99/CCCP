"""Measure one loaded Qwen3.5 Dense VQ image across CPU thread counts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine" / "CCCP-Engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from cccp.qwen35_model import Qwen35DenseVQModel  # noqa: E402
from cccp.cpuext import (  # noqa: E402
    reset_resident_projection_profile,
    reset_three_projection_phase_profile,
    resident_projection_profile,
    three_projection_phase_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threads", default="24,48,72,96")
    parser.add_argument("--prefill-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=64)
    args = parser.parse_args()
    candidates = tuple(
        dict.fromkeys(
            max(1, int(value.strip()))
            for value in args.threads.split(",")
            if value.strip()
        )
    )
    if not candidates:
        raise SystemExit("at least one thread count is required")

    torch.set_num_threads(candidates[0])
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    model = Qwen35DenseVQModel(
        args.model,
        device="cpu",
        max_ctx=max(256, args.prefill_tokens + args.decode_tokens + 8),
    )
    model.preload()
    load_seconds = time.perf_counter() - started
    results = []
    for threads in candidates:
        torch.set_num_threads(threads)
        model.reset_kv()
        prompt = [248000] * args.prefill_tokens
        tick = time.perf_counter()
        logits = model.forward(prompt)
        prefill_seconds = time.perf_counter() - tick
        for index in range(2):
            logits = model.forward([248001 + index])
        reset_resident_projection_profile()
        reset_three_projection_phase_profile()
        tick = time.perf_counter()
        for index in range(args.decode_tokens):
            logits = model.forward([248003 + index % 64])
        decode_seconds = time.perf_counter() - tick
        projection_profile = resident_projection_profile()
        mlp_profile = three_projection_phase_profile()
        results.append({
            "threads": threads,
            "prefill_tokens_per_second": (
                args.prefill_tokens / prefill_seconds
            ),
            "decode_tokens_per_second": args.decode_tokens / decode_seconds,
            "finite_logits": bool(torch.isfinite(logits).all()),
            "resident_projection_profile": projection_profile,
            "q4_swiglu_profile": mlp_profile,
        })
    print("[cccp-qwen35-cpu-sweep] " + json.dumps({
        "load_seconds": load_seconds,
        "logical_cpus": __import__("os").cpu_count(),
        "results": results,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
