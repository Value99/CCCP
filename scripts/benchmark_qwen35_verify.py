"""Measure Qwen3.5 resident main-model speculative verification only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from cccp.qwen35_model import Qwen35DenseVQModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=12)
    args = parser.parse_args()

    batch = max(2, int(args.batch))
    rounds = max(1, int(args.rounds))
    model = Qwen35DenseVQModel(
        str(args.model),
        device="cuda",
        max_ctx=max(512, 32 + batch * (rounds + 3)),
    )
    model.preload()
    model.reset_kv()
    prompt = [1000 + index for index in range(16)]
    model.forward_hidden(prompt)
    block = [2000 + index for index in range(batch)]

    # First call initializes vendor plans; the second captures the fixed-size
    # verifier graph. Neither call is part of the steady-state measurement.
    model.forward_hidden_verify(block)
    model.forward_hidden_verify(block)
    torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    for _ in range(rounds):
        model.forward_hidden_verify(block)
    torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - started
    print("[cccp-qwen35-verify-benchmark] " + json.dumps({
        "batch": batch,
        "rounds": rounds,
        "seconds": elapsed,
        "block_ms": elapsed * 1000.0 / rounds,
        "verified_tokens_per_second": batch * rounds / elapsed,
        "mode": model._gpu_mode,
        "cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
