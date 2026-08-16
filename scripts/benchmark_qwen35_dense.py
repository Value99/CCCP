"""Real-model Qwen3.5 Dense VQ smoke/performance benchmark.

This script intentionally exercises the same model adapter used by the API
server.  It does not alter or train the model.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from cccp.qwen35_model import Qwen35DenseVQModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--prefill-tokens", type=int, default=19)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--decode-warmup", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--profile-linears", action="store_true")
    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(max(1, args.torch_threads))
    started = time.perf_counter()
    model = Qwen35DenseVQModel(
        args.model,
        device=args.device,
        max_ctx=max(
            128,
            args.prefill_tokens + args.decode_tokens + args.decode_warmup + 8,
        ),
    )
    model.preload()
    load_seconds = time.perf_counter() - started

    linear_samples: dict[str, list[float]] = defaultdict(list)
    original_linear_forward = None
    if args.profile_linears:
        from cccp.dense_vq import DenseVQLinear

        original_linear_forward = DenseVQLinear.forward

        def timed_linear(module, value):
            tick = time.perf_counter()
            result = original_linear_forward(module, value)
            linear_samples[module.name].append(time.perf_counter() - tick)
            return result

        DenseVQLinear.forward = timed_linear

    prompt = [248000] * args.prefill_tokens
    started = time.perf_counter()
    logits = model.forward(prompt)
    if args.device == "cuda":
        torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("non-finite logits after prefill")

    vocab_size = int(getattr(model.network.config, "vocab_size", 0) or 0)
    decode_base = min(248001, max(0, vocab_size - 128))
    decode_span = max(1, min(100, vocab_size - decode_base))
    for index in range(max(0, args.decode_warmup)):
        logits = model.forward([decode_base + index % decode_span])
    if args.device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for index in range(args.decode_tokens):
        offset = index + max(0, args.decode_warmup)
        logits = model.forward([decode_base + offset % decode_span])
    if args.device == "cuda":
        torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - started
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("non-finite logits after decode")

    result = {
        "device": args.device,
        "load_seconds": load_seconds,
        "prefill_tokens": args.prefill_tokens,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": args.prefill_tokens / prefill_seconds,
        "decode_tokens": args.decode_tokens,
        "decode_warmup": max(0, args.decode_warmup),
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": args.decode_tokens / decode_seconds,
        "finite_logits": True,
        "cuda_memory_gib": (
            torch.cuda.memory_allocated() / 2**30
            if args.device == "cuda"
            else 0.0
        ),
    }
    if linear_samples:
        grouped: dict[str, float] = defaultdict(float)
        totals = {
            name: sum(samples) / max(1, len(samples))
            for name, samples in linear_samples.items()
        }
        for name, seconds in totals.items():
            leaf = name.rsplit(".", 2)[-2]
            grouped[leaf] += seconds
        result["linear_profile"] = {
            "sum_ms_per_token": sum(totals.values()) * 1000.0,
            "groups_ms_per_token": {
                key: value * 1000.0
                for key, value in sorted(
                    grouped.items(), key=lambda item: item[1], reverse=True
                )
            },
            "slowest": [
                [name, seconds * 1000.0]
                for name, seconds in sorted(
                    totals.items(), key=lambda item: item[1], reverse=True
                )[:12]
            ],
        }
    print("[cccp-qwen35-benchmark] " + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
