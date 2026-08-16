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
    parser.add_argument("--profile-prefill", action="store_true")
    parser.add_argument("--torch-profile", action="store_true")
    parser.add_argument(
        "--torch-profile-phase",
        choices=("prefill", "decode"),
        default="decode",
    )
    parser.add_argument("--torch-profile-rows", type=int, default=40)
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
    module_samples: dict[str, list[float]] = defaultdict(list)
    module_hooks = []
    original_linear_forward = None
    original_group_forward = None
    group_names: dict[int, str] = {}
    if args.profile_linears:
        from cccp.dense_vq import DenseVQLinear, DenseVQLinearGroup

        group_names = {
            id(module): name
            for name, module in model.network.named_modules()
            if isinstance(module, DenseVQLinearGroup)
        }

        original_linear_forward = DenseVQLinear.forward
        original_group_forward = DenseVQLinearGroup._forward_combined

        def timed_linear(module, value):
            tick = time.perf_counter()
            result = original_linear_forward(module, value)
            linear_samples[module.name].append(time.perf_counter() - tick)
            return result

        def timed_group(module, value):
            tick = time.perf_counter()
            result = original_group_forward(module, value)
            name = group_names.get(id(module), f"group@{id(module):x}")
            linear_samples[name].append(time.perf_counter() - tick)
            return result

        DenseVQLinear.forward = timed_linear
        DenseVQLinearGroup._forward_combined = timed_group

    if args.profile_prefill:
        def attach_timer(name, module):
            starts: list[float] = []

            def before(_module, _args):
                starts.append(time.perf_counter())

            def after(_module, _args, output):
                del output
                module_samples[name].append(
                    time.perf_counter() - starts.pop()
                )

            module_hooks.append(module.register_forward_pre_hook(before))
            module_hooks.append(module.register_forward_hook(after))

        for layer_index, layer in enumerate(model.network.model.layers):
            mixer = getattr(layer, "linear_attn", None)
            mixer_kind = "delta_attention"
            if mixer is None:
                mixer = getattr(layer, "self_attn", None)
                mixer_kind = "full_attention"
            if mixer is not None:
                attach_timer(f"{mixer_kind}.L{layer_index}", mixer)
            attach_timer(f"mlp.L{layer_index}", layer.mlp)
        attach_timer("final_norm", model.network.model.norm)
        attach_timer("lm_head", model.network.lm_head)

    prompt = [248000] * args.prefill_tokens
    profiler = None
    if args.torch_profile and args.torch_profile_phase == "prefill":
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        )
        profiler.__enter__()
    started = time.perf_counter()
    try:
        logits = model.forward(prompt)
    finally:
        if profiler is not None:
            profiler.__exit__(None, None, None)
    if args.device == "cuda":
        torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("non-finite logits after prefill")
    prefill_linear_samples = (
        {name: list(samples) for name, samples in linear_samples.items()}
        if args.profile_prefill
        else {}
    )
    for hook in module_hooks:
        hook.remove()

    vocab_size = int(getattr(model.network.config, "vocab_size", 0) or 0)
    decode_base = min(248001, max(0, vocab_size - 128))
    decode_span = max(1, min(100, vocab_size - decode_base))
    for index in range(max(0, args.decode_warmup)):
        logits = model.forward([decode_base + index % decode_span])
    if args.device == "cuda":
        torch.cuda.synchronize()
    # Prefill and warmup use different matrix shapes and must not pollute the
    # single-token Decode profile below.
    linear_samples.clear()
    if args.profile_linears and args.device == "cpu":
        from cccp.cpuext import (
            reset_resident_projection_profile,
            reset_three_projection_phase_profile,
        )

        reset_resident_projection_profile()
        reset_three_projection_phase_profile()
    started = time.perf_counter()
    decode_profiler = None
    if args.torch_profile and args.torch_profile_phase == "decode":
        activities = [torch.profiler.ProfilerActivity.CPU]
        if args.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        decode_profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        )
        decode_profiler.__enter__()
    try:
        for index in range(args.decode_tokens):
            offset = index + max(0, args.decode_warmup)
            logits = model.forward([decode_base + offset % decode_span])
    finally:
        if decode_profiler is not None:
            decode_profiler.__exit__(None, None, None)
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
    if prefill_linear_samples:
        prefill_totals = {
            name: sum(samples) / max(1, len(samples))
            for name, samples in prefill_linear_samples.items()
        }
        prefill_grouped: dict[str, float] = defaultdict(float)
        for name, seconds in prefill_totals.items():
            leaf = name.rsplit(".", 2)[-2]
            prefill_grouped[leaf] += seconds
        divisor = max(1, args.prefill_tokens)
        result["prefill_linear_profile"] = {
            "sum_seconds": sum(prefill_totals.values()),
            "sum_ms_per_token": (
                sum(prefill_totals.values()) * 1000.0 / divisor
            ),
            "groups_seconds": {
                key: value
                for key, value in sorted(
                    prefill_grouped.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
            "slowest": [
                [name, seconds]
                for name, seconds in sorted(
                    prefill_totals.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:12]
            ],
        }
    if module_samples:
        module_totals = {
            name: sum(samples)
            for name, samples in module_samples.items()
        }
        module_grouped: dict[str, float] = defaultdict(float)
        for name, seconds in module_totals.items():
            module_grouped[name.split(".", 1)[0]] += seconds
        result["prefill_module_profile"] = {
            "groups_seconds": dict(sorted(
                module_grouped.items(),
                key=lambda item: item[1],
                reverse=True,
            )),
            "slowest": [
                [name, seconds]
                for name, seconds in sorted(
                    module_totals.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:16]
            ],
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
        if args.device == "cpu":
            from cccp.cpuext import (
                resident_projection_profile,
                three_projection_phase_profile,
            )

            result["native_projection_profile"] = (
                resident_projection_profile()
            )
            result["native_three_projection_profile"] = (
                three_projection_phase_profile()
            )
    active_profiler = decode_profiler or profiler
    if active_profiler is not None:
        print(
            f"[cccp-qwen35-torch-profile phase={args.torch_profile_phase}]\n"
            + active_profiler.key_averages(
                group_by_input_shape=True
            ).table(
                sort_by=(
                    "self_cuda_time_total"
                    if args.device == "cuda"
                    else "self_cpu_time_total"
                ),
                row_limit=max(1, args.torch_profile_rows),
            ),
            flush=True,
        )
    print("[cccp-qwen35-benchmark] " + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
