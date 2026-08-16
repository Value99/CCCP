"""Measure production Prefill and Decode scaling without changing weights.

The benchmark loads one CCCP model once, runs exact-length synthetic prompts
through the public model executor, and reports Decode throughput in consecutive
windows.  The same script is used for a fully resident execution image and a
hard VRAM limit, so H2D/cache traffic cannot be hidden by different drivers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from cccp.engine import Engine
from cccp.presets import apply_preset_environment, resolve_preset


def _sync(engine: Engine) -> None:
    device = getattr(engine.model, "device", torch.device("cpu"))
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _pool_counters(engine: Engine) -> dict[str, float | int]:
    pool = getattr(engine.model, "pool", None)
    if pool is None:
        return {}
    names = (
        "hits",
        "miss",
        "prefetch_hits",
        "uploaded_bytes",
        "transfer_seconds",
        "bytes",
        "ram_bytes",
        "_host_pinned_bytes",
    )
    result: dict[str, float | int] = {}
    for name in names:
        value = getattr(pool, name, None)
        if isinstance(value, (int, float)):
            result[name] = value
    return result


def _counter_delta(
    before: dict[str, float | int],
    after: dict[str, float | int],
) -> dict[str, float | int]:
    return {
        key: after[key] - before.get(key, 0)
        for key in after
        if isinstance(after[key], (int, float))
    }


def _tokens(engine: Engine, count: int, seed: str) -> list[int]:
    encoded = list(engine.encode(seed))
    if not encoded:
        raise RuntimeError("tokenizer returned an empty seed")
    repeat = (int(count) + len(encoded) - 1) // len(encoded)
    return (encoded * repeat)[: int(count)]


def _cuda_memory(engine: Engine) -> dict[str, float]:
    device = getattr(engine.model, "device", torch.device("cpu"))
    if torch.device(device).type != "cuda":
        return {}
    return {
        "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
        "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }


def _logit_health(logits: torch.Tensor) -> dict[str, float | int | bool]:
    values = logits.float().reshape(-1)
    finite = torch.isfinite(values)
    finite_count = int(finite.sum().item())
    finite_values = values[finite]
    return {
        "values": int(values.numel()),
        "finite": finite_count,
        "nan": int(torch.isnan(values).sum().item()),
        "positive_inf": int(torch.isposinf(values).sum().item()),
        "negative_inf": int(torch.isneginf(values).sum().item()),
        "finite_min": (
            float(finite_values.min().item()) if finite_count else 0.0
        ),
        "finite_max": (
            float(finite_values.max().item()) if finite_count else 0.0
        ),
        "usable": finite_count > 0 and not bool(torch.isnan(values).any()),
    }


class _CudaStageTimer:
    """Collect exact GPU time for the two expensive DSV4 block stages."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.enabled = False
        self.events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def wrap(self, name: str, callback):
        def timed(*args, **kwargs):
            if not self.enabled:
                return callback(*args, **kwargs)
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            result = callback(*args, **kwargs)
            end.record()
            self.events.append((name, begin, end))
            return result

        return timed

    def begin(self) -> None:
        self.events.clear()
        self.enabled = True

    def finish(self, wall_seconds: float) -> dict[str, float | int]:
        self.enabled = False
        _sync(self.engine)
        totals: dict[str, float] = {}
        for name, begin, end in self.events:
            totals[name] = totals.get(name, 0.0) + float(
                begin.elapsed_time(end)
            )
        gpu_total = sum(totals.values())
        return {
            "attention_ms": totals.get("attention", 0.0),
            "moe_ms": totals.get("moe", 0.0),
            "measured_gpu_ms": gpu_total,
            "wall_ms": float(wall_seconds) * 1000.0,
            "unattributed_wall_ms": max(
                0.0, float(wall_seconds) * 1000.0 - gpu_total
            ),
            "component_calls": len(self.events),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--profile",
        choices=("ram", "mapped", "resident"),
        default="resident",
    )
    parser.add_argument(
        "--prefill-lengths",
        type=int,
        nargs="+",
        default=(32, 512, 2048, 4096),
    )
    parser.add_argument(
        "--decode-contexts", type=int, nargs="+", default=(32, 4096)
    )
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--decode-window", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument(
        "--trace-numerics",
        action="store_true",
        help="Report the first non-finite hidden state after each model block.",
    )
    parser.add_argument(
        "--stage-timing",
        action="store_true",
        help="Report CUDA Attention and routed-MoE time for each case.",
    )
    parser.add_argument(
        "--decode-stage-probe",
        action="store_true",
        help="Time one eager Decode token with the model's stage profiler.",
    )
    parser.add_argument(
        "--seed",
        default="动态专家推理需要稳定、准确并保持上下文一致。",
    )
    args = parser.parse_args()
    lengths = sorted({max(1, int(value)) for value in args.prefill_lengths})
    contexts = sorted({max(1, int(value)) for value in args.decode_contexts})
    if args.device == "cpu" and args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))

    preset = resolve_preset(args.model, profile=args.profile, tp=1)
    apply_preset_environment(preset)
    max_ctx = max(lengths + contexts) + max(0, args.decode_tokens) + 16
    loaded = time.perf_counter()
    engine = Engine(
        str(preset.model_dir),
        device=args.device,
        max_ctx=max_ctx,
        tp_size=1,
        dense_residency="gpu" if args.device == "cuda" else "auto",
    )
    _sync(engine)
    load_seconds = time.perf_counter() - loaded

    # Build lazy kernels/graphs outside every measured interval.
    engine.reset()
    warm = _tokens(engine, 4, args.seed)
    logits = engine.model.forward(warm)
    logits = engine.model.forward([int(torch.argmax(logits.float()).item())])
    _sync(engine)
    warmup_health = _logit_health(logits)
    if not bool(warmup_health["usable"]):
        print(
            "[cccp-context-scaling] warmup-logits="
            + json.dumps(warmup_health, sort_keys=True),
            flush=True,
        )

    stage_timer = None
    if (
        args.stage_timing
        and args.device == "cuda"
        and hasattr(engine.model, "_attn_batch")
        and hasattr(engine.model, "_moe")
    ):
        stage_timer = _CudaStageTimer(engine)
        engine.model._attn_batch = stage_timer.wrap(
            "attention", engine.model._attn_batch
        )
        engine.model._moe = stage_timer.wrap("moe", engine.model._moe)

    if args.trace_numerics and hasattr(engine.model, "_block"):
        original_block = engine.model._block
        original_attention = engine.model._attn_batch
        original_moe = engine.model._moe

        def report_component(component, layer, hidden, result):
            if int(hidden.shape[1]) <= 32 or int(layer) != 0:
                return
            tensors = result if isinstance(result, tuple) else (result,)
            for part, tensor in enumerate(tensors):
                health = _logit_health(tensor)
                print(
                    "[cccp-context-component] "
                    + json.dumps({
                        "component": component,
                        "layer": int(layer),
                        "part": int(part),
                        "tokens": int(hidden.shape[1]),
                        **health,
                    }, sort_keys=True),
                    flush=True,
                )

        def traced_attention(hidden, layer, *values, **keywords):
            result = original_attention(hidden, layer, *values, **keywords)
            report_component("attention", layer, hidden, result)
            return result

        def traced_moe(hidden, layer, *values, **keywords):
            result = original_moe(hidden, layer, *values, **keywords)
            report_component("moe", layer, hidden, result)
            return result

        def traced_block(hidden, layer, *values, **keywords):
            result = original_block(hidden, layer, *values, **keywords)
            if int(hidden.shape[1]) > 32 and int(layer) == 0:
                health = _logit_health(result)
                print(
                    "[cccp-context-numerics] "
                    + json.dumps({
                        "layer": int(layer),
                        "tokens": int(hidden.shape[1]),
                        **health,
                    }, sort_keys=True),
                    flush=True,
                )
            return result

        engine.model._attn_batch = traced_attention
        engine.model._moe = traced_moe
        engine.model._block = traced_block

    prefill_results: list[dict[str, object]] = []
    for length in lengths:
        engine.reset()
        prompt = _tokens(engine, length, args.seed)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats(engine.model.device)
        counters_before = _pool_counters(engine)
        if stage_timer is not None:
            stage_timer.begin()
        started = time.perf_counter()
        logits = engine.model.forward(prompt)
        _sync(engine)
        seconds = time.perf_counter() - started
        stages = (
            stage_timer.finish(seconds) if stage_timer is not None else {}
        )
        counters_after = _pool_counters(engine)
        prefill_results.append({
            "tokens": length,
            "seconds": seconds,
            "tokens_per_second": length / max(seconds, 1.0e-9),
            "logits": _logit_health(logits),
            "cache_delta": _counter_delta(counters_before, counters_after),
            "memory": _cuda_memory(engine),
            "stages": stages,
        })

    decode_results: list[dict[str, object]] = []
    for context in contexts:
        engine.reset()
        prompt = _tokens(engine, context, args.seed)
        logits = engine.model.forward(prompt)
        _sync(engine)
        current = int(torch.argmax(logits.float()).item())
        decode_probe: dict[str, object] = {}
        if (
            args.decode_stage_probe
            and hasattr(engine.model, "start_profile")
            and hasattr(engine.model, "finish_profile")
        ):
            engine.model.start_profile()
            logits = engine.model.forward([current])
            _sync(engine)
            decode_probe = engine.model.finish_profile()
            current = int(torch.argmax(logits.float()).item())
        counters_before = _pool_counters(engine)
        windows: list[dict[str, float | int]] = []
        window_started = time.perf_counter()
        total_started = window_started
        if stage_timer is not None:
            stage_timer.begin()
        for index in range(max(0, int(args.decode_tokens))):
            logits = engine.model.forward([current])
            current = int(torch.argmax(logits.float()).item())
            boundary = (
                (index + 1) % max(1, int(args.decode_window)) == 0
                or index + 1 == int(args.decode_tokens)
            )
            if boundary:
                _sync(engine)
                now = time.perf_counter()
                previous = windows[-1]["end_token"] if windows else 0
                count = index + 1 - int(previous)
                elapsed = now - window_started
                windows.append({
                    "start_token": int(previous) + 1,
                    "end_token": index + 1,
                    "seconds": elapsed,
                    "tokens_per_second": count / max(elapsed, 1.0e-9),
                })
                window_started = now
        _sync(engine)
        total_seconds = time.perf_counter() - total_started
        stages = (
            stage_timer.finish(total_seconds)
            if stage_timer is not None
            else {}
        )
        counters_after = _pool_counters(engine)
        decode_results.append({
            "context_tokens": context,
            "decode_tokens": int(args.decode_tokens),
            "seconds": total_seconds,
            "tokens_per_second": (
                int(args.decode_tokens) / max(total_seconds, 1.0e-9)
            ),
            "logits": _logit_health(logits),
            "windows": windows,
            "cache_delta": _counter_delta(counters_before, counters_after),
            "memory": _cuda_memory(engine),
            "stages": stages,
            "stage_probe": decode_probe,
        })

    report = {
        "architecture": preset.architecture,
        "device": args.device,
        "profile": args.profile,
        "load_seconds": load_seconds,
        "max_ctx": max_ctx,
        "operator": getattr(engine.model, "packed_operator_name", None),
        "warmup_logits": warmup_health,
        "prefill": prefill_results,
        "decode": decode_results,
    }
    print("[cccp-context-scaling] " + json.dumps(report, sort_keys=True))
    return 0 if all(
        item["logits"]["usable"]
        for item in prefill_results + decode_results
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
