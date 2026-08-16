"""Benchmark one real CCCP mixed-VQ expert layer without loading the model.

The probe keeps the model's native top-k and matrix dimensions, but reads only
the selected experts.  It is intended for CPU scheduling/operator A/B tests;
no expert is removed from a launcher configuration by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine" / "CCCP-Engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))


def measure(call, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1_000.0)
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": ordered[0],
        "p90_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "maximum_ms": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="模型绑定配置；省略时选取 <model>/profiles 下第一份配置",
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--rank-offset", type=int, default=8)
    parser.add_argument(
        "--experts",
        help="comma-separated expert IDs; overrides profile rank selection",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    os.environ["CCCP_CPU_THREADS"] = str(max(1, args.threads))
    os.environ.setdefault("CCCP_CPU_AUTOBUILD", "0")

    from cccp.cpuext import (  # noqa: PLC0415
        configure_cpu_threads,
        extension_status,
        make_packed_three_layer_cpu,
    )
    from cccp.store import CCCPStore  # noqa: PLC0415

    actual_threads = configure_cpu_threads()
    profile_path = args.profile
    if profile_path is None:
        candidates = sorted((args.model / "profiles").glob("*.json"))
        if not candidates:
            raise SystemExit("模型目录下没有 profiles/*.json；请用 --profile 指定配置")
        profile_path = candidates[0]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    ranked = sorted(
        (
            (int(str(item["key"]).split(":", 1)[1]), int(item.get("route_count", 0)))
            for item in profile["experts"]
            if int(str(item["key"]).split(":", 1)[0]) == args.layer
        ),
        key=lambda item: (-item[1], item[0]),
    )
    selected = (
        [int(value) for value in args.experts.split(",")]
        if args.experts
        else [
            expert
            for expert, _ in ranked[
                args.rank_offset : args.rank_offset + args.top_k
            ]
        ]
    )
    if len(selected) != args.top_k:
        raise RuntimeError("profile does not contain enough experts for probe")

    store = CCCPStore(str(args.model))
    bundles = []
    metadata = []
    for expert in selected:
        bundle = store.load_expert_packed(args.layer, expert)
        if not all(weight.optimize_cpu_row_tile(8) for weight in bundle):
            raise RuntimeError(f"row-tile optimization failed for expert {expert}")
        bundles.append(bundle)
        metadata.append(
            {
                "expert": expert,
                "bits": [int(weight.bits) for weight in bundle],
                "dims": [int(weight.dim) for weight in bundle],
            }
        )

    executor = make_packed_three_layer_cpu(tuple(bundles), force_mixed=True)
    if executor is None:
        raise RuntimeError("mixed resident executor is unavailable")
    torch.manual_seed(5090 + args.layer)
    hidden = int(bundles[0][0].cols)
    value = torch.randn((1, hidden), dtype=torch.float32)
    ids = torch.arange(args.top_k, dtype=torch.int64)
    route = torch.full(
        (args.top_k,), 1.0 / float(args.top_k), dtype=torch.float32
    )

    result = {
        "layer": args.layer,
        "experts": metadata,
        "threads": actual_threads,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CCCP_CPU_L2_TASK_TILES",
                "CCCP_CPU_PACKED_DIRECT_ROWS8",
                "CCCP_CPU_PACKED_FUSED_GATE_UP",
                "CCCP_CPU_PACKED_FUSED_DOWN_REDUCE",
            )
        },
        "extension": extension_status(),
        "latency": measure(
            lambda: executor.forward(
                value, ids, route, 10.0, "swiglu", 1.0, -1.0
            ),
            args.warmup,
            args.repeats,
        ),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
