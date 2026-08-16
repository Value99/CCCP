"""Benchmark exact routed CPU packed-MoE rows on a real CCCP archive."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = ROOT / "engine" / "CCCP-Engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

os.environ.setdefault("CCCP_CPU_AUTOBUILD", "0")
os.environ.setdefault("CCCP_CPU_THREADS", "auto")

from cccp.cpuext import configure_cpu_threads, extension_status  # noqa: E402
from cccp.ops import packed_moe_selected_rows, packed_moe_selected_topk  # noqa: E402
from cccp.store import CCCPStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=5090)
    args = parser.parse_args()

    threads = configure_cpu_threads()
    store = CCCPStore(str(Path(args.model).resolve()))
    hidden = int(store.cfg["hidden"])
    top_k = int(store.cfg["top_k"])
    expert_count = int(store.cfg["n_experts"])
    generator = torch.Generator().manual_seed(args.seed)
    route_ids = torch.empty(args.rows, top_k, dtype=torch.long)
    for row in range(args.rows):
        route_ids[row] = torch.tensor(
            [((row * 37) + (slot * 41)) % expert_count for slot in range(top_k)]
        )
    unique = sorted({int(item) for item in route_ids.reshape(-1)})
    loaded = {
        expert: store.load_expert_packed(args.layer, expert)
        for expert in unique
    }
    nested = [
        [loaded[int(expert)] for expert in row]
        for row in route_ids
    ]
    values = torch.randn(args.rows, hidden, generator=generator)
    route_weights = torch.rand(args.rows, top_k, generator=generator)
    route_weights /= route_weights.sum(dim=1, keepdim=True)

    def batch() -> torch.Tensor:
        result = packed_moe_selected_rows(
            values,
            nested,
            route_weights,
            activation="swiglu",
            activation_beta=4.0,
            activation_linear_beta=-1.0,
            limit=float(store.cfg.get("swiglu_limit", 0.0)),
        )
        if result is None:
            raise RuntimeError("batched packed-MoE rows operator unavailable")
        return result.clone()

    def serial() -> torch.Tensor:
        output = []
        for row in range(args.rows):
            result = packed_moe_selected_topk(
                values[row : row + 1],
                nested[row],
                route_weights[row],
                activation="swiglu",
                activation_beta=4.0,
                activation_linear_beta=-1.0,
                limit=float(store.cfg.get("swiglu_limit", 0.0)),
            )
            if result is None:
                raise RuntimeError("single-row packed-MoE operator unavailable")
            output.append(result.clone())
        return torch.stack(output)

    batch()
    serial()
    started = time.perf_counter()
    batched = batch()
    batch_seconds = time.perf_counter() - started
    started = time.perf_counter()
    sequential = serial()
    serial_seconds = time.perf_counter() - started
    difference = (batched - sequential).abs()
    print(json.dumps({
        "model": str(Path(args.model).resolve()),
        "layer": args.layer,
        "rows": args.rows,
        "top_k": top_k,
        "unique_experts": len(unique),
        "threads": threads,
        "extension": extension_status(),
        "batch_seconds": round(batch_seconds, 6),
        "serial_seconds": round(serial_seconds, 6),
        "speedup": round(serial_seconds / max(batch_seconds, 1e-12), 4),
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "finite": bool(torch.isfinite(batched).all()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
