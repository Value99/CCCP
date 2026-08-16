"""Autotune the public packed three-projection Decode operator in one load."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import torch

from cccp.engine import Engine
from cccp.ops import packed_moe_topk
from cccp.presets import apply_preset_environment, resolve_preset


_TUNABLES = (
    "CCCP_PROJECTION_WARPS",
    "CCCP_PROJECTION_ROWS",
    "CCCP_PROJECTION_DOWN_REDUCE",
    "CCCP_PROJECTION_DOWN_ROWS",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    os.environ["CCCP_DSV4_TOKEN_GRAPH"] = "0"

    preset = resolve_preset(args.model, profile="resident", tp=1)
    apply_preset_environment(preset)
    engine = Engine(
        str(preset.model_dir),
        device="cuda",
        max_ctx=512,
        tp_size=1,
        dense_residency="gpu",
    )
    # Populate the fixed inputs and exact routes for every layer.
    engine.model.forward([1, 2, 3, 4])
    engine.model.forward([5])
    torch.cuda.synchronize(engine.model.device)
    model = engine.model
    pool = model.pool
    cfg = model.cfg
    layers = tuple(sorted(pool._metadata))
    activation = str(cfg.get("activation", "situ"))
    activation_beta = float(cfg.get("situ_beta", 4.0))
    linear_beta_value = cfg.get("situ_linear_beta")
    activation_linear_beta = (
        0.0 if linear_beta_value is None else float(linear_beta_value)
    )
    limit = float(cfg.get("swiglu_limit", 0.0))

    def run_layer(layer: int) -> torch.Tensor:
        hidden, output, result = pool._workspaces[0]
        return packed_moe_topk(
            model._tp_shared_mlp.input_hidden(layer).replicas[0],
            model._tp_route_buffers[layer][2][0].reshape(-1),
            model._tp_route_buffers[layer][1][0].reshape(-1),
            pool._metadata[layer][0],
            activation=activation,
            activation_beta=activation_beta,
            activation_linear_beta=activation_linear_beta,
            limit=limit,
            hidden_workspace=hidden,
            output_workspace=output,
            result=result,
            grouped_prefix=-1,
            **model.store.man.projection_operator_capability(layer),
        )

    for key in _TUNABLES:
        os.environ.pop(key, None)
    references = {}
    for layer in layers:
        references[layer] = run_layer(layer).clone()
    torch.cuda.synchronize(model.device)

    variants = itertools.product(
        (8, 16, 32),
        (1, 2, 4),
        (0, 1),
        (1, 2, 4),
    )
    results = []
    for warps, rows, reduce, down_rows in variants:
        os.environ["CCCP_PROJECTION_WARPS"] = str(warps)
        os.environ["CCCP_PROJECTION_ROWS"] = str(rows)
        os.environ["CCCP_PROJECTION_DOWN_REDUCE"] = str(reduce)
        os.environ["CCCP_PROJECTION_DOWN_ROWS"] = str(down_rows)
        for layer in layers:
            run_layer(layer)
        torch.cuda.synchronize(model.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_started = time.perf_counter()
        start_event.record()
        for _ in range(max(1, args.repeats)):
            for layer in layers:
                run_layer(layer)
        end_event.record()
        end_event.synchronize()
        cuda_ms = start_event.elapsed_time(end_event) / max(1, args.repeats)
        wall_ms = (
            (time.perf_counter() - wall_started) * 1000.0
            / max(1, args.repeats)
        )
        max_error = 0.0
        finite = True
        for layer in layers:
            value = run_layer(layer)
            finite = finite and bool(torch.isfinite(value).all())
            max_error = max(
                max_error,
                float((value - references[layer]).abs().max().item()),
            )
        results.append({
            "warps": warps,
            "rows": rows,
            "down_reduce": reduce,
            "down_rows": down_rows,
            "cuda_ms_per_token_moe": cuda_ms,
            "wall_ms_per_token_moe": wall_ms,
            "finite": finite,
            "max_abs_error": max_error,
        })
    results.sort(key=lambda item: item["cuda_ms_per_token_moe"])
    print(
        "[cccp-packed-moe-autotune] "
        + json.dumps(results, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
