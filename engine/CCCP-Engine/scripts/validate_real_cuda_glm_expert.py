"""Validate the legacy two-projection GLM CUDA expert path with real weights.

The probe compares the production top-k slot kernel against an explicit dense
reconstruction.  It is intentionally decode-only: GLM Prefill uses expert-
grouped row batches and must never call this single-token operator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from cccp.grouped import moe_mlp_grouped_mixed
from cccp.store import CCCPStore, ExpertPool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare real GLM CUDA slot experts with dense reference"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--experts", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--prefill-rows", type=int, default=64)
    parser.add_argument("--prefill-top-k", type=int, default=4)
    parser.add_argument("--json")
    return parser


def _stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    difference = actual_f - expected_f
    return {
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "relative_l2": float(
            difference.norm().div(expected_f.norm().clamp_min(1e-12)).item()
        ),
        "actual_norm": float(actual_f.norm().item()),
        "expected_norm": float(expected_f.norm().item()),
    }


def main() -> None:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda:0")
    store = CCCPStore(str(Path(args.model).resolve()))
    expert_ids = tuple(
        int(item.strip()) for item in args.experts.split(",") if item.strip()
    )
    if not expert_ids or len(expert_ids) > 8:
        raise SystemExit("--experts must contain between one and eight ids")
    layer = int(args.layer)
    hidden = int(store.cfg.get("routed_hidden", store.cfg["hidden"]))
    intermediate = int(store.cfg["moe_inter"])
    torch.manual_seed(int(args.seed))
    host_experts = [store.load_expert(layer, expert) for expert in expert_ids]
    device_experts = [
        tuple(weight.to(device) for weight in expert)
        for expert in host_experts
    ]
    value = torch.randn(1, hidden, dtype=torch.bfloat16, device=device)
    route_weights = torch.rand(
        len(expert_ids), dtype=torch.float32, device=device
    )
    route_weights.div_(route_weights.sum())
    actual = moe_mlp_grouped_mixed(
        value,
        device_experts,
        route_weights,
        activation="silu",
    ).clone()

    expected = torch.zeros(hidden, dtype=torch.float32, device=device)
    for position, (gu, down) in enumerate(host_experts):
        gu_dense = gu.dequant().to(device=device, dtype=torch.bfloat16)
        projected = value @ gu_dense.t()
        del gu_dense
        activated = F.silu(projected[:, :intermediate]) * projected[:, intermediate:]
        down_dense = down.dequant().to(device=device, dtype=torch.bfloat16)
        expert_output = (activated @ down_dense.t()).reshape(-1).float()
        del down_dense
        expected.add_(expert_output, alpha=float(route_weights[position].item()))
    torch.cuda.synchronize(device)

    # Exercise the production multi-token path independently from the model
    # wrapper: full token rows, expert-chunk dequantization, two grouped GEMMs
    # and FP32 weighted route accumulation.
    prefill_rows = max(2, int(args.prefill_rows))
    prefill_top_k = min(
        len(expert_ids),
        max(1, int(args.prefill_top_k)),
    )
    prefill_value = torch.randn(
        prefill_rows,
        hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    route_ids = torch.empty(
        prefill_rows,
        prefill_top_k,
        dtype=torch.long,
        device=device,
    )
    for row in range(prefill_rows):
        route_ids[row] = torch.tensor(
            [
                expert_ids[(row + offset) % len(expert_ids)]
                for offset in range(prefill_top_k)
            ],
            dtype=torch.long,
            device=device,
        )
    prefill_weights = torch.rand(
        prefill_rows,
        prefill_top_k,
        dtype=torch.float32,
        device=device,
    )
    prefill_weights.div_(prefill_weights.sum(dim=1, keepdim=True))
    pool = object.__new__(ExpertPool)
    pool.gpu = True
    pool.device = device
    pool.store = store
    pool._prefill_executor_announced = False
    pool._prefill_dequant_workspace = None
    pool.prefill_batch_rows = 0
    pool.prefill_batch_submissions = 0
    pool.prefill_batch_max = 0
    pool.prefill_expert_chunk_capacity = 0
    pool.prefill_expert_chunk_submissions = 0
    pool.prefill_layer_unique_max = 0
    by_key = {
        (layer, expert_id): expert
        for expert_id, expert in zip(expert_ids, device_experts)
    }
    pool.get_many = lambda keys: {key: by_key[key] for key in keys}
    prefill_actual = pool.run_rows(
        layer,
        prefill_value,
        route_ids,
        prefill_weights,
        activation="silu",
        activation_beta=4.0,
        activation_linear_beta=None,
    ).clone()

    prefill_expected = torch.zeros_like(prefill_actual)
    for expert_id, (gu, down) in zip(expert_ids, host_experts):
        token_rows, slots = (route_ids == expert_id).nonzero(as_tuple=True)
        if token_rows.numel() == 0:
            continue
        gu_dense = gu.dequant().to(device=device, dtype=torch.bfloat16)
        projected = prefill_value.index_select(0, token_rows) @ gu_dense.t()
        del gu_dense
        activated = F.silu(projected[:, :intermediate]) * projected[:, intermediate:]
        down_dense = down.dequant().to(device=device, dtype=torch.bfloat16)
        expert_output = activated @ down_dense.t()
        del down_dense
        prefill_expected.index_add_(
            0,
            token_rows,
            expert_output.float()
            * prefill_weights[token_rows, slots].unsqueeze(1),
        )
    torch.cuda.synchronize(device)
    report = {
        "model": str(Path(args.model).resolve()),
        "layer": layer,
        "experts": list(expert_ids),
        "layouts": [
            [
                {
                    "index_dtype": str(weight.idx.dtype),
                    "index_shape": list(weight.idx.shape),
                    "codebook_shape": list(weight.cb.shape),
                }
                for weight in expert
            ]
            for expert in host_experts
        ],
        "slot_kernel_vs_dense": _stats(actual, expected),
        "prefill": {
            "rows": prefill_rows,
            "top_k": prefill_top_k,
            "executor": "cuda.chunked-dequant-grouped-gemm",
            "single_token_projection": "forbidden",
            "grouped_vs_dense": _stats(prefill_actual, prefill_expected),
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    if (
        report["slot_kernel_vs_dense"]["relative_l2"] > 0.08
        or report["prefill"]["grouped_vs_dense"]["relative_l2"] > 0.03
    ):
        raise SystemExit("real GLM CUDA expert validation failed")


if __name__ == "__main__":
    main()
