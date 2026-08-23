"""Validate public CUDA packed expert kernels with real CCCP weights.

This probe deliberately bypasses the hybrid cache and uploads a small set of
real experts into dedicated CUDA allocations.  It compares both production
paths against an explicit dense reconstruction:

* decode: one-token ``packed_moe_topk``;
* prefill: ``projection_dequant`` followed by grouped dense GEMMs.

It is a release gate, not a benchmark.  Multi-token execution is intentionally
absent from the decode operator so Prefill cannot silently regress to GEMV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cccp.grouped import activate_gate_up
from cccp.ops import packed_moe_topk, projection_dequant
from cccp.ops.packed_view import build_runtime_metadata_rows
from cccp.packed_hybrid import DevicePackedWeight
from cccp.store import CCCPStore, PackedVQWeight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare real CCCP packed CUDA experts with dense reference"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--experts", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--activation")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--json")
    return parser


def _device_weight(weight: PackedVQWeight, device: torch.device):
    return DevicePackedWeight(
        raw=weight.raw.to(device),
        cb=weight.cb.to(device=device, dtype=torch.bfloat16),
        rows=weight.rows,
        cols=weight.cols,
        blocks=weight.blocks,
        dim=weight.dim,
        bits=weight.bits,
    )


def _dense_weight(weight: PackedVQWeight, device: torch.device):
    indices = weight.unpack().long()
    return (
        weight.cb.index_select(0, indices.reshape(-1))
        .reshape(weight.rows, weight.blocks, weight.dim)
        .reshape(weight.rows, weight.cols)
        .to(device=device, dtype=torch.bfloat16)
    )


def _stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    difference = (actual_f - expected_f).abs()
    denominator = expected_f.norm().clamp_min(1e-12)
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "relative_l2": float((actual_f - expected_f).norm().div(denominator).item()),
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
    if not expert_ids:
        raise SystemExit("--experts must contain at least one expert id")
    layer = int(args.layer)
    hidden = int(store.cfg.get("routed_hidden", store.cfg["hidden"]))
    intermediate = int(store.cfg["moe_inter"])
    activation = str(
        args.activation
        or store.cfg.get("moe_activation")
        or store.cfg.get("activation")
        or "swiglu"
    ).strip().lower()
    activation_beta = float(store.cfg.get("situ_beta", 4.0))
    activation_linear_beta = store.cfg.get("situ_linear_beta")
    activation_limit = float(store.cfg.get("swiglu_limit", 0.0))

    torch.manual_seed(int(args.seed))
    host_experts = [store.load_expert_packed(layer, eid) for eid in expert_ids]
    if any(len(expert) != 3 for expert in host_experts):
        raise SystemExit("the real CUDA probe currently requires three projections")
    device_experts = [
        tuple(_device_weight(weight, device) for weight in expert)
        for expert in host_experts
    ]
    metadata_rows = build_runtime_metadata_rows(device_experts)
    metadata = torch.tensor(
        metadata_rows,
        dtype=torch.long,
        device=device,
    )
    dequant_metadata = torch.tensor(
        metadata_rows[:15],
        dtype=torch.long,
        device=device,
    )
    count = len(expert_ids)
    value = torch.randn(1, hidden, dtype=torch.bfloat16, device=device)
    route_ids = torch.arange(count, dtype=torch.long, device=device)
    route_weights = torch.rand(count, dtype=torch.float32, device=device)
    route_weights.div_(route_weights.sum())
    hidden_workspace = torch.empty(
        count, 2 * intermediate, dtype=torch.bfloat16, device=device
    )
    output_workspace = torch.empty(
        count, hidden, dtype=torch.bfloat16, device=device
    )
    result = torch.empty(hidden, dtype=torch.float32, device=device)
    capability = store.man.projection_operator_capability(layer)
    print(
        json.dumps(
            {
                "input": [str(value.device), str(value.dtype), list(value.shape)],
                "routes": [str(route_ids.device), str(route_ids.dtype), list(route_ids.shape)],
                "weights": [
                    str(route_weights.device),
                    str(route_weights.dtype),
                    list(route_weights.shape),
                ],
                "metadata": [
                    str(metadata.device), str(metadata.dtype), list(metadata.shape)
                ],
                "activation": activation,
                "capability": capability,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    packed_output = packed_moe_topk(
        value,
        route_ids,
        route_weights,
        metadata,
        activation=activation,
        activation_beta=activation_beta,
        activation_linear_beta=(
            0.0 if activation_linear_beta is None else float(activation_linear_beta)
        ),
        limit=activation_limit,
        hidden_workspace=hidden_workspace,
        output_workspace=output_workspace,
        result=result,
        grouped_prefix=-1,
        **capability,
    ).clone()

    dense_experts = [
        tuple(_dense_weight(weight, device) for weight in expert)
        for expert in host_experts
    ]
    dense_outputs = []
    for gate_weight, up_weight, down_weight in dense_experts:
        gate = value @ gate_weight.t()
        up = value @ up_weight.t()
        if activation_limit > 0.0:
            gate = gate.clamp(max=activation_limit)
            up = up.clamp(min=-activation_limit, max=activation_limit)
        activated = activate_gate_up(
            gate,
            up,
            activation=activation,
            situ_beta=activation_beta,
            situ_linear_beta=(
                None
                if activation_linear_beta is None
                else float(activation_linear_beta)
            ),
        )
        dense_outputs.append((activated @ down_weight.t()).float())
    dense_output = torch.stack(dense_outputs, dim=0).mul(
        route_weights.view(-1, 1, 1)
    ).sum(dim=0).reshape(-1)

    dequant_gu = torch.empty(
        count,
        2 * intermediate,
        hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    dequant_down = torch.empty(
        count,
        hidden,
        intermediate,
        dtype=torch.bfloat16,
        device=device,
    )
    projection_dequant(dequant_metadata, dequant_gu, dequant_down)
    torch.cuda.synchronize(device)
    dense_gate_up = torch.stack(
        [torch.cat((expert[0], expert[1]), dim=0) for expert in dense_experts]
    )
    dense_down = torch.stack([expert[2] for expert in dense_experts])

    report = {
        "model": str(Path(args.model).resolve()),
        "layer": layer,
        "experts": list(expert_ids),
        "activation": activation,
        "layouts": [
            [
                {
                    "bits": weight.bits,
                    "dim": weight.dim,
                    "codebook": int(weight.cb.shape[0]),
                    "shape": [weight.rows, weight.cols],
                }
                for weight in expert
            ]
            for expert in host_experts
        ],
        "packed_decode_vs_dense": _stats(packed_output, dense_output),
        "projection_gate_up_vs_dense": _stats(dequant_gu, dense_gate_up),
        "projection_down_vs_dense": _stats(dequant_down, dense_down),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")

    maximum_relative_l2 = max(
        report[key]["relative_l2"]
        for key in (
            "packed_decode_vs_dense",
            "projection_gate_up_vs_dense",
            "projection_down_vs_dense",
        )
    )
    if maximum_relative_l2 > 0.08:
        raise SystemExit(
            f"real CUDA expert validation failed: relative_l2={maximum_relative_l2:.6f}"
        )


if __name__ == "__main__":
    main()
