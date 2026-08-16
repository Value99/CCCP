"""Numerically validate legacy [10,E] packed grouped MoE on one GPU.

The legacy Kimi archive stores one fused Gate+Up projection followed by Down.
This probe constructs that exact directory shape, executes the public grouped
operator, and compares it with a BF16-rounded dense reference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine" / "CCCP-Engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from cccp.ops import packed_moe_topk_grouped  # noqa: E402


def _bf16_round(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16).float()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("a CUDA/HIP device is required")

    torch.manual_seed(509035)
    tokens, experts, top_k = 7, 5, 3
    hidden, intermediate, vector = 64, 48, 4
    codebook_size = 256
    values = (
        torch.randn(tokens, hidden, dtype=torch.float32) * 0.15
    ).to(device=device, dtype=torch.bfloat16)
    gu_codebook = (
        torch.randn(codebook_size, vector, dtype=torch.float32) * 0.04
    ).to(device=device, dtype=torch.bfloat16)
    down_codebook = (
        torch.randn(codebook_size, vector, dtype=torch.float32) * 0.04
    ).to(device=device, dtype=torch.bfloat16)

    retained: list[torch.Tensor] = [gu_codebook, down_codebook]
    gu_indices: list[torch.Tensor] = []
    down_indices: list[torch.Tensor] = []
    metadata_host = torch.zeros(10, experts, dtype=torch.long)
    for expert in range(experts):
        gu_index = torch.randint(
            codebook_size,
            (2 * intermediate, hidden // vector),
            dtype=torch.uint8,
        )
        down_index = torch.randint(
            codebook_size,
            (hidden, intermediate // vector),
            dtype=torch.uint8,
        )
        gu_indices.append(gu_index)
        down_indices.append(down_index)
        gu_packed = gu_index.reshape(-1).to(device)
        down_packed = down_index.reshape(-1).to(device)
        retained.extend((gu_packed, down_packed))
        metadata_host[0:5, expert] = torch.tensor(
            [
                gu_packed.data_ptr(),
                gu_codebook.data_ptr(),
                hidden // vector,
                vector,
                0,
            ],
            dtype=torch.long,
        )
        metadata_host[5:10, expert] = torch.tensor(
            [
                down_packed.data_ptr(),
                down_codebook.data_ptr(),
                intermediate // vector,
                vector,
                0,
            ],
            dtype=torch.long,
        )
    metadata = metadata_host.to(device)

    route_ids = torch.tensor(
        [
            [0, 2, 4],
            [1, 2, 3],
            [4, 0, 1],
            [3, 2, 0],
            [2, 4, 1],
            [0, 3, 4],
            [1, 4, 2],
        ],
        dtype=torch.long,
        device=device,
    )
    route_weights = torch.rand(tokens, top_k, device=device)
    route_weights.div_(route_weights.sum(dim=1, keepdim=True))
    flat_experts = route_ids.reshape(-1)
    flat_tokens = (
        torch.arange(tokens, dtype=torch.long, device=device)
        .view(-1, 1)
        .expand(tokens, top_k)
        .reshape(-1)
    )
    order = torch.argsort(flat_experts)
    sorted_experts = flat_experts.index_select(0, order).contiguous()
    sorted_tokens = flat_tokens.index_select(0, order).contiguous()
    sorted_weights = route_weights.reshape(-1).index_select(0, order).float()
    group_experts, group_counts = torch.unique_consecutive(
        sorted_experts, return_counts=True
    )
    group_offsets = torch.empty(
        group_experts.numel() + 1, dtype=torch.int32, device=device
    )
    group_offsets[0] = 0
    group_offsets[1:] = torch.cumsum(group_counts, dim=0).to(torch.int32)
    hidden_workspace = torch.empty(
        sorted_tokens.numel(),
        2 * intermediate,
        dtype=torch.bfloat16,
        device=device,
    )
    actual = torch.empty(tokens, hidden, dtype=torch.float32, device=device)

    actual = packed_moe_topk_grouped(
        values,
        sorted_tokens,
        group_experts.contiguous(),
        group_offsets,
        sorted_weights,
        metadata,
        activation="swiglu",
        activation_beta=1.0,
        activation_linear_beta=0.0,
        hidden_workspace=hidden_workspace,
        result=actual,
    )
    torch.cuda.synchronize(device)

    expected = torch.zeros(tokens, hidden, dtype=torch.float32)
    values_cpu = values.cpu().float()
    gu_cb_cpu = gu_codebook.cpu().float()
    down_cb_cpu = down_codebook.cpu().float()
    route_ids_cpu = route_ids.cpu()
    route_weights_cpu = route_weights.cpu()
    for token in range(tokens):
        for slot in range(top_k):
            expert = int(route_ids_cpu[token, slot])
            dense_gu = gu_cb_cpu[gu_indices[expert].long()].reshape(
                2 * intermediate, hidden
            )
            dense_down = down_cb_cpu[down_indices[expert].long()].reshape(
                hidden, intermediate
            )
            projected = _bf16_round(dense_gu @ values_cpu[token])
            gate, up = projected.chunk(2)
            activated = (torch.nn.functional.silu(gate) * up).to(
                torch.bfloat16
            ).float()
            down = _bf16_round(dense_down @ activated)
            expected[token].add_(
                down, alpha=float(route_weights_cpu[token, slot])
            )

    actual_cpu = actual.cpu()
    difference = (actual_cpu - expected).abs()
    summary = {
        "backend": "hip" if torch.version.hip is not None else "cuda",
        "device": torch.cuda.get_device_name(device),
        "metadata_rows": int(metadata.shape[0]),
        "tokens": tokens,
        "experts": experts,
        "top_k": top_k,
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "finite": bool(torch.isfinite(actual_cpu).all()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    torch.testing.assert_close(actual_cpu, expected, rtol=0.04, atol=0.004)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
