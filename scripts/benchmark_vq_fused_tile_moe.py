"""Benchmark a complete Top-K MoE layer without materialized Native8 weights.

Each Triton program group owns one routed expert.  It unpacks VQ indices,
gathers E4M3 codewords into register tiles, and immediately issues Tensor Core
dots.  Gate, Up, activation, Down and final route reduction are all included
in the reported wall time; no full expert matrix is written to global memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from cccp import fusedext
from cccp.store import CCCPStore


@triton.jit
def _grouped_vq_fp8_tile_mm_kernel(
    source_ptr,
    packed_ptr,
    codebook_ptr,
    expert_ids_ptr,
    source_scales_ptr,
    codebook_scales_ptr,
    output_ptr,
    rows: tl.constexpr,
    columns: tl.constexpr,
    blocks: tl.constexpr,
    packed_bytes: tl.constexpr,
    codebook_values: tl.constexpr,
    bits: tl.constexpr,
    vector: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    local_expert = tl.program_id(1)
    expert = tl.load(expert_ids_ptr + local_expert).to(tl.int64)
    offsets_m = tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    code_mask = (1 << bits) - 1
    packed_base = packed_ptr + local_expert * packed_bytes
    codebook_base = codebook_ptr + local_expert * codebook_values

    for k0 in range(0, columns, BLOCK_K):
        offsets_k = k0 + tl.arange(0, BLOCK_K)
        source = tl.load(
            source_ptr
            + expert * columns
            + offsets_m[:, None] * columns
            + offsets_k[None, :],
            mask=(offsets_m[:, None] == 0)
            & (offsets_k[None, :] < columns),
            other=0.0,
        )
        block = offsets_k // vector
        component = offsets_k - block * vector
        linear_index = offsets_n[:, None] * blocks + block[None, :]
        bit_offset = linear_index * bits
        byte_offset = bit_offset >> 3
        shift = bit_offset & 7
        valid = (offsets_n[:, None] < rows) & (offsets_k[None, :] < columns)
        low = tl.load(
            packed_base + byte_offset,
            mask=valid & (byte_offset < packed_bytes),
            other=0,
        ).to(tl.uint32)
        middle = tl.load(
            packed_base + byte_offset + 1,
            mask=valid & (byte_offset + 1 < packed_bytes),
            other=0,
        ).to(tl.uint32)
        high = tl.load(
            packed_base + byte_offset + 2,
            mask=valid & (byte_offset + 2 < packed_bytes),
            other=0,
        ).to(tl.uint32)
        code = ((low | (middle << 8) | (high << 16)) >> shift) & code_mask
        weight = tl.load(
            codebook_base + code * vector
            + component[None, :],
            mask=valid,
            other=0.0,
        )
        accumulator += tl.dot(source, tl.trans(weight), out_dtype=tl.float32)

    scale = tl.load(source_scales_ptr + expert) * tl.load(
        codebook_scales_ptr + local_expert
    )
    result = accumulator * scale
    tl.store(
        output_ptr
        + expert * rows
        + offsets_m[:, None] * rows
        + offsets_n[None, :],
        result,
        mask=(offsets_m[:, None] == 0) & (offsets_n[None, :] < rows),
    )


def _measure(call, repeats: int) -> tuple[torch.Tensor, float]:
    for _ in range(6):
        output = call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, repeats)):
        output = call()
    end.record()
    end.synchronize()
    return output, float(start.elapsed_time(end) / max(1, repeats))


def _quality(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate = candidate.float()
    reference = reference.float()
    delta = candidate - reference
    denominator = reference.abs().mean().clamp_min(1.0e-12)
    return {
        "cosine": float(F.cosine_similarity(
            candidate.reshape(-1), reference.reshape(-1), dim=0
        )),
        "max_abs_error": float(delta.abs().max()),
        "relative_mae": float(delta.abs().mean() / denominator),
    }


def _projection_storage(weights, projection: int, device: torch.device):
    selected = [expert[projection] for expert in weights]
    shapes = {
        (int(weight.rows), int(weight.blocks * weight.dim))
        for weight in selected
    }
    if len(shapes) != 1:
        raise RuntimeError("routed projection output/input shapes differ")
    rows, columns = shapes.pop()
    grouped: dict[tuple[int, int, int, int], list[tuple[int, object]]] = {}
    for expert_index, weight in enumerate(selected):
        signature = (
            int(weight.rows),
            int(weight.blocks),
            int(weight.bits),
            int(weight.dim),
        )
        grouped.setdefault(signature, []).append((expert_index, weight))

    groups = []
    exact: list[torch.Tensor | None] = [None] * len(selected)
    empty_rows = torch.empty(0, dtype=torch.long, device=device)
    for signature, members in grouped.items():
        signature_rows, blocks, bits, vector = signature
        packed_tensors = []
        codebooks = []
        scales = []
        expert_ids = []
        for expert_index, weight in members:
            packed = weight.raw.reshape(-1).to(device).contiguous()
            codebook = weight.cb.to(
                device=device, dtype=torch.float32
            ).contiguous()
            scale = max(float(codebook.abs().amax()) / 448.0, 1.0e-12)
            codebook8 = (
                codebook.div(scale).clamp(-448.0, 448.0)
                .to(torch.float8_e4m3fn).contiguous()
            )
            packed_tensors.append(packed)
            codebooks.append(codebook8)
            scales.append(scale)
            expert_ids.append(expert_index)
            exact[expert_index] = fusedext._EXT.dense_vq_dequant_packed(
                packed,
                codebook,
                signature_rows,
                blocks,
                bits,
                empty_rows,
            )
        groups.append({
            "rows": signature_rows,
            "blocks": blocks,
            "bits": bits,
            "vector": vector,
            "columns": blocks * vector,
            "packed": torch.stack(packed_tensors),
            "codebooks": torch.stack(codebooks),
            "expert_ids": torch.tensor(
                expert_ids, dtype=torch.int64, device=device
            ),
            "scales": torch.tensor(
                scales, dtype=torch.float32, device=device
            ),
        })
    if any(tensor is None for tensor in exact):
        raise RuntimeError("exact routed projection reference is incomplete")
    return {
        "rows": rows,
        "columns": columns,
        "groups": groups,
        "bits": [int(weight.bits) for weight in selected],
        "exact": [tensor for tensor in exact if tensor is not None],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument("--num-warps", type=int, default=2)
    args = parser.parse_args()

    if not fusedext.prebuild():
        raise RuntimeError(f"GPU extension unavailable: {fusedext.last_error()}")
    if torch.cuda.get_device_capability() < (8, 9):
        raise RuntimeError("FP8 tile prototype requires SM89+")
    device = torch.device("cuda")
    store = CCCPStore(str(args.model))
    experts = [
        store.load_expert_packed(args.layer, args.expert + offset)
        for offset in range(args.top_k)
    ]
    if not experts or any(len(expert) != 3 for expert in experts):
        raise RuntimeError("benchmark requires Gate/Up/Down VQ experts")
    gate = _projection_storage(experts, 0, device)
    up = _projection_storage(experts, 1, device)
    down = _projection_storage(experts, 2, device)
    hidden = int(gate["columns"])
    intermediate = int(gate["rows"])
    top_k = len(experts)
    if int(up["columns"]) != hidden or int(up["rows"]) != intermediate:
        raise RuntimeError("Gate and Up dimensions differ")
    if int(down["columns"]) != intermediate or int(down["rows"]) != hidden:
        raise RuntimeError("Down dimensions do not close the MoE transform")

    source = torch.randn(1, hidden, dtype=torch.bfloat16, device=device)
    source_scale = max(float(source.float().abs().amax()) / 448.0, 1.0e-12)
    source8 = (
        source.float().div(source_scale).clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn).repeat(top_k, 1).contiguous()
    )
    source_scales = torch.full(
        (top_k,), source_scale, dtype=torch.float32, device=device
    )
    gate_output = torch.empty(
        top_k, intermediate, dtype=torch.bfloat16, device=device
    )
    up_output = torch.empty_like(gate_output)
    activated8 = torch.empty(
        top_k, intermediate, dtype=torch.float8_e4m3fn, device=device
    )
    activated_scales = torch.empty(
        top_k, 1, dtype=torch.float32, device=device
    )
    down_output = torch.empty(
        top_k, hidden, dtype=torch.bfloat16, device=device
    )
    result = torch.empty(1, hidden, dtype=torch.bfloat16, device=device)
    route_weights = torch.softmax(
        torch.randn(top_k, dtype=torch.float32, device=device), dim=0
    )

    def launch_projection(storage, inputs, input_scales, output):
        for group in storage["groups"]:
            rows = int(group["rows"])
            columns = int(group["columns"])
            group_size = int(group["expert_ids"].numel())
            _grouped_vq_fp8_tile_mm_kernel[
                (triton.cdiv(rows, int(args.block_n)), group_size)
            ](
                inputs,
                group["packed"],
                group["codebooks"],
                group["expert_ids"],
                input_scales,
                group["scales"],
                output,
                rows=rows,
                columns=columns,
                blocks=int(group["blocks"]),
                packed_bytes=int(group["packed"].shape[1]),
                codebook_values=int(
                    group["codebooks"].shape[1]
                    * group["codebooks"].shape[2]
                ),
                bits=int(group["bits"]),
                vector=int(group["vector"]),
                BLOCK_M=16,
                BLOCK_N=int(args.block_n),
                BLOCK_K=int(args.block_k),
                num_warps=int(args.num_warps),
            )

    def execute() -> torch.Tensor:
        launch_projection(gate, source8, source_scales, gate_output)
        launch_projection(up, source8, source_scales, up_output)
        activated = (
            F.silu(gate_output.float())
            * up_output.float()
            * route_weights.view(-1, 1)
        ).to(torch.bfloat16)
        if fusedext.dense_fp8_quantize_rows_fused(
            activated, activated8, activated_scales
        ) is None:
            raise RuntimeError("FP8 activation quantizer rejected routed rows")
        launch_projection(
            down,
            activated8,
            activated_scales.view(-1),
            down_output,
        )
        result.copy_(down_output.float().sum(dim=0, keepdim=True))
        return result

    output, elapsed_ms = _measure(execute, int(args.repeats))
    exact_gate = torch.stack([
        F.linear(source.float(), weight.float()).squeeze(0)
        for weight in gate["exact"]
    ])
    exact_up = torch.stack([
        F.linear(source.float(), weight.float()).squeeze(0)
        for weight in up["exact"]
    ])
    exact_activated = (
        F.silu(exact_gate) * exact_up * route_weights.view(-1, 1)
    )
    reference = sum(
        F.linear(exact_activated[index:index + 1], weight.float())
        for index, weight in enumerate(down["exact"])
    )
    report = {
        "backend": "vq-register-tile-topk-tensorcore",
        "bits": {
            "gate": gate["bits"],
            "up": up["bits"],
            "down": down["bits"],
        },
        "elapsed_ms": elapsed_ms,
        "equivalent_routed_layers_per_second": 1000.0 / elapsed_ms,
        "materialized_weight_bytes": 0,
        "quality": _quality(output, reference),
        "top_k": top_k,
    }
    print("[cccp-vq-fused-tile-moe] " + json.dumps(report, sort_keys=True))
    return 0 if (
        report["quality"]["cosine"] >= 0.99
        and report["quality"]["relative_mae"] <= 0.15
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
