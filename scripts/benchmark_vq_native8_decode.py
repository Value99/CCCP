"""Profile one Top-K VQ->E4M3->Tensor-Core Decode layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from cccp import fusedext
from cccp.ops import projection_expand_native8
from cccp.store import CCCPStore


_DTYPE_TAG = {
    8: 0, 16: 1, 12: 2, 14: 3, 10: 4, 9: 5,
    11: 6, 13: 7, 15: 8,
}


def _time(call, repeats: int) -> float:
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        call()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end) / repeats)


def _quantize_codebook(
    codebook: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, float]:
    limit = 127.0 if dtype == torch.int8 else 448.0
    scale = max(float(codebook.abs().amax()) / limit, 1.0e-12)
    normalized = codebook.div(scale).clamp(-limit, limit)
    if dtype == torch.int8:
        normalized = normalized.round()
    return normalized.to(dtype).contiguous(), scale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--execution-format",
        choices=("e4m3", "int8"),
        default="e4m3",
    )
    args = parser.parse_args()

    if not fusedext.prebuild():
        raise RuntimeError(f"GPU extension unavailable: {fusedext.last_error()}")
    device = torch.device("cuda")
    store = CCCPStore(str(args.model))
    experts = [
        store.load_expert_packed(args.layer, args.expert + offset)
        for offset in range(args.top_k)
    ]
    if not experts or any(len(expert) != 3 for expert in experts):
        raise RuntimeError("benchmark requires three-projection experts")
    intermediate = int(experts[0][0].rows)
    hidden = int(experts[0][0].cols)
    native_dtype = (
        torch.int8
        if args.execution_format == "int8"
        else torch.float8_e4m3fn
    )
    rows = [[0 for _ in experts] for _ in range(15)]
    scales = torch.empty(len(experts), 3, dtype=torch.float32, device=device)
    keepalive: list[torch.Tensor] = []
    for expert_index, expert in enumerate(experts):
        for projection, weight in enumerate(expert):
            payload = weight.raw.to(device).contiguous().reshape(-1)
            codebook = weight.cb.to(
                device=device, dtype=torch.float32
            ).contiguous()
            codebook8, scale = _quantize_codebook(codebook, native_dtype)
            keepalive.extend((payload, codebook, codebook8))
            base = projection * 5
            rows[base + 0][expert_index] = int(payload.data_ptr())
            rows[base + 1][expert_index] = int(codebook8.data_ptr())
            rows[base + 2][expert_index] = int(weight.blocks)
            rows[base + 3][expert_index] = int(weight.dim)
            rows[base + 4][expert_index] = _DTYPE_TAG[int(weight.bits)]
            scales[expert_index, projection] = scale
    metadata = torch.tensor(rows, dtype=torch.long, device=device)
    top_k = len(experts)
    options = {"dtype": native_dtype, "device": device}
    gu = torch.empty(top_k, 2 * intermediate, hidden, **options)
    down_tc = torch.empty(hidden, top_k * intermediate, **options)
    source = torch.randn(1, hidden, dtype=torch.bfloat16, device=device)
    source8 = torch.empty(1, hidden, **options)
    source_scale = torch.empty(1, 1, dtype=torch.float32, device=device)
    activated8 = torch.empty(1, top_k * intermediate, **options)
    activated_scale = torch.empty(1, 1, dtype=torch.float32, device=device)
    route_weights = torch.softmax(
        torch.randn(top_k, dtype=torch.float32, device=device), dim=0
    )
    gu_scale = torch.empty(
        1, top_k * 2 * intermediate, dtype=torch.float32, device=device
    )
    gu_scale_view = gu_scale.view(top_k, 2 * intermediate)
    gu_scale_view[:, :intermediate].copy_(scales[:, 0:1])
    gu_scale_view[:, intermediate:].copy_(scales[:, 1:2])
    unit_scale = torch.ones(1, 1, dtype=torch.float32, device=device)

    projection_expand_native8(metadata, gu, down_tc)
    layout_mismatches = 0
    checked_bytes = 0
    for expert_index, expert in enumerate(experts):
        for projection, weight in enumerate(expert):
            codebook = weight.cb.to(
                device=device, dtype=torch.float32
            ).contiguous()
            codebook8, _scale = _quantize_codebook(
                codebook, native_dtype
            )
            sample_rows = torch.tensor(
                sorted({0, int(weight.rows) // 2, int(weight.rows) - 1}),
                dtype=torch.long,
                device=device,
            )
            indices = weight.unpack().index_select(
                0, sample_rows.cpu()
            ).long().to(device)
            expected = codebook8.index_select(
                0, indices.reshape(-1)
            ).reshape(len(sample_rows), int(weight.cols))
            if projection == 0:
                actual = gu[expert_index, :intermediate].index_select(
                    0, sample_rows
                )
            elif projection == 1:
                actual = gu[expert_index, intermediate:].index_select(
                    0, sample_rows
                )
            else:
                actual = down_tc[
                    :, expert_index * intermediate :
                    (expert_index + 1) * intermediate
                ].index_select(0, sample_rows)
            mismatch = actual.view(torch.uint8).ne(
                expected.view(torch.uint8)
            )
            layout_mismatches += int(mismatch.sum().item())
            checked_bytes += int(mismatch.numel())
    repeats = max(1, int(args.repeats))
    expand_ms = _time(
        lambda: projection_expand_native8(metadata, gu, down_tc), repeats
    )
    if native_dtype == torch.int8:
        report = {
            "backend": "vq-int8-execution-image",
            "top_k": top_k,
            "expand_ms": expand_ms,
            "layout_checked_bytes": checked_bytes,
            "layout_mismatches": layout_mismatches,
            "layout_exact": layout_mismatches == 0,
        }
        print(
            "[cccp-vq-native8-decode] "
            + json.dumps(report, sort_keys=True)
        )
        return 0 if layout_mismatches == 0 else 2
    fusedext.dense_fp8_quantize_rows_fused(source, source8, source_scale)

    def gate_up_call():
        return torch._scaled_mm(
            source8,
            gu.reshape(top_k * 2 * intermediate, hidden).t(),
            scale_a=source_scale,
            scale_b=gu_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

    gate_up = gate_up_call().view(top_k, 2 * intermediate)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = F.silu(gate.float()) * up.float()
    weighted = (
        activated * route_weights.view(-1, 1) * scales[:, 2:3]
    ).reshape(1, top_k * intermediate)
    fusedext.dense_fp8_quantize_rows_fused(
        weighted, activated8, activated_scale
    )
    def down_call():
        return torch._scaled_mm(
            activated8,
            down_tc.t(),
            scale_a=activated_scale,
            scale_b=unit_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

    def full_call():
        projection_expand_native8(metadata, gu, down_tc)
        fusedext.dense_fp8_quantize_rows_fused(source, source8, source_scale)
        combined = gate_up_call().view(top_k, 2 * intermediate)
        gate_value, up_value = combined.chunk(2, dim=-1)
        weighted_value = (
            F.silu(gate_value.float())
            * up_value.float()
            * route_weights.view(-1, 1)
            * scales[:, 2:3]
        ).reshape(1, top_k * intermediate)
        fusedext.dense_fp8_quantize_rows_fused(
            weighted_value, activated8, activated_scale
        )
        return down_call()

    report = {
        "backend": "vq-e4m3-two-gemm-decode",
        "top_k": top_k,
        "layout_checked_bytes": checked_bytes,
        "layout_mismatches": layout_mismatches,
        "layout_exact": layout_mismatches == 0,
        "expand_ms": expand_ms,
        "gate_up_mm_ms": _time(gate_up_call, repeats),
        "down_mm_ms": _time(down_call, repeats),
        "full_ms": _time(full_call, repeats),
    }
    print("[cccp-vq-native8-decode] " + json.dumps(report, sort_keys=True))
    return 0 if layout_mismatches == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
