"""Numerical and speed gate for VQ->E4M3 grouped Tensor Core MoE.

This benchmark exercises the public operator boundary used by DSV4, Kimi and
GLM: shared codebooks are quantized once, three packed projections are
expanded in batches, and both matrix multiplications use scaled grouped GEMM.
"""

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


def _measure(call, repeats: int) -> tuple[torch.Tensor, float]:
    for _ in range(3):
        output = call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        output = call()
    end.record()
    end.synchronize()
    return output, begin.elapsed_time(end) / repeats


def _quality(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate = candidate.float()
    reference = reference.float()
    delta = candidate - reference
    denominator = reference.abs().mean().clamp_min(1.0e-12)
    return {
        "cosine": float(F.cosine_similarity(
            candidate.reshape(-1), reference.reshape(-1), dim=0
        )),
        "relative_mae": float(delta.abs().mean() / denominator),
        "max_abs_error": float(delta.abs().max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--expert", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    if not fusedext.prebuild():
        raise RuntimeError(f"GPU extension unavailable: {fusedext.last_error()}")
    store = CCCPStore(str(args.model))
    packed = store.load_expert_packed(args.layer, args.expert)
    if len(packed) != 3:
        raise RuntimeError("native8 MoE benchmark requires Gate/Up/Down VQ")
    device = torch.device("cuda")
    payloads: list[torch.Tensor] = []
    codebooks: list[torch.Tensor] = []
    scales: list[float] = []
    exact: list[torch.Tensor] = []
    rows: list[list[int]] = []
    for weight in packed:
        payload = weight.raw.to(device).contiguous().reshape(-1)
        codebook = weight.cb.to(device=device, dtype=torch.float32).contiguous()
        scale = max(float(codebook.abs().amax()) / 448.0, 1.0e-12)
        quantized = (
            codebook.div(scale).clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn).contiguous()
        )
        payloads.append(payload)
        codebooks.append(quantized)
        scales.append(scale)
        exact.append(fusedext._EXT.dense_vq_dequant_packed(
            payload, codebook, int(weight.rows), int(weight.blocks),
            int(weight.bits), torch.empty(0, dtype=torch.long, device=device),
        ))
        rows.extend((
            [int(payload.data_ptr())],
            [int(quantized.data_ptr())],
            [int(weight.blocks)],
            [int(weight.dim)],
            [_DTYPE_TAG[int(weight.bits)]],
        ))
    metadata = torch.tensor(rows, dtype=torch.long, device=device)
    intermediate, hidden = int(packed[0].rows), int(packed[0].cols)
    gu = torch.empty(
        1, 2 * intermediate, hidden,
        dtype=torch.float8_e4m3fn, device=device,
    )
    down = torch.empty(
        1, hidden, intermediate,
        dtype=torch.float8_e4m3fn, device=device,
    )
    batch = int(args.batch)
    source = torch.randn(batch, hidden, dtype=torch.bfloat16, device=device)
    source8 = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    source_scale = torch.empty(batch, 1, dtype=torch.float32, device=device)
    activated8 = torch.empty(
        batch, intermediate, dtype=torch.float8_e4m3fn, device=device
    )
    activated_scale = torch.empty(
        batch, 1, dtype=torch.float32, device=device
    )
    gu_scale = torch.empty(
        1, 2 * intermediate, dtype=torch.float32, device=device
    )
    gu_scale[:, :intermediate].fill_(scales[0])
    gu_scale[:, intermediate:].fill_(scales[1])
    down_scale = torch.full(
        (1, hidden), scales[2], dtype=torch.float32, device=device
    )
    offsets = torch.tensor([batch], dtype=torch.int32, device=device)

    def execute() -> torch.Tensor:
        projection_expand_native8(metadata, gu, down)
        if fusedext.dense_fp8_quantize_rows_fused(
            source, source8, source_scale
        ) is None:
            raise RuntimeError("FP8 source quantizer rejected benchmark")
        gate_up = torch._scaled_grouped_mm(
            source8,
            gu.transpose(1, 2),
            scale_a=source_scale.view(-1),
            scale_b=gu_scale,
            offs=offsets,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        activated = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        if fusedext.dense_fp8_quantize_rows_fused(
            activated, activated8, activated_scale
        ) is None:
            raise RuntimeError("FP8 hidden quantizer rejected benchmark")
        return torch._scaled_grouped_mm(
            activated8,
            down.transpose(1, 2),
            scale_a=activated_scale.view(-1),
            scale_b=down_scale,
            offs=offsets,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

    output, elapsed_ms = _measure(execute, max(1, int(args.repeats)))
    gate = F.linear(source.float(), exact[0].float())
    up = F.linear(source.float(), exact[1].float())
    reference = F.linear(F.silu(gate) * up, exact[2].float())
    report = {
        "backend": "vq-e4m3-scaled-grouped-gemm",
        "batch": batch,
        "bits": [int(weight.bits) for weight in packed],
        "code_dimensions": [int(weight.dim) for weight in packed],
        "elapsed_ms": elapsed_ms,
        "tokens_per_second": batch / max(elapsed_ms / 1000.0, 1.0e-12),
        "quality": _quality(output, reference),
    }
    print("[cccp-vq-native8-moe] " + json.dumps(report, sort_keys=True))
    return 0 if (
        report["quality"]["cosine"] >= 0.995
        and report["quality"]["relative_mae"] <= 0.10
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
