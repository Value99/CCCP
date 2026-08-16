"""Speed/accuracy gate for VQ -> native FP8/INT8 execution images.

The VQ codebook is quantized once. The timed conversion therefore measures
only packed-index extraction plus aligned copies of 4/8/16-byte codewords into
a caller-owned execution buffer. Vendor FP8/INT8 matrix multiplication is
measured separately so a slow conversion cannot be hidden by Tensor Core work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from cccp import fusedext
from cccp.store import CCCPStore


def _measure(call, repeats: int) -> tuple[torch.Tensor, float]:
    for _ in range(4):
        result = call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        result = call()
    end.record()
    end.synchronize()
    return result, begin.elapsed_time(end) / repeats


def _quality(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate = candidate.float()
    reference = reference.float()
    delta = candidate - reference
    denominator = reference.abs().mean().clamp_min(1.0e-12)
    return {
        "cosine": float(F.cosine_similarity(
            candidate.reshape(-1), reference.reshape(-1), dim=0
        ).item()),
        "max_abs_error": float(delta.abs().max().item()),
        "relative_mae": float(delta.abs().mean().div(denominator).item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--expert", type=int, default=5)
    parser.add_argument(
        "--projection", choices=("gate", "up", "down"), default="gate"
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--batches", default="1,16,128,4096")
    args = parser.parse_args()

    if not fusedext.prebuild():
        raise RuntimeError(f"GPU extension unavailable: {fusedext.last_error()}")
    extension = fusedext._EXT
    store = CCCPStore(str(args.model))
    projection_index = tuple(store.man.projection_names).index(args.projection)
    host = store.load_expert_packed(args.layer, args.expert)[projection_index]
    device = torch.device("cuda")
    payload = host.raw.to(device).contiguous().reshape(-1)
    codebook = host.cb.to(device=device, dtype=torch.float32).contiguous()
    row_ids = torch.empty(0, dtype=torch.int64, device=device)
    columns = int(host.blocks * host.dim)

    fp8_scale = max(float(codebook.abs().max().item()) / 448.0, 1.0e-12)
    fp8_codebook = (
        codebook.div(fp8_scale)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    fp8_weight = torch.empty(
        (host.rows, columns), device=device, dtype=torch.float8_e4m3fn
    )
    int8_scale = max(float(codebook.abs().max().item()) / 127.0, 1.0e-12)
    int8_codebook = (
        codebook.div(int8_scale)
        .round()
        .clamp(-127, 127)
        .to(torch.int8)
        .contiguous()
    )
    int8_weight = torch.empty(
        (host.rows, columns), device=device, dtype=torch.int8
    )

    def expand_fp8() -> torch.Tensor:
        return extension.dense_vq_expand_native8(
            payload, fp8_codebook, fp8_weight,
            int(host.rows), int(host.blocks), int(host.bits), row_ids,
        )

    def expand_int8() -> torch.Tensor:
        return extension.dense_vq_expand_native8(
            payload, int8_codebook, int8_weight,
            int(host.rows), int(host.blocks), int(host.bits), row_ids,
        )

    fp8_result, fp8_ms = _measure(expand_fp8, max(1, args.repeats))
    int8_result, int8_ms = _measure(expand_int8, max(1, args.repeats))
    exact = extension.dense_vq_dequant_packed(
        payload,
        codebook,
        int(host.rows),
        int(host.blocks),
        int(host.bits),
        row_ids,
    )
    torch.cuda.synchronize()
    expanded_bytes = int(host.rows * columns)
    report: dict[str, object] = {
        "backend": "vq-native8-execution-image",
        "bits": int(host.bits),
        "code_dimension": int(host.dim),
        "columns": columns,
        "expert": args.expert,
        "expanded_mib": expanded_bytes / 2**20,
        "fp8_conversion_ms": fp8_ms,
        "fp8_conversion_gib_s": expanded_bytes / 2**30 / (fp8_ms / 1000.0),
        "fp8_quality": _quality(fp8_result.float() * fp8_scale, exact),
        "int8_conversion_ms": int8_ms,
        "int8_conversion_gib_s": expanded_bytes / 2**30 / (int8_ms / 1000.0),
        "int8_quality": _quality(int8_result.float() * int8_scale, exact),
        "layer": args.layer,
        "projection": args.projection,
        "rows": int(host.rows),
    }

    gemm: dict[str, object] = {}
    fp8_weight_scale = torch.tensor(
        [[fp8_scale]], device=device, dtype=torch.float32
    )
    for batch in (int(value) for value in args.batches.split(",") if value):
        torch.manual_seed(20260816 + batch)
        source = torch.randn(
            batch, columns, device=device, dtype=torch.bfloat16
        )
        batch_report: dict[str, object] = {}
        if batch == 1:
            source_f32 = source.float().contiguous()
            _, exact_ms = _measure(
                lambda: extension.dense_vq_gemv_packed(
                    source_f32, payload, codebook, int(host.rows),
                    int(host.blocks), int(host.bits)
                ),
                max(1, args.repeats),
            )
            batch_report["compact_exact_gemv_ms"] = exact_ms

        fp8_input = torch.empty_like(source, dtype=torch.float8_e4m3fn)
        fp8_input_scale = torch.empty((1, 1), device=device, dtype=torch.float32)
        if fusedext.dense_fp8_quantize_rows_fused(
            source, fp8_input, fp8_input_scale
        ) is None:
            raise RuntimeError("fused FP8 activation conversion unavailable")

        def fp8_mm() -> torch.Tensor:
            return torch._scaled_mm(
                fp8_input,
                fp8_weight.t(),
                scale_a=fp8_input_scale,
                scale_b=fp8_weight_scale,
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
            )

        repetitions = max(
            1, min(args.repeats, 20 if batch >= 128 else args.repeats)
        )
        try:
            fp8_output, fp8_gemm_ms = _measure(fp8_mm, repetitions)
            reference = F.linear(source, exact)
            batch_report.update({
                "fp8_gemm_ms": fp8_gemm_ms,
                "fp8_quality": _quality(fp8_output, reference),
            })
        except (RuntimeError, NotImplementedError) as error:
            batch_report["fp8_error"] = str(error).splitlines()[0]

        activation_scale = max(
            float(source.float().abs().max().item()) / 127.0, 1.0e-12
        )
        int8_input = (
            source.float().div(activation_scale).round().clamp(-127, 127)
            .to(torch.int8).contiguous()
        )

        def int8_mm() -> torch.Tensor:
            return torch._int_mm(int8_input, int8_weight.t())

        try:
            int32_output, int8_gemm_ms = _measure(int8_mm, repetitions)
            batch_report.update({
                "int8_gemm_ms": int8_gemm_ms,
                "int8_quality": _quality(
                    int32_output.float() * (activation_scale * int8_scale),
                    F.linear(source, exact),
                ),
            })
        except (RuntimeError, NotImplementedError) as error:
            batch_report["int8_error"] = str(error).splitlines()[0]
        gemm[str(batch)] = batch_report
    report["gemm"] = gemm
    print("[cccp-vq-native8] " + json.dumps(report, sort_keys=True))

    fp8_quality = report["fp8_quality"]
    int8_quality = report["int8_quality"]
    assert isinstance(fp8_quality, dict) and isinstance(int8_quality, dict)
    return 0 if (
        fp8_quality["cosine"] >= 0.999
        and fp8_quality["relative_mae"] <= 0.06
        and int8_quality["cosine"] >= 0.995
        and int8_quality["relative_mae"] <= 0.10
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
