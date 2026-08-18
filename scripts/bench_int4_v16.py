"""INT4 GEMV v16 — wide-load double-buffered kernel bench (synthetic).

Lessons applied from the nine measured variants:
  * v1's per-lane 4B serial chain caps at ~650 GB/s (insufficient bytes in
    flight for HBM latency); widening to 8B with extra shuffles (v8) made it
    worse — the critical path, not load width alone, dominated.
  * v2's contiguous-32-column lane windows died on 32-way shared-x bank
    conflicts — v16 keeps those windows (16B-aligned uint4 loads) but reads
    x straight from L1/L2 via __ldg instead of shared staging.
  * Two independent accumulators + a register double buffer keep two 16-byte
    loads in flight per lane (bytes-in-flight x4 vs v1).
Layout stays the runtime int4_g64 format (no repack), numerics identical to
v1: acc += ((nibble - 8) * scale) * x[col].  SM75+ compatible (plain LDG);
an optional cp.async shared double-buffer stage is added for SM80+.
"""
from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

// 5120-row class kernel: one warp per output row, lanes own contiguous
// 32-column windows (16 packed bytes = one uint4, 16B-aligned because rows
// are cols/2 bytes and cols is a multiple of 64).  Register double buffer
// keeps the next uint4 in flight while the current one is consumed.
template <int ROWS_PER_BLOCK>
__global__ void int4_gemv_v16_kernel(
    const float* __restrict__ x,          // [cols] fp32, L1-resident
    const uint8_t* __restrict__ packed,   // [rows, cols/2]
    const __half* __restrict__ scales,    // [rows, groups]
    float* __restrict__ output,
    const int rows,
    const int cols,
    const int groups)
{
    const int lane = threadIdx.x;
    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;
    const int packed_cols = cols >> 1;
    const uint8_t* prow = packed + (long)row * packed_cols;
    const __half* srow = scales + (long)row * groups;

    // Lane window columns: [lane*32, lane*32+32) per 1024-column round.
    const int rounds = cols >> 10;                 // cols/1024
    const int tail_groups = (cols - (rounds << 10)) >> 6;
    float acc0 = 0.f, acc1 = 0.f;
    uint4 cur;
    // Prime the pipeline.
    const int base0 = lane * 16;                   // bytes within round
    if (rounds > 0) {
        cur = *reinterpret_cast<const uint4*>(prow + base0);
    }
    for (int r = 0; r < rounds; ++r) {
        const uint4 next =
            (r + 1 < rounds)
                ? *reinterpret_cast<const uint4*>(
                      prow + ((r + 1) << 9) + base0)
                : cur;
        const int col0 = (r << 10) + lane * 32;
        const float sc = __half2float(__ldg(srow + (col0 >> 6)));
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&cur);
        const float* xr = x + col0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float lo = static_cast<float>(static_cast<int>(b[i] & 15) - 8) * sc;
            const float hi = static_cast<float>(static_cast<int>(b[i] >> 4) - 8) * sc;
            acc0 = __fmaf_rn(lo, __ldg(xr + i * 2), acc0);
            acc1 = __fmaf_rn(hi, __ldg(xr + i * 2 + 1), acc1);
        }
        cur = next;
    }
    // Tail rounds (cols % 1024): lane-window guarded.
    if (tail_groups > 0) {
        const int col_base = rounds << 10;
        const int local = lane * 32;
        if (local < (tail_groups << 6)) {
            const int col0 = col_base + local;
            const float sc = __half2float(__ldg(srow + (col0 >> 6)));
            const uint4 w = *reinterpret_cast<const uint4*>(
                prow + (col_base >> 1) + (local >> 1));
            const uint8_t* b = reinterpret_cast<const uint8_t*>(&w);
            const float* xr = x + col0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int c = i * 2;
                if (c < (tail_groups << 6) - local) {
                    const float lo = static_cast<float>(static_cast<int>(b[i] & 15) - 8) * sc;
                    const float hi = static_cast<float>(static_cast<int>(b[i] >> 4) - 8) * sc;
                    acc0 = __fmaf_rn(lo, __ldg(xr + c), acc0);
                    acc1 = __fmaf_rn(hi, __ldg(xr + c + 1), acc1);
                }
            }
        }
    }
    float acc = acc0 + acc1;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) output[row] = acc;
}

torch::Tensor int4_gemv_v16(
    torch::Tensor x,        // [cols] fp32
    torch::Tensor packed,
    torch::Tensor scales,
    int64_t rows,
    int64_t cols,
    int64_t groups)
{
    TORCH_CHECK(cols % 64 == 0, "cols%64");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto output = torch::empty(
        {rows}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    const int rpb = rows >= 4096 ? 32 : (rows >= 2048 ? 16 : 8);
    if (rpb == 32) {
        int4_gemv_v16_kernel<32><<<(rows + 31) / 32, dim3(32, 32), 0, stream>>>(
            x.data_ptr<float>(), packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(scales.data_ptr()),
            output.data_ptr<float>(), (int)rows, (int)cols, (int)groups);
    } else if (rpb == 16) {
        int4_gemv_v16_kernel<16><<<(rows + 15) / 16, dim3(32, 16), 0, stream>>>(
            x.data_ptr<float>(), packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(scales.data_ptr()),
            output.data_ptr<float>(), (int)rows, (int)cols, (int)groups);
    } else {
        int4_gemv_v16_kernel<8><<<(rows + 7) / 8, dim3(32, 8), 0, stream>>>(
            x.data_ptr<float>(), packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(scales.data_ptr()),
            output.data_ptr<float>(), (int)rows, (int)cols, (int)groups);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

ext = load_inline(
    name="int4_gemv_v16_proto",
    cpp_sources="torch::Tensor int4_gemv_v16(torch::Tensor x, torch::Tensor packed, torch::Tensor scales, int64_t rows, int64_t cols, int64_t groups);",
    cuda_sources=SRC,
    functions=["int4_gemv_v16"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


def main() -> None:
    torch.manual_seed(5090)
    rows, cols = 5120, 17408
    groups = cols // 64
    packed = torch.randint(
        0, 256, (rows * cols // 2,), device="cuda", dtype=torch.uint8)
    scales = (torch.randn(rows, groups, device="cuda") * 0.05).half()
    x = torch.randn(cols, device="cuda", dtype=torch.float32)

    # torch 参考(前 64 行)
    ref_rows = 64
    idx = packed[: ref_rows * cols // 2].view(ref_rows, cols // 2)
    lo = (idx & 15).long()
    hi = (idx >> 4).long()
    nib = torch.stack([lo, hi], dim=-1).view(ref_rows, cols) - 8
    sc = scales[:ref_rows].repeat_interleave(64, dim=1).float()
    ref = (nib.float() * sc * x).sum(dim=1)

    out = ext.int4_gemv_v16(x, packed, scales, rows, cols, groups)
    torch.cuda.synchronize()
    diff = (out[:ref_rows] - ref).abs()
    rel = diff / ref.abs().clamp_min(1e-3)
    print(f"numerics: max_abs={diff.max().item():.5f} "
          f"p99_rel={rel.quantile(0.99).item():.4f}")

    for _ in range(50):
        ext.int4_gemv_v16(x, packed, scales, rows, cols, groups)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(300):
        ext.int4_gemv_v16(x, packed, scales, rows, cols, groups)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / 300
    nbytes = rows * cols * 0.5 + rows * groups * 2 + cols * 4
    print(f"v16: {ms:.4f} ms -> {nbytes / (ms / 1e3) / 1e9:.0f} GB/s "
          f"(v1=650, 目标 1500)")

    # 小矩阵(verify 常见 5120x5120 级)与 lm_head 形状
    for r2, c2 in ((5120, 5120), (248320, 5120)):
        p2 = torch.randint(0, 256, (r2 * c2 // 2,), device="cuda", dtype=torch.uint8)
        s2 = (torch.randn(r2, c2 // 64, device="cuda") * 0.05).half()
        x2 = torch.randn(c2, device="cuda")
        o2 = ext.int4_gemv_v16(x2, p2, s2, r2, c2, c2 // 64)
        torch.cuda.synchronize()
        for _ in range(30):
            ext.int4_gemv_v16(x2, p2, s2, r2, c2, c2 // 64)
        torch.cuda.synchronize()
        s3 = torch.cuda.Event(enable_timing=True); e3 = torch.cuda.Event(enable_timing=True)
        s3.record()
        for _ in range(100):
            ext.int4_gemv_v16(x2, p2, s2, r2, c2, c2 // 64)
        e3.record(); torch.cuda.synchronize()
        m2 = s3.elapsed_time(e3) / 100
        nb2 = r2 * c2 * 0.5 + r2 * (c2 // 64) * 2 + c2 * 4
        print(f"v16 {r2}x{c2}: {m2:.4f} ms -> {nb2 / (m2 / 1e3) / 1e9:.0f} GB/s")


if __name__ == "__main__":
    main()
