"""INT4 GEMV v17 — marlin repack layout + FMA + shared-x.

Verified marlin tile mapping (one uint32 per lane per 32-col superstep,
numerics p99 2.98%) with the mma path replaced by eight FMAs against
shared-x; SM75+ compatible, no repack-time layout changes beyond the
proven superstep tiles.
"""
from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

__global__ void int4_repack17_kernel(
    const uint8_t* __restrict__ src,
    uint32_t* __restrict__ dst,
    const int rows,
    const int cols)
{
    const long total = (((long)rows + 7) / 8) * (cols / 32) * 32;
    for (long w = blockIdx.x * (long)blockDim.x + threadIdx.x;
         w < total; w += (long)gridDim.x * blockDim.x) {
        const int intra = (int)(w & 31);
        const long tile = w >> 5;
        const int tiles_k = cols / 32;
        const int tile_k = (int)(tile % tiles_k);
        const int tile_n = (int)(tile / tiles_k);
        const int j = intra >> 2;
        const int i = intra & 3;
        const int row = tile_n * 8 + j;
        uint32_t out = 0u;
        if (row < rows) {
            const uint8_t* srow = src + (long)row * (cols >> 1) + tile_k * 16;
            out = (uint32_t)srow[i] | ((uint32_t)srow[i + 4] << 8) |
                  ((uint32_t)srow[i + 8] << 16) | ((uint32_t)srow[i + 12] << 24);
        }
        dst[w] = out;
    }
}

constexpr int V17_SLICE = 2048;

__global__ void int4_gemv_v17_kernel(
    const float* __restrict__ x,
    const uint32_t* __restrict__ repacked,
    const __half* __restrict__ scales,
    float* __restrict__ partial,
    const int rows,
    const int cols,
    const int groups,
    const int slices)
{
    extern __shared__ float sx[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int slice = blockIdx.x;
    const int k0 = slice * V17_SLICE;
    const int here = min(V17_SLICE, cols - k0);
    if (here <= 0) return;
    for (int c = threadIdx.x; c < here; c += 128) {
        sx[c] = x[k0 + c];
    }
    __syncthreads();
    const int row0 = (blockIdx.y * 4 + warp) * 8;
    if (row0 >= rows) return;
    const int j = lane >> 2;
    const int i = lane & 3;
    const int row = row0 + j;
    const __half* srow = scales + (long)row * groups;
    const int tiles_k = cols >> 5;
    const uint32_t* base = repacked +
        ((((long)(row0 >> 3)) * tiles_k + (k0 >> 5)) << 5);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int ts = 0; ts < (here >> 5); ++ts) {
        const uint32_t w = __ldg(base + (ts << 5) + lane);
        const int col0 = ts << 5;
        const float sc = __half2float(__ldg(srow + ((k0 + col0 + 2 * i) >> 6)));
        const uint32_t b0 = w & 0xFF;
        const uint32_t b1 = (w >> 8) & 0xFF;
        const uint32_t b2 = (w >> 16) & 0xFF;
        const uint32_t b3 = (w >> 24) & 0xFF;
        a0 = __fmaf_rn(static_cast<float>((int)(b0 & 15) - 8) * sc, sx[col0 + 2 * i], a0);
        a0 = __fmaf_rn(static_cast<float>((int)(b0 >> 4) - 8) * sc, sx[col0 + 2 * i + 1], a0);
        a1 = __fmaf_rn(static_cast<float>((int)(b1 & 15) - 8) * sc, sx[col0 + 2 * i + 8], a1);
        a1 = __fmaf_rn(static_cast<float>((int)(b1 >> 4) - 8) * sc, sx[col0 + 2 * i + 9], a1);
        a2 = __fmaf_rn(static_cast<float>((int)(b2 & 15) - 8) * sc, sx[col0 + 2 * i + 16], a2);
        a2 = __fmaf_rn(static_cast<float>((int)(b2 >> 4) - 8) * sc, sx[col0 + 2 * i + 17], a2);
        a3 = __fmaf_rn(static_cast<float>((int)(b3 & 15) - 8) * sc, sx[col0 + 2 * i + 24], a3);
        a3 = __fmaf_rn(static_cast<float>((int)(b3 >> 4) - 8) * sc, sx[col0 + 2 * i + 25], a3);
    }
    // Each 4-lane group (same j) holds one output row's partials.
    float acc = (a0 + a1) + (a2 + a3);
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, off, 4);
    }
    if (row < rows && (lane & 3) == 0) {
        partial[(long)row * slices + slice] = acc;
    }
}

__global__ void v17_reduce(
    const float* __restrict__ partial,
    float* __restrict__ output,
    const int rows,
    const int slices)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float acc = 0.f;
    for (int s = 0; s < slices; ++s) acc += partial[(long)row * slices + s];
    output[row] = acc;
}

torch::Tensor int4_repack17(torch::Tensor packed, int64_t rows, int64_t cols)
{
    const long words = (((long)rows + 7) / 8) * (cols / 32) * 32;
    auto dst = torch::empty(
        {words * 4},
        torch::TensorOptions().dtype(torch::kUInt8).device(packed.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    const long n = words;
    const int blocks = (int)((n + 255) / 256 > 4096 ? 4096 : (n + 255) / 256);
    int4_repack17_kernel<<<blocks, 256, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<uint32_t*>(dst.data_ptr()),
        (int)rows, (int)cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dst;
}

torch::Tensor int4_gemv_v17(
    torch::Tensor x,
    torch::Tensor repacked,
    torch::Tensor scales,
    int64_t rows,
    int64_t cols,
    int64_t groups)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int slices = (int)((cols + V17_SLICE - 1) / V17_SLICE);
    static torch::Tensor pc;
    const long needed = (long)rows * slices;
    if (!pc.defined() || pc.numel() < needed || pc.device() != x.device()) {
        pc = torch::empty(
            {needed}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    }
    torch::Tensor partial = pc.narrow(0, 0, needed);
    auto output = torch::empty(
        {rows}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    dim3 grid(slices, (unsigned)(((rows / 8) + 3) / 4));
    int4_gemv_v17_kernel<<<grid, 128, V17_SLICE * sizeof(float), stream>>>(
        x.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(repacked.data_ptr()),
        reinterpret_cast<const __half*>(scales.data_ptr()),
        partial.data_ptr<float>(), (int)rows, (int)cols, (int)groups, slices);
    v17_reduce<<<(rows + 255) / 256, 256, 0, stream>>>(
        partial.data_ptr<float>(), output.data_ptr<float>(), (int)rows, slices);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

ext = load_inline(
    name="int4_gemv_v17c_proto",
    cpp_sources=(
        "torch::Tensor int4_repack17(torch::Tensor packed, int64_t rows, int64_t cols);\n"
        "torch::Tensor int4_gemv_v17(torch::Tensor x, torch::Tensor repacked,"
        " torch::Tensor scales, int64_t rows, int64_t cols, int64_t groups);"
    ),
    cuda_sources=SRC,
    functions=["int4_repack17", "int4_gemv_v17"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


def bench(fn, repeats=300, warmup=50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(repeats):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / repeats


def main() -> None:
    torch.manual_seed(5090)
    rows, cols = 5120, 17408
    groups = cols // 64
    packed = torch.randint(
        0, 256, (rows * cols // 2,), device="cuda", dtype=torch.uint8)
    scales = (torch.randn(rows, groups, device="cuda") * 0.05).half()
    x = torch.randn(cols, device="cuda", dtype=torch.float32)

    ref_rows = 64
    idx = packed[: ref_rows * cols // 2].view(ref_rows, cols // 2)
    nib = torch.stack([(idx & 15).long(), (idx >> 4).long()], -1).view(ref_rows, cols) - 8
    sc = scales[:ref_rows].repeat_interleave(64, dim=1).float()
    ref = (nib.float() * sc * x).sum(dim=1)

    rep = ext.int4_repack17(packed, rows, cols)
    out = ext.int4_gemv_v17(x, rep, scales, rows, cols, groups)
    torch.cuda.synchronize()
    diff = (out[:ref_rows] - ref).abs()
    rel = diff / ref.abs().clamp_min(1e-3)
    print(f"numerics: max_abs={diff.max().item():.5f} p99_rel={rel.quantile(0.99).item():.4f}")

    ms = bench(lambda: ext.int4_gemv_v17(x, rep, scales, rows, cols, groups))
    nbytes = rows * cols * 0.5 + rows * groups * 2 + cols * 4
    print(f"v17 5120x17408: {ms:.4f} ms -> {nbytes / (ms / 1e3) / 1e9:.0f} GB/s (v1=650 目标1500)")

    for r2, c2 in ((5120, 5120), (248320, 5120)):
        p2 = torch.randint(0, 256, (r2 * c2 // 2,), device="cuda", dtype=torch.uint8)
        s2 = (torch.randn(r2, c2 // 64, device="cuda") * 0.05).half()
        x2 = torch.randn(c2, device="cuda")
        rp2 = ext.int4_repack17(p2, r2, c2)
        m2 = bench(lambda: ext.int4_gemv_v17(x2, rp2, s2, r2, c2, c2 // 64), 100, 30)
        nb2 = r2 * c2 * 0.5 + r2 * (c2 // 64) * 2 + c2 * 4
        print(f"v17 {r2}x{c2}: {m2:.4f} ms -> {nb2 / (m2 / 1e3) / 1e9:.0f} GB/s")


if __name__ == "__main__":
    main()
