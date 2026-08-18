"""Minimal device-side stall probe for the v21 GEMV loop (no ncu needed).

Runs one 5120x17408 v21-layout GEMV with clock64() sampling inside the
4-superstep-group main loop: load-phase (uint4 fetch) vs compute-phase
(FMA + scale) cycle counts for one probed warp, reported from the kernel.
"""
from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <vector>

__global__ void repack_kernel(
    const uint8_t* __restrict__ src,
    uint32_t* __restrict__ dst,
    const int rows,
    const int cols)
{
    const long total = (((long)rows + 7) / 8) * (cols / 128) * 128;
    for (long w = blockIdx.x * (long)blockDim.x + threadIdx.x;
         w < total; w += (long)gridDim.x * blockDim.x) {
        const int intra = (int)(w & 127);
        const long tile = w >> 7;
        const int groups_k = cols / 128;
        const int sg = (int)(tile % groups_k);
        const int tile_n = (int)(tile / groups_k);
        const int j = intra >> 4;
        const int i = (intra >> 2) & 3;
        const int u = intra & 3;
        const int row = tile_n * 8 + j;
        uint32_t out = 0u;
        if (row < rows) {
            const uint8_t* srow = src + (long)row * (cols >> 1) + (sg * 4 + u) * 16;
            out = (uint32_t)srow[i] | ((uint32_t)srow[i + 4] << 8) |
                  ((uint32_t)srow[i + 8] << 16) | ((uint32_t)srow[i + 12] << 24);
        }
        dst[w] = out;
    }
}

constexpr int P_SLICE = 2048;

__global__ void probe_kernel(
    const float* __restrict__ x,
    const uint32_t* __restrict__ repacked,
    const __half* __restrict__ scales,
    float* __restrict__ partial,
    unsigned long long* __restrict__ dbg,
    const int rows,
    const int cols,
    const int groups,
    const int slices)
{
    extern __shared__ float sx[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int slice = blockIdx.x;
    const int k0 = slice * P_SLICE;
    const int here = min(P_SLICE, cols - k0);
    if (here <= 0) return;
    for (int c = threadIdx.x; c < here; c += 128) sx[c] = x[k0 + c];
    __syncthreads();
    const int row0 = (blockIdx.y * 4 + warp) * 8;
    if (row0 >= rows) return;
    const int j = lane >> 2;
    const int i = lane & 3;
    const int row = row0 + j;
    const __half* srow = scales + (long)row * groups;
    const int groups_k = cols >> 7;
    const uint32_t* base = repacked +
        ((((long)(row0 >> 3)) * groups_k + (k0 >> 7)) << 7);
    const uint32_t* base_li = base + (lane >> 2) * 16 + (lane & 3) * 4;

    float p[32];
#pragma unroll
    for (int u = 0; u < 32; ++u) p[u] = 0.f;
    const int ss = here >> 5;
    const bool probe = (row0 == 0 && lane == 0);
    unsigned long long t_pre = 0, t_ld = 0, t_fma = 0;
    int sg = 0;
    for (; sg + 4 <= (ss >> 2); sg += 4) {
        if (probe && sg == 4) t_pre = clock64();
        const uint4 pw[4] = {
            *reinterpret_cast<const uint4*>(base_li + (sg << 7)),
            *reinterpret_cast<const uint4*>(base_li + ((sg + 1) << 7)),
            *reinterpret_cast<const uint4*>(base_li + ((sg + 2) << 7)),
            *reinterpret_cast<const uint4*>(base_li + ((sg + 3) << 7)),
        };
        if (probe && sg == 4) t_ld = clock64();
        const uint32_t wv[16] = {
            pw[0].x, pw[0].y, pw[0].z, pw[0].w,
            pw[1].x, pw[1].y, pw[1].z, pw[1].w,
            pw[2].x, pw[2].y, pw[2].z, pw[2].w,
            pw[3].x, pw[3].y, pw[3].z, pw[3].w,
        };
#pragma unroll
        for (int s2 = 0; s2 < 16; ++s2) {
            const int ts = sg * 4 + s2;
            const int ca = ts << 5;
            const float sa = __half2float(
                __ldg(srow + ((k0 + ca + 2 * i) >> 6)));
            const uint32_t w = wv[s2];
            const uint32_t b0 = w & 0xFF, b1 = (w >> 8) & 0xFF;
            const uint32_t b2 = (w >> 16) & 0xFF, b3 = (w >> 24) & 0xFF;
            p[s2 * 2] = __fmaf_rn(static_cast<float>((int)(b0 & 15) - 8) * sa, sx[ca + 2 * i], p[s2 * 2]);
            p[s2 * 2] = __fmaf_rn(static_cast<float>((int)(b0 >> 4) - 8) * sa, sx[ca + 2 * i + 1], p[s2 * 2]);
            p[s2 * 2 + 1] = __fmaf_rn(static_cast<float>((int)(b1 & 15) - 8) * sa, sx[ca + 2 * i + 8], p[s2 * 2 + 1]);
            p[s2 * 2 + 1] = __fmaf_rn(static_cast<float>((int)(b1 >> 4) - 8) * sa, sx[ca + 2 * i + 9], p[s2 * 2 + 1]);
            p[s2 * 2] = __fmaf_rn(static_cast<float>((int)(b2 & 15) - 8) * sa, sx[ca + 2 * i + 16], p[s2 * 2]);
            p[s2 * 2] = __fmaf_rn(static_cast<float>((int)(b2 >> 4) - 8) * sa, sx[ca + 2 * i + 17], p[s2 * 2]);
            p[s2 * 2 + 1] = __fmaf_rn(static_cast<float>((int)(b3 & 15) - 8) * sa, sx[ca + 2 * i + 24], p[s2 * 2 + 1]);
            p[s2 * 2 + 1] = __fmaf_rn(static_cast<float>((int)(b3 >> 4) - 8) * sa, sx[ca + 2 * i + 25], p[s2 * 2 + 1]);
        }
        if (probe && sg == 4) {
            t_fma = clock64();
            dbg[0] = t_ld - t_pre;
            dbg[1] = t_fma - t_ld;
        }
    }
    for (int ts2 = sg; ts2 < (ss >> 2); ++ts2) {
        const uint4 pw = *reinterpret_cast<const uint4*>(base_li + (ts2 << 7));
        const uint32_t wt[4] = {pw.x, pw.y, pw.z, pw.w};
        for (int u2 = 0; u2 < 4; ++u2) {
            const int ts = ts2 * 4 + u2;
            const uint32_t w = wt[u2];
            const int ca = ts << 5;
            const float sa = __half2float(
                __ldg(srow + ((k0 + ca + 2 * i) >> 6)));
            const uint32_t b0 = w & 0xFF, b1 = (w >> 8) & 0xFF;
            const uint32_t b2 = (w >> 16) & 0xFF, b3 = (w >> 24) & 0xFF;
            p[u2 * 2] = __fmaf_rn(static_cast<float>((int)(b0 & 15) - 8) * sa, sx[ca + 2 * i], p[u2 * 2]);
            p[u2 * 2] = __fmaf_rn(static_cast<float>((int)(b0 >> 4) - 8) * sa, sx[ca + 2 * i + 1], p[u2 * 2]);
            p[u2 * 2 + 1] = __fmaf_rn(static_cast<float>((int)(b1 & 15) - 8) * sa, sx[ca + 2 * i + 8], p[u2 * 2 + 1]);
            p[u2 * 2 + 1] = __fmaf_rn(static_cast<float>((int)(b1 >> 4) - 8) * sa, sx[ca + 2 * i + 9], p[u2 * 2 + 1]);
            p[u2 * 2] = __fmaf_rn(static_cast<float>((int)(b2 & 15) - 8) * sa, sx[ca + 2 * i + 16], p[u2 * 2]);
            p[u2 * 2] = __fmaf_rn(static_cast<float>((int)(b2 >> 4) - 8) * sa, sx[ca + 2 * i + 17], p[u2 * 2]);
            p[u2 * 2 + 1] = __fmaf_rn(static_cast<float>((int)(b3 & 15) - 8) * sa, sx[ca + 2 * i + 24], p[u2 * 2 + 1]);
            p[u2 * 2 + 1] = __fmaf_rn(static_cast<float>((int)(b3 >> 4) - 8) * sa, sx[ca + 2 * i + 25], p[u2 * 2 + 1]);
        }
    }
    float acc = 0.f;
#pragma unroll
    for (int u = 0; u < 32; ++u) acc += p[u];
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, off, 4);
    }
    if (row < rows && (lane & 3) == 0) {
        partial[(long)row * slices + slice] = acc;
    }
}

__global__ void probe_reduce(
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

torch::Tensor repack(torch::Tensor packed, int64_t rows, int64_t cols)
{
    const long words = (((long)rows + 7) / 8) * (cols / 128) * 128;
    auto dst = torch::empty(
        {words * 4},
        torch::TensorOptions().dtype(torch::kUInt8).device(packed.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (int)((words + 255) / 256 > 4096 ? 4096 : (words + 255) / 256);
    repack_kernel<<<blocks, 256, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<uint32_t*>(dst.data_ptr()),
        (int)rows, (int)cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dst;
}

static long g_probe[2] = {0, 0};

torch::Tensor probe_gemv(
    torch::Tensor x,
    torch::Tensor repacked,
    torch::Tensor scales,
    int64_t rows,
    int64_t cols,
    int64_t groups)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int slices = (int)((cols + P_SLICE - 1) / P_SLICE);
    static torch::Tensor pc;
    const long needed = (long)rows * slices;
    if (!pc.defined() || pc.numel() < needed || pc.device() != x.device()) {
        pc = torch::empty(
            {needed}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    }
    torch::Tensor partial = pc.narrow(0, 0, needed);
    auto dbg = torch::zeros(
        {2}, torch::TensorOptions().dtype(torch::kInt64).device(x.device()));
    auto output = torch::empty(
        {rows}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    dim3 grid(slices, (unsigned)(((rows / 8) + 3) / 4));
    probe_kernel<<<grid, 128, P_SLICE * sizeof(float), stream>>>(
        x.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(repacked.data_ptr()),
        reinterpret_cast<const __half*>(scales.data_ptr()),
        partial.data_ptr<float>(),
        reinterpret_cast<unsigned long long*>(dbg.data_ptr()),
        (int)rows, (int)cols, (int)groups, slices);
    probe_reduce<<<(rows + 255) / 256, 256, 0, stream>>>(
        partial.data_ptr<float>(), output.data_ptr<float>(), (int)rows, slices);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    g_probe[0] = dbg[0].item<long>();
    g_probe[1] = dbg[1].item<long>();
    return output;
}

std::vector<long> probe_read()
{
    return std::vector<long>(g_probe, g_probe + 2);
}
"""

ext = load_inline(
    name="int4_probe_min",
    cpp_sources=(
        "torch::Tensor repack(torch::Tensor packed, int64_t rows, int64_t cols);\n"
        "torch::Tensor probe_gemv(torch::Tensor x, torch::Tensor repacked,"
        " torch::Tensor scales, int64_t rows, int64_t cols, int64_t groups);\n"
        "std::vector<long> probe_read();"
    ),
    cuda_sources=SRC,
    functions=["repack", "probe_gemv", "probe_read"],
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
    rep = ext.repack(packed, rows, cols)
    out = ext.probe_gemv(x, rep, scales, rows, cols, groups)
    torch.cuda.synchronize()
    ref_rows = 64
    idx = packed[: ref_rows * cols // 2].view(ref_rows, cols // 2)
    nib = torch.stack([(idx & 15).long(), (idx >> 4).long()], -1).view(ref_rows, cols) - 8
    ref = (nib.float() * scales[:ref_rows].repeat_interleave(64, 1).float() * x).sum(1)
    diff = (out[:ref_rows] - ref).abs()
    print(f"numerics max_abs={diff.max().item():.5f}")
    pr = ext.probe_read()
    total = pr[0] + pr[1]
    print(f"probe cycles (sg=4 iter, 128 columns): load={pr[0]} "
          f"compute={pr[1]} total={total} load占比={pr[0] * 100 // max(total, 1)}%")


if __name__ == "__main__":
    main()
