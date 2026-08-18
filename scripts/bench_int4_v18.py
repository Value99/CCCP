"""INT4 GEMV v18 — v17 layout + cp.async double-buffered weight prefetch.

Attacks the measured end-to-end bottleneck directly: decode GEMV streams
cold weights from DRAM every token, and v17's on-demand __ldg pays full
latency per superstep.  v18 prefetches the next stage of repacked weight
tiles into a shared double buffer via cp.async (SM80+; SM75 compiles a
plain LDG+st.shared fallback), overlapping DRAM latency with FMA compute.

The bench streams 60 different matrices per pass (> L2) so numbers reflect
the real decode cold-read pattern instead of an L2-resident microbench.
"""
from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
#if __CUDA_ARCH__ >= 800
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s),
                 "l"(gmem));
#else
    *reinterpret_cast<uint4*>(smem) = *reinterpret_cast<const uint4*>(gmem);
#endif
}
__device__ __forceinline__ void cp_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
#endif
}
template <int N>
__device__ __forceinline__ void cp_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

// Reuse the verified marlin tile layout via the engine's repacker entry is
// not available here; local copy for the standalone bench.
__global__ void repack18_kernel(
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

// Block = 4 warps, each warp owns 8 rows; k-slice 2048 columns.
// Stage = 8 supersteps => 1 KB weight per warp; double buffer 2 KB/warp.
// smem: x[2048] floats (8 KB) + w[2][4 warps][256 words] (8 KB).
constexpr int V18_SLICE = 2048;
constexpr int V18_STAGE_SS = 8;               // supersteps per stage
constexpr int V18_STAGES = V18_SLICE / 32 / V18_STAGE_SS;  // 8
constexpr int V18_WBUF_WORDS = V18_STAGE_SS * 32;          // 256 per warp

__global__ void int4_gemv_v18_kernel(
    const float* __restrict__ x,
    const uint32_t* __restrict__ repacked,
    const __half* __restrict__ scales,
    float* __restrict__ partial,
    const int rows,
    const int cols,
    const int groups,
    const int slices)
{
    extern __shared__ unsigned char smem_raw[];
    float* sx = reinterpret_cast<float*>(smem_raw);
    uint32_t* wbuf = reinterpret_cast<uint32_t*>(smem_raw + V18_SLICE * 4);

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int slice = blockIdx.x;
    const int k0 = slice * V18_SLICE;
    const int here = min(V18_SLICE, cols - k0);
    if (here <= 0) return;
    for (int c = threadIdx.x; c < here; c += 128) {
        sx[c] = x[k0 + c];
    }
    const int row0 = (blockIdx.y * 4 + warp) * 8;
    if (row0 >= rows) {
        __syncthreads();
        return;
    }
    const int j = lane >> 2;
    const int i = lane & 3;
    const int row = row0 + j;
    const __half* srow = scales + (long)row * groups;
    const int tiles_k = cols >> 5;
    const uint32_t* wbase = repacked +
        ((((long)(row0 >> 3)) * tiles_k + (k0 >> 5)) << 5);
    // Stage prefetch target for this warp: consecutive tiles.
    uint32_t* mybuf[2] = {
        wbuf + warp * V18_WBUF_WORDS,
        wbuf + (4 + warp) * V18_WBUF_WORDS,
    };
    const int ss_total = here >> 5;

    auto prefetch = [&](int stage) {
        uint32_t* dstbuf = mybuf[stage & 1];
        const int ss0 = stage * V18_STAGE_SS;
#pragma unroll
        for (int u = 0; u < V18_STAGE_SS; u += 2) {   // 8 lanes x 16B = 128B
            const int ss = ss0 + u;
            if (ss < ss_total && lane < 8) {
                cp_async16(dstbuf + u * 32 + lane * 4,
                           wbase + ss * 32 + lane * 4);
                cp_async16(dstbuf + (u + 1) * 32 + lane * 4,
                           wbase + (ss + 1) * 32 + lane * 4);
            }
        }
        cp_commit();
    };

    prefetch(0);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int st = 0; st < V18_STAGES; ++st) {
        if (st + 1 < V18_STAGES) {
            prefetch(st + 1);
            cp_wait<1>();                 // current stage ready
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        const uint32_t* src = mybuf[st & 1];
        const int ss0 = st * V18_STAGE_SS;
        const int ss_end = min(V18_STAGE_SS, ss_total - ss0);
        for (int u = 0; u < ss_end; ++u) {
            const uint32_t w = src[u * 32 + lane];
            const int col0 = (ss0 + u) << 5;
            const float sc = __half2float(
                __ldg(srow + ((k0 + col0 + 2 * i) >> 6)));
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
        __syncthreads();
    }
    float acc = (a0 + a1) + (a2 + a3);
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, off, 4);
    }
    if (row < rows && (lane & 3) == 0) {
        partial[(long)row * slices + slice] = acc;
    }
}

__global__ void v18_reduce(
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

torch::Tensor repack18(torch::Tensor packed, int64_t rows, int64_t cols)
{
    const long words = (((long)rows + 7) / 8) * (cols / 32) * 32;
    auto dst = torch::empty(
        {words * 4},
        torch::TensorOptions().dtype(torch::kUInt8).device(packed.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (int)((words + 255) / 256 > 4096 ? 4096 : (words + 255) / 256);
    repack18_kernel<<<blocks, 256, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<uint32_t*>(dst.data_ptr()),
        (int)rows, (int)cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dst;
}

torch::Tensor int4_gemv_v18(
    torch::Tensor x,
    torch::Tensor repacked,
    torch::Tensor scales,
    int64_t rows,
    int64_t cols,
    int64_t groups)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int slices = (int)((cols + V18_SLICE - 1) / V18_SLICE);
    static torch::Tensor pc;
    const long needed = (long)rows * slices;
    if (!pc.defined() || pc.numel() < needed || pc.device() != x.device()) {
        pc = torch::empty(
            {needed}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    }
    torch::Tensor partial = pc.narrow(0, 0, needed);
    auto output = torch::empty(
        {rows}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    const size_t smem = V18_SLICE * 4 + 2 * 4 * V18_WBUF_WORDS * 4;
    dim3 grid(slices, (unsigned)(((rows / 8) + 3) / 4));
    int4_gemv_v18_kernel<<<grid, 128, smem, stream>>>(
        x.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(repacked.data_ptr()),
        reinterpret_cast<const __half*>(scales.data_ptr()),
        partial.data_ptr<float>(), (int)rows, (int)cols, (int)groups, slices);
    v18_reduce<<<(rows + 255) / 256, 256, 0, stream>>>(
        partial.data_ptr<float>(), output.data_ptr<float>(), (int)rows, slices);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

ext = load_inline(
    name="int4_gemv_v18_proto",
    cpp_sources=(
        "torch::Tensor repack18(torch::Tensor packed, int64_t rows, int64_t cols);\n"
        "torch::Tensor int4_gemv_v18(torch::Tensor x, torch::Tensor repacked,"
        " torch::Tensor scales, int64_t rows, int64_t cols, int64_t groups);"
    ),
    cuda_sources=SRC,
    functions=["repack18", "int4_gemv_v18"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


def main() -> None:
    torch.manual_seed(5090)
    rows, cols = 5120, 17408
    groups = cols // 64

    # 冷读流:60 个不同矩阵循环(780MB >> L2),模拟 decode 每 token 冷权重
    N_MAT = 60
    mats = []
    for _ in range(N_MAT):
        pk = torch.randint(
            0, 256, (rows * cols // 2,), device="cuda", dtype=torch.uint8)
        sc = (torch.randn(rows, groups, device="cuda") * 0.05).half()
        mats.append((ext.repack18(pk, rows, cols), pk, sc))
    x = torch.randn(cols, device="cuda", dtype=torch.float32)

    # 数值(第一个矩阵前 64 行,用原始 packed 参考)
    rep0, pk0, sc0 = mats[0]
    ref_rows = 64
    idx = pk0[: ref_rows * cols // 2].view(ref_rows, cols // 2)
    nib = torch.stack([(idx & 15).long(), (idx >> 4).long()], -1).view(ref_rows, cols) - 8
    ref = (nib.float() * sc0[:ref_rows].repeat_interleave(64, 1).float() * x).sum(1)
    out = ext.int4_gemv_v18(x, rep0, sc0, rows, cols, groups)
    torch.cuda.synchronize()
    diff = (out[:ref_rows] - ref).abs()
    rel = diff / ref.abs().clamp_min(1e-3)
    print(f"numerics: max_abs={diff.max().item():.5f} p99_rel={rel.quantile(0.99).item():.4f}")

    def stream_pass() -> None:
        for rep, _pk, sc in mats:
            ext.int4_gemv_v18(x, rep, sc, rows, cols, groups)

    for _ in range(3):
        stream_pass()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(5):
        stream_pass()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / 5 / N_MAT
    nbytes = rows * cols * 0.5 + rows * groups * 2 + cols * 4
    print(f"v18 cold-stream: {ms:.4f} ms/matrix -> {nbytes / (ms / 1e3) / 1e9:.0f} GB/s "
          f"(v17 冷读对照见同脚本 v17 输出;目标 1500)")


if __name__ == "__main__":
    main()
