"""v21b — batched int4 GEMV (B activations share one weight stream).

The MTP verify path calls every projection with batch=5 tokens.  v21's
kernel reads the 12.5 GiB weight stream once per token, so naive looping
pays 5x DRAM traffic for a memory-bound op.  v21b stages B activation rows
in shared memory (B<=5: 2048*5*4B = 40 KiB) and emits B outputs per weight
pass — verify cost approaches 1x instead of Bx.
"""
from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

__global__ void repack_b_kernel(
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

constexpr int VB_SLICE = 2048;
constexpr int VB_MAXB = 5;

__global__ void int4_gemv_v21b_kernel(
    const float* __restrict__ x,          // [B, cols]
    const uint32_t* __restrict__ repacked,
    const __half* __restrict__ scales,
    float* __restrict__ partial,          // [B, rows, slices]
    const int rows,
    const int cols,
    const int groups,
    const int slices,
    const int B)
{
    extern __shared__ float sxB[];        // [VB_MAXB][VB_SLICE]
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int slice = blockIdx.x;
    const int k0 = slice * VB_SLICE;
    const int here = min(VB_SLICE, cols - k0);
    if (here <= 0) return;
    for (int idx = threadIdx.x; idx < B * here; idx += 128) {
        const int b = idx / here;
        const int c = idx % here;
        sxB[b * VB_SLICE + c] = x[(long)b * cols + k0 + c];
    }
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

    float acc[VB_MAXB][32];
#pragma unroll
    for (int b = 0; b < VB_MAXB; ++b)
#pragma unroll
        for (int u = 0; u < 32; ++u) acc[b][u] = 0.f;

    const int ss = here >> 5;
    int sg = 0;
    for (; sg + 4 <= (ss >> 2); sg += 4) {
        const uint4 pw[4] = {
            *reinterpret_cast<const uint4*>(base_li + (sg << 7)),
            *reinterpret_cast<const uint4*>(base_li + ((sg + 1) << 7)),
            *reinterpret_cast<const uint4*>(base_li + ((sg + 2) << 7)),
            *reinterpret_cast<const uint4*>(base_li + ((sg + 3) << 7)),
        };
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
#pragma unroll
            for (int b = 0; b < VB_MAXB; ++b) {
                if (b >= B) break;
                const float* xb = sxB + b * VB_SLICE;
                const int a2 = s2 * 2;
                acc[b][a2] = __fmaf_rn(static_cast<float>((int)(b0 & 15) - 8) * sa, xb[ca + 2 * i], acc[b][a2]);
                acc[b][a2] = __fmaf_rn(static_cast<float>((int)(b0 >> 4) - 8) * sa, xb[ca + 2 * i + 1], acc[b][a2]);
                acc[b][a2 + 1] = __fmaf_rn(static_cast<float>((int)(b1 & 15) - 8) * sa, xb[ca + 2 * i + 8], acc[b][a2 + 1]);
                acc[b][a2 + 1] = __fmaf_rn(static_cast<float>((int)(b1 >> 4) - 8) * sa, xb[ca + 2 * i + 9], acc[b][a2 + 1]);
                acc[b][a2] = __fmaf_rn(static_cast<float>((int)(b2 & 15) - 8) * sa, xb[ca + 2 * i + 16], acc[b][a2]);
                acc[b][a2] = __fmaf_rn(static_cast<float>((int)(b2 >> 4) - 8) * sa, xb[ca + 2 * i + 17], acc[b][a2]);
                acc[b][a2 + 1] = __fmaf_rn(static_cast<float>((int)(b3 & 15) - 8) * sa, xb[ca + 2 * i + 24], acc[b][a2 + 1]);
                acc[b][a2 + 1] = __fmaf_rn(static_cast<float>((int)(b3 >> 4) - 8) * sa, xb[ca + 2 * i + 25], acc[b][a2 + 1]);
            }
        }
    }
    for (int ts2 = sg; ts2 < (ss >> 2); ++ts2) {
        const uint4 pw = *reinterpret_cast<const uint4*>(base_li + (ts2 << 7));
        const uint32_t wt[4] = {pw.x, pw.y, pw.z, pw.w};
        for (int u2 = 0; u2 < 4; ++u2) {
            const int ts = ts2 * 4 + u2;
            const int ca = ts << 5;
            const float sa = __half2float(
                __ldg(srow + ((k0 + ca + 2 * i) >> 6)));
            const uint32_t w = wt[u2];
            const uint32_t b0 = w & 0xFF, b1 = (w >> 8) & 0xFF;
            const uint32_t b2 = (w >> 16) & 0xFF, b3 = (w >> 24) & 0xFF;
#pragma unroll
            for (int b = 0; b < VB_MAXB; ++b) {
                if (b >= B) break;
                const float* xb = sxB + b * VB_SLICE;
                acc[b][u2 * 2] = __fmaf_rn(static_cast<float>((int)(b0 & 15) - 8) * sa, xb[ca + 2 * i], acc[b][u2 * 2]);
                acc[b][u2 * 2] = __fmaf_rn(static_cast<float>((int)(b0 >> 4) - 8) * sa, xb[ca + 2 * i + 1], acc[b][u2 * 2]);
                acc[b][u2 * 2 + 1] = __fmaf_rn(static_cast<float>((int)(b1 & 15) - 8) * sa, xb[ca + 2 * i + 8], acc[b][u2 * 2 + 1]);
                acc[b][u2 * 2 + 1] = __fmaf_rn(static_cast<float>((int)(b1 >> 4) - 8) * sa, xb[ca + 2 * i + 9], acc[b][u2 * 2 + 1]);
                acc[b][u2 * 2] = __fmaf_rn(static_cast<float>((int)(b2 & 15) - 8) * sa, xb[ca + 2 * i + 16], acc[b][u2 * 2]);
                acc[b][u2 * 2] = __fmaf_rn(static_cast<float>((int)(b2 >> 4) - 8) * sa, xb[ca + 2 * i + 17], acc[b][u2 * 2]);
                acc[b][u2 * 2 + 1] = __fmaf_rn(static_cast<float>((int)(b3 & 15) - 8) * sa, xb[ca + 2 * i + 24], acc[b][u2 * 2 + 1]);
                acc[b][u2 * 2 + 1] = __fmaf_rn(static_cast<float>((int)(b3 >> 4) - 8) * sa, xb[ca + 2 * i + 25], acc[b][u2 * 2 + 1]);
            }
        }
    }
#pragma unroll
    for (int b = 0; b < VB_MAXB; ++b) {
        if (b >= B) break;
        float accf = 0.f;
#pragma unroll
        for (int u = 0; u < 32; ++u) accf += acc[b][u];
#pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            accf += __shfl_xor_sync(0xffffffffu, accf, off, 4);
        }
        if (row < rows && (lane & 3) == 0) {
            partial[(((long)b * rows) + row) * slices + slice] = accf;
        }
    }
}

__global__ void vb_reduce(
    const float* __restrict__ partial,
    float* __restrict__ output,           // [B, rows]
    const int rows,
    const int slices,
    const int B)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * rows;
    if (idx >= total) return;
    const int b = idx / rows;
    const int row = idx % rows;
    float acc = 0.f;
    for (int s = 0; s < slices; ++s) {
        acc += partial[((long)b * rows + row) * slices + s];
    }
    output[idx] = acc;
}

torch::Tensor repack_b(torch::Tensor packed, int64_t rows, int64_t cols)
{
    const long words = (((long)rows + 7) / 8) * (cols / 128) * 128;
    auto dst = torch::empty(
        {words * 4},
        torch::TensorOptions().dtype(torch::kUInt8).device(packed.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (int)((words + 255) / 256 > 4096 ? 4096 : (words + 255) / 256);
    repack_b_kernel<<<blocks, 256, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<uint32_t*>(dst.data_ptr()),
        (int)rows, (int)cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dst;
}

torch::Tensor int4_gemv_v21b(
    torch::Tensor x,                      // [B, cols]
    torch::Tensor repacked,
    torch::Tensor scales,
    int64_t rows,
    int64_t cols,
    int64_t groups)
{
    const int B = (int)x.size(0);
    TORCH_CHECK(B >= 1 && B <= VB_MAXB, "batch 1..5");
    auto stream = at::cuda::getCurrentCUDAStream();
    const int slices = (int)((cols + VB_SLICE - 1) / VB_SLICE);
    static torch::Tensor pc;
    const long needed = (long)B * rows * slices;
    if (!pc.defined() || pc.numel() < needed || pc.device() != x.device()) {
        pc = torch::empty(
            {needed}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    }
    torch::Tensor partial = pc.narrow(0, 0, needed);
    auto output = torch::empty(
        {B, rows}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    dim3 grid(slices, (unsigned)(((rows / 8) + 3) / 4));
    int4_gemv_v21b_kernel<<<grid, 128, VB_MAXB * VB_SLICE * sizeof(float), stream>>>(
        x.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(repacked.data_ptr()),
        reinterpret_cast<const __half*>(scales.data_ptr()),
        partial.data_ptr<float>(),
        (int)rows, (int)cols, (int)groups, slices, B);
    vb_reduce<<<(B * rows + 255) / 256, 256, 0, stream>>>(
        partial.data_ptr<float>(), output.data_ptr<float>(),
        (int)rows, slices, B);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

ext = load_inline(
    name="int4_gemv_v21b_proto",
    cpp_sources=(
        "torch::Tensor repack_b(torch::Tensor packed, int64_t rows, int64_t cols);\n"
        "torch::Tensor int4_gemv_v21b(torch::Tensor x, torch::Tensor repacked,"
        " torch::Tensor scales, int64_t rows, int64_t cols, int64_t groups);"
    ),
    cuda_sources=SRC,
    functions=["repack_b", "int4_gemv_v21b"],
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
    rep = ext.repack_b(packed, rows, cols)

    for B in (1, 5):
        x = torch.randn(B, cols, device="cuda", dtype=torch.float32)
        out = ext.int4_gemv_v21b(x, rep, scales, rows, cols, groups)
        torch.cuda.synchronize()
        # 参考: 每 batch 独立对前 64 行
        ref_rows = 64
        idx = packed[: ref_rows * cols // 2].view(ref_rows, cols // 2)
        nib = torch.stack([(idx & 15).long(), (idx >> 4).long()], -1).view(ref_rows, cols) - 8
        sc = scales[:ref_rows].repeat_interleave(64, 1).float()
        ok = True
        for b in range(B):
            ref = (nib.float() * sc * x[b]).sum(1)
            d = (out[b, :ref_rows] - ref).abs()
            rel = d / ref.abs().clamp_min(1e-3)
            if rel.quantile(0.99).item() > 0.01:
                ok = False
        print(f"B={B}: numerics {'OK (p99<1%)' if ok else 'FAIL'}")

    # 冷读流: batch=5 vs 5x batch=1 等效带宽
    N_MAT = 60
    mats = [(ext.repack_b(
        torch.randint(0, 256, (rows * cols // 2,), device="cuda", dtype=torch.uint8),
        rows, cols),
        (torch.randn(rows, groups, device="cuda") * 0.05).half()) for _ in range(N_MAT)]
    x5 = torch.randn(5, cols, device="cuda")

    def one_b5():
        for r2, s2 in mats:
            ext.int4_gemv_v21b(x5, r2, s2, rows, cols, groups)

    for _ in range(3):
        one_b5()
    torch.cuda.synchronize()
    s3 = torch.cuda.Event(enable_timing=True)
    e3 = torch.cuda.Event(enable_timing=True)
    s3.record()
    for _ in range(5):
        one_b5()
    e3.record()
    torch.cuda.synchronize()
    ms = s3.elapsed_time(e3) / 5 / N_MAT
    # 权重字节读一遍,产出 5 行
    nbytes = rows * cols * 0.5 + rows * groups * 2 + 5 * cols * 4
    eff = nbytes / (ms / 1e3) / 1e9
    per_tok = ms / 5
    print(f"v21b B=5 cold: {ms:.4f} ms/pass -> 权重流带宽 {eff:.0f} GB/s "
          f"(= {per_tok:.4f} ms/token; v21 逐token 0.0531)")
    print(f"verify 提速比: {5 * 0.0531 / ms:.2f}x")


if __name__ == "__main__":
    main()
