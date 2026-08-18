"""v1b — batched int4 GEMV on the v1 (row-major packed) layout.

Unlike v21b (which needs the interleave repack and breaks MTP greedy
acceptance), v1b keeps the exact v1 kernel structure: same per-lane FMA
order per batch row, so outputs are bit-identical to looping v1 over the
batch — MTP's strict greedy comparison keeps its 5/12 acceptance while the
weight rows stream from DRAM once for all B activations.
"""
from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

constexpr int V1B_MAXB = 5;

constexpr int V1B_SLICE = 4096;   // per-slice shared: B*4096*4B <= 80KB

template <int ROWS_PER_BLOCK>
__global__ void int4_gemv_v1b_kernel(
    const float* __restrict__ x,          // [B, cols]
    const uint8_t* __restrict__ packed,   // [rows, cols/2]
    const __half* __restrict__ scales,    // [rows, groups]
    float* __restrict__ output,           // [B, rows]
    const int rows,
    const int cols,
    const int groups,
    const int B)
{
    extern __shared__ float sx[];         // [B][V1B_SLICE]
    const int lane = threadIdx.x;
    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;
    const int packed_cols = cols >> 1;
    const uint8_t* qrow = packed + (long)row * packed_cols;
    const __half* srow = scales + (long)row * groups;
    const int slice_groups = V1B_SLICE >> 6;   // 64
    const int slices = (cols + V1B_SLICE - 1) / V1B_SLICE;

    float accs[V1B_MAXB];
#pragma unroll
    for (int b = 0; b < V1B_MAXB; ++b) accs[b] = 0.f;

    for (int slice = 0; slice < slices; ++slice) {
        const int g0 = slice * slice_groups;
        const int g1 = min(groups, g0 + slice_groups);
        const int here = (min(cols, (g1 << 6)) - (g0 << 6));
        // Stage this slice's activations for all B rows.
        const int linear = threadIdx.y * 32 + lane;
        for (int c = linear; c < B * here; c += 32 * ROWS_PER_BLOCK) {
            const int b = c / here;
            const int k = c % here;
            sx[b * V1B_SLICE + k] = x[(long)b * cols + (g0 << 6) + k];
        }
        __syncthreads();
        for (int group = g0; group < g1; ++group) {
            float scale = lane == 0 ? __half2float(srow[group]) : 0.f;
            scale = __shfl_sync(0xffffffffu, scale, 0);
            const int byte_index = group * 32 + lane;
            const uint8_t q = __ldg(qrow + byte_index);
            const int col = group * 64 + lane * 2;
            const float lo = __fmul_rn(
                static_cast<float>((q & 15) - 8), scale);
            const float hi = __fmul_rn(
                static_cast<float>((q >> 4) - 8), scale);
#pragma unroll
            for (int b = 0; b < V1B_MAXB; ++b) {
                if (b >= B) break;
                const float* xb = sx + b * V1B_SLICE;
                const int local = col - (g0 << 6);
                accs[b] = __fmaf_rn(lo, xb[local], accs[b]);
                accs[b] = __fmaf_rn(hi, xb[local + 1], accs[b]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int b = 0; b < V1B_MAXB; ++b) {
        if (b >= B) break;
        float acc = accs[b];
        acc += __shfl_down_sync(0xffffffffu, acc, 16);
        acc += __shfl_down_sync(0xffffffffu, acc, 8);
        acc += __shfl_down_sync(0xffffffffu, acc, 4);
        acc += __shfl_down_sync(0xffffffffu, acc, 2);
        acc += __shfl_down_sync(0xffffffffu, acc, 1);
        if (lane == 0) {
            output[(long)b * rows + row] = acc;
        }
    }
}

torch::Tensor int4_gemv_v1b(
    torch::Tensor x,                      // [B, cols] fp32
    torch::Tensor packed,
    torch::Tensor scales,
    int64_t rows,
    int64_t cols,
    int64_t groups)
{
    const int B = (int)x.size(0);
    TORCH_CHECK(B >= 1 && B <= V1B_MAXB, "batch 1..5");
    TORCH_CHECK(cols % 64 == 0, "cols%64");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto output = torch::empty(
        {B, rows}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    const size_t shared = (size_t)B * V1B_SLICE * sizeof(float);
    static size_t configured = 0;
    if (configured < shared) {
        auto set_attr = [&](auto kernel) {
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)shared);
        };
        set_attr(int4_gemv_v1b_kernel<32>);
        set_attr(int4_gemv_v1b_kernel<16>);
        set_attr(int4_gemv_v1b_kernel<8>);
        configured = shared;
    }
    const int rpb = rows >= 4096 ? 32 : (rows >= 2048 ? 16 : 8);
    if (rpb == 32) {
        int4_gemv_v1b_kernel<32><<<(rows + 31) / 32, dim3(32, 32), shared, stream>>>(
            x.data_ptr<float>(), packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(scales.data_ptr()),
            output.data_ptr<float>(), (int)rows, (int)cols, (int)groups, B);
    } else if (rpb == 16) {
        int4_gemv_v1b_kernel<16><<<(rows + 15) / 16, dim3(32, 16), shared, stream>>>(
            x.data_ptr<float>(), packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(scales.data_ptr()),
            output.data_ptr<float>(), (int)rows, (int)cols, (int)groups, B);
    } else {
        int4_gemv_v1b_kernel<8><<<(rows + 7) / 8, dim3(32, 8), shared, stream>>>(
            x.data_ptr<float>(), packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(scales.data_ptr()),
            output.data_ptr<float>(), (int)rows, (int)cols, (int)groups, B);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

ext = load_inline(
    name="int4_gemv_v1b_proto",
    cpp_sources=(
        "torch::Tensor int4_gemv_v1b(torch::Tensor x, torch::Tensor packed,"
        " torch::Tensor scales, int64_t rows, int64_t cols, int64_t groups);"
    ),
    cuda_sources=SRC,
    functions=["int4_gemv_v1b"],
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

    B = 5
    x = torch.randn(B, cols, device="cuda", dtype=torch.float32)
    out_b = ext.int4_gemv_v1b(x, packed, scales, rows, cols, groups)
    # 对照: 逐 batch 调 B=1(bit 一致性验证)
    max_diff = 0.0
    for b in range(B):
        ref = ext.int4_gemv_v1b(
            x[b:b+1].contiguous(), packed, scales, rows, cols, groups)
        max_diff = max(max_diff, (out_b[b] - ref[0]).abs().max().item())
    print(f"bit-consistency vs per-token v1b: max_abs={max_diff:.2e}")
    # torch 参考数值
    ref_rows = 64
    idx = packed[: ref_rows * cols // 2].view(ref_rows, cols // 2)
    nib = torch.stack([(idx & 15).long(), (idx >> 4).long()], -1).view(ref_rows, cols) - 8
    sc = scales[:ref_rows].repeat_interleave(64, 1).float()
    ref0 = (nib.float() * sc * x[0]).sum(1)
    print(f"torch-ref numerics b0: max_abs={(out_b[0, :ref_rows] - ref0).abs().max().item():.5f}")

    # 冷读流 batch=5
    N_MAT = 60
    mats = [(torch.randint(
        0, 256, (rows * cols // 2,), device="cuda", dtype=torch.uint8),
        (torch.randn(rows, groups, device="cuda") * 0.05).half()) for _ in range(N_MAT)]

    def one():
        for pk2, sc2 in mats:
            ext.int4_gemv_v1b(x, pk2, sc2, rows, cols, groups)

    for _ in range(3):
        one()
    torch.cuda.synchronize()
    s3 = torch.cuda.Event(enable_timing=True)
    e3 = torch.cuda.Event(enable_timing=True)
    s3.record()
    for _ in range(5):
        one()
    e3.record()
    torch.cuda.synchronize()
    ms = s3.elapsed_time(e3) / 5 / N_MAT
    nbytes = rows * cols * 0.5 + rows * groups * 2 + B * cols * 4
    print(f"v1b B=5 cold: {ms:.4f} ms/pass -> {nbytes / (ms / 1e3) / 1e9:.0f} GB/s "
          f"({ms/B:.4f} ms/token; v21b 0.0414, v21 0.0531)")


if __name__ == "__main__":
    main()
