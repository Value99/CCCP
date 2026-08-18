"""Marlin-style INT4 mma.sync GEMV prototype bench (synthetic data).

Builds a standalone extension with:
  * marlin_int4_gemv kernel — mma.sync.m16n8k16 (SM75+, all architectures),
    INT4 dequantized in registers (nibble -> fp16 with G64 group scale) and
    fed straight into the tensor-core fragment; activation padded to m=16.
Compares numerics against int4_gemv_packed_f32 (v1) on random data and
reports achieved device bandwidth, to judge whether a 2 TB/s-class kernel
makes the 15 GiB tier capable of 100 tok/s.
"""
from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Marlin-style INT4 GEMV prototype.
//   out[n] = sum_k ((nibble(n,k) - 8) * scale(n, k/64)) * x[k]
//
// Weight stays in the runtime int4_g64 layout (row-major nibbles, one
// __half scale per 64 columns).  Each warp owns 8 output rows (one
// mma.sync n=8 tile) and a K-slice; lanes assemble their B-fragment from
// 8-byte contiguous reads, dequantize in registers, and run
// mma.sync.aligned.m16n8k16.f32.f16.f16.f32 (Turing baseline — SM75+,
// no wgmma/cluster, works on every architecture we ship).
// ---------------------------------------------------------------------------

#define FULL_MASK 0xffffffffu

__device__ __forceinline__ void mma_m16n8k16(
    const uint32_t* a, const uint32_t* b, float* c)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t half2_bits(__half2 v)
{
    return *reinterpret_cast<uint32_t*>(&v);
}

// Assemble an fp16x2 register from two raw half values.
__device__ __forceinline__ uint32_t h2(__half lo, __half hi)
{
    uint32_t v;
    asm("mov.b32 %0, {%1, %2};" : "=r"(v) : "h"(__half_as_ushort(lo)),
        "h"(__half_as_ushort(hi)));
    return v;
}

// Dequantize one 32-bit word (8 nibbles) into 4 fp16x2 registers given the
// group scale.  Nibble storage: byte i holds columns 2i (low) and 2i+1
// (high); all 8 nibbles share one G64 group scale, which the caller has
// already loaded when the 16-column slice is group-aligned.
__device__ __forceinline__ void dequant_word(
    uint32_t word, float scale, uint32_t* out4)
{
    __half s = __float2half_rn(scale);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t byte = (word >> (i * 8)) & 0xFF;
        const float lo = static_cast<float>(static_cast<int>(byte & 15) - 8) * scale;
        const float hi = static_cast<float>(static_cast<int>(byte >> 4) - 8) * scale;
        out4[i] = h2(__float2half_rn(lo), __float2half_rn(hi));
    }
    (void)s;
}

// ---------------------------------------------------------------------------
// Kernel: grid.x = k-slices, grid.y = row tiles (8 rows per warp, 4 warps
// per block => 32 rows per block).  m=16 tiles of the activation reuse the
// same x row (rows 1..15 are zero), only row 0 of the accumulator matters.
// ---------------------------------------------------------------------------
constexpr int MARLIN_KSPLIT = 2048;   // columns per k-slice
constexpr int MARLIN_WARPS = 4;       // 8 output rows per warp

__global__ void marlin_int4_gemv_kernel(
    const __half* __restrict__ x,        // [cols]
    const uint8_t* __restrict__ packed,  // [rows, cols/2]
    const __half* __restrict__ scales,   // [rows, groups]
    float* __restrict__ partial,         // [rows, slices]
    const int rows,
    const int cols,
    const int groups,
    const int slices)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row0 = (blockIdx.y * MARLIN_WARPS + warp) * 8;
    const int slice = blockIdx.x;
    const int k0 = slice * MARLIN_KSPLIT;
    const int k1 = min(cols, k0 + MARLIN_KSPLIT);
    if (row0 >= rows) return;

    // Per-lane accumulators: mma m16n8k16 c-fragment has 4 floats per lane
    // covering rows {lane/4, lane/4+8} x cols {2*(lane%4), +1}.
    float c[4] = {0.f, 0.f, 0.f, 0.f};

    const uint8_t* prow8 = packed + (long)(row0 + lane / 4) * (cols >> 1);
    const __half* srow = scales + (long)(row0 + lane / 4) * groups;

    // 32-column superstep.  Runtime int4_g64 layout cannot serve wide
    // aligned loads to lanes with odd byte offsets (t%4 = 1/3), so the
    // four sub-lanes of each row cooperatively load one aligned 4-byte
    // word (16 bytes per row per step) and exchange the two bytes each
    // lane needs through shuffles — the no-repack equivalent of Marlin's
    // offline weight repack.
    for (int k = k0; k < k1; k += 32) {
        const int kk = 2 * (lane % 4);
        const uint8_t* base = prow8 + (k >> 1);
        // Lane loads aligned word (k>>1 is 16-byte aligned; sub-lane adds
        // 4*(t%4) -> 4-byte aligned).  Row r's 16 bytes live in lanes
        // 4r..4r+3, one word each.
        const uint32_t my_word = __ldg(reinterpret_cast<const uint32_t*>(
            base + 4 * (lane % 4)));
        // Bytes needed by this lane: index kk/2 (+kk/2+8) within the row's
        // 16-byte window -> owner lane = row_base + byte_idx/4.
        const int row_base = lane & ~3;
        const int o0 = row_base + ((kk >> 1) >> 2);
        const int o1 = row_base + (((kk >> 1) + 4) >> 2);
        const uint32_t w0 = __shfl_sync(0xffffffffu, my_word, o0);
        const uint32_t w1 = __shfl_sync(0xffffffffu, my_word, o1);
        const int o2 = row_base + (((kk >> 1) + 8) >> 2);
        const int o3 = row_base + (((kk >> 1) + 12) >> 2);
        const uint32_t w2 = __shfl_sync(0xffffffffu, my_word, o2);
        const uint32_t w3 = __shfl_sync(0xffffffffu, my_word, o3);
        const uint32_t bw0 = (w0 >> (8 * ((kk >> 1) & 3))) & 0xFF;
        const uint32_t bw1 = (w1 >> (8 * ((kk >> 1) & 3))) & 0xFF;
        const uint32_t bw2 = (w2 >> (8 * ((kk >> 1) & 3))) & 0xFF;
        const uint32_t bw3 = (w3 >> (8 * ((kk >> 1) & 3))) & 0xFF;

        const float sc = __half2float(__ldg(srow + ((k + kk) >> 6)));
        uint32_t a0[4] = {0u, 0u, 0u, 0u}, a1[4] = {0u, 0u, 0u, 0u};
        if (lane < 4) {
            const int ak = 2 * (lane & 3);
            a0[0] = half2_bits(__halves2half2(x[k + ak], x[k + ak + 1]));
            a0[2] = half2_bits(__halves2half2(x[k + ak + 8], x[k + ak + 9]));
            a1[0] = half2_bits(__halves2half2(x[k + 16 + ak], x[k + 16 + ak + 1]));
            a1[2] = half2_bits(__halves2half2(x[k + 16 + ak + 8], x[k + 16 + ak + 9]));
        }
#pragma unroll
        for (int half_step = 0; half_step < 2; ++half_step) {
            const uint32_t x0 = half_step == 0 ? bw0 : bw2;
            const uint32_t x1 = half_step == 0 ? bw1 : bw3;
            const float l0 = static_cast<float>(static_cast<int>(x0 & 15) - 8) * sc;
            const float h0 = static_cast<float>(static_cast<int>(x0 >> 4) - 8) * sc;
            const float l1 = static_cast<float>(static_cast<int>(x1 & 15) - 8) * sc;
            const float h1 = static_cast<float>(static_cast<int>(x1 >> 4) - 8) * sc;
            uint32_t bfrag[2] = {
                h2(__float2half_rn(l0), __float2half_rn(h0)),
                h2(__float2half_rn(l1), __float2half_rn(h1)),
            };
            mma_m16n8k16(half_step == 0 ? a0 : a1, bfrag, c);
        }
    }

    // C layout: thread t holds {m = t/4, n = 2*(t%4)} and {+1 n} (plus the
    // m+8 rows in c2/c3, which are zero for m=1 activations).  With m=1
    // only lanes 0..3 (m==0) carry results, and each lane owns TWO whole
    // output rows n=2*lane and 2*lane+1 — no cross-lane reduce needed.
    if (lane < 4) {
        const int n0 = row0 + 2 * lane;
        if (n0 < rows) partial[(long)n0 * slices + slice] = c[0];
        if (n0 + 1 < rows) partial[(long)(n0 + 1) * slices + slice] = c[1];
    }
}

__global__ void marlin_reduce(
    const float* __restrict__ partial,
    float* __restrict__ output,
    int rows,
    int slices);

torch::Tensor marlin_int4_gemv(
    torch::Tensor x,        // [cols] half
    torch::Tensor packed,   // [rows*cols/2] uint8
    torch::Tensor scales,   // [rows, groups] half
    int64_t rows,
    int64_t cols,
    int64_t groups)
{
    TORCH_CHECK(cols % 64 == 0, "cols must be a multiple of 64");
    auto stream = at::cuda::getCurrentCUDAStream();
    const int slices = (int)((cols + MARLIN_KSPLIT - 1) / MARLIN_KSPLIT);
    static torch::Tensor partial_cache;
    const long needed = (long)rows * slices;
    if (!partial_cache.defined() ||
        partial_cache.numel() < needed ||
        partial_cache.device() != x.device()) {
        partial_cache = torch::empty(
            {needed},
            torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    }
    torch::Tensor partial = partial_cache.narrow(0, 0, needed);
    auto output = torch::empty(
        {rows}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    dim3 block(32 * MARLIN_WARPS);
    dim3 grid(slices, (unsigned)((rows / 8 + MARLIN_WARPS - 1) / MARLIN_WARPS));
    marlin_int4_gemv_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr()),
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr()),
        partial.data_ptr<float>(),
        (int)rows, (int)cols, (int)groups, slices);
    // reduce
    marlin_reduce<<<(rows + 255) / 256, 256, 0, stream>>>(
        partial.data_ptr<float>(), output.data_ptr<float>(),
        (int)rows, slices);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void marlin_reduce(
    const float* __restrict__ partial,
    float* __restrict__ output,
    int rows,
    int slices)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float acc = 0.f;
    for (int s = 0; s < slices; ++s) {
        acc += partial[(long)row * slices + s];
    }
    output[row] = acc;
}
"""

ext = load_inline(
    name="marlin_int4_proto",
    cpp_sources="torch::Tensor marlin_int4_gemv(torch::Tensor x, torch::Tensor packed, torch::Tensor scales, int64_t rows, int64_t cols, int64_t groups);",
    cuda_sources=SRC,
    functions=["marlin_int4_gemv"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


def main() -> None:
    torch.manual_seed(5090)
    rows, cols = 5120, 17408
    groups = cols // 64
    packed = torch.randint(
        0, 256, (rows * cols // 2,), device="cuda", dtype=torch.uint8
    )
    scales = (torch.randn(rows, groups, device="cuda") * 0.05).half()
    x = torch.randn(cols, device="cuda", dtype=torch.float16)

    # 参考:v1 语义 (nibble-8)*scale*x,用纯 torch 慢速核对小样本
    ref_rows = 64
    idx = packed[: ref_rows * cols // 2].view(ref_rows, cols // 2)
    lo = (idx & 15).long()
    hi = (idx >> 4).long()
    nib = torch.stack([lo, hi], dim=-1).view(ref_rows, cols) - 8
    sc = scales[:ref_rows].repeat_interleave(64, dim=1).float()
    ref = (nib.float() * sc * x.float()).sum(dim=1)

    out = ext.marlin_int4_gemv(x, packed, scales, rows, cols, groups)
    torch.cuda.synchronize()
    diff = (out[:ref_rows] - ref).abs()
    rel = diff / ref.abs().clamp_min(1e-3)
    print(f"numerics: max_abs={diff.max().item():.5f} p99_rel={rel.quantile(0.99).item():.4f}")

    # 结构化诊断:小矩阵 + 冲激 x,直接对照权重系数
    rows_d, cols_d = 8, 256
    groups_d = cols_d // 64
    pk = torch.zeros(rows_d * cols_d // 2, device="cuda", dtype=torch.uint8)
    # 行 r 的列 2c 处放 nibble=(r+c)%16
    for r in range(rows_d):
        row_bytes = torch.zeros(cols_d // 2, dtype=torch.uint8)
        for c in range(cols_d // 2):
            lo = (r + 2 * c) % 16
            hi = (r + 2 * c + 1) % 16
            row_bytes[c] = lo | (hi << 4)
        pk[r * cols_d // 2:(r + 1) * cols_d // 2] = row_bytes
    sc_d = torch.ones(rows_d, groups_d, device="cuda", dtype=torch.float16)
    x_d = torch.zeros(cols_d, device="cuda", dtype=torch.float16)
    x_d[0] = 1.0   # 冲激列 0:out[r] 应=( r%16 -8 )
    x_d2 = torch.zeros(cols_d, device="cuda", dtype=torch.float16)
    x_d2[17] = 1.0
    got = ext.marlin_int4_gemv(x_d, pk, sc_d, rows_d, cols_d, groups_d)
    got2 = ext.marlin_int4_gemv(x_d2, pk, sc_d, rows_d, cols_d, groups_d)
    torch.cuda.synchronize()
    expect = torch.tensor(
        [(r % 16) - 8 for r in range(rows_d)], device="cuda", dtype=torch.float32)
    expect2 = torch.tensor(
        [((r + 17) % 16) - 8 for r in range(rows_d)], device="cuda", dtype=torch.float32)
    print("impulse@0  got :", [round(v, 2) for v in got.tolist()])
    print("impulse@0  want:", expect.tolist())
    print("impulse@17 got :", [round(v, 2) for v in got2.tolist()])
    print("impulse@17 want:", expect2.tolist())

    # ---- Marlin repack 版(引擎内同款 kernel,经 vq_gemv 扩展) ----
    try:
        import sys
        sys.path.insert(0, "/media/tyh20/disk22/cccp-qwen-gemv-20260818/engine/CCCP-Engine")
        from cccp.fusedext import int4_gemv_marlin_fused, int4_repack_marlin_fused
        nbytes_r = rows * cols * 0.5 + rows * groups * 2 + cols * 2
        rep = int4_repack_marlin_fused(packed, cols)
        out_m = int4_gemv_marlin_fused(x, rep, scales, cols, groups)
        torch.cuda.synchronize()
        # 冲激诊断 repack 版
        pk8 = pk.clone()
        sc8 = sc_d.clone()
        x0 = torch.zeros(cols_d, device="cuda", dtype=torch.float16)
        x0[0] = 1.0
        rep8 = int4_repack_marlin_fused(pk8, cols_d)
        g8 = int4_gemv_marlin_fused(x0, rep8, sc8, cols_d, groups_d)
        torch.cuda.synchronize()
        print("repack impulse@0 got :", [round(v, 2) for v in g8.tolist()])
        x17 = torch.zeros(cols_d, device="cuda", dtype=torch.float16)
        x17[17] = 1.0
        g17 = int4_gemv_marlin_fused(x17, rep8, sc8, cols_d, groups_d)
        print("repack impulse@17 got:", [round(v, 2) for v in g17.tolist()])
        diff_m = (out_m[:ref_rows] - ref).abs()
        rel_m = diff_m / ref.abs().clamp_min(1e-3)
        print(f"repack+marlin numerics: max_abs={diff_m.max().item():.5f} "
              f"p99_rel={rel_m.quantile(0.99).item():.4f}")
        for _ in range(50):
            int4_gemv_marlin_fused(x, rep, scales, cols, groups)
        torch.cuda.synchronize()
        s2 = torch.cuda.Event(enable_timing=True); e2 = torch.cuda.Event(enable_timing=True)
        s2.record()
        for _ in range(300):
            int4_gemv_marlin_fused(x, rep, scales, cols, groups)
        e2.record(); torch.cuda.synchronize()
        ms2 = s2.elapsed_time(e2) / 300
        print(f"repack+marlin: {ms2:.4f} ms -> {nbytes_r / (ms2 / 1e3) / 1e9:.0f} GB/s")
    except Exception as exc:  # noqa: BLE001
        print("repack path unavailable:", type(exc).__name__, exc)

    # 带宽
    nbytes = rows * cols * 0.5 + rows * groups * 2 + cols * 2
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(50):
        ext.marlin_int4_gemv(x, packed, scales, rows, cols, groups)
    torch.cuda.synchronize()
    start.record()
    for _ in range(300):
        ext.marlin_int4_gemv(x, packed, scales, rows, cols, groups)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 300
    print(f"marlin proto: {ms:.4f} ms  -> {nbytes / (ms / 1e3) / 1e9:.0f} GB/s "
          f"(目标 2000 GB/s;v1 实测 ~650)")


if __name__ == "__main__":
    main()
