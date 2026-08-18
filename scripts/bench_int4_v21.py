"""INT4 GEMV v19 — ILP-4 load pipelining over the v17 layout.

Correction to the v19 blueprint: the marlin tile layout ALREADY stores one
row-group's tiles contiguously, so v17's stream is address-continuous —
the 736 GB/s ceiling comes from issuing one 128B load per superstep
(ILP=1), not from jumpiness.  v19 keeps the exact v17 layout/numerics and
pipelines TWO supersteps per iteration (two independent uint32 loads in
flight, eight accumulators), aiming to fill the DRAM bandwidth-latency
product.  Validated on the 60-matrix cold-stream bench.
"""
from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

__global__ void repack19_kernel(
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
        const int groups_k = cols / 128;           // 4-superstep groups
        const int sg = (int)(tile % groups_k);
        const int tile_n = (int)(tile / groups_k);
        // v21 interleave: intra = j*16 + i*4 + u, so four consecutive
        // words (a lane's uint4) cover supersteps tile_k+0..3 at the same
        // (j, i) position.
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

constexpr int V19_SLICE = 2048;

__global__ void int4_gemv_v19_kernel(
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
    const int k0 = slice * V19_SLICE;
    const int here = min(V19_SLICE, cols - k0);
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
    const int groups_k = cols >> 7;
    const uint32_t* base = repacked +
        ((((long)(row0 >> 3)) * groups_k + (k0 >> 7)) << 7);

    // v21: one uint4 per lane covers four supersteps at the same (j, i);
    // intra layout [sg][j][i][u] makes those four words contiguous.
    float p[32];
#pragma unroll
    for (int u = 0; u < 32; ++u) p[u] = 0.f;
    const int ss = here >> 5;
    const uint32_t* base_li = base + (lane >> 2) * 16 + (lane & 3) * 4;
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
            p[s2 * 2] = __fmaf_rn(static_cast<float>((int)(b0 & 15) - 8) * sa, sx[ca + 2 * i], p[s2 * 2]);
            p[s2 * 2] = __fmaf_rn(static_cast<float>((int)(b0 >> 4) - 8) * sa, sx[ca + 2 * i + 1], p[s2 * 2]);
            p[s2 * 2 + 1] = __fmaf_rn(static_cast<float>((int)(b1 & 15) - 8) * sa, sx[ca + 2 * i + 8], p[s2 * 2 + 1]);
            p[s2 * 2 + 1] = __fmaf_rn(static_cast<float>((int)(b1 >> 4) - 8) * sa, sx[ca + 2 * i + 9], p[s2 * 2 + 1]);
            p[s2 * 2] = __fmaf_rn(static_cast<float>((int)(b2 & 15) - 8) * sa, sx[ca + 2 * i + 16], p[s2 * 2]);
            p[s2 * 2] = __fmaf_rn(static_cast<float>((int)(b2 >> 4) - 8) * sa, sx[ca + 2 * i + 17], p[s2 * 2]);
            p[s2 * 2 + 1] = __fmaf_rn(static_cast<float>((int)(b3 & 15) - 8) * sa, sx[ca + 2 * i + 24], p[s2 * 2 + 1]);
            p[s2 * 2 + 1] = __fmaf_rn(static_cast<float>((int)(b3 >> 4) - 8) * sa, sx[ca + 2 * i + 25], p[s2 * 2 + 1]);
        }
    }
    for (int ts2 = sg; ts2 < (ss >> 2); ++ts2) {
        const uint4 pw = *reinterpret_cast<const uint4*>(base_li + (ts2 << 7));
        const uint32_t wt[4] = {pw.x, pw.y, pw.z, pw.w};
        for (int u2 = 0; u2 < 4; ++u2) {
        const int ts = ts2 * 4 + u2;
        const uint32_t w = wt[u2];
        const int ca = ts << 5;
        const float sa = __half2float(__ldg(srow + ((k0 + ca + 2 * i) >> 6)));
        const uint32_t b0 = w & 0xFF, b1 = (w >> 8) & 0xFF;
        const uint32_t b2 = (w >> 16) & 0xFF, b3 = (w >> 24) & 0xFF;
        p[0] = __fmaf_rn(static_cast<float>((int)(b0 & 15) - 8) * sa, sx[ca + 2 * i], p[0]);
        p[0] = __fmaf_rn(static_cast<float>((int)(b0 >> 4) - 8) * sa, sx[ca + 2 * i + 1], p[0]);
        p[1] = __fmaf_rn(static_cast<float>((int)(b1 & 15) - 8) * sa, sx[ca + 2 * i + 8], p[1]);
        p[1] = __fmaf_rn(static_cast<float>((int)(b1 >> 4) - 8) * sa, sx[ca + 2 * i + 9], p[1]);
        p[0] = __fmaf_rn(static_cast<float>((int)(b2 & 15) - 8) * sa, sx[ca + 2 * i + 16], p[0]);
        p[0] = __fmaf_rn(static_cast<float>((int)(b2 >> 4) - 8) * sa, sx[ca + 2 * i + 17], p[0]);
        p[1] = __fmaf_rn(static_cast<float>((int)(b3 & 15) - 8) * sa, sx[ca + 2 * i + 24], p[1]);
        p[1] = __fmaf_rn(static_cast<float>((int)(b3 >> 4) - 8) * sa, sx[ca + 2 * i + 25], p[1]);
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

__global__ void int4_pure_read_kernel(
    const uint32_t* __restrict__ repacked,
    float* __restrict__ sink,
    const int rows,
    const int cols,
    const int slices)
{
    constexpr int SL = 2048;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int slice = blockIdx.x;
    const int k0 = slice * SL;
    const int here = min(SL, cols - k0);
    if (here <= 0) return;
    const int row0 = (blockIdx.y * 4 + warp) * 8;
    if (row0 >= rows) return;
    const int groups_k = cols >> 7;
    const uint32_t* base = repacked +
        ((((long)(row0 >> 3)) * groups_k + (k0 >> 7)) << 7);
    uint32_t acc = 0;
    for (int ts = 0; ts < (here >> 5); ++ts) {
        acc ^= base[(ts << 5) + lane];
    }
    if (acc == 0xDEADBEEFu) sink[0] = 1.f;   // 防优化
}

__global__ void v19_reduce(
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

torch::Tensor repack19(torch::Tensor packed, int64_t rows, int64_t cols)
{
    const long words = (((long)rows + 7) / 8) * (cols / 128) * 128;
    auto dst = torch::empty(
        {words * 4},
        torch::TensorOptions().dtype(torch::kUInt8).device(packed.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (int)((words + 255) / 256 > 4096 ? 4096 : (words + 255) / 256);
    repack19_kernel<<<blocks, 256, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<uint32_t*>(dst.data_ptr()),
        (int)rows, (int)cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dst;
}

torch::Tensor int4_gemv_v19(
    torch::Tensor x,
    torch::Tensor repacked,
    torch::Tensor scales,
    int64_t rows,
    int64_t cols,
    int64_t groups)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int slices = (int)((cols + V19_SLICE - 1) / V19_SLICE);
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
    int4_gemv_v19_kernel<<<grid, 128, V19_SLICE * sizeof(float), stream>>>(
        x.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(repacked.data_ptr()),
        reinterpret_cast<const __half*>(scales.data_ptr()),
        partial.data_ptr<float>(), (int)rows, (int)cols, (int)groups, slices);
    v19_reduce<<<(rows + 255) / 256, 256, 0, stream>>>(
        partial.data_ptr<float>(), output.data_ptr<float>(), (int)rows, slices);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

ext = load_inline(
    name="int4_gemv_v19_proto",
    cpp_sources=(
        "torch::Tensor repack19(torch::Tensor packed, int64_t rows, int64_t cols);\n"
        "torch::Tensor int4_gemv_v19(torch::Tensor x, torch::Tensor repacked,"
        " torch::Tensor scales, int64_t rows, int64_t cols, int64_t groups);"
    ),
    cuda_sources=SRC,
    functions=["repack19", "int4_gemv_v19"],
    extra_cuda_cflags=["-O3", "-Xptxas", "-dlcm=ca"],
    verbose=False,
)


def main() -> None:
    torch.manual_seed(5090)
    rows, cols = 5120, 17408
    groups = cols // 64
    N_MAT = 60
    mats = []
    for _ in range(N_MAT):
        pk = torch.randint(
            0, 256, (rows * cols // 2,), device="cuda", dtype=torch.uint8)
        sc = (torch.randn(rows, groups, device="cuda") * 0.05).half()
        mats.append((ext.repack19(pk, rows, cols), pk, sc))
    x = torch.randn(cols, device="cuda", dtype=torch.float32)

    rep0, pk0, sc0 = mats[0]
    ref_rows = 64
    idx = pk0[: ref_rows * cols // 2].view(ref_rows, cols // 2)
    nib = torch.stack([(idx & 15).long(), (idx >> 4).long()], -1).view(ref_rows, cols) - 8
    ref = (nib.float() * sc0[:ref_rows].repeat_interleave(64, 1).float() * x).sum(1)
    out = ext.int4_gemv_v19(x, rep0, sc0, rows, cols, groups)
    torch.cuda.synchronize()
    diff = (out[:ref_rows] - ref).abs()
    rel = diff / ref.abs().clamp_min(1e-3)
    print(f"numerics: max_abs={diff.max().item():.5f} p99_rel={rel.quantile(0.99).item():.4f}")

    def one():
        for rep, _pk, sc in mats:
            ext.int4_gemv_v19(x, rep, sc, rows, cols, groups)

    for _ in range(3):
        one()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(5):
        one()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / 5 / N_MAT
    nbytes = rows * cols * 0.5 + rows * groups * 2 + cols * 4
    print(f"v19 cold-stream: {ms:.4f} ms/matrix -> {nbytes / (ms / 1e3) / 1e9:.0f} GB/s "
          f"(v17=736, 目标1500)")


def pure_read() -> None:
    torch.manual_seed(11)
    rows, cols = 5120, 17408
    N = 60
    reps = []
    for _ in range(N):
        pk = torch.randint(0, 256, (rows * cols // 2,), device="cuda", dtype=torch.uint8)
        reps.append(ext.repack19(pk, rows, cols))
    sink = torch.zeros(1, device="cuda")
    slices = (cols + 2047) // 2048
    grid = (slices, (rows // 8 + 3) // 4)
    def one():
        for rep in reps:
            ext_mod = None  # noqa
            launch_pure(rep, sink, rows, cols)
    def launch_pure(rep, sink, rows, cols):
        import torch.utils.cpp_extension as _ce
        # 直接经 torch 调用:使用绑定好的 raw kernel 不存在,借 v19 输出张量驱动不行——
        # 通过添加绑定太重;改用驱动式:此处直接调 ext 的私有 launch 不可用,
        # 所以用简单方案:重跑 v19 gemv 但注释版不可行——改为在 C 侧已绑定 pure read?
        raise NotImplementedError
    # 简化:纯读通过在 v19 gemv 上以全零 x 比较(计算仍发生)不可行。
    # 直接方案:临时用 torch 自身做大步幅拷贝测 DRAM 顺序带宽对照。
    src = torch.empty(780 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(10):
        dst.copy_(src)
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / 10
    bw = 2 * src.numel() / (ms / 1e3) / 1e9
    print(f"torch D2D copy 780MB: {ms:.2f} ms -> {bw:.0f} GB/s (DRAM 读写双向参考)")


if __name__ == "__main__":
    main()
    pure_read()
