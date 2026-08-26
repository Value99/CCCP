// CCCP fused kernels:
//   1) VQ grouped GEMV (single-kernel codebook lookup + dot), u8/u16 index.
//   2) HC sinkhorn (Hyper-Connections 4x4 doubly-stochastic iteration, one
//      kernel for softmax + 20 rounds, replacing ~85 tiny launches per call).
//   3) DSV4 decode attention core (score + masked sink-softmax + value reduce
//      + inverse RoPE, one block per head).
// Chinese docs live in cccp/fusedext.py (this file stays pure ASCII: CJK
// comments make MSVC emit C4819 and echo source lines into the build log,
// which crashes the torch builder with UnicodeDecodeError on GBK consoles).
//
// vq_gemv math: y[n, r] = sum_b dot(cb[n, idx[n, r, b], :], x[n, b*D:(b+1)*D])
//   Element-wise identical to the LUT algorithm of VQWeight.matmul_T in
//   cccp/kernels.py, but avoids materializing the [N, B, R] gather buffer.
//   cb batch broadcast supported (cbStrideN==0: all N experts share the layer
//   codebook -- saves the per-call N-fold stack copy of grouped.py).
//
// Parallelization: one warp per (n, r) row; block = 32x8 (8 rows),
// grid = (ceil(R/8), N). The x row is staged in dynamic shared memory
// (<=16KB at C=4096), then each warp walks b blocks: fetch codebook row,
// lane-strided dot, warp reduce via __shfl_down_sync. x batch broadcast is
// supported (xStrideN==0: all N experts share one input row at T=1 decode).
//
// hc_sinkhorn math (per row of mixes [N,24], hc=4; mirrors CCCP/dsv4.hc_split):
//   pre[j]  = sigmoid(m[j]*scale[0] + base[j]) + eps
//   post[j] = 2*sigmoid(m[4+j]*scale[1] + base[4+j])
//   comb[j][k] = m[8+4j+k]*scale[2] + base[8+4j+k]
//   comb = softmax(comb, dim=-1) + eps; col-normalize; then (iters-1) rounds
//   of (row-normalize, col-normalize), each divide by (sum + eps).
//   One thread per row, 4x4 kept in registers; output packed [N,24]:
//   pre | post | comb(row-major).
//
// Build: python -c "from cccp import fusedext; fusedext.prebuild()"
//        or scripts/prebuild_gpu_ops.ps1 (uses the bundled toolchain).

#include <torch/extension.h>
#include <mma.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/warp/warp_merge_sort.cuh>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <climits>
#include <condition_variable>
#include <cfloat>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__)
// CUDA uses a 32-bit warp mask; AMD wavefront builtins require a 64-bit mask.
// HIPIFY keeps CUDA's literal/variable type, so normalize it at the call site.
#define __shfl_sync(mask, value, source, ...) \
    __shfl_sync( \
        static_cast<unsigned long long>(mask), \
        value, \
        source __VA_OPT__(,) __VA_ARGS__)
#define __shfl_down_sync(mask, value, delta, ...) \
    __shfl_down_sync( \
        static_cast<unsigned long long>(mask), \
        value, \
        delta __VA_OPT__(,) __VA_ARGS__)
#endif

template <typename Kernel>
inline const void* cccp_gpu_kernel_pointer(Kernel kernel)
{
    return reinterpret_cast<const void*>(kernel);
}

template <typename Kernel, typename Attribute>
inline cudaError_t cccp_gpu_func_set_attribute(
    Kernel kernel,
    Attribute attribute,
    int value)
{
#if defined(__HIP_PLATFORM_AMD__)
    return hipFuncSetAttribute(
        cccp_gpu_kernel_pointer(kernel),
        attribute,
        value);
#else
    return cudaFuncSetAttribute(kernel, attribute, value);
#endif
}

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || \
    defined(_M_X64)
#include <immintrin.h>
#endif

#define ROWS_PER_BLOCK 32  // warps per block = output rows per block

template <typename idx_t>
__global__ void vq_gemv_kernel(
    const float* __restrict__ x,     // [Nx, C], batch-broadcast when xStrideN==0
    const idx_t* __restrict__ idx,   // [Ni, R, B], batch-broadcast when idxStrideN==0
    const float* __restrict__ cb,    // [N, K, D], batch-broadcast when cbStrideN==0
    float* __restrict__ out,         // [N, R]
    const int R, const int B, const int D,
    const long xStrideN, const long cbStrideN, const long idxStrideN)
{
    const int n = blockIdx.y;
    const int r = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    extern __shared__ float xs[];                 // [C] staged x row
    const float* xrow = x + (long)n * xStrideN;
    const int C = B * D;
    for (int i = threadIdx.y * 32 + threadIdx.x; i < C; i += 32 * ROWS_PER_BLOCK)
        xs[i] = xrow[i];
    __syncthreads();
    if (r >= R) return;

    const idx_t* irow = idx + (long)n * idxStrideN + (long)r * B;
    const float* cbn = cb + (long)n * cbStrideN;
    // lane-parallel over b blocks: each lane finishes its own D-dim dot locally
    // (D=4/8, no cross-lane work), one single warp reduce at the end (the old
    // version reduced per b-block = 5B shuffles per row, with most lanes idle
    // when D<32).
    float acc = 0.f;
    for (int b = threadIdx.x; b < B; b += 32) {
        const float* crow = cbn + (long)irow[b] * D;   // codebook row (D floats)
        const float* xb = xs + b * D;
        float part = 0.f;
        #pragma unroll 8
        for (int i = 0; i < D; ++i)
            part += crow[i] * xb[i];
        acc += part;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (threadIdx.x == 0)
        out[(long)n * R + r] = acc;
}

torch::Tensor vq_gemv(torch::Tensor x, torch::Tensor idx, torch::Tensor cb) {
    TORCH_CHECK(x.is_cuda() && idx.is_cuda() && cb.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat && cb.scalar_type() == at::kFloat,
                "x/cb must be float32");
    TORCH_CHECK(idx.scalar_type() == at::kByte || idx.scalar_type() == at::kUInt16,
                "idx must be uint8 or uint16");
    TORCH_CHECK(x.is_contiguous() && idx.is_contiguous() && cb.is_contiguous(),
                "tensors must be contiguous");
    const long R = idx.size(1), B = idx.size(2);
    const long D = cb.size(2);
    // batch N = the non-broadcast side of x/idx: (x[N],idx[N]) | (x[1],idx[N]) | (x[N],idx[1])
    const long N = x.size(0) > idx.size(0) ? x.size(0) : idx.size(0);
    TORCH_CHECK(x.size(0) == 1 || x.size(0) == N, "x batch must be 1 or N");
    TORCH_CHECK(idx.size(0) == 1 || idx.size(0) == N, "idx batch must be 1 or N");
    TORCH_CHECK(cb.size(0) == N || cb.size(0) == 1, "cb batch must be 1 or N");
    TORCH_CHECK(x.size(1) == B * D, "x cols must equal B*D");

    auto out = torch::empty({N, R}, x.options());
    const long xStrideN = x.size(0) == 1 ? 0 : x.stride(0);
    const long cbStrideN = cb.size(0) == 1 ? 0 : cb.stride(0);
    const long idxStrideN = idx.size(0) == 1 ? 0 : (long)R * B;
    dim3 block(32, ROWS_PER_BLOCK);
    dim3 grid((unsigned)((R + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK), (unsigned)N);
    const size_t smem = (size_t)B * D * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (idx.scalar_type() == at::kByte) {
        vq_gemv_kernel<uint8_t><<<grid, block, smem, stream>>>(
            x.data_ptr<float>(), idx.data_ptr<uint8_t>(), cb.data_ptr<float>(),
            out.data_ptr<float>(), (int)R, (int)B, (int)D, xStrideN, cbStrideN, idxStrideN);
    } else {
        vq_gemv_kernel<uint16_t><<<grid, block, smem, stream>>>(
            x.data_ptr<float>(), (const uint16_t*)idx.data_ptr(), cb.data_ptr<float>(),
            out.data_ptr<float>(), (int)R, (int)B, (int)D, xStrideN, cbStrideN, idxStrideN);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ---- Kimi KDA one-token recurrent update (V-first FP32 state) ----

template <typename weight_t>
__device__ __forceinline__ float kimi_conv_weight(
    const weight_t* value)
{
    return static_cast<float>(*value);
}

template <>
__device__ __forceinline__ float kimi_conv_weight(
    const __nv_bfloat16* value)
{
    return __bfloat162float(*value);
}

template <typename weight_t>
__global__ void kimi_short_conv3_kernel(
    __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ query_state,
    __nv_bfloat16* __restrict__ key_state,
    __nv_bfloat16* __restrict__ value_state,
    const weight_t* __restrict__ query_weight,
    const weight_t* __restrict__ key_weight,
    const weight_t* __restrict__ value_weight,
    const int channels,
    const int history)
{
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    const int stream = blockIdx.y;
    if (channel >= channels || stream >= 3) return;
    __nv_bfloat16* input = stream == 0 ? query : (
        stream == 1 ? key : value);
    __nv_bfloat16* state = stream == 0 ? query_state : (
        stream == 1 ? key_state : value_state);
    const weight_t* weight = stream == 0 ? query_weight : (
        stream == 1 ? key_weight : value_weight);
    __nv_bfloat16* state_row =
        state + static_cast<long>(channel) * history;
    const weight_t* weight_row =
        weight + static_cast<long>(channel) * (history + 1);
    float result = 0.0f;
    for (int item = 0; item < history; ++item) {
        result = fmaf(
            __bfloat162float(state_row[item]),
            kimi_conv_weight(weight_row + item),
            result);
    }
    const float current = __bfloat162float(input[channel]);
    result = fmaf(
        current,
        kimi_conv_weight(weight_row + history),
        result);
    for (int item = 0; item + 1 < history; ++item)
        state_row[item] = state_row[item + 1];
    if (history > 0)
        state_row[history - 1] = input[channel];
    const float silu = result / (1.0f + expf(-result));
    input[channel] = __float2bfloat16_rn(silu);
}

bool kimi_short_conv3(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor query_state,
    torch::Tensor key_state,
    torch::Tensor value_state,
    torch::Tensor query_weight,
    torch::Tensor key_weight,
    torch::Tensor value_weight)
{
    TORCH_CHECK(
        query.is_cuda() && key.is_cuda() && value.is_cuda() &&
        query_state.is_cuda() && key_state.is_cuda() &&
        value_state.is_cuda() && query_weight.is_cuda() &&
        key_weight.is_cuda() && value_weight.is_cuda(),
        "Kimi short convolution tensors must be CUDA");
    TORCH_CHECK(
        query.scalar_type() == at::kBFloat16 &&
        key.scalar_type() == at::kBFloat16 &&
        value.scalar_type() == at::kBFloat16 &&
        query_state.scalar_type() == at::kBFloat16 &&
        key_state.scalar_type() == at::kBFloat16 &&
        value_state.scalar_type() == at::kBFloat16 &&
        (
            query_weight.scalar_type() == at::kBFloat16 ||
            query_weight.scalar_type() == at::kFloat
        ) &&
        key_weight.scalar_type() == query_weight.scalar_type() &&
        value_weight.scalar_type() == query_weight.scalar_type(),
        "Kimi short convolution requires BF16 state and matching "
        "BF16/FP32 weights");
    TORCH_CHECK(
        query.is_contiguous() && key.is_contiguous() &&
        value.is_contiguous() && query_state.is_contiguous() &&
        key_state.is_contiguous() && value_state.is_contiguous() &&
        query_weight.is_contiguous() && key_weight.is_contiguous() &&
        value_weight.is_contiguous(),
        "Kimi short convolution tensors must be contiguous");
    TORCH_CHECK(
        query.dim() == 1 && key.sizes() == query.sizes() &&
        value.sizes() == query.sizes() &&
        query_state.dim() == 2 &&
        key_state.sizes() == query_state.sizes() &&
        value_state.sizes() == query_state.sizes() &&
        query_state.size(0) == query.size(0),
        "Kimi short convolution input/state shapes do not match");
    const int channels = static_cast<int>(query.numel());
    const int history = static_cast<int>(query_state.size(1));
    const long weight_items =
        static_cast<long>(channels) * (history + 1);
    TORCH_CHECK(
        query_weight.numel() == weight_items &&
        key_weight.numel() == weight_items &&
        value_weight.numel() == weight_items,
        "Kimi short convolution weight shapes do not match");
    const int device = query.get_device();
    TORCH_CHECK(
        key.get_device() == device && value.get_device() == device &&
        query_state.get_device() == device &&
        key_state.get_device() == device &&
        value_state.get_device() == device &&
        query_weight.get_device() == device &&
        key_weight.get_device() == device &&
        value_weight.get_device() == device,
        "Kimi short convolution tensors must share one device");
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid((channels + 255) / 256, 3);
    if (query_weight.scalar_type() == at::kBFloat16) {
        kimi_short_conv3_kernel<<<grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(
                query.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                key.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                value.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                query_state.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                key_state.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                value_state.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                query_weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                key_weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                value_weight.data_ptr<at::BFloat16>()),
            channels,
            history);
    } else {
        kimi_short_conv3_kernel<<<grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(
                query.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                key.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                value.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                query_state.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                key_state.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                value_state.data_ptr<at::BFloat16>()),
            query_weight.data_ptr<float>(),
            key_weight.data_ptr<float>(),
            value_weight.data_ptr<float>(),
            channels,
            history);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

// Qwen3.5 single-stream cached depthwise convolution. The public
// Transformers fallback rolls the state and dispatches several pointwise
// operators per layer; one thread owns one channel here.
template <typename weight_t>
__global__ void qwen35_conv1d_update_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ state,
    const weight_t* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    const int channels,
    const int width)
{
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    __nv_bfloat16* state_row = state + (long)channel * width;
    const weight_t* weight_row = weight + (long)channel * width;
    const float current = __bfloat162float(input[channel]);
    float sum = 0.f;
    for (int item = 0; item + 1 < width; ++item) {
        const __nv_bfloat16 previous = state_row[item + 1];
        sum = fmaf(
            __bfloat162float(previous),
            kimi_conv_weight(weight_row + item),
            sum);
        state_row[item] = previous;
    }
    sum = fmaf(
        current,
        kimi_conv_weight(weight_row + width - 1),
        sum);
    state_row[width - 1] = input[channel];
    output[channel] = __float2bfloat16_rn(
        sum / (1.f + expf(-sum)));
}

torch::Tensor qwen35_conv1d_update(
    torch::Tensor input,
    torch::Tensor state,
    torch::Tensor weight,
    torch::Tensor output)
{
    TORCH_CHECK(
        input.is_cuda() && state.is_cuda() && weight.is_cuda()
            && output.is_cuda(),
        "Qwen3.5 convolution tensors must be CUDA");
    TORCH_CHECK(
        input.scalar_type() == at::kBFloat16
            && state.scalar_type() == at::kBFloat16
            && output.scalar_type() == at::kBFloat16
            && (weight.scalar_type() == at::kBFloat16
                || weight.scalar_type() == at::kFloat),
        "Qwen3.5 convolution requires BF16 input/state/output and "
        "BF16/FP32 weights");
    TORCH_CHECK(
        input.is_contiguous() && state.is_contiguous()
            && weight.is_contiguous() && output.is_contiguous(),
        "Qwen3.5 convolution tensors must be contiguous");
    TORCH_CHECK(
        input.dim() == 3 && input.size(0) == 1 && input.size(2) == 1
            && state.dim() == 3 && state.size(0) == 1
            && state.size(1) == input.size(1)
            && output.sizes() == input.sizes(),
        "Qwen3.5 convolution shape mismatch");
    const int channels = static_cast<int>(input.size(1));
    const int width = static_cast<int>(state.size(2));
    TORCH_CHECK(
        width > 0 && width <= 16
            && weight.numel() == (long)channels * width,
        "Qwen3.5 convolution weight width mismatch");
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (channels + 255) / 256;
    if (weight.scalar_type() == at::kBFloat16) {
        qwen35_conv1d_update_kernel<<<blocks, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                input.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                state.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            channels,
            width);
    } else {
        qwen35_conv1d_update_kernel<<<blocks, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                input.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                state.data_ptr<at::BFloat16>()),
            weight.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            channels,
            width);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__device__ __forceinline__ float qwen35_optional_scalar(
    const float* fp32,
    const __nv_bfloat16* bf16,
    const int index)
{
    return fp32 != nullptr ? fp32[index] : __bfloat162float(bf16[index]);
}

// One CTA owns one recurrent head. Threads map to adjacent value columns,
// so every state row is read and written coalesced despite Qwen's [H,K,V]
// state layout.
__global__ void qwen35_delta_recurrent_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate_f,
    const __nv_bfloat16* __restrict__ gate_b,
    const float* __restrict__ beta_f,
    const __nv_bfloat16* __restrict__ beta_b,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    const int heads,
    const int key_dim,
    const int value_dim)
{
    const int head = blockIdx.x;
    const int item = threadIdx.x;
    if (head >= heads) return;
    extern __shared__ float shared[];
    float* q_norm = shared;
    float* k_norm = q_norm + blockDim.x;
    float q = 0.f;
    float k = 0.f;
    if (item < key_dim) {
        const long offset = (long)head * key_dim + item;
        q = __bfloat162float(query[offset]);
        k = __bfloat162float(key[offset]);
    }
    q_norm[item] = q * q;
    k_norm[item] = k * k;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (item < stride) {
            q_norm[item] += q_norm[item + stride];
            k_norm[item] += k_norm[item + stride];
        }
        __syncthreads();
    }
    const float q_scale = rsqrtf(fmaxf(q_norm[0], 1.0e-12f));
    const float k_scale = rsqrtf(fmaxf(k_norm[0], 1.0e-12f));
    if (item < key_dim) {
        q_norm[item] = q * q_scale;
        k_norm[item] = k * k_scale;
    }
    __syncthreads();

    if (item >= value_dim) return;
    const float decay = expf(qwen35_optional_scalar(
        gate_f, gate_b, head));
    const float beta = qwen35_optional_scalar(
        beta_f, beta_b, head);
    float prediction = 0.f;
    const long state_head = (long)head * key_dim * value_dim;
    for (int row = 0; row < key_dim; ++row) {
        const long offset = state_head + (long)row * value_dim + item;
        const float current = state[offset] * decay;
        state[offset] = current;
        prediction = fmaf(current, k_norm[row], prediction);
    }
    prediction = (
        __bfloat162float(value[(long)head * value_dim + item])
        - prediction) * beta;
    float result = 0.f;
    for (int row = 0; row < key_dim; ++row) {
        const long offset = state_head + (long)row * value_dim + item;
        const float current = fmaf(k_norm[row], prediction, state[offset]);
        state[offset] = current;
        result = fmaf(current, q_norm[row], result);
    }
    output[(long)head * value_dim + item] = __float2bfloat16_rn(
        result * rsqrtf((float)key_dim));
}

// Qwen3.5 text layers use 48 heads with K=V=128.  One CTA per head leaves
// more than half of a Hopper GPU idle and the generic kernel writes the
// decayed state only to read and write it again during the rank-one update.
// Split each head into two 64-column CTAs and commit the state once.  The two
// CTAs own disjoint value columns, so no atomics or cross-CTA barrier is
// required and the recurrent update remains exact for each state element.
__global__ void qwen35_delta_recurrent_128_split2_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate_f,
    const __nv_bfloat16* __restrict__ gate_b,
    const float* __restrict__ beta_f,
    const __nv_bfloat16* __restrict__ beta_b,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    const int heads)
{
    constexpr int width = 128;
    constexpr int columns_per_block = 64;
    const int head = blockIdx.x >> 1;
    const int column =
        (blockIdx.x & 1) * columns_per_block + threadIdx.x;
    const int lane = threadIdx.x;
    if (head >= heads) return;

    extern __shared__ float shared[];
    float* q_norm = shared;
    float* k_norm = q_norm + width;
    const long qk_head = (long)head * width;
    const float q0 = __bfloat162float(query[qk_head + lane]);
    const float q1 = __bfloat162float(
        query[qk_head + lane + columns_per_block]);
    const float k0 = __bfloat162float(key[qk_head + lane]);
    const float k1 = __bfloat162float(
        key[qk_head + lane + columns_per_block]);
    q_norm[lane] = fmaf(q0, q0, q1 * q1);
    k_norm[lane] = fmaf(k0, k0, k1 * k1);
    __syncthreads();
    for (int stride = columns_per_block / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            q_norm[lane] += q_norm[lane + stride];
            k_norm[lane] += k_norm[lane + stride];
        }
        __syncthreads();
    }
    const float q_scale = rsqrtf(fmaxf(q_norm[0], 1.0e-12f));
    const float k_scale = rsqrtf(fmaxf(k_norm[0], 1.0e-12f));
    q_norm[lane] = q0 * q_scale;
    q_norm[lane + columns_per_block] = q1 * q_scale;
    k_norm[lane] = k0 * k_scale;
    k_norm[lane + columns_per_block] = k1 * k_scale;
    __syncthreads();

    const float decay = expf(qwen35_optional_scalar(
        gate_f, gate_b, head));
    const float beta = qwen35_optional_scalar(
        beta_f, beta_b, head);
    const long state_head = (long)head * width * width;
    float prediction = 0.f;
    for (int row = 0; row < width; ++row) {
        const long offset = state_head + (long)row * width + column;
        const float decayed = state[offset] * decay;
        prediction = fmaf(decayed, k_norm[row], prediction);
    }
    prediction = (
        __bfloat162float(value[qk_head + column]) - prediction) * beta;
    float result = 0.f;
    for (int row = 0; row < width; ++row) {
        const long offset = state_head + (long)row * width + column;
        const float current = fmaf(
            k_norm[row], prediction, state[offset] * decay);
        state[offset] = current;
        result = fmaf(current, q_norm[row], result);
    }
    output[qk_head + column] = __float2bfloat16_rn(
        result * 0.08838834764831845f);
}

torch::Tensor qwen35_delta_recurrent(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor gate,
    torch::Tensor beta,
    torch::Tensor state,
    torch::Tensor output)
{
    TORCH_CHECK(
        query.is_cuda() && key.is_cuda() && value.is_cuda()
            && gate.is_cuda() && beta.is_cuda() && state.is_cuda()
            && output.is_cuda(),
        "Qwen3.5 delta tensors must be CUDA");
    TORCH_CHECK(
        query.scalar_type() == at::kBFloat16
            && key.scalar_type() == at::kBFloat16
            && value.scalar_type() == at::kBFloat16
            && output.scalar_type() == at::kBFloat16
            && state.scalar_type() == at::kFloat
            && (gate.scalar_type() == at::kFloat
                || gate.scalar_type() == at::kBFloat16)
            && (beta.scalar_type() == at::kFloat
                || beta.scalar_type() == at::kBFloat16),
        "Qwen3.5 delta dtype mismatch");
    TORCH_CHECK(
        query.is_contiguous() && key.is_contiguous()
            && value.is_contiguous() && gate.is_contiguous()
            && beta.is_contiguous() && state.is_contiguous()
            && output.is_contiguous(),
        "Qwen3.5 delta tensors must be contiguous");
    TORCH_CHECK(
        query.dim() == 2 && key.sizes() == query.sizes()
            && value.dim() == 2 && value.size(0) == query.size(0)
            && output.sizes() == value.sizes(),
        "Qwen3.5 delta activation shape mismatch");
    const int heads = static_cast<int>(query.size(0));
    const int key_dim = static_cast<int>(query.size(1));
    const int value_dim = static_cast<int>(value.size(1));
    TORCH_CHECK(
        key_dim > 0 && key_dim <= 256
            && value_dim > 0 && value_dim <= 256
            && state.sizes() == torch::IntArrayRef(
                {heads, key_dim, value_dim})
            && gate.numel() == heads && beta.numel() == heads,
        "Qwen3.5 delta state or scalar shape mismatch");
    int threads = 1;
    while (threads < key_dim || threads < value_dim) threads <<= 1;
    TORCH_CHECK(threads <= 256, "Qwen3.5 delta width is unsupported");
    const float* gate_f = gate.scalar_type() == at::kFloat
        ? gate.data_ptr<float>() : nullptr;
    const __nv_bfloat16* gate_b = gate.scalar_type() == at::kBFloat16
        ? reinterpret_cast<const __nv_bfloat16*>(
            gate.data_ptr<at::BFloat16>()) : nullptr;
    const float* beta_f = beta.scalar_type() == at::kFloat
        ? beta.data_ptr<float>() : nullptr;
    const __nv_bfloat16* beta_b = beta.scalar_type() == at::kBFloat16
        ? reinterpret_cast<const __nv_bfloat16*>(
            beta.data_ptr<at::BFloat16>()) : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (key_dim == 128 && value_dim == 128 && threads == 128) {
        qwen35_delta_recurrent_128_split2_kernel<<<
            heads * 2,
            64,
            2LL * 128 * sizeof(float),
            stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    query.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    key.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    value.data_ptr<at::BFloat16>()),
                gate_f,
                gate_b,
                beta_f,
                beta_b,
                state.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(
                    output.data_ptr<at::BFloat16>()),
                heads);
    } else {
        qwen35_delta_recurrent_kernel<<<
            heads,
            threads,
            2LL * threads * sizeof(float),
            stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                query.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                key.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                value.data_ptr<at::BFloat16>()),
            gate_f,
            gate_b,
            beta_f,
            beta_b,
            state.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            heads,
            key_dim,
            value_dim);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void qwen35_delta_recurrent_batch_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate_f,
    const __nv_bfloat16* __restrict__ gate_b,
    const float* __restrict__ beta_f,
    const __nv_bfloat16* __restrict__ beta_b,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ checkpoints,
    const int tokens,
    const int heads,
    const int key_dim,
    const int value_dim)
{
    const int head = blockIdx.x;
    const int item = threadIdx.x;
    if (head >= heads) return;
    extern __shared__ float shared[];
    float* q_norm = shared;
    float* k_norm = q_norm + blockDim.x;
    const long state_head = (long)head * key_dim * value_dim;
    for (int token = 0; token < tokens; ++token) {
        float q = 0.f;
        float k = 0.f;
        if (item < key_dim) {
            const long qk_offset =
                ((long)token * heads + head) * key_dim + item;
            q = __bfloat162float(query[qk_offset]);
            k = __bfloat162float(key[qk_offset]);
        }
        q_norm[item] = q * q;
        k_norm[item] = k * k;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (item < stride) {
                q_norm[item] += q_norm[item + stride];
                k_norm[item] += k_norm[item + stride];
            }
            __syncthreads();
        }
        const float q_scale = rsqrtf(fmaxf(q_norm[0], 1.0e-12f));
        const float k_scale = rsqrtf(fmaxf(k_norm[0], 1.0e-12f));
        if (item < key_dim) {
            q_norm[item] = q * q_scale;
            k_norm[item] = k * k_scale;
        }
        __syncthreads();
        if (item < value_dim) {
            const int scalar_index = token * heads + head;
            const float decay = expf(qwen35_optional_scalar(
                gate_f, gate_b, scalar_index));
            const float beta = qwen35_optional_scalar(
                beta_f, beta_b, scalar_index);
            float prediction = 0.f;
            for (int row = 0; row < key_dim; ++row) {
                const long offset =
                    state_head + (long)row * value_dim + item;
                const float current = state[offset] * decay;
                state[offset] = current;
                prediction = fmaf(current, k_norm[row], prediction);
            }
            const long value_offset =
                ((long)token * heads + head) * value_dim + item;
            prediction = (
                __bfloat162float(value[value_offset]) - prediction) * beta;
            float result = 0.f;
            for (int row = 0; row < key_dim; ++row) {
                const long offset =
                    state_head + (long)row * value_dim + item;
                const float current = fmaf(
                    k_norm[row], prediction, state[offset]);
                state[offset] = current;
                result = fmaf(current, q_norm[row], result);
            }
            output[value_offset] = __float2bfloat16_rn(
                result * rsqrtf((float)key_dim));
        }
        if (checkpoints != nullptr && item < value_dim) {
            const long checkpoint_head =
                ((long)token * heads + head) * key_dim * value_dim;
            for (int row = 0; row < key_dim; ++row) {
                const long state_offset =
                    state_head + (long)row * value_dim + item;
                checkpoints[
                    checkpoint_head + (long)row * value_dim + item
                ] = state[state_offset];
            }
        }
        __syncthreads();
    }
}

torch::Tensor qwen35_delta_recurrent_batch(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor gate,
    torch::Tensor beta,
    torch::Tensor state,
    torch::Tensor output)
{
    TORCH_CHECK(
        query.is_cuda() && key.is_cuda() && value.is_cuda()
            && gate.is_cuda() && beta.is_cuda() && state.is_cuda()
            && output.is_cuda(),
        "Qwen3.5 batched delta tensors must be CUDA");
    TORCH_CHECK(
        query.scalar_type() == at::kBFloat16
            && key.scalar_type() == at::kBFloat16
            && value.scalar_type() == at::kBFloat16
            && output.scalar_type() == at::kBFloat16
            && state.scalar_type() == at::kFloat
            && (gate.scalar_type() == at::kFloat
                || gate.scalar_type() == at::kBFloat16)
            && (beta.scalar_type() == at::kFloat
                || beta.scalar_type() == at::kBFloat16),
        "Qwen3.5 batched delta dtype mismatch");
    TORCH_CHECK(
        query.is_contiguous() && key.is_contiguous()
            && value.is_contiguous() && gate.is_contiguous()
            && beta.is_contiguous() && state.is_contiguous()
            && output.is_contiguous(),
        "Qwen3.5 batched delta tensors must be contiguous");
    TORCH_CHECK(
        query.dim() == 3 && key.sizes() == query.sizes()
            && value.dim() == 3
            && value.size(0) == query.size(0)
            && value.size(1) == query.size(1)
            && output.sizes() == value.sizes(),
        "Qwen3.5 batched delta activation shape mismatch");
    const int tokens = static_cast<int>(query.size(0));
    const int heads = static_cast<int>(query.size(1));
    const int key_dim = static_cast<int>(query.size(2));
    const int value_dim = static_cast<int>(value.size(2));
    TORCH_CHECK(
        tokens > 0 && key_dim > 0 && key_dim <= 256
            && value_dim > 0 && value_dim <= 256
            && state.sizes() == torch::IntArrayRef(
                {heads, key_dim, value_dim})
            && gate.numel() == (long)tokens * heads
            && beta.numel() == (long)tokens * heads,
        "Qwen3.5 batched delta state or scalar shape mismatch");
    int threads = 1;
    while (threads < key_dim || threads < value_dim) threads <<= 1;
    const float* gate_f = gate.scalar_type() == at::kFloat
        ? gate.data_ptr<float>() : nullptr;
    const __nv_bfloat16* gate_b = gate.scalar_type() == at::kBFloat16
        ? reinterpret_cast<const __nv_bfloat16*>(
            gate.data_ptr<at::BFloat16>()) : nullptr;
    const float* beta_f = beta.scalar_type() == at::kFloat
        ? beta.data_ptr<float>() : nullptr;
    const __nv_bfloat16* beta_b = beta.scalar_type() == at::kBFloat16
        ? reinterpret_cast<const __nv_bfloat16*>(
            beta.data_ptr<at::BFloat16>()) : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream();
    qwen35_delta_recurrent_batch_kernel<<<
        heads,
        threads,
        2LL * threads * sizeof(float),
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                query.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                key.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                value.data_ptr<at::BFloat16>()),
            gate_f,
            gate_b,
            beta_f,
            beta_b,
            state.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            nullptr,
            tokens,
            heads,
            key_dim,
            value_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor qwen35_delta_recurrent_batch_checkpoint(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor gate,
    torch::Tensor beta,
    torch::Tensor state,
    torch::Tensor output,
    torch::Tensor checkpoints)
{
    TORCH_CHECK(
        query.is_cuda() && key.is_cuda() && value.is_cuda()
            && gate.is_cuda() && beta.is_cuda() && state.is_cuda()
            && output.is_cuda() && checkpoints.is_cuda(),
        "Qwen3.5 checkpoint tensors must be CUDA");
    TORCH_CHECK(
        query.scalar_type() == at::kBFloat16
            && key.scalar_type() == at::kBFloat16
            && value.scalar_type() == at::kBFloat16
            && output.scalar_type() == at::kBFloat16
            && state.scalar_type() == at::kFloat
            && checkpoints.scalar_type() == at::kFloat
            && (gate.scalar_type() == at::kFloat
                || gate.scalar_type() == at::kBFloat16)
            && (beta.scalar_type() == at::kFloat
                || beta.scalar_type() == at::kBFloat16),
        "Qwen3.5 checkpoint dtype mismatch");
    TORCH_CHECK(
        query.is_contiguous() && key.is_contiguous()
            && value.is_contiguous() && gate.is_contiguous()
            && beta.is_contiguous() && state.is_contiguous()
            && output.is_contiguous() && checkpoints.is_contiguous(),
        "Qwen3.5 checkpoint tensors must be contiguous");
    TORCH_CHECK(
        query.dim() == 3 && key.sizes() == query.sizes()
            && value.dim() == 3
            && value.size(0) == query.size(0)
            && value.size(1) == query.size(1)
            && output.sizes() == value.sizes(),
        "Qwen3.5 checkpoint activation shape mismatch");
    const int tokens = static_cast<int>(query.size(0));
    const int heads = static_cast<int>(query.size(1));
    const int key_dim = static_cast<int>(query.size(2));
    const int value_dim = static_cast<int>(value.size(2));
    TORCH_CHECK(
        tokens > 0 && key_dim > 0 && key_dim <= 256
            && value_dim > 0 && value_dim <= 256
            && state.sizes() == torch::IntArrayRef(
                {heads, key_dim, value_dim})
            && checkpoints.sizes() == torch::IntArrayRef(
                {tokens, heads, key_dim, value_dim})
            && gate.numel() == (long)tokens * heads
            && beta.numel() == (long)tokens * heads,
        "Qwen3.5 checkpoint state shape mismatch");
    int threads = 1;
    while (threads < key_dim || threads < value_dim) threads <<= 1;
    const float* gate_f = gate.scalar_type() == at::kFloat
        ? gate.data_ptr<float>() : nullptr;
    const __nv_bfloat16* gate_b = gate.scalar_type() == at::kBFloat16
        ? reinterpret_cast<const __nv_bfloat16*>(
            gate.data_ptr<at::BFloat16>()) : nullptr;
    const float* beta_f = beta.scalar_type() == at::kFloat
        ? beta.data_ptr<float>() : nullptr;
    const __nv_bfloat16* beta_b = beta.scalar_type() == at::kBFloat16
        ? reinterpret_cast<const __nv_bfloat16*>(
            beta.data_ptr<at::BFloat16>()) : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream();
    qwen35_delta_recurrent_batch_kernel<<<
        heads,
        threads,
        2LL * threads * sizeof(float),
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                query.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                key.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                value.data_ptr<at::BFloat16>()),
            gate_f,
            gate_b,
            beta_f,
            beta_b,
            state.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            checkpoints.data_ptr<float>(),
            tokens,
            heads,
            key_dim,
            value_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void kimi_kda_prepare_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ gate,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ query_norm,
    float* __restrict__ key_norm,
    float* __restrict__ decay,
    const int heads,
    const int key_dim,
    const float lower_bound)
{
    const int head = blockIdx.x;
    const int item = threadIdx.x;
    if (head >= heads) return;
    extern __shared__ float shared[];
    float* q_square = shared;
    float* k_square = shared + blockDim.x;
    float q = 0.f;
    float k = 0.f;
    if (item < key_dim) {
        const long offset = (long)head * key_dim + item;
        q = __bfloat162float(query[offset]);
        k = __bfloat162float(key[offset]);
    }
    q_square[item] = q * q;
    k_square[item] = k * k;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (item < stride) {
            q_square[item] += q_square[item + stride];
            k_square[item] += k_square[item + stride];
        }
        __syncthreads();
    }
    if (item < key_dim) {
        const long offset = (long)head * key_dim + item;
        const float q_scale = rsqrtf(q_square[0] + 1e-6f);
        const float k_scale = rsqrtf(k_square[0] + 1e-6f);
        query_norm[offset] = q * q_scale;
        key_norm[offset] = k * k_scale;
        const float a = expf(a_log[head]);
        const float raw = a * (
            __bfloat162float(gate[offset]) + dt_bias[offset]);
        const float sigmoid = 1.f / (1.f + expf(-raw));
        decay[offset] = expf(lower_bound * sigmoid);
    }
}

constexpr int KIMI_KDA_VALUES_PER_BLOCK = 4;

__global__ void kimi_kda_update_kernel(
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ query_norm,
    const float* __restrict__ key_norm,
    const float* __restrict__ decay,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    const int heads,
    const int key_dim,
    const int value_dim)
{
    const int head = blockIdx.y;
    const int value_start = blockIdx.x * KIMI_KDA_VALUES_PER_BLOCK;
    const int item = threadIdx.x;
    if (head >= heads) return;
    extern __shared__ float shared[];
    float* prediction = shared;
    float* old_output =
        prediction + KIMI_KDA_VALUES_PER_BLOCK * blockDim.x;
    float* key_query =
        old_output + KIMI_KDA_VALUES_PER_BLOCK * blockDim.x;
    float* deltas = key_query + blockDim.x;

    const long qk_offset = (long)head * key_dim + item;
    const float q = item < key_dim ? query_norm[qk_offset] : 0.f;
    const float k = item < key_dim ? key_norm[qk_offset] : 0.f;
    const float d = item < key_dim ? decay[qk_offset] : 0.f;
    key_query[item] = q * k;
    #pragma unroll
    for (int row = 0; row < KIMI_KDA_VALUES_PER_BLOCK; ++row) {
        const int value_index = value_start + row;
        float current = 0.f;
        if (value_index < value_dim && item < key_dim) {
            const long state_offset =
                ((long)head * value_dim + value_index) * key_dim + item;
            current = state[state_offset] * d;
            state[state_offset] = current;
        }
        prediction[row * blockDim.x + item] = current * k;
        old_output[row * blockDim.x + item] = current * q;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (item < stride) {
            key_query[item] += key_query[item + stride];
            #pragma unroll
            for (int row = 0; row < KIMI_KDA_VALUES_PER_BLOCK; ++row) {
                const int base = row * blockDim.x + item;
                prediction[base] += prediction[base + stride];
                old_output[base] += old_output[base + stride];
            }
        }
        __syncthreads();
    }

    if (item < KIMI_KDA_VALUES_PER_BLOCK) {
        const int value_index = value_start + item;
        float delta = 0.f;
        if (value_index < value_dim) {
            const float beta_value = 1.f / (1.f + expf(-beta[head]));
            const float source = __bfloat162float(
                value[(long)head * value_dim + value_index]);
            delta = (source - prediction[item * blockDim.x]) * beta_value;
        }
        deltas[item] = delta;
    }
    __syncthreads();

    #pragma unroll
    for (int row = 0; row < KIMI_KDA_VALUES_PER_BLOCK; ++row) {
        const int value_index = value_start + row;
        if (value_index < value_dim && item < key_dim) {
            const long state_offset =
                ((long)head * value_dim + value_index) * key_dim + item;
            state[state_offset] += deltas[row] * k;
        }
        if (item == 0 && value_index < value_dim) {
            const float result = (
                old_output[row * blockDim.x]
                + deltas[row] * key_query[0]
            ) * rsqrtf((float)key_dim);
            output[(long)head * value_dim + value_index] =
                __float2bfloat16_rn(result);
        }
    }
}

torch::Tensor kimi_kda_recurrent(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor gate,
    torch::Tensor beta,
    torch::Tensor a_log,
    torch::Tensor dt_bias,
    torch::Tensor state,
    torch::Tensor workspace,
    torch::Tensor output,
    double lower_bound)
{
    TORCH_CHECK(
        query.is_cuda() && key.is_cuda() && value.is_cuda() &&
        gate.is_cuda() && beta.is_cuda() && a_log.is_cuda() &&
        dt_bias.is_cuda() && state.is_cuda() && workspace.is_cuda() &&
        output.is_cuda(),
        "Kimi KDA tensors must be CUDA");
    TORCH_CHECK(
        query.scalar_type() == at::kBFloat16 &&
        key.scalar_type() == at::kBFloat16 &&
        value.scalar_type() == at::kBFloat16 &&
        gate.scalar_type() == at::kBFloat16 &&
        output.scalar_type() == at::kBFloat16,
        "Kimi KDA activations must be BF16");
    TORCH_CHECK(
        beta.scalar_type() == at::kFloat &&
        a_log.scalar_type() == at::kFloat &&
        dt_bias.scalar_type() == at::kFloat &&
        state.scalar_type() == at::kFloat &&
        workspace.scalar_type() == at::kFloat,
        "Kimi KDA parameters/state/workspace must be FP32");
    TORCH_CHECK(
        query.is_contiguous() && key.is_contiguous() &&
        value.is_contiguous() && gate.is_contiguous() &&
        beta.is_contiguous() && a_log.is_contiguous() &&
        dt_bias.is_contiguous() && state.is_contiguous() &&
        workspace.is_contiguous() && output.is_contiguous(),
        "Kimi KDA tensors must be contiguous");
    TORCH_CHECK(
        query.dim() == 2 && key.sizes() == query.sizes() &&
        gate.sizes() == query.sizes() && value.dim() == 2,
        "Kimi KDA q/k/g must be [H,K] and v must be [H,V]");
    const int heads = (int)query.size(0);
    const int key_dim = (int)query.size(1);
    const int value_dim = (int)value.size(1);
    TORCH_CHECK(
        value.size(0) == heads &&
        state.dim() == 3 &&
        state.size(0) == heads &&
        state.size(1) == value_dim &&
        state.size(2) == key_dim &&
        output.sizes() == value.sizes(),
        "Kimi KDA value/state/output shape mismatch");
    TORCH_CHECK(
        key_dim > 0 && key_dim <= 256 &&
        (key_dim & (key_dim - 1)) == 0,
        "Kimi KDA key dimension must be a power of two <= 256");
    TORCH_CHECK(
        workspace.numel() >= 3LL * heads * key_dim &&
        beta.numel() >= heads && a_log.numel() >= heads &&
        dt_bias.numel() >= (long)heads * key_dim,
        "Kimi KDA workspace or parameter shape mismatch");

    auto query_norm = workspace.narrow(
        0, 0, (long)heads * key_dim).view({heads, key_dim});
    auto key_norm = workspace.narrow(
        0, (long)heads * key_dim,
        (long)heads * key_dim).view({heads, key_dim});
    auto decay = workspace.narrow(
        0, 2LL * heads * key_dim,
        (long)heads * key_dim).view({heads, key_dim});
    auto stream = at::cuda::getCurrentCUDAStream();
    kimi_kda_prepare_kernel<<<
        heads,
        key_dim,
        2LL * key_dim * sizeof(float),
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                query.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                key.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                gate.data_ptr<at::BFloat16>()),
            a_log.data_ptr<float>(),
            dt_bias.data_ptr<float>(),
            query_norm.data_ptr<float>(),
            key_norm.data_ptr<float>(),
            decay.data_ptr<float>(),
            heads,
            key_dim,
            (float)lower_bound);
    const int value_blocks =
        (value_dim + KIMI_KDA_VALUES_PER_BLOCK - 1)
        / KIMI_KDA_VALUES_PER_BLOCK;
    const size_t update_smem = (
        2 * KIMI_KDA_VALUES_PER_BLOCK * key_dim
        + key_dim
        + KIMI_KDA_VALUES_PER_BLOCK
    ) * sizeof(float);
    kimi_kda_update_kernel<<<
        dim3(value_blocks, heads),
        key_dim,
        update_smem,
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                value.data_ptr<at::BFloat16>()),
            beta.data_ptr<float>(),
            query_norm.data_ptr<float>(),
            key_norm.data_ptr<float>(),
            decay.data_ptr<float>(),
            state.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            heads,
            key_dim,
            value_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// Ordered block-prefill counterpart.  One CTA owns a (head, value-tile) and
// walks the token dimension in order, so the recurrent state has exactly the
// same semantics as repeated decode calls while Python submits one kernel.
__global__ void kimi_kda_recurrent_batch_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ gate,
    const float* __restrict__ beta,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    const int tokens,
    const int heads,
    const int key_dim,
    const int value_dim,
    const float lower_bound)
{
    const int head = blockIdx.y;
    const int value_start = blockIdx.x * KIMI_KDA_VALUES_PER_BLOCK;
    const int item = threadIdx.x;
    if (head >= heads) return;
    extern __shared__ float shared[];
    float* q_norm = shared;
    float* k_norm = q_norm + key_dim;
    float* decay = k_norm + key_dim;
    float* prediction = decay + key_dim;
    float* old_output = prediction + KIMI_KDA_VALUES_PER_BLOCK * key_dim;
    float* key_query = old_output + KIMI_KDA_VALUES_PER_BLOCK * key_dim;
    float* deltas = key_query + key_dim;
    float* norm_scales = deltas + KIMI_KDA_VALUES_PER_BLOCK;
    for (int token = 0; token < tokens; ++token) {
        const long qk_base = ((long)token * heads + head) * key_dim;
        float q = item < key_dim
            ? __bfloat162float(query[qk_base + item]) : 0.f;
        float k = item < key_dim
            ? __bfloat162float(key[qk_base + item]) : 0.f;
        q_norm[item] = q * q;
        k_norm[item] = k * k;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (item < stride) {
                q_norm[item] += q_norm[item + stride];
                k_norm[item] += k_norm[item + stride];
            }
            __syncthreads();
        }
        if (item == 0) {
            norm_scales[0] = rsqrtf(q_norm[0] + 1e-6f);
            norm_scales[1] = rsqrtf(k_norm[0] + 1e-6f);
        }
        __syncthreads();
        if (item < key_dim) {
            const float q_scale = norm_scales[0];
            const float k_scale = norm_scales[1];
            q_norm[item] = q * q_scale;
            k_norm[item] = k * k_scale;
            const long gate_offset = qk_base + item;
            const long bias_offset = (long)head * key_dim + item;
            const float raw = expf(a_log[head]) * (
                __bfloat162float(gate[gate_offset]) + dt_bias[bias_offset]);
            decay[item] = expf(lower_bound * (1.f / (1.f + expf(-raw))));
            key_query[item] = q_norm[item] * k_norm[item];
        }
        __syncthreads();
        const long state_head = (long)head * value_dim * key_dim;
        #pragma unroll
        for (int row = 0; row < KIMI_KDA_VALUES_PER_BLOCK; ++row) {
            const int value_index = value_start + row;
            float current = 0.f;
            if (value_index < value_dim && item < key_dim) {
                const long state_offset = state_head
                    + (long)value_index * key_dim + item;
                current = state[state_offset] * decay[item];
                state[state_offset] = current;
            }
            prediction[row * key_dim + item] = current * k_norm[item];
            old_output[row * key_dim + item] = current * q_norm[item];
        }
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (item < stride) {
                key_query[item] += key_query[item + stride];
                #pragma unroll
                for (int row = 0; row < KIMI_KDA_VALUES_PER_BLOCK; ++row) {
                    const int base = row * key_dim + item;
                    prediction[base] += prediction[base + stride];
                    old_output[base] += old_output[base + stride];
                }
            }
            __syncthreads();
        }
        if (item < KIMI_KDA_VALUES_PER_BLOCK) {
            const int value_index = value_start + item;
            float delta = 0.f;
            if (value_index < value_dim) {
                const long value_offset = ((long)token * heads + head)
                    * value_dim + value_index;
                const float b = 1.f / (1.f + expf(-beta[(long)token * heads + head]));
                delta = (__bfloat162float(value[value_offset])
                    - prediction[item * key_dim]) * b;
            }
            deltas[item] = delta;
        }
        __syncthreads();
        #pragma unroll
        for (int row = 0; row < KIMI_KDA_VALUES_PER_BLOCK; ++row) {
            const int value_index = value_start + row;
            if (value_index < value_dim && item < key_dim) {
                const long state_offset = state_head
                    + (long)value_index * key_dim + item;
                state[state_offset] += deltas[row] * k_norm[item];
            }
            if (item == 0 && value_index < value_dim) {
                const float result = (old_output[row * key_dim]
                    + deltas[row] * key_query[0]) * rsqrtf((float)key_dim);
                const long output_offset = ((long)token * heads + head)
                    * value_dim + value_index;
                output[output_offset] = __float2bfloat16_rn(result);
            }
        }
        __syncthreads();
    }
}

torch::Tensor kimi_kda_recurrent_batch(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor gate, torch::Tensor beta, torch::Tensor a_log,
    torch::Tensor dt_bias, torch::Tensor state, torch::Tensor output,
    double lower_bound)
{
    TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda()
        && gate.is_cuda() && beta.is_cuda() && a_log.is_cuda()
        && dt_bias.is_cuda() && state.is_cuda() && output.is_cuda(),
        "Kimi KDA batch tensors must be CUDA");
    TORCH_CHECK(query.scalar_type() == at::kBFloat16
        && key.scalar_type() == at::kBFloat16
        && value.scalar_type() == at::kBFloat16
        && gate.scalar_type() == at::kBFloat16
        && output.scalar_type() == at::kBFloat16
        && beta.scalar_type() == at::kFloat
        && a_log.scalar_type() == at::kFloat
        && dt_bias.scalar_type() == at::kFloat
        && state.scalar_type() == at::kFloat,
        "Kimi KDA batch dtype mismatch");
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous()
        && value.is_contiguous() && gate.is_contiguous()
        && beta.is_contiguous() && a_log.is_contiguous()
        && dt_bias.is_contiguous() && state.is_contiguous()
        && output.is_contiguous(), "Kimi KDA batch tensors must be contiguous");
    TORCH_CHECK(query.dim() == 3 && key.sizes() == query.sizes()
        && gate.sizes() == query.sizes() && value.dim() == 3
        && value.size(0) == query.size(0) && value.size(1) == query.size(1)
        && beta.sizes() == query.sizes().slice(0, 2)
        && output.sizes() == value.sizes(), "Kimi KDA batch shapes mismatch");
    const int tokens = (int)query.size(0);
    const int heads = (int)query.size(1);
    const int key_dim = (int)query.size(2);
    const int value_dim = (int)value.size(2);
    TORCH_CHECK(key_dim > 0 && key_dim <= 256
        && (key_dim & (key_dim - 1)) == 0
        && state.sizes() == torch::IntArrayRef({heads, value_dim, key_dim})
        && beta.numel() >= (long)tokens * heads
        && a_log.numel() >= heads
        && dt_bias.numel() >= (long)heads * key_dim,
        "Kimi KDA batch state or parameter shape mismatch");
    const int value_blocks = (value_dim + KIMI_KDA_VALUES_PER_BLOCK - 1)
        / KIMI_KDA_VALUES_PER_BLOCK;
    const size_t smem = (
        (4LL + 2LL * KIMI_KDA_VALUES_PER_BLOCK) * key_dim
        + KIMI_KDA_VALUES_PER_BLOCK + 2) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    kimi_kda_recurrent_batch_kernel<<<dim3(value_blocks, heads), key_dim,
        smem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(key.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(value.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr<at::BFloat16>()),
            beta.data_ptr<float>(), a_log.data_ptr<float>(),
            dt_bias.data_ptr<float>(), state.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            tokens, heads, key_dim, value_dim, (float)lower_bound);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void kimi_gated_rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    const int heads,
    const int width,
    const float eps)
{
    const int head = blockIdx.x;
    const int item = threadIdx.x;
    if (head >= heads) return;
    extern __shared__ float reduction[];
    float square = 0.0f;
    if (item < width) {
        const float value = __bfloat162float(
            input[static_cast<long>(head) * width + item]);
        square = value * value;
    }
    reduction[item] = square;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (item < stride)
            reduction[item] += reduction[item + stride];
        __syncthreads();
    }
    if (item < width) {
        const long offset = static_cast<long>(head) * width + item;
        const float scale = rsqrtf(
            reduction[0] / static_cast<float>(width) + eps);
        const __nv_bfloat16 normalized = __float2bfloat16_rn(
            __bfloat162float(input[offset]) * scale);
        const __nv_bfloat16 weighted = __float2bfloat16_rn(
            __bfloat162float(normalized)
            * __bfloat162float(weight[item]));
        const float gate_value = __bfloat162float(
            __float2bfloat16_rn(
                1.0f / (
                    1.0f
                    + expf(-__bfloat162float(gate[offset])))));
        output[offset] = __float2bfloat16_rn(
            __bfloat162float(weighted) * gate_value);
    }
}

torch::Tensor kimi_gated_rmsnorm(
    torch::Tensor input,
    torch::Tensor gate,
    torch::Tensor weight,
    torch::Tensor output,
    double eps)
{
    TORCH_CHECK(
        input.is_cuda() && gate.is_cuda() &&
        weight.is_cuda() && output.is_cuda(),
        "Kimi gated RMSNorm tensors must be CUDA");
    TORCH_CHECK(
        input.scalar_type() == at::kBFloat16 &&
        gate.scalar_type() == at::kBFloat16 &&
        weight.scalar_type() == at::kBFloat16 &&
        output.scalar_type() == at::kBFloat16,
        "Kimi gated RMSNorm currently requires BF16");
    TORCH_CHECK(
        input.is_contiguous() && gate.is_contiguous() &&
        weight.is_contiguous() && output.is_contiguous(),
        "Kimi gated RMSNorm tensors must be contiguous");
    TORCH_CHECK(
        input.dim() == 2 && gate.sizes() == input.sizes() &&
        output.sizes() == input.sizes() &&
        weight.dim() == 1 && weight.size(0) == input.size(1),
        "Kimi gated RMSNorm shapes do not match");
    const int heads = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    TORCH_CHECK(
        width > 0 && width <= 256 &&
        (width & (width - 1)) == 0,
        "Kimi gated RMSNorm width must be a power of two <= 256");
    const int device = input.get_device();
    TORCH_CHECK(
        gate.get_device() == device &&
        weight.get_device() == device &&
        output.get_device() == device,
        "Kimi gated RMSNorm tensors must share one device");
    auto stream = at::cuda::getCurrentCUDAStream();
    kimi_gated_rmsnorm_kernel<<<
        heads,
        width,
        width * sizeof(float),
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                gate.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            heads,
            width,
            static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// ---- Stable-slot grouped VQ MLP (top-k <= 8, BF16 I/O) ----

constexpr int MAX_SLOT_EXPERTS = 16;

template <typename idx_t>
struct IndexPointerPack {
    const idx_t* ptrs[MAX_SLOT_EXPERTS];
};

template <typename scalar_t>
struct CodebookPointerPack {
    const scalar_t* ptrs[MAX_SLOT_EXPERTS];
};

struct SlotIntPack {
    int values[MAX_SLOT_EXPERTS];
};

template <typename scalar_t>
__device__ __forceinline__ float vq_scalar_to_float(const scalar_t* p) {
    return (float)(*p);
}

template <>
__device__ __forceinline__ float vq_scalar_to_float<__nv_bfloat16>(
    const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t vq_float_to_scalar(float v) {
    return (scalar_t)v;
}

template <>
__device__ __forceinline__ __nv_bfloat16
vq_float_to_scalar<__nv_bfloat16>(float v) {
    return __float2bfloat16_rn(v);
}

template <typename scalar_t>
__device__ __forceinline__ float vq_block_dot(
    const scalar_t* cb, const scalar_t* x, int D) {
    float part = 0.f;
    #pragma unroll 8
    for (int i = 0; i < D; ++i)
        part = fmaf(
            vq_scalar_to_float(cb + i),
            vq_scalar_to_float(x + i),
            part);
    return part;
}

template <>
__device__ __forceinline__ float vq_block_dot<__nv_bfloat16>(
    const __nv_bfloat16* cb, const __nv_bfloat16* x, int D) {
    const auto* cb2 = reinterpret_cast<const __nv_bfloat162*>(cb);
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    float part = 0.f;
    #pragma unroll 4
    for (int i = 0; i < D / 2; ++i) {
        const float2 cv = __bfloat1622float2(cb2[i]);
        const float2 xv = __bfloat1622float2(x2[i]);
        part = fmaf(cv.x, xv.x, part);
        part = fmaf(cv.y, xv.y, part);
    }
    return part;
}

template <typename idx_t, typename scalar_t>
__global__ void vq_gemv_slots_kernel(
    const scalar_t* __restrict__ x,
    const IndexPointerPack<idx_t> indices,
    const CodebookPointerPack<scalar_t> codebooks,
    const SlotIntPack blocks,
    const SlotIntPack code_dims,
    scalar_t* __restrict__ out,
    const int N, const int R, const int C,
    const long x_stride_n)
{
    const int n = blockIdx.y;
    if (n >= N) return;
    const int B = blocks.values[n];
    const int D = code_dims.values[n];
    const int r = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    extern __shared__ unsigned char raw_smem[];
    scalar_t* xs = reinterpret_cast<scalar_t*>(raw_smem);
    const scalar_t* xrow = x + (long)n * x_stride_n;
    for (int i = threadIdx.y * 32 + threadIdx.x;
         i < C; i += 32 * ROWS_PER_BLOCK)
        xs[i] = xrow[i];
    __syncthreads();
    if (r >= R) return;

    const idx_t* irow = indices.ptrs[n] + (long)r * B;
    const scalar_t* cb = codebooks.ptrs[n];
    float acc = 0.f;
    for (int b = threadIdx.x; b < B; b += 32) {
        const scalar_t* crow = cb + (long)irow[b] * D;
        const scalar_t* xb = xs + b * D;
        acc += vq_block_dot(crow, xb, D);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (threadIdx.x == 0)
        out[(long)n * R + r] = vq_float_to_scalar<scalar_t>(acc);
}

template <typename idx_t, typename scalar_t>
void launch_vq_gemv_slots(
    const torch::Tensor& x,
    const std::vector<torch::Tensor>& indices,
    const std::vector<torch::Tensor>& codebooks,
    torch::Tensor& out)
{
    const int N = (int)indices.size();
    const int R = (int)indices[0].size(0);
    const int C = (int)x.size(1);
    IndexPointerPack<idx_t> index_pack{};
    CodebookPointerPack<scalar_t> codebook_pack{};
    SlotIntPack block_pack{};
    SlotIntPack dim_pack{};
    for (int n = 0; n < N; ++n) {
        index_pack.ptrs[n] =
            reinterpret_cast<const idx_t*>(indices[n].data_ptr());
        codebook_pack.ptrs[n] =
            reinterpret_cast<const scalar_t*>(codebooks[n].data_ptr());
        block_pack.values[n] = (int)indices[n].size(1);
        dim_pack.values[n] = (int)codebooks[n].size(1);
    }
    const long x_stride_n = x.size(0) == 1 ? 0 : x.stride(0);
    dim3 block(32, ROWS_PER_BLOCK);
    dim3 grid(
        (unsigned)((R + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK),
        (unsigned)N);
    const size_t smem = (size_t)C * sizeof(scalar_t);
    auto stream = at::cuda::getCurrentCUDAStream();
    vq_gemv_slots_kernel<idx_t, scalar_t><<<grid, block, smem, stream>>>(
        reinterpret_cast<const scalar_t*>(x.data_ptr()),
        index_pack,
        codebook_pack,
        block_pack,
        dim_pack,
        reinterpret_cast<scalar_t*>(out.data_ptr()),
        N, R, C, x_stride_n);
}

void vq_gemv_slots_out(
    torch::Tensor x,
    const std::vector<torch::Tensor>& indices,
    const std::vector<torch::Tensor>& codebooks,
    torch::Tensor out)
{
    TORCH_CHECK(!indices.empty() && indices.size() <= MAX_SLOT_EXPERTS,
                "slot expert count must be in [1,8]");
    TORCH_CHECK(codebooks.size() == indices.size(),
                "slot codebook count mismatch");
    TORCH_CHECK(x.is_cuda() && out.is_cuda(),
                "x/out must be CUDA");
    TORCH_CHECK(
        x.scalar_type() == at::kBFloat16 &&
        out.scalar_type() == at::kBFloat16,
        "slot VQ x/out must be bfloat16");
    TORCH_CHECK(x.stride(1) == 1 && out.is_contiguous(),
                "slot VQ x rows and out must be contiguous");
    TORCH_CHECK(x.dim() == 2 && out.dim() == 2,
                "slot VQ tensors must be 2D");
    const auto dtype = indices[0].scalar_type();
    TORCH_CHECK(dtype == at::kByte || dtype == at::kUInt16,
                "slot indices must be uint8 or uint16");
    const auto rows = indices[0].size(0);
    TORCH_CHECK((long)indices.size() == out.size(0) && rows == out.size(1),
                "slot VQ output shape mismatch");
    TORCH_CHECK(x.size(0) == 1 || x.size(0) == (long)indices.size(),
                "slot VQ x batch mismatch");
    for (int n = 0; n < (int)indices.size(); ++n) {
        const auto& idx = indices[n];
        const auto& cb = codebooks[n];
        TORCH_CHECK(
            idx.is_cuda() && idx.is_contiguous() &&
            idx.scalar_type() == dtype &&
            idx.dim() == 2 && idx.size(0) == rows,
            "slot index row/dtype mismatch");
        TORCH_CHECK(
            cb.is_cuda() && cb.is_contiguous() &&
            cb.scalar_type() == at::kBFloat16 &&
            cb.dim() == 2,
            "slot codebook must be contiguous CUDA BF16 [K,D]");
        TORCH_CHECK(
            idx.size(1) * cb.size(1) == x.size(1),
            "slot VQ input width mismatch");
    }
    if (dtype == at::kByte) {
        launch_vq_gemv_slots<uint8_t, __nv_bfloat16>(
            x, indices, codebooks, out);
    } else {
        launch_vq_gemv_slots<uint16_t, __nv_bfloat16>(
            x, indices, codebooks, out);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void swiglu_bf16_inplace_kernel(
    __nv_bfloat16* __restrict__ h,
    const int N, const int inter, const float limit)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * inter;
    if (i >= total) return;
    const int n = i / inter;
    const int m = i - n * inter;
    const long row = (long)n * 2 * inter;
    float gate = __bfloat162float(h[row + m]);
    float up = __bfloat162float(h[row + inter + m]);
    if (limit > 0.f) {
        gate = fminf(gate, limit);
        up = fminf(fmaxf(up, -limit), limit);
    }
    const float silu = gate / (1.f + expf(-gate));
    h[row + m] = __float2bfloat16_rn(silu * up);
}

__global__ void gated_activation_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    const int count,
    const int activation,
    const float beta,
    const float linear_beta,
    const float limit)
{
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= count)
        return;
    float gate_value = __bfloat162float(gate[item]);
    float up_value = __bfloat162float(up[item]);
    if (activation == 0) {
        if (limit > 0.0f) {
            gate_value = fminf(gate_value, limit);
            up_value = fminf(fmaxf(up_value, -limit), limit);
        }
        const __nv_bfloat16 silu = __float2bfloat16_rn(
            gate_value / (1.0f + expf(-gate_value)));
        output[item] = __float2bfloat16_rn(
            __bfloat162float(silu) * up_value);
        return;
    }
    const float activated = (
        beta
        * tanhf(gate_value / beta)
        / (1.0f + expf(-gate_value)));
    float bounded_up = up_value;
    if (linear_beta > 0.0f)
        bounded_up = linear_beta * tanhf(up_value / linear_beta);
    output[item] = __float2bfloat16_rn(activated * bounded_up);
}

torch::Tensor gated_activation_bf16(
    torch::Tensor gate,
    torch::Tensor up,
    long activation,
    double beta,
    double linear_beta,
    double limit,
    c10::optional<torch::Tensor> output_buffer)
{
    TORCH_CHECK(
        gate.is_cuda() && up.is_cuda() &&
        gate.scalar_type() == at::kBFloat16 &&
        up.scalar_type() == at::kBFloat16 &&
        gate.is_contiguous() && up.is_contiguous() &&
        gate.sizes() == up.sizes() &&
        gate.get_device() == up.get_device(),
        "Gated activation needs colocated contiguous BF16 tensors");
    TORCH_CHECK(
        activation == 0 || activation == 1,
        "Gated activation kind must be 0 (SiLU) or 1 (SiTU)");
    TORCH_CHECK(
        activation == 0 || beta > 0.0,
        "SiTU beta must be positive");
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty_like(gate);
    TORCH_CHECK(
        output.is_cuda() &&
        output.scalar_type() == at::kBFloat16 &&
        output.is_contiguous() &&
        output.sizes() == gate.sizes() &&
        output.get_device() == gate.get_device(),
        "Gated activation output must match input");
    const int count = static_cast<int>(gate.numel());
    auto stream = at::cuda::getCurrentCUDAStream();
    gated_activation_bf16_kernel<<<
        (count + 255) / 256,
        256,
        0,
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                gate.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                up.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            count,
            static_cast<int>(activation),
            static_cast<float>(beta),
            static_cast<float>(linear_beta),
            static_cast<float>(limit));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void weighted_sum_bf16_kernel(
    const __nv_bfloat16* __restrict__ rows,
    const float* __restrict__ weights,
    __nv_bfloat16* __restrict__ result,
    const int N, const int D)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    #pragma unroll
    for (int n = 0; n < MAX_SLOT_EXPERTS; ++n) {
        if (n < N)
            acc = fmaf(
                __bfloat162float(rows[(long)n * D + d]),
                weights[n],
                acc);
    }
    result[d] = __float2bfloat16_rn(acc);
}

__global__ void weighted_sum_f32_kernel(
    const __nv_bfloat16* __restrict__ rows,
    const float* __restrict__ weights,
    float* __restrict__ result,
    const int N, const int D)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    #pragma unroll
    for (int n = 0; n < MAX_SLOT_EXPERTS; ++n) {
        if (n < N)
            acc = fmaf(
                __bfloat162float(rows[(long)n * D + d]),
                weights[n],
                acc);
    }
    result[d] = acc;
}

torch::Tensor moe_mlp_slots(
    torch::Tensor x,
    const std::vector<torch::Tensor>& gu_indices,
    const std::vector<torch::Tensor>& gu_codebooks,
    const std::vector<torch::Tensor>& dn_indices,
    const std::vector<torch::Tensor>& dn_codebooks,
    torch::Tensor weights,
    double limit,
    torch::Tensor hidden_workspace,
    torch::Tensor out_workspace,
    torch::Tensor result)
{
    const int N = (int)gu_indices.size();
    TORCH_CHECK(N > 0 && N <= MAX_SLOT_EXPERTS &&
                dn_indices.size() == gu_indices.size(),
                "GU/DN slot expert count mismatch");
    TORCH_CHECK(
        weights.is_cuda() && weights.scalar_type() == at::kFloat &&
        weights.is_contiguous() && weights.numel() == N,
        "slot route weights must be contiguous float32 [N]");
    TORCH_CHECK(
        hidden_workspace.is_cuda() &&
        hidden_workspace.scalar_type() == at::kBFloat16 &&
        hidden_workspace.is_contiguous() &&
        hidden_workspace.size(0) == N,
        "hidden workspace must be contiguous BF16 [N,2I]");
    TORCH_CHECK(
        out_workspace.is_cuda() &&
        out_workspace.scalar_type() == at::kBFloat16 &&
        out_workspace.is_contiguous() &&
        out_workspace.size(0) == N,
        "out workspace must be contiguous BF16 [N,D]");
    TORCH_CHECK(
        result.is_cuda() &&
        (
            result.scalar_type() == at::kBFloat16 ||
            result.scalar_type() == at::kFloat
        ) &&
        result.is_contiguous() && result.dim() == 1,
        "result must be contiguous BF16 or float32 [D]");

    vq_gemv_slots_out(x, gu_indices, gu_codebooks, hidden_workspace);
    const int inter = (int)hidden_workspace.size(1) / 2;
    const int activation_items = N * inter;
    auto stream = at::cuda::getCurrentCUDAStream();
    swiglu_bf16_inplace_kernel<<<
        (activation_items + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(
            hidden_workspace.data_ptr<at::BFloat16>()),
        N, inter, (float)limit);
    auto activation = hidden_workspace.narrow(1, 0, inter);
    vq_gemv_slots_out(
        activation, dn_indices, dn_codebooks, out_workspace);
    const int D = (int)out_workspace.size(1);
    TORCH_CHECK(result.numel() == D, "slot result width mismatch");
    if (result.scalar_type() == at::kFloat) {
        weighted_sum_f32_kernel<<<
            (D + 255) / 256, 256, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    out_workspace.data_ptr<at::BFloat16>()),
                weights.data_ptr<float>(),
                result.data_ptr<float>(),
                N, D);
    } else {
        weighted_sum_bf16_kernel<<<
            (D + 255) / 256, 256, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    out_workspace.data_ptr<at::BFloat16>()),
                weights.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(
                    result.data_ptr<at::BFloat16>()),
                N, D);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

// ---- Device-routed stable VQ MLP for full-resident Expert Parallel ----
//
// metadata is [10,E] int64 on the owner GPU:
//   GU index pointer, codebook pointer, blocks, code dim, index dtype tag,
//   DN index pointer, codebook pointer, blocks, code dim, index dtype tag.
// A zero index pointer means that the expert belongs to another rank.  The
// Top-K IDs stay on CUDA; every rank processes its owned positions directly.

constexpr int ROUTED_META_ROWS = 10;

void ensure_peer_access(
    const int current,
    const int peer,
    const char* operation)
{
    if (current == peer) return;
    constexpr int MAX_CACHED_DEVICES = 64;
    static bool peer_enabled[
        MAX_CACHED_DEVICES][MAX_CACHED_DEVICES] = {};
    TORCH_CHECK(
        current >= 0 && current < MAX_CACHED_DEVICES &&
        peer >= 0 && peer < MAX_CACHED_DEVICES,
        operation, " CUDA device index out of cache range");
    if (peer_enabled[current][peer]) return;
    int can_access = 0;
    const auto query_status = cudaDeviceCanAccessPeer(
        &can_access, current, peer);
    TORCH_CHECK(
        query_status == cudaSuccess && can_access,
        operation, " requires CUDA peer access");
    const auto enable_status = cudaDeviceEnablePeerAccess(peer, 0);
    if (enable_status == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
    } else {
        TORCH_CHECK(
            enable_status == cudaSuccess,
            "failed to enable ", operation, " peer access: ",
            cudaGetErrorString(enable_status));
    }
    peer_enabled[current][peer] = true;
}

#include "codegemm_vq.cuh"

// One peer-reading launch replaces three tiny cross-device copies for the
// full-resident expert path. x keeps the model's FP32 -> BF16 boundary.
template <typename input_t>
__global__ void expert_dispatch_pack_kernel(
    const input_t* __restrict__ x,
    const int64_t* __restrict__ route_ids,
    const float* __restrict__ weights,
    __nv_bfloat16* __restrict__ x_out,
    int64_t* __restrict__ route_ids_out,
    float* __restrict__ weights_out,
    const int hidden,
    const int K)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden)
        x_out[i] = __float2bfloat16_rn(
            vq_scalar_to_float(x + i));
    if (i < K) {
        route_ids_out[i] = route_ids[i];
        weights_out[i] = weights[i];
    }
}

void expert_dispatch_pack(
    torch::Tensor x,
    torch::Tensor route_ids,
    torch::Tensor weights,
    torch::Tensor x_out,
    torch::Tensor route_ids_out,
    torch::Tensor weights_out)
{
    TORCH_CHECK(
        x.is_cuda() &&
        (
            x.scalar_type() == at::kFloat ||
            x.scalar_type() == at::kBFloat16
        ) &&
        x.is_contiguous() && x.dim() == 2 && x.size(0) == 1,
        "expert dispatch x must be contiguous CUDA FP32/BF16 [1,D]");
    TORCH_CHECK(
        route_ids.is_cuda() && route_ids.scalar_type() == at::kLong &&
        route_ids.is_contiguous() && route_ids.dim() == 1,
        "expert dispatch IDs must be contiguous CUDA int64 [K]");
    TORCH_CHECK(
        weights.is_cuda() && weights.scalar_type() == at::kFloat &&
        weights.is_contiguous() && weights.sizes() == route_ids.sizes(),
        "expert dispatch weights must be contiguous CUDA FP32 [K]");
    TORCH_CHECK(
        x_out.is_cuda() && x_out.scalar_type() == at::kBFloat16 &&
        x_out.is_contiguous() && x_out.sizes() == x.sizes(),
        "expert dispatch x output must be contiguous CUDA BF16 [1,D]");
    TORCH_CHECK(
        route_ids_out.is_cuda() &&
        route_ids_out.scalar_type() == at::kLong &&
        route_ids_out.is_contiguous() &&
        route_ids_out.sizes() == route_ids.sizes(),
        "expert dispatch ID output shape mismatch");
    TORCH_CHECK(
        weights_out.is_cuda() &&
        weights_out.scalar_type() == at::kFloat &&
        weights_out.is_contiguous() &&
        weights_out.sizes() == weights.sizes(),
        "expert dispatch weight output shape mismatch");
    const int source = x.get_device();
    const int target = x_out.get_device();
    TORCH_CHECK(
        route_ids.get_device() == source &&
        weights.get_device() == source,
        "expert dispatch sources must share one CUDA device");
    TORCH_CHECK(
        route_ids_out.get_device() == target &&
        weights_out.get_device() == target,
        "expert dispatch outputs must share one CUDA device");

    int current = -1;
    const auto current_status = cudaGetDevice(&current);
    TORCH_CHECK(
        current_status == cudaSuccess && current == target,
        "expert dispatch must run under the output CUDA device");
    ensure_peer_access(target, source, "expert dispatch");
    const int hidden = static_cast<int>(x.numel());
    const int K = static_cast<int>(route_ids.numel());
    const int count = std::max(hidden, K);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == at::kBFloat16) {
        expert_dispatch_pack_kernel<<<
            (count + 255) / 256, 256, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    x.data_ptr<at::BFloat16>()),
                route_ids.data_ptr<int64_t>(),
                weights.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(
                    x_out.data_ptr<at::BFloat16>()),
                route_ids_out.data_ptr<int64_t>(),
                weights_out.data_ptr<float>(),
                hidden,
                K);
    } else {
        expert_dispatch_pack_kernel<<<
            (count + 255) / 256, 256, 0, stream>>>(
                x.data_ptr<float>(),
                route_ids.data_ptr<int64_t>(),
                weights.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(
                    x_out.data_ptr<at::BFloat16>()),
                route_ids_out.data_ptr<int64_t>(),
                weights_out.data_ptr<float>(),
                hidden,
                K);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
__global__ void tp_peer_copy_kernel(
    const scalar_t* __restrict__ source,
    scalar_t* __restrict__ destination,
    const long count)
{
    for (
        long index =
            static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < count;
        index += static_cast<long>(blockDim.x) * gridDim.x
    ) {
        destination[index] = source[index];
    }
}

void tp_peer_copy(
    torch::Tensor source,
    torch::Tensor destination)
{
    TORCH_CHECK(
        source.is_cuda() && destination.is_cuda() &&
        source.is_contiguous() && destination.is_contiguous() &&
        source.sizes() == destination.sizes() &&
        source.scalar_type() == destination.scalar_type(),
        "TP peer copy tensors must be matching contiguous CUDA tensors");
    TORCH_CHECK(
        source.scalar_type() == at::kFloat ||
        source.scalar_type() == at::kLong ||
        source.scalar_type() == at::kBFloat16,
        "TP peer copy currently supports float32, bfloat16 and int64");
    const int source_device = source.get_device();
    const int target_device = destination.get_device();
    int current = -1;
    C10_CUDA_CHECK(cudaGetDevice(&current));
    TORCH_CHECK(
        current == target_device || current == source_device,
        "TP peer copy must run under its source or destination CUDA device");
    ensure_peer_access(
        current,
        current == target_device ? source_device : target_device,
        "TP peer copy");
    const long count = source.numel();
    const int blocks = static_cast<int>(
        std::min<long>((count + 255) / 256, 4096));
    auto stream = at::cuda::getCurrentCUDAStream();
    if (source.scalar_type() == at::kFloat) {
        tp_peer_copy_kernel<<<blocks, 256, 0, stream>>>(
            source.data_ptr<float>(),
            destination.data_ptr<float>(),
            count);
    } else if (source.scalar_type() == at::kLong) {
        tp_peer_copy_kernel<<<blocks, 256, 0, stream>>>(
            source.data_ptr<int64_t>(),
            destination.data_ptr<int64_t>(),
            count);
    } else {
        tp_peer_copy_kernel<<<blocks, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                source.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                destination.data_ptr<at::BFloat16>()),
            count);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool POSITION_FROM_POINTER>
__global__ void tp_attention_dispatch_kernel(
    const float* __restrict__ source_q,
    float* __restrict__ destination_q,
    const long q_count,
    const float* __restrict__ source_c,
    float* __restrict__ destination_c,
    const long c_count,
    const float* __restrict__ source_k,
    float* __restrict__ destination_k,
    const long k_count,
    const int64_t* __restrict__ source_position,
    int64_t* __restrict__ destination_position,
    const int64_t position_value)
{
    for (
        long index =
            static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < q_count;
        index += static_cast<long>(blockDim.x) * gridDim.x
    ) {
        destination_q[index] = source_q[index];
    }
    for (
        long index =
            static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < c_count;
        index += static_cast<long>(blockDim.x) * gridDim.x
    ) {
        destination_c[index] = source_c[index];
    }
    for (
        long index =
            static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < k_count;
        index += static_cast<long>(blockDim.x) * gridDim.x
    ) {
        destination_k[index] = source_k[index];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        destination_position[0] = (
            POSITION_FROM_POINTER
                ? source_position[0]
                : position_value);
    }
}

void tp_attention_peer_dispatch(
    torch::Tensor source_q,
    torch::Tensor source_c,
    torch::Tensor source_k,
    torch::Tensor source_position,
    torch::Tensor destination_q,
    torch::Tensor destination_c,
    torch::Tensor destination_k,
    torch::Tensor destination_position)
{
    const torch::Tensor sources[] = {
        source_q,
        source_c,
        source_k,
    };
    const torch::Tensor destinations[] = {
        destination_q,
        destination_c,
        destination_k,
    };
    for (int index = 0; index < 3; ++index) {
        TORCH_CHECK(
            sources[index].is_cuda() &&
            destinations[index].is_cuda() &&
            sources[index].scalar_type() == at::kFloat &&
            destinations[index].scalar_type() == at::kFloat &&
            sources[index].is_contiguous() &&
            destinations[index].is_contiguous() &&
            sources[index].sizes() == destinations[index].sizes(),
            "Attention TP peer tensors must be matching contiguous "
            "CUDA float32 tensors");
    }
    TORCH_CHECK(
        source_position.is_cuda() &&
        destination_position.is_cuda() &&
        source_position.scalar_type() == at::kLong &&
        destination_position.scalar_type() == at::kLong &&
        source_position.is_contiguous() &&
        destination_position.is_contiguous() &&
        source_position.numel() == 1 &&
        destination_position.numel() == 1,
        "Attention TP position tensors must be scalar CUDA int64");
    const int source_device = source_q.get_device();
    const int target_device = destination_q.get_device();
    TORCH_CHECK(
        source_c.get_device() == source_device &&
        source_k.get_device() == source_device &&
        source_position.get_device() == source_device &&
        destination_c.get_device() == target_device &&
        destination_k.get_device() == target_device &&
        destination_position.get_device() == target_device,
        "Attention TP source and destination tensors must each share "
        "one device");
    int current = -1;
    C10_CUDA_CHECK(cudaGetDevice(&current));
    TORCH_CHECK(
        current == target_device,
        "Attention TP peer dispatch must run under the destination device");
    ensure_peer_access(
        target_device,
        source_device,
        "Attention TP source dispatch");
    const long q_count = (
        source_q.data_ptr() == destination_q.data_ptr()
            ? 0
            : source_q.numel());
    const long c_count = (
        source_c.data_ptr() == destination_c.data_ptr()
            ? 0
            : source_c.numel());
    const long k_count = (
        source_k.data_ptr() == destination_k.data_ptr()
            ? 0
            : source_k.numel());
    const long count = std::max(q_count, std::max(c_count, k_count));
    const int blocks = static_cast<int>(
        std::min<long>((count + 255) / 256, 4096));
    auto stream = at::cuda::getCurrentCUDAStream();
    tp_attention_dispatch_kernel<true><<<blocks, 256, 0, stream>>>(
        source_q.data_ptr<float>(),
        destination_q.data_ptr<float>(),
        q_count,
        source_c.data_ptr<float>(),
        destination_c.data_ptr<float>(),
        c_count,
        source_k.data_ptr<float>(),
        destination_k.data_ptr<float>(),
        k_count,
        source_position.data_ptr<int64_t>(),
        destination_position.data_ptr<int64_t>(),
        0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tp_attention_source_pack(
    torch::Tensor source_q,
    torch::Tensor source_c,
    torch::Tensor source_k,
    torch::Tensor destination_q,
    torch::Tensor destination_c,
    torch::Tensor destination_k,
    torch::Tensor destination_position,
    int64_t position)
{
    const torch::Tensor sources[] = {
        source_q,
        source_c,
        source_k,
    };
    const torch::Tensor destinations[] = {
        destination_q,
        destination_c,
        destination_k,
    };
    const int device = source_q.get_device();
    for (int index = 0; index < 3; ++index) {
        TORCH_CHECK(
            sources[index].is_cuda() &&
            destinations[index].is_cuda() &&
            sources[index].scalar_type() == at::kFloat &&
            destinations[index].scalar_type() == at::kFloat &&
            sources[index].is_contiguous() &&
            destinations[index].is_contiguous() &&
            sources[index].sizes() == destinations[index].sizes() &&
            sources[index].get_device() == device &&
            destinations[index].get_device() == device,
            "Attention TP source-pack tensors must be matching contiguous "
            "CUDA float32 tensors on one device");
    }
    TORCH_CHECK(
        destination_position.is_cuda() &&
        destination_position.scalar_type() == at::kLong &&
        destination_position.is_contiguous() &&
        destination_position.numel() == 1 &&
        destination_position.get_device() == device,
        "Attention TP source-pack position must be scalar CUDA int64 "
        "on the input device");
    int current = -1;
    C10_CUDA_CHECK(cudaGetDevice(&current));
    TORCH_CHECK(
        current == device,
        "Attention TP source pack must run under the input device");
    const long q_count = (
        source_q.data_ptr() == destination_q.data_ptr()
            ? 0
            : source_q.numel());
    const long c_count = (
        source_c.data_ptr() == destination_c.data_ptr()
            ? 0
            : source_c.numel());
    const long k_count = (
        source_k.data_ptr() == destination_k.data_ptr()
            ? 0
            : source_k.numel());
    const long count = std::max(q_count, std::max(c_count, k_count));
    // All three sources may already alias their fixed graph buffers.  We
    // still launch one block to publish the new position scalar; a zero-block
    // launch is an invalid CUDA configuration.
    const int blocks = static_cast<int>(
        std::max<long>(
            1,
            std::min<long>((count + 255) / 256, 4096)));
    auto stream = at::cuda::getCurrentCUDAStream();
    tp_attention_dispatch_kernel<false><<<blocks, 256, 0, stream>>>(
        source_q.data_ptr<float>(),
        destination_q.data_ptr<float>(),
        q_count,
        source_c.data_ptr<float>(),
        destination_c.data_ptr<float>(),
        c_count,
        source_k.data_ptr<float>(),
        destination_k.data_ptr<float>(),
        k_count,
        nullptr,
        destination_position.data_ptr<int64_t>(),
        position);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__device__ __forceinline__ int routed_index_value(
    const int64_t address,
    const int dtype_tag,
    const long offset)
{
    const uintptr_t raw = static_cast<uintptr_t>(address);
    if (dtype_tag == 0)
        return static_cast<int>(
            reinterpret_cast<const uint8_t*>(raw)[offset]);
    if (dtype_tag == 1)
        return static_cast<int>(
            reinterpret_cast<const uint16_t*>(raw)[offset]);
    const auto* bytes = reinterpret_cast<const uint8_t*>(raw);
    if (dtype_tag == 2) {
        // Two little-endian 12-bit indices are stored in three bytes.
        const long base = (offset >> 1) * 3;
        if ((offset & 1) == 0)
            return static_cast<int>(
                bytes[base] | ((bytes[base + 1] & 0x0f) << 8));
        return static_cast<int>(
            (bytes[base + 1] >> 4) | (bytes[base + 2] << 4));
    }
    if (dtype_tag == 5) {
        // Consecutive little-endian 9-bit indices.  Reading two bytes is
        // sufficient because the bit offset within the first byte is 0..7.
        const long bit_offset = offset * 9;
        const long base = bit_offset >> 3;
        const int shift = static_cast<int>(bit_offset & 7);
        const unsigned word =
            static_cast<unsigned>(bytes[base]) |
            (static_cast<unsigned>(bytes[base + 1]) << 8);
        return static_cast<int>((word >> shift) & 0x1ffu);
    }
    if (dtype_tag >= 6 && dtype_tag <= 8) {
        // New projection archives use every odd width through p15.  Decode
        // directly from the little-endian bit stream; the final index reads
        // only two bytes when its 15 bits end exactly on the payload edge.
        const int bits = 2 * dtype_tag - 1;
        const long bit_offset = offset * bits;
        const long base = bit_offset >> 3;
        const int shift = static_cast<int>(bit_offset & 7);
        unsigned word =
            static_cast<unsigned>(bytes[base]) |
            (static_cast<unsigned>(bytes[base + 1]) << 8);
        if (shift + bits > 16)
            word |= static_cast<unsigned>(bytes[base + 2]) << 16;
        return static_cast<int>(
            (word >> shift) & ((1u << bits) - 1u));
    }
    if (dtype_tag == 4) {
        // Four little-endian 10-bit indices are stored in five bytes.
        const long base = (offset >> 2) * 5;
        unsigned long long word = 0;
        #pragma unroll
        for (int byte = 0; byte < 5; ++byte)
            word |= static_cast<unsigned long long>(bytes[base + byte])
                    << (8 * byte);
        return static_cast<int>(
            (word >> (10 * (offset & 3))) & 0x3ffu);
    }
    // Four little-endian 14-bit indices are stored in seven bytes.  Assemble
    // explicitly so the read remains valid for arbitrary byte alignment.
    const long base = (offset >> 2) * 7;
    unsigned long long word = 0;
    #pragma unroll
    for (int byte = 0; byte < 7; ++byte)
        word |= static_cast<unsigned long long>(bytes[base + byte])
                << (8 * byte);
    return static_cast<int>(
        (word >> (14 * (offset & 3))) & 0x3fffu);
}

// Generic dense-VQ projections keep one row-major p8--p16 byte stream per
// matrix. Unlike routed MoE metadata, these weights are ordinary Linear and
// Embedding modules, so the public operator takes explicit logical shape.
template <typename input_t>
__device__ __forceinline__ float dense_fp8_input_value(input_t value)
{
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float dense_fp8_input_value(
    __nv_bfloat16 value)
{
    return __bfloat162float(value);
}

// Capture-safe row-wise activation conversion for the native scaled-GEMM
// path. One block computes one scale and writes one E4M3 row, replacing the
// many Torch reductions/conversions which otherwise cost more than GEMM at
// batch one.
template <typename input_t>
__global__ void dense_fp8_quantize_rows_kernel(
    const input_t* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    const int64_t rows,
    const int64_t cols)
{
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows)
        return;
    __shared__ float reductions[256];
    __shared__ float row_scale;
    float maximum = 0.0f;
    const int64_t row_offset = row * cols;
    for (int64_t column = threadIdx.x;
         column < cols;
         column += blockDim.x)
        maximum = fmaxf(
            maximum,
            fabsf(dense_fp8_input_value(input[row_offset + column])));
    reductions[threadIdx.x] = maximum;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width >>= 1) {
        if (threadIdx.x < width)
            reductions[threadIdx.x] = fmaxf(
                reductions[threadIdx.x], reductions[threadIdx.x + width]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_scale = fmaxf(reductions[0] / 448.0f, 1.0e-12f);
        scales[row] = row_scale;
    }
    __syncthreads();
    const float inverse_scale = 1.0f / row_scale;
    for (int64_t column = threadIdx.x;
         column < cols;
         column += blockDim.x) {
        const float value = fminf(
            448.0f,
            fmaxf(
                -448.0f,
                dense_fp8_input_value(input[row_offset + column])
                    * inverse_scale));
        __nv_fp8_e4m3 quantized(value);
        output[row_offset + column] = quantized.__x;
    }
}

// CUDA 12.8 exposes the fast tensor-scaled FP8 GEMM but not the newer
// row-scaled form.  Reduce the source directly in its native dtype, then
// quantize in a second launch.  This replaces Torch's FP32 expansion plus
// abs/amax/div/clamp/cast chain without allocating a full-size temporary.
template <typename input_t>
__global__ void dense_fp8_tensor_amax_kernel(
    const input_t* __restrict__ input,
    unsigned int* __restrict__ maximum_bits,
    const int64_t items)
{
    float maximum = 0.0f;
    for (int64_t item =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < items;
         item += static_cast<int64_t>(gridDim.x) * blockDim.x)
        maximum = fmaxf(
            maximum,
            fabsf(dense_fp8_input_value(input[item])));
    __shared__ float reductions[256];
    reductions[threadIdx.x] = maximum;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width >>= 1) {
        if (threadIdx.x < width)
            reductions[threadIdx.x] = fmaxf(
                reductions[threadIdx.x], reductions[threadIdx.x + width]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicMax(maximum_bits, __float_as_uint(reductions[0]));
}

template <typename input_t>
__global__ void dense_fp8_quantize_tensor_kernel(
    const input_t* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scale,
    const int64_t items)
{
    const float value_scale = fmaxf(scale[0] / 448.0f, 1.0e-12f);
    const float inverse_scale = 1.0f / value_scale;
    for (int64_t item =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < items;
         item += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const float value = fminf(
            448.0f,
            fmaxf(
                -448.0f,
                dense_fp8_input_value(input[item]) * inverse_scale));
        __nv_fp8_e4m3 quantized(value);
        output[item] = quantized.__x;
    }
}

__global__ void dense_fp8_finalize_tensor_scale_kernel(float* scale)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        scale[0] = fmaxf(scale[0] / 448.0f, 1.0e-12f);
}

torch::Tensor dense_fp8_quantize_rows(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales)
{
    TORCH_CHECK(input.is_cuda(), "Dense FP8 input must be CUDA");
    TORCH_CHECK(input.dim() == 2, "Dense FP8 input must be rank two");
    TORCH_CHECK(
        input.scalar_type() == torch::kBFloat16
            || input.scalar_type() == torch::kFloat32,
        "Dense FP8 input must be BF16 or F32");
    TORCH_CHECK(input.is_contiguous(), "Dense FP8 input must be contiguous");
    TORCH_CHECK(output.is_cuda(), "Dense FP8 output must be CUDA");
    TORCH_CHECK(
        output.scalar_type() == at::ScalarType::Float8_e4m3fn,
        "Dense FP8 output must be E4M3FN");
    TORCH_CHECK(
        output.sizes() == input.sizes() && output.is_contiguous(),
        "Dense FP8 output shape/stride mismatch");
    TORCH_CHECK(
        scales.is_cuda() && scales.scalar_type() == torch::kFloat32,
        "Dense FP8 scales must be CUDA F32");
    TORCH_CHECK(
        scales.dim() == 2
            && scales.size(1) == 1
            && (scales.size(0) == 1 || scales.size(0) == input.size(0))
            && scales.is_contiguous(),
        "Dense FP8 scales must be contiguous [1,1] or [rows,1]");
    const int64_t rows = input.size(0);
    const int64_t cols = input.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    // Decode has exactly one row and must stay on the original single-block
    // quantizer.  The scalar tensor reduction below is only for true batches;
    // using its three launches for batch one adds hundreds of needless graph
    // nodes across a full token.
    if (scales.size(0) == 1 && rows > 1) {
        const int64_t items = rows * cols;
        const int blocks = static_cast<int>(std::min<int64_t>(
            1024, (items + 255) / 256));
        C10_CUDA_CHECK(cudaMemsetAsync(
            scales.data_ptr<float>(), 0, sizeof(float), stream));
        if (input.scalar_type() == torch::kBFloat16) {
            dense_fp8_tensor_amax_kernel<<<blocks, 256, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                reinterpret_cast<unsigned int*>(scales.data_ptr<float>()),
                items);
            dense_fp8_quantize_tensor_kernel<<<blocks, 256, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                static_cast<uint8_t*>(output.data_ptr()),
                scales.data_ptr<float>(),
                items);
        } else {
            dense_fp8_tensor_amax_kernel<<<blocks, 256, 0, stream>>>(
                input.data_ptr<float>(),
                reinterpret_cast<unsigned int*>(scales.data_ptr<float>()),
                items);
            dense_fp8_quantize_tensor_kernel<<<blocks, 256, 0, stream>>>(
                input.data_ptr<float>(),
                static_cast<uint8_t*>(output.data_ptr()),
                scales.data_ptr<float>(),
                items);
        }
        dense_fp8_finalize_tensor_scale_kernel<<<1, 1, 0, stream>>>(
            scales.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    }
    const dim3 grid(static_cast<unsigned int>(rows));
    if (input.scalar_type() == torch::kBFloat16) {
        dense_fp8_quantize_rows_kernel<<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            static_cast<uint8_t*>(output.data_ptr()),
            scales.data_ptr<float>(),
            rows,
            cols);
    } else {
        dense_fp8_quantize_rows_kernel<<<grid, 256, 0, stream>>>(
            input.data_ptr<float>(),
            static_cast<uint8_t*>(output.data_ptr()),
            scales.data_ptr<float>(),
            rows,
            cols);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// Fuse the gated activation and the row-wise E4M3 conversion used between
// the two routed expert GEMMs.  The old chain wrote a full BF16 activation,
// reread it to find amax, then reread it again for quantization.  One CTA per
// routed row keeps the rounded BF16 activation in shared memory, preserving
// the former numerical boundary while removing both global-memory passes.
__global__ void gated_activation_fp8_quantize_rows_kernel(
    const __nv_bfloat16* __restrict__ gate_up,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    const int64_t rows,
    const int intermediate,
    const int activation,
    const float beta,
    const float linear_beta,
    const float limit)
{
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows)
        return;
    extern __shared__ __nv_bfloat16 activated[];
    __shared__ float reductions[256];
    __shared__ float row_scale;
    const int64_t source = row * static_cast<int64_t>(2 * intermediate);
    float maximum = 0.0f;
    for (int column = threadIdx.x; column < intermediate;
         column += blockDim.x) {
        float gate = __bfloat162float(gate_up[source + column]);
        float up = __bfloat162float(
            gate_up[source + intermediate + column]);
        float value;
        if (activation == 0) {
            if (limit > 0.0f) {
                gate = fminf(gate, limit);
                up = fminf(fmaxf(up, -limit), limit);
            }
            value = (gate / (1.0f + expf(-gate))) * up;
        } else {
            const float nonlinear =
                beta * tanhf(gate / beta) / (1.0f + expf(-gate));
            if (linear_beta > 0.0f)
                up = linear_beta * tanhf(up / linear_beta);
            value = nonlinear * up;
        }
        const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
        activated[column] = rounded;
        maximum = fmaxf(maximum, fabsf(__bfloat162float(rounded)));
    }
    reductions[threadIdx.x] = maximum;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width >>= 1) {
        if (threadIdx.x < width)
            reductions[threadIdx.x] = fmaxf(
                reductions[threadIdx.x], reductions[threadIdx.x + width]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_scale = fmaxf(reductions[0] / 448.0f, 1.0e-12f);
        scales[row] = row_scale;
    }
    __syncthreads();
    const float inverse_scale = 1.0f / row_scale;
    const int64_t destination = row * static_cast<int64_t>(intermediate);
    for (int column = threadIdx.x; column < intermediate;
         column += blockDim.x) {
        const float value = fminf(
            448.0f,
            fmaxf(
                -448.0f,
                __bfloat162float(activated[column]) * inverse_scale));
        __nv_fp8_e4m3 quantized(value);
        output[destination + column] = quantized.__x;
    }
}

torch::Tensor gated_activation_fp8_quantize_rows(
    torch::Tensor gate_up,
    torch::Tensor output,
    torch::Tensor scales,
    int64_t activation,
    double beta,
    double linear_beta,
    double limit)
{
    TORCH_CHECK(
        gate_up.is_cuda() && gate_up.scalar_type() == at::kBFloat16 &&
        gate_up.is_contiguous() && gate_up.dim() == 2 &&
        gate_up.size(1) % 2 == 0,
        "Gated FP8 activation requires contiguous CUDA BF16 [N,2I]");
    const int64_t rows = gate_up.size(0);
    const int64_t intermediate = gate_up.size(1) / 2;
    TORCH_CHECK(
        output.is_cuda() &&
        output.scalar_type() == at::ScalarType::Float8_e4m3fn &&
        output.is_contiguous() && output.dim() == 2 &&
        output.size(0) == rows && output.size(1) == intermediate &&
        scales.is_cuda() && scales.scalar_type() == at::kFloat &&
        scales.is_contiguous() && scales.size(0) == rows &&
        scales.size(1) == 1 && (activation == 0 || activation == 1),
        "Gated FP8 activation output/scale mismatch");
    auto stream = at::cuda::getCurrentCUDAStream(gate_up.get_device());
    gated_activation_fp8_quantize_rows_kernel<<<
        static_cast<unsigned>(rows), 256,
        static_cast<size_t>(intermediate) * sizeof(__nv_bfloat16),
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                gate_up.data_ptr<at::BFloat16>()),
            static_cast<uint8_t*>(output.data_ptr()),
            scales.data_ptr<float>(), rows, static_cast<int>(intermediate),
            static_cast<int>(activation), static_cast<float>(beta),
            static_cast<float>(linear_beta), static_cast<float>(limit));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void routed_inverse_order_kernel(
    const int64_t* __restrict__ sorted_positions,
    int64_t* __restrict__ inverse,
    const int64_t count)
{
    const int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (item < count)
        inverse[sorted_positions[item]] = item;
}

__global__ void routed_weighted_reduce_kernel(
    const __nv_bfloat16* __restrict__ rows,
    const int64_t* __restrict__ inverse,
    const float* __restrict__ weights,
    float* __restrict__ output,
    const int tokens,
    const int top_k,
    const int hidden)
{
    const int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const int64_t total = static_cast<int64_t>(tokens) * hidden;
    if (item >= total)
        return;
    const int token = static_cast<int>(item / hidden);
    const int column = static_cast<int>(item - static_cast<int64_t>(token) * hidden);
    float sum = 0.0f;
    #pragma unroll
    for (int route = 0; route < top_k; ++route) {
        const int64_t logical = static_cast<int64_t>(token) * top_k + route;
        const int64_t packed = inverse[logical];
        sum += __bfloat162float(
            rows[packed * static_cast<int64_t>(hidden) + column])
            * weights[logical];
    }
    output[item] = sum;
}

torch::Tensor routed_weighted_reduce(
    torch::Tensor rows,
    torch::Tensor sorted_positions,
    torch::Tensor weights,
    torch::Tensor inverse,
    torch::Tensor output,
    int64_t top_k)
{
    TORCH_CHECK(
        rows.is_cuda() && rows.scalar_type() == at::kBFloat16 &&
        rows.is_contiguous() && rows.dim() == 2 &&
        sorted_positions.is_cuda() &&
        sorted_positions.scalar_type() == at::kLong &&
        sorted_positions.is_contiguous() &&
        weights.is_cuda() && weights.scalar_type() == at::kFloat &&
        weights.is_contiguous() && inverse.is_cuda() &&
        inverse.scalar_type() == at::kLong && inverse.is_contiguous() &&
        output.is_cuda() && output.scalar_type() == at::kFloat &&
        output.is_contiguous() && output.dim() == 2 && top_k > 0,
        "Routed weighted reduction operand mismatch");
    const int64_t routes = rows.size(0);
    TORCH_CHECK(
        sorted_positions.numel() == routes && weights.numel() == routes &&
        inverse.numel() >= routes && output.size(0) * top_k == routes &&
        output.size(1) == rows.size(1),
        "Routed weighted reduction shape mismatch");
    auto stream = at::cuda::getCurrentCUDAStream(rows.get_device());
    routed_inverse_order_kernel<<<
        static_cast<unsigned>((routes + 255) / 256), 256, 0, stream>>>(
            sorted_positions.data_ptr<int64_t>(), inverse.data_ptr<int64_t>(),
            routes);
    const int64_t items = output.numel();
    routed_weighted_reduce_kernel<<<
        static_cast<unsigned>((items + 255) / 256), 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                rows.data_ptr<at::BFloat16>()),
            inverse.data_ptr<int64_t>(), weights.data_ptr<float>(),
            output.data_ptr<float>(), static_cast<int>(output.size(0)),
            static_cast<int>(top_k), static_cast<int>(output.size(1)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

inline int dense_vq_dtype_tag(const int bits)
{
    switch (bits) {
        case 8: return 0;
        case 16: return 1;
        case 12: return 2;
        case 14: return 3;
        case 10: return 4;
        case 9: return 5;
        case 11: return 6;
        case 13: return 7;
        case 15: return 8;
        default: return -1;
    }
}

constexpr int CCCP_DENSE_VQ_ROWS_PER_BLOCK = 16;

template <bool STAGE_INPUT>
__global__ void dense_vq_gemv_packed_kernel(
    const float* __restrict__ input,
    const uint8_t* __restrict__ packed,
    const float* __restrict__ codebook,
    float* __restrict__ output,
    const int tokens,
    const int rows,
    const int blocks,
    const int vector,
    const int dtype_tag)
{
    const int token = blockIdx.y;
    const int row = blockIdx.x * CCCP_DENSE_VQ_ROWS_PER_BLOCK + threadIdx.y;
    const int columns = blocks * vector;
    extern __shared__ float staged_input[];
    const float* source = input + static_cast<long>(token) * columns;
    if constexpr (STAGE_INPUT) {
        for (int column = threadIdx.y * 32 + threadIdx.x;
             column < columns;
             column += 32 * CCCP_DENSE_VQ_ROWS_PER_BLOCK)
            staged_input[column] = source[column];
        __syncthreads();
        source = staged_input;
    }
    if (token >= tokens || row >= rows) return;

    const int64_t address = static_cast<int64_t>(
        reinterpret_cast<uintptr_t>(packed));
    float sum = 0.0f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        const int code = routed_index_value(
            address, dtype_tag,
            static_cast<long>(row) * blocks + block);
        const float* code_row = codebook + static_cast<long>(code) * vector;
        const float* input_block = source + block * vector;
        float partial = 0.0f;
        for (int component = 0; component < vector; ++component)
            partial = fmaf(
                code_row[component], input_block[component], partial);
        sum += partial;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset, 32);
    if (threadIdx.x == 0)
        output[static_cast<long>(token) * rows + row] = sum;
}

torch::Tensor dense_vq_gemv_packed(
    torch::Tensor input,
    torch::Tensor packed,
    torch::Tensor codebook,
    int64_t rows,
    int64_t blocks,
    int64_t bits)
{
    TORCH_CHECK(
        input.is_cuda() && packed.is_cuda() && codebook.is_cuda(),
        "dense packed VQ operands must be CUDA tensors");
    TORCH_CHECK(
        input.scalar_type() == at::kFloat && input.dim() == 2 &&
        packed.scalar_type() == at::kByte && packed.dim() == 1 &&
        codebook.scalar_type() == at::kFloat && codebook.dim() == 2,
        "dense packed VQ requires FP32 [T,C], uint8 [bytes], FP32 [K,D]");
    TORCH_CHECK(
        input.is_contiguous() && packed.is_contiguous() &&
        codebook.is_contiguous(),
        "dense packed VQ operands must be contiguous");
    const int tag = dense_vq_dtype_tag(static_cast<int>(bits));
    const int vector = static_cast<int>(codebook.size(1));
    const int64_t expected_bits = rows * blocks * bits;
    TORCH_CHECK(
        tag >= 0 && rows > 0 && blocks > 0 && vector > 0 &&
        expected_bits % 8 == 0 && packed.numel() == expected_bits / 8 &&
        input.size(1) == blocks * vector,
        "dense packed VQ metadata mismatch");
    TORCH_CHECK(
        codebook.size(0) <= (int64_t{1} << bits),
        "dense packed width cannot represent the codebook");
    const int tokens = static_cast<int>(input.size(0));
    auto output = torch::empty({tokens, rows}, input.options());
    const dim3 block(32, CCCP_DENSE_VQ_ROWS_PER_BLOCK);
    const dim3 grid(
        static_cast<unsigned>((rows + CCCP_DENSE_VQ_ROWS_PER_BLOCK - 1) /
                              CCCP_DENSE_VQ_ROWS_PER_BLOCK),
        static_cast<unsigned>(tokens));
    const size_t input_bytes = static_cast<size_t>(input.size(1)) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    // 48 KiB is portable across released CUDA and Windows HIP targets. Wide
    // down projections read the input row from L2 instead of requiring a
    // device-specific dynamic shared-memory opt-in.
    if (input_bytes <= 48 * 1024) {
        dense_vq_gemv_packed_kernel<true><<<grid, block, input_bytes, stream>>>(
            input.data_ptr<float>(), packed.data_ptr<uint8_t>(),
            codebook.data_ptr<float>(), output.data_ptr<float>(), tokens,
            static_cast<int>(rows), static_cast<int>(blocks), vector, tag);
    } else {
        dense_vq_gemv_packed_kernel<false><<<grid, block, 0, stream>>>(
            input.data_ptr<float>(), packed.data_ptr<uint8_t>(),
            codebook.data_ptr<float>(), output.data_ptr<float>(), tokens,
            static_cast<int>(rows), static_cast<int>(blocks), vector, tag);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// GGUF-style compact Decode.  One warp owns one output row, reads packed VQ
// codes, looks up the small E4M3 codebook, and accumulates the dot product
// without ever materializing a weight row.  Multiple projections which share
// the same activation are described by one immutable device metadata table so
// QKV or Gate/Up execute in a single kernel launch.
constexpr int CCCP_DENSE_VQ_GROUP_META_FIELDS = 8;

__global__ void dense_vq_gemv_grouped_fp8_codebook_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int64_t* __restrict__ metadata,
    __nv_bfloat16* __restrict__ output,
    const int projections,
    const int total_rows,
    const int columns)
{
    const int global_row =
        blockIdx.x * CCCP_DENSE_VQ_ROWS_PER_BLOCK + threadIdx.y;
    extern __shared__ __nv_bfloat16 grouped_staged_input[];
    for (int column = threadIdx.y * 32 + threadIdx.x;
         column < columns;
         column += 32 * CCCP_DENSE_VQ_ROWS_PER_BLOCK)
        grouped_staged_input[column] = input[column];
    __syncthreads();
    if (global_row >= total_rows) return;

    int projection = 0;
    int output_offset = 0;
    int projection_rows = 0;
    #pragma unroll 1
    for (int candidate = 0; candidate < projections; ++candidate) {
        const int64_t* item =
            metadata + candidate * CCCP_DENSE_VQ_GROUP_META_FIELDS;
        const int candidate_offset = static_cast<int>(item[2]);
        const int candidate_rows = static_cast<int>(item[3]);
        if (global_row >= candidate_offset &&
            global_row < candidate_offset + candidate_rows) {
            projection = candidate;
            output_offset = candidate_offset;
            projection_rows = candidate_rows;
            break;
        }
    }
    const int64_t* item =
        metadata + projection * CCCP_DENSE_VQ_GROUP_META_FIELDS;
    const auto* packed = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(item[0]));
    const auto* codebook = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(item[1]));
    const int blocks = static_cast<int>(item[4]);
    const int vector = static_cast<int>(item[5]);
    const int dtype_tag = static_cast<int>(item[6]);
    const auto* scale_pointer = reinterpret_cast<const float*>(
        static_cast<uintptr_t>(item[7]));
    const float codebook_scale = *scale_pointer;
    const int row = global_row - output_offset;
    if (row < 0 || row >= projection_rows) return;

    const int64_t address = static_cast<int64_t>(
        reinterpret_cast<uintptr_t>(packed));
    float sum = 0.0f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        const int code = routed_index_value(
            address, dtype_tag,
            static_cast<long>(row) * blocks + block);
        const auto* code_row =
            codebook + static_cast<long>(code) * vector;
        const auto* input_row = grouped_staged_input + block * vector;
        float partial = 0.0f;
        #pragma unroll
        for (int component = 0; component < 16; component += 4) {
            if (component >= vector) break;
            if (component + 3 < vector) {
                __nv_fp8x4_e4m3 packed_code;
                packed_code.__x = __ldg(reinterpret_cast<const uint32_t*>(
                    code_row + component));
                const float4 code_value = static_cast<float4>(packed_code);
                const float2 input01 = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(
                        input_row + component));
                const float2 input23 = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(
                        input_row + component + 2));
                partial = fmaf(
                    code_value.x * codebook_scale, input01.x, partial);
                partial = fmaf(
                    code_value.y * codebook_scale, input01.y, partial);
                partial = fmaf(
                    code_value.z * codebook_scale, input23.x, partial);
                partial = fmaf(
                    code_value.w * codebook_scale, input23.y, partial);
            } else {
                __nv_fp8x2_e4m3 packed_code;
                packed_code.__x = __ldg(reinterpret_cast<const uint16_t*>(
                    code_row + component));
                const float2 code_value = static_cast<float2>(packed_code);
                const float2 input_value = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(
                        input_row + component));
                partial = fmaf(
                    code_value.x * codebook_scale, input_value.x, partial);
                partial = fmaf(
                    code_value.y * codebook_scale, input_value.y, partial);
            }
        }
        sum += partial;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset, 32);
    if (threadIdx.x == 0)
        output[global_row] = __float2bfloat16_rn(sum);
}

torch::Tensor dense_vq_gemv_grouped_fp8_codebook(
    torch::Tensor input,
    torch::Tensor metadata,
    int64_t total_rows)
{
    TORCH_CHECK(
        input.is_cuda() && metadata.is_cuda(),
        "Dense VQ grouped Decode operands must be CUDA tensors");
    TORCH_CHECK(
        input.scalar_type() == at::kBFloat16 && input.dim() == 2 &&
        input.size(0) == 1 && metadata.scalar_type() == at::kLong &&
        metadata.dim() == 2 &&
        metadata.size(1) == CCCP_DENSE_VQ_GROUP_META_FIELDS,
        "Dense VQ grouped Decode requires BF16 [1,C] and I64 [P,8]");
    TORCH_CHECK(
        input.is_contiguous() && metadata.is_contiguous() &&
        input.get_device() == metadata.get_device(),
        "Dense VQ grouped Decode operands must be contiguous and colocated");
    const int columns = static_cast<int>(input.size(1));
    const int projections = static_cast<int>(metadata.size(0));
    TORCH_CHECK(
        columns > 0 && projections > 0 && total_rows > 0,
        "Dense VQ grouped Decode metadata is empty");
    const size_t input_bytes = static_cast<size_t>(columns) * sizeof(__nv_bfloat16);
    TORCH_CHECK(
        input_bytes <= 48 * 1024,
        "Dense VQ grouped Decode input exceeds the shared-memory contract");
    auto output = torch::empty(
        {1, total_rows}, input.options().dtype(at::kBFloat16));
    const dim3 block(32, CCCP_DENSE_VQ_ROWS_PER_BLOCK);
    const dim3 grid(static_cast<unsigned>(
        (total_rows + CCCP_DENSE_VQ_ROWS_PER_BLOCK - 1) /
        CCCP_DENSE_VQ_ROWS_PER_BLOCK));
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    dense_vq_gemv_grouped_fp8_codebook_kernel<<<
        grid, block, input_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        metadata.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        projections,
        static_cast<int>(total_rows),
        columns);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// Compact-codebook Decode probe and reusable single-projection primitive.
// The packed VQ indices remain in their archive representation.  Each warp
// owns one output row, converts only the referenced E4M3 codewords in
// registers, and immediately accumulates the dot product.  No full FP8/BF16
// weight row is ever allocated or written to global memory.
__global__ void dense_vq_gemv_packed_fp8_codebook_kernel(
    const __nv_bfloat16* __restrict__ input,
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ codebook,
    float* __restrict__ output,
    const int rows,
    const int blocks,
    const int vector,
    const int dtype_tag,
    const float codebook_scale)
{
    const int row =
        blockIdx.x * CCCP_DENSE_VQ_ROWS_PER_BLOCK + threadIdx.y;
    extern __shared__ __nv_bfloat16 compact_fp8_staged_input[];
    const int columns = blocks * vector;
    for (int column = threadIdx.y * 32 + threadIdx.x;
         column < columns;
         column += 32 * CCCP_DENSE_VQ_ROWS_PER_BLOCK)
        compact_fp8_staged_input[column] = input[column];
    __syncthreads();
    if (row >= rows) return;

    const int64_t address = static_cast<int64_t>(
        reinterpret_cast<uintptr_t>(packed));
    float sum = 0.0f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        const int code = routed_index_value(
            address,
            dtype_tag,
            static_cast<long>(row) * blocks + block);
        const uint8_t* code_row =
            codebook + static_cast<long>(code) * vector;
        const __nv_bfloat16* input_row =
            compact_fp8_staged_input + block * vector;
        float partial = 0.0f;
        #pragma unroll
        for (int component = 0; component < 16; component += 4) {
            if (component >= vector) break;
            __nv_fp8x4_e4m3 packed_code;
            packed_code.__x = __ldg(reinterpret_cast<const uint32_t*>(
                code_row + component));
            const float4 code_value = static_cast<float4>(packed_code);
            const float2 input01 = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(
                    input_row + component));
            const float2 input23 = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(
                    input_row + component + 2));
            partial = fmaf(
                code_value.x * codebook_scale, input01.x, partial);
            partial = fmaf(
                code_value.y * codebook_scale, input01.y, partial);
            partial = fmaf(
                code_value.z * codebook_scale, input23.x, partial);
            partial = fmaf(
                code_value.w * codebook_scale, input23.y, partial);
        }
        sum += partial;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset, 32);
    if (threadIdx.x == 0) output[row] = sum;
}

torch::Tensor dense_vq_gemv_packed_fp8_codebook(
    torch::Tensor input,
    torch::Tensor packed,
    torch::Tensor codebook,
    double codebook_scale,
    int64_t rows,
    int64_t blocks,
    int64_t bits)
{
#if defined(__HIP_PLATFORM_AMD__)
    TORCH_CHECK(false, "compact E4M3 codebook GEMV is unavailable on HIP");
#else
    TORCH_CHECK(
        input.is_cuda() && packed.is_cuda() && codebook.is_cuda() &&
        input.scalar_type() == at::kBFloat16 && input.dim() == 2 &&
        input.size(0) == 1 && input.is_contiguous() &&
        packed.scalar_type() == at::kByte && packed.dim() == 1 &&
        packed.is_contiguous() &&
        codebook.scalar_type() == at::ScalarType::Float8_e4m3fn &&
        codebook.dim() == 2 && codebook.is_contiguous(),
        "compact E4M3 codebook GEMV requires CUDA BF16 [1,C], packed "
        "uint8, and E4M3 [K,D]");
    const int tag = dense_vq_dtype_tag(static_cast<int>(bits));
    const int vector = static_cast<int>(codebook.size(1));
    const int64_t columns = blocks * vector;
    const int64_t expected_bits = rows * blocks * bits;
    TORCH_CHECK(
        tag >= 0 && rows > 0 && blocks > 0 &&
        (vector == 4 || vector == 8 || vector == 16) &&
        expected_bits % 8 == 0 &&
        packed.numel() == expected_bits / 8 &&
        input.size(1) == columns &&
        codebook.size(0) > 0 &&
        codebook.size(0) <= (int64_t{1} << bits) &&
        std::isfinite(codebook_scale) && codebook_scale > 0.0,
        "compact E4M3 codebook GEMV metadata mismatch");
    const int device = input.get_device();
    TORCH_CHECK(
        packed.get_device() == device && codebook.get_device() == device,
        "compact E4M3 codebook GEMV tensors must share one device");
    const size_t input_bytes =
        static_cast<size_t>(columns) * sizeof(__nv_bfloat16);
    TORCH_CHECK(
        input_bytes <= 48 * 1024,
        "compact E4M3 codebook GEMV input exceeds shared-memory contract");
    auto output = torch::empty(
        {1, rows}, input.options().dtype(at::kFloat));
    const dim3 block(32, CCCP_DENSE_VQ_ROWS_PER_BLOCK);
    const dim3 grid(static_cast<unsigned>(
        (rows + CCCP_DENSE_VQ_ROWS_PER_BLOCK - 1) /
        CCCP_DENSE_VQ_ROWS_PER_BLOCK));
    dense_vq_gemv_packed_fp8_codebook_kernel<<<
        grid,
        block,
        input_bytes,
        at::cuda::getCurrentCUDAStream(device)>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            packed.data_ptr<uint8_t>(),
            static_cast<const uint8_t*>(codebook.data_ptr()),
            output.data_ptr<float>(),
            static_cast<int>(rows),
            static_cast<int>(blocks),
            vector,
            tag,
            static_cast<float>(codebook_scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
#endif
}

// Compact-codebook Q8 Decode primitive.  This is deliberately different
// from an INT8 weight image: only the small VQ codebook is quantized once.
// The packed expert indices remain authoritative, the activation is
// quantized once per VQ block into CTA shared memory, and each referenced
// codeword is consumed immediately by DP4A.  No decoded/expanded weight row
// or matrix exists in global memory.
template <int DTYPE_TAG>
__device__ __forceinline__ int routed_index_value_t(
    const int64_t address,
    const long offset,
    const long index_count)
{
    const uintptr_t raw = static_cast<uintptr_t>(address);
    if constexpr (DTYPE_TAG == 0)
        return static_cast<int>(
            reinterpret_cast<const uint8_t*>(raw)[offset]);
    if constexpr (DTYPE_TAG == 1)
        return static_cast<int>(
            reinterpret_cast<const uint16_t*>(raw)[offset]);
    const auto* bytes = reinterpret_cast<const uint8_t*>(raw);
    if constexpr (DTYPE_TAG == 2) {
        const long base = (offset >> 1) * 3;
        if ((offset & 1) == 0)
            return static_cast<int>(
                bytes[base] | ((bytes[base + 1] & 0x0f) << 8));
        return static_cast<int>(
            (bytes[base + 1] >> 4) | (bytes[base + 2] << 4));
    }
    if constexpr (DTYPE_TAG == 5) {
        const long bit_offset = offset * 9;
        const long base = bit_offset >> 3;
        const int shift = static_cast<int>(bit_offset & 7);
        const unsigned word =
            static_cast<unsigned>(bytes[base]) |
            (static_cast<unsigned>(bytes[base + 1]) << 8);
        return static_cast<int>((word >> shift) & 0x1ffu);
    }
    if constexpr (DTYPE_TAG >= 6 && DTYPE_TAG <= 8) {
        constexpr int bits = 2 * DTYPE_TAG - 1;
        const long bit_offset = offset * bits;
        const long base = bit_offset >> 3;
        const int shift = static_cast<int>(bit_offset & 7);
        unsigned word =
            static_cast<unsigned>(bytes[base]) |
            (static_cast<unsigned>(bytes[base + 1]) << 8);
        if (shift + bits > 16)
            word |= static_cast<unsigned>(bytes[base + 2]) << 16;
        return static_cast<int>(
            (word >> shift) & ((1u << bits) - 1u));
    }
    if constexpr (DTYPE_TAG == 4) {
        const long group = offset >> 2;
        const long base = (offset >> 2) * 5;
        const long group_count = (index_count + 3) >> 2;
        // Four p10 indices occupy five bytes.  Read their common 40-bit word
        // through two naturally aligned 32-bit loads instead of rebuilding
        // it with five dependent byte loads for every output row.  The last
        // group keeps the exact scalar path so no padding/over-read contract
        // is imposed on a packed archive.
        if ((raw & 3u) == 0 && group + 1 < group_count) {
            const uintptr_t absolute_address = raw + base;
            const uintptr_t aligned_address = absolute_address & ~uintptr_t{3};
            const int byte_shift = static_cast<int>(
                (absolute_address & uintptr_t{3}) * 8);
            const unsigned long long low = __ldg(reinterpret_cast<const uint32_t*>(
                aligned_address));
            const unsigned long long high = __ldg(reinterpret_cast<const uint32_t*>(
                aligned_address + 4));
            const unsigned long long word =
                (low | (high << 32)) >> byte_shift;
            return static_cast<int>(
                (word >> (10 * (offset & 3))) & 0x3ffu);
        }
        unsigned long long word = 0;
        #pragma unroll
        for (int byte = 0; byte < 5; ++byte)
            word |= static_cast<unsigned long long>(bytes[base + byte])
                    << (8 * byte);
        return static_cast<int>(
            (word >> (10 * (offset & 3))) & 0x3ffu);
    }
    const long base = (offset >> 2) * 7;
    unsigned long long word = 0;
    #pragma unroll
    for (int byte = 0; byte < 7; ++byte)
        word |= static_cast<unsigned long long>(bytes[base + byte])
                << (8 * byte);
    return static_cast<int>(
        (word >> (14 * (offset & 3))) & 0x3fffu);
}

__device__ __forceinline__ void routed_p10_index_quad(
    const int64_t address,
    const long offset,
    const long index_count,
    int (&codes)[4])
{
    const uintptr_t raw = static_cast<uintptr_t>(address);
    const long group = offset >> 2;
    const long group_count = (index_count + 3) >> 2;
    if ((offset & 3) == 0 && offset + 3 < index_count &&
        (raw & 3u) == 0 && group + 1 < group_count) {
        const uintptr_t absolute_address = raw + group * 5;
        const uintptr_t aligned_address = absolute_address & ~uintptr_t{3};
        const int byte_shift = static_cast<int>(
            (absolute_address & uintptr_t{3}) * 8);
        const unsigned long long low = __ldg(
            reinterpret_cast<const uint32_t*>(aligned_address));
        const unsigned long long high = __ldg(
            reinterpret_cast<const uint32_t*>(aligned_address + 4));
        const unsigned long long word =
            (low | (high << 32)) >> byte_shift;
        codes[0] = static_cast<int>(word & 0x3ffu);
        codes[1] = static_cast<int>((word >> 10) & 0x3ffu);
        codes[2] = static_cast<int>((word >> 20) & 0x3ffu);
        codes[3] = static_cast<int>((word >> 30) & 0x3ffu);
        return;
    }
    #pragma unroll
    for (int item = 0; item < 4; ++item)
        codes[item] = routed_index_value_t<4>(
            address, offset + item, index_count);
}

template <int VECTOR>
__device__ __forceinline__ int vq_block_dot_routed_q8_codebook_i32_t(
    const int8_t* __restrict__ codebook,
    const int8_t* __restrict__ input)
{
    if constexpr (VECTOR == 16) {
        const uint4 code = __ldg(
            reinterpret_cast<const uint4*>(codebook));
        const uint4 value = *reinterpret_cast<const uint4*>(input);
        int sum = __dp4a(
            static_cast<int>(code.x), static_cast<int>(value.x), 0);
        sum = __dp4a(
            static_cast<int>(code.y), static_cast<int>(value.y), sum);
        sum = __dp4a(
            static_cast<int>(code.z), static_cast<int>(value.z), sum);
        return __dp4a(
            static_cast<int>(code.w), static_cast<int>(value.w), sum);
    }
    if constexpr (VECTOR == 8) {
        const uint2 code = __ldg(
            reinterpret_cast<const uint2*>(codebook));
        const uint2 value = *reinterpret_cast<const uint2*>(input);
        const int first = __dp4a(
            static_cast<int>(code.x), static_cast<int>(value.x), 0);
        return __dp4a(
            static_cast<int>(code.y), static_cast<int>(value.y), first);
    }
    const int code = __ldg(reinterpret_cast<const int*>(codebook));
    const int value = *reinterpret_cast<const int*>(input);
    return __dp4a(code, value, 0);
}

__device__ __forceinline__ int vq_block_dot_routed_q8_codebook_i32(
    const int8_t* __restrict__ codebook,
    const int8_t* __restrict__ input,
    const int vector)
{
    if (vector == 16)
        return vq_block_dot_routed_q8_codebook_i32_t<16>(codebook, input);
    if (vector == 8)
        return vq_block_dot_routed_q8_codebook_i32_t<8>(codebook, input);
    return vq_block_dot_routed_q8_codebook_i32_t<4>(codebook, input);
}

#if !defined(__HIP_PLATFORM_AMD__)

__device__ __forceinline__ int compact_q8_bits_from_tag(const int tag)
{
    if (tag == 0) return 8;
    if (tag == 1) return 16;
    if (tag == 2) return 12;
    if (tag == 3) return 14;
    if (tag == 4) return 10;
    if (tag == 5) return 9;
    if (tag == 6) return 11;
    if (tag == 7) return 13;
    if (tag == 8) return 15;
    return 0;
}

// Route-local codebooks are only a few MiB even though the model-wide Q8
// codebook set is larger than H20's L2.  The cache-control stream issues L2
// prefetch hints for the selected layer while the default stream evaluates
// the independent shared expert.  Duplicate semantic codebooks in one route
// are skipped; packed expert bytes and expanded weight residency are unchanged.
__global__ void compact_q8_codebook_l2_prefetch_kernel(
    const int64_t* __restrict__ metadata,
    const int expert_count)
{
    const int item = static_cast<int>(blockIdx.x);
    const int projection = item / expert_count;
    const int expert = item - projection * expert_count;
    if (projection >= 3 || expert >= expert_count) return;
    const int pointer_row = projection * 5 + 1;
    const int vector_row = projection * 5 + 3;
    const int tag_row = projection * 5 + 4;
    const int64_t pointer = metadata[
        static_cast<long>(pointer_row) * expert_count + expert];
    const int vector = static_cast<int>(metadata[
        static_cast<long>(vector_row) * expert_count + expert]);
    const int tag = static_cast<int>(metadata[
        static_cast<long>(tag_row) * expert_count + expert]);
    if (pointer == 0 || (vector != 4 && vector != 8 && vector != 16)) return;
    // A route may select several experts sharing the same projection codebook.
    // Let only its first occurrence populate L2.
    for (int prior = 0; prior < expert; ++prior) {
        if (metadata[
                static_cast<long>(pointer_row) * expert_count + prior] ==
            pointer)
            return;
    }
    const int bits = compact_q8_bits_from_tag(tag);
    if (bits == 0) return;
    const long bytes = (long{1} << bits) * vector;
    const auto* base = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(pointer));
    // One hint per cache line.  It is intentionally a hint rather than a
    // compulsory load so demand traffic from the concurrent shared branch
    // retains scheduler priority.
    for (long offset = static_cast<long>(threadIdx.x) * 128;
         offset < bytes;
         offset += static_cast<long>(blockDim.x) * 128) {
        const void* address = base + offset;
        asm volatile("prefetch.global.L2 [%0];" : : "l"(address));
    }
}
#endif

__global__ void dense_vq_gemv_packed_q8_codebook_kernel(
    const __nv_bfloat16* __restrict__ input,
    const uint8_t* __restrict__ packed,
    const int8_t* __restrict__ codebook,
    float* __restrict__ output,
    const int rows,
    const int blocks,
    const int vector,
    const int dtype_tag,
    const float codebook_scale)
{
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    const int row =
        blockIdx.x * CCCP_DENSE_VQ_ROWS_PER_BLOCK + threadIdx.y;
    const int columns = blocks * vector;
    extern __shared__ unsigned char compact_q8_raw[];
    auto* quantized_input = reinterpret_cast<int8_t*>(compact_q8_raw);
    auto* input_scales = reinterpret_cast<float*>(
        compact_q8_raw + ((columns + 15) & ~15));

    for (int block = linear_thread; block < blocks;
         block += 32 * CCCP_DENSE_VQ_ROWS_PER_BLOCK) {
        const __nv_bfloat16* source = input + block * vector;
        float absolute_max = 0.0f;
        #pragma unroll
        for (int component = 0; component < 16; ++component) {
            if (component >= vector) break;
            absolute_max = fmaxf(
                absolute_max,
                fabsf(__bfloat162float(source[component])));
        }
        const float scale = fmaxf(absolute_max, 1.0e-12f) / 127.0f;
        input_scales[block] = scale;
        #pragma unroll
        for (int component = 0; component < 16; ++component) {
            if (component >= vector) break;
            const float value = __bfloat162float(source[component]) / scale;
            quantized_input[block * vector + component] =
                static_cast<int8_t>(__float2int_rn(
                    fminf(fmaxf(value, -127.0f), 127.0f)));
        }
    }
    __syncthreads();
    if (row >= rows) return;

    const int64_t address = static_cast<int64_t>(
        reinterpret_cast<uintptr_t>(packed));
    float sum = 0.0f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        const int code = routed_index_value(
            address,
            dtype_tag,
            static_cast<long>(row) * blocks + block);
        const int integer_sum = vq_block_dot_routed_q8_codebook_i32(
            codebook + static_cast<long>(code) * vector,
            quantized_input + block * vector,
            vector);
        sum += static_cast<float>(integer_sum) *
            (codebook_scale * input_scales[block]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset, 32);
    if (threadIdx.x == 0) output[row] = sum;
}

torch::Tensor dense_vq_gemv_packed_q8_codebook(
    torch::Tensor input,
    torch::Tensor packed,
    torch::Tensor codebook,
    double codebook_scale,
    int64_t rows,
    int64_t blocks,
    int64_t bits)
{
#if defined(__HIP_PLATFORM_AMD__)
    TORCH_CHECK(false, "compact Q8 codebook GEMV is unavailable on HIP");
#else
    TORCH_CHECK(
        input.is_cuda() && packed.is_cuda() && codebook.is_cuda() &&
        input.scalar_type() == at::kBFloat16 && input.dim() == 2 &&
        input.size(0) == 1 && input.is_contiguous() &&
        packed.scalar_type() == at::kByte && packed.dim() == 1 &&
        packed.is_contiguous() &&
        codebook.scalar_type() == at::kChar && codebook.dim() == 2 &&
        codebook.is_contiguous(),
        "compact Q8 codebook GEMV requires CUDA BF16 [1,C], packed "
        "uint8, and INT8 [K,D]");
    const int tag = dense_vq_dtype_tag(static_cast<int>(bits));
    const int vector = static_cast<int>(codebook.size(1));
    const int64_t columns = blocks * vector;
    const int64_t expected_bits = rows * blocks * bits;
    TORCH_CHECK(
        tag >= 0 && rows > 0 && blocks > 0 &&
        (vector == 4 || vector == 8 || vector == 16) &&
        expected_bits % 8 == 0 &&
        packed.numel() == expected_bits / 8 &&
        input.size(1) == columns && codebook.size(0) > 0 &&
        codebook.size(0) <= (int64_t{1} << bits) &&
        std::isfinite(codebook_scale) && codebook_scale > 0.0,
        "compact Q8 codebook GEMV metadata mismatch");
    const int device = input.get_device();
    TORCH_CHECK(
        packed.get_device() == device && codebook.get_device() == device,
        "compact Q8 codebook GEMV tensors must share one device");
    const size_t quantized_bytes = static_cast<size_t>(columns);
    const size_t scale_offset = (quantized_bytes + 15) & ~size_t{15};
    const size_t shared_bytes =
        scale_offset + static_cast<size_t>(blocks) * sizeof(float);
    TORCH_CHECK(
        shared_bytes <= 48 * 1024,
        "compact Q8 codebook GEMV input exceeds shared-memory contract");
    auto output = torch::empty(
        {1, rows}, input.options().dtype(at::kFloat));
    const dim3 block(32, CCCP_DENSE_VQ_ROWS_PER_BLOCK);
    const dim3 grid(static_cast<unsigned>(
        (rows + CCCP_DENSE_VQ_ROWS_PER_BLOCK - 1) /
            CCCP_DENSE_VQ_ROWS_PER_BLOCK));
    dense_vq_gemv_packed_q8_codebook_kernel<<<
        grid,
        block,
        shared_bytes,
        at::cuda::getCurrentCUDAStream(device)>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            packed.data_ptr<uint8_t>(),
            codebook.data_ptr<int8_t>(),
            output.data_ptr<float>(),
            static_cast<int>(rows),
            static_cast<int>(blocks),
            vector,
            tag,
            static_cast<float>(codebook_scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
#endif
}

__global__ void dense_vq_dequant_packed_kernel(
    const uint8_t* __restrict__ packed,
    const float* __restrict__ codebook,
    __nv_bfloat16* __restrict__ output,
    const long selected_rows,
    const long blocks,
    const int vector,
    const int dtype_tag,
    const int64_t* __restrict__ row_ids)
{
    const long item = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long count = selected_rows * blocks;
    if (item >= count) return;
    const long selected_row = item / blocks;
    const long block = item - selected_row * blocks;
    const long source_row = row_ids == nullptr ? selected_row : row_ids[selected_row];
    const int64_t address = static_cast<int64_t>(
        reinterpret_cast<uintptr_t>(packed));
    const int code = routed_index_value(
        address, dtype_tag, source_row * blocks + block);
    const float* source = codebook + static_cast<long>(code) * vector;
    __nv_bfloat16* destination = output +
        (selected_row * blocks + block) * vector;
    for (int component = 0; component < vector; ++component)
        destination[component] = __float2bfloat16_rn(source[component]);
}


__global__ void dense_vq_expand_fp8_packed_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ fp8_codebook,
    uint8_t* __restrict__ output,
    const long selected_rows,
    const long row_start,
    const long blocks,
    const int vector,
    const int dtype_tag,
    const int64_t* __restrict__ row_ids)
{
    const long item = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long count = selected_rows * blocks;
    if (item >= count) return;
    const long selected_row = item / blocks;
    const long block = item - selected_row * blocks;
    const long source_row = row_ids == nullptr
        ? row_start + selected_row
        : row_ids[selected_row];
    const int64_t address = static_cast<int64_t>(
        reinterpret_cast<uintptr_t>(packed));
    const int code = routed_index_value(
        address, dtype_tag, source_row * blocks + block);
    const uint8_t* source = fp8_codebook + static_cast<long>(code) * vector;
    uint8_t* destination = output +
        (selected_row * blocks + block) * vector;
    for (int component = 0; component < vector; ++component)
        destination[component] = source[component];
}

std::vector<torch::Tensor> dense_vq_quantize_fp8_codebook(
    torch::Tensor codebook)
{
    TORCH_CHECK(
        codebook.is_cuda() && codebook.scalar_type() == at::kFloat &&
        codebook.dim() == 2 && codebook.is_contiguous() &&
        codebook.numel() > 0,
        "Dense VQ FP8 codebook conversion requires contiguous CUDA FP32");
    auto fp8_options = codebook.options().dtype(
        at::ScalarType::Float8_e4m3fn);
    auto fp8_codebook = torch::empty(codebook.sizes(), fp8_options);
    auto scale = torch::empty(
        {1, 1}, codebook.options().dtype(at::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        scale.data_ptr<float>(), 0, sizeof(float), stream));
    const int64_t items = codebook.numel();
    const int reduction_blocks = static_cast<int>(std::min<int64_t>(
        1024, (items + 255) / 256));
    dense_fp8_tensor_amax_kernel<float><<<
        reduction_blocks, 256, 0, stream>>>(
            codebook.data_ptr<float>(),
            reinterpret_cast<unsigned int*>(scale.data_ptr<float>()),
            items);
    dense_fp8_quantize_tensor_kernel<float><<<
        reduction_blocks, 256, 0, stream>>>(
            codebook.data_ptr<float>(),
            static_cast<uint8_t*>(fp8_codebook.data_ptr()),
            scale.data_ptr<float>(),
            items);
    dense_fp8_finalize_tensor_scale_kernel<<<1, 1, 0, stream>>>(
        scale.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {fp8_codebook, scale};
}

std::vector<torch::Tensor> dense_vq_dequant_fp8_packed(
    torch::Tensor packed,
    torch::Tensor codebook,
    int64_t rows,
    int64_t blocks,
    int64_t bits,
    torch::Tensor row_ids)
{
    TORCH_CHECK(
        packed.is_cuda() && codebook.is_cuda() && row_ids.is_cuda(),
        "Dense VQ FP8 conversion operands must be CUDA tensors");
    TORCH_CHECK(
        packed.scalar_type() == at::kByte && packed.dim() == 1 &&
        codebook.scalar_type() == at::kFloat && codebook.dim() == 2 &&
        row_ids.scalar_type() == at::kLong && row_ids.dim() == 1 &&
        packed.is_contiguous() && codebook.is_contiguous() &&
        row_ids.is_contiguous(),
        "Dense VQ FP8 conversion operand layout mismatch");
    const int tag = dense_vq_dtype_tag(static_cast<int>(bits));
    const int vector = static_cast<int>(codebook.size(1));
    const int64_t entries = codebook.size(0);
    const int64_t expected_bits = rows * blocks * bits;
    TORCH_CHECK(
        tag >= 0 && rows > 0 && blocks > 0 && vector > 0 &&
        expected_bits % 8 == 0 && packed.numel() == expected_bits / 8 &&
        entries > 0 && entries <= (int64_t{1} << bits),
        "Dense VQ FP8 conversion metadata mismatch");
    const int64_t selected = row_ids.numel() == 0 ? rows : row_ids.numel();
    auto converted_codebook = dense_vq_quantize_fp8_codebook(codebook);
    auto fp8_codebook = converted_codebook[0];
    auto scale = converted_codebook[1];
    auto output = torch::empty(
        {selected, blocks * vector}, fp8_codebook.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    // A VQ matrix only contains values from its codebook. Quantize that
    // compact table once, then expand byte-sized E4M3 values while decoding
    // the packed indices. This never materializes the old full BF16 matrix.
    const int64_t count = selected * blocks;
    dense_vq_expand_fp8_packed_kernel<<<
        static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
            packed.data_ptr<uint8_t>(),
            static_cast<const uint8_t*>(fp8_codebook.data_ptr()),
            static_cast<uint8_t*>(output.data_ptr()),
            selected,
            0,
            blocks,
            vector,
            tag,
            row_ids.numel() ? row_ids.data_ptr<int64_t>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, scale};
}

__global__ void dense_vq_compile_int4_g64_kernel(
    const uint8_t* __restrict__ packed,
    const float* __restrict__ codebook,
    uint8_t* __restrict__ output,
    __half* __restrict__ scales,
    const int rows,
    const int blocks,
    const int vector,
    const int dtype_tag,
    const int groups)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int64_t address = static_cast<int64_t>(
        reinterpret_cast<uintptr_t>(packed));
    const int columns = blocks * vector;
    for (int group = warp; group < groups; group += warps) {
        const int first_column = group * 64 + lane * 2;
        const int first_block = first_column / vector;
        const int first_component = first_column - first_block * vector;
        const int second_column = first_column + 1;
        const int second_block = second_column / vector;
        const int second_component = second_column - second_block * vector;
        const int first_code = routed_index_value(
            address, dtype_tag,
            static_cast<long>(row) * blocks + first_block);
        const int second_code = routed_index_value(
            address, dtype_tag,
            static_cast<long>(row) * blocks + second_block);
        const float first = codebook[
            static_cast<long>(first_code) * vector + first_component];
        const float second = codebook[
            static_cast<long>(second_code) * vector + second_component];
        float signed_max = fabsf(first) >= fabsf(second) ? first : second;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_down_sync(
                0xffffffffu, signed_max, offset, 32);
            if (fabsf(other) > fabsf(signed_max)) signed_max = other;
        }
        signed_max = __shfl_sync(0xffffffffu, signed_max, 0, 32);
        float scale = signed_max == 0.0f ? 0.0f : signed_max / -8.0f;
        const __half rounded_scale = __float2half_rn(scale);
        scale = __half2float(rounded_scale);
        const float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
        const int low = max(
            0, min(15, __float2int_rn(first * inverse + 8.0f)));
        const int high = max(
            0, min(15, __float2int_rn(second * inverse + 8.0f)));
        output[static_cast<long>(row) * (columns / 2) + group * 32 + lane] =
            static_cast<uint8_t>(low | (high << 4));
        if (lane == 0)
            scales[static_cast<long>(row) * groups + group] = rounded_scale;
    }
}

std::vector<torch::Tensor> dense_vq_compile_int4_g64(
    torch::Tensor packed,
    torch::Tensor codebook,
    int64_t rows,
    int64_t blocks,
    int64_t bits)
{
    TORCH_CHECK(
        packed.is_cuda() && codebook.is_cuda() &&
        packed.scalar_type() == at::kByte && packed.dim() == 1 &&
        codebook.scalar_type() == at::kFloat && codebook.dim() == 2 &&
        packed.is_contiguous() && codebook.is_contiguous(),
        "Dense VQ INT4 compilation requires CUDA packed/codebook tensors");
    const int tag = dense_vq_dtype_tag(static_cast<int>(bits));
    const int vector = static_cast<int>(codebook.size(1));
    const int64_t columns = blocks * vector;
    const int64_t expected_bits = rows * blocks * bits;
    TORCH_CHECK(
        tag >= 0 && rows > 0 && blocks > 0 && vector > 0 &&
        columns % 64 == 0 && expected_bits % 8 == 0 &&
        packed.numel() == expected_bits / 8 &&
        codebook.size(0) <= (int64_t{1} << bits),
        "Dense VQ INT4 compilation metadata mismatch");
    const int64_t groups = columns / 64;
    auto output = torch::empty({rows, columns / 2}, packed.options());
    auto scales = torch::empty(
        {rows, groups}, packed.options().dtype(at::kHalf));
    dense_vq_compile_int4_g64_kernel<<<
        static_cast<unsigned>(rows), 256, 0,
        at::cuda::getCurrentCUDAStream()>>>(
            packed.data_ptr<uint8_t>(), codebook.data_ptr<float>(),
            output.data_ptr<uint8_t>(),
            reinterpret_cast<__half*>(scales.data_ptr<at::Half>()),
            static_cast<int>(rows), static_cast<int>(blocks), vector, tag,
            static_cast<int>(groups));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, scales};
}


torch::Tensor dense_vq_expand_fp8_tile_out(
    torch::Tensor packed,
    torch::Tensor fp8_codebook,
    int64_t rows,
    int64_t blocks,
    int64_t bits,
    int64_t row_start,
    int64_t row_count,
    torch::Tensor output)
{
    TORCH_CHECK(
        packed.is_cuda() && fp8_codebook.is_cuda() && output.is_cuda(),
        "Dense VQ FP8 tile operands must be CUDA tensors");
    TORCH_CHECK(
        packed.scalar_type() == at::kByte && packed.dim() == 1 &&
        fp8_codebook.scalar_type() == at::ScalarType::Float8_e4m3fn &&
        fp8_codebook.dim() == 2 &&
        output.scalar_type() == at::ScalarType::Float8_e4m3fn &&
        output.dim() == 2 && packed.is_contiguous() &&
        fp8_codebook.is_contiguous() && output.is_contiguous(),
        "Dense VQ FP8 tile operand layout mismatch");
    const int tag = dense_vq_dtype_tag(static_cast<int>(bits));
    const int vector = static_cast<int>(fp8_codebook.size(1));
    const int64_t expected_bits = rows * blocks * bits;
    TORCH_CHECK(
        tag >= 0 && rows > 0 && blocks > 0 && vector > 0 &&
        expected_bits % 8 == 0 && packed.numel() == expected_bits / 8 &&
        fp8_codebook.size(0) > 0 &&
        fp8_codebook.size(0) <= (int64_t{1} << bits) &&
        row_start >= 0 && row_count > 0 && row_start + row_count <= rows &&
        output.size(0) >= row_count && output.size(1) == blocks * vector,
        "Dense VQ FP8 tile metadata/workspace mismatch");
    const int device = packed.get_device();
    TORCH_CHECK(
        fp8_codebook.get_device() == device && output.get_device() == device,
        "Dense VQ FP8 tile tensors must share one device");
    const int64_t count = row_count * blocks;
    dense_vq_expand_fp8_packed_kernel<<<
        static_cast<unsigned>((count + 255) / 256), 256, 0,
        at::cuda::getCurrentCUDAStream()>>>(
            packed.data_ptr<uint8_t>(),
            static_cast<const uint8_t*>(fp8_codebook.data_ptr()),
            static_cast<uint8_t*>(output.data_ptr()),
            row_count,
            row_start,
            blocks,
            vector,
            tag,
            nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output.narrow(0, 0, row_count);
}


__global__ void dense_vq_mma_packed_m1_kernel(
    const float* __restrict__ input,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ codebook,
    float* __restrict__ output,
    const int rows,
    const int blocks,
    const int vector,
    const int dtype_tag)
{
    using namespace nvcuda;
    constexpr int tile = 16;
    const int lane = threadIdx.x;
    const int row_start = static_cast<int>(blockIdx.x) * tile;
    const int columns = blocks * vector;
    const int64_t address = static_cast<int64_t>(
        reinterpret_cast<uintptr_t>(packed));
    __shared__ __align__(32) __half a_shared[tile * tile];
    __shared__ __align__(32) __half b_shared[tile * tile];
    __shared__ __align__(32) float c_shared[tile * tile];
    wmma::fragment<wmma::matrix_a, tile, tile, tile, __half,
                   wmma::row_major> a_fragment;
    wmma::fragment<wmma::matrix_b, tile, tile, tile, __half,
                   wmma::col_major> b_fragment;
    wmma::fragment<wmma::accumulator, tile, tile, tile, float>
        accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (int item = lane; item < tile * tile; item += 32)
        a_shared[item] = __float2half_rn(0.0f);
    __syncwarp();

    for (int column_start = 0; column_start < columns;
         column_start += tile) {
        if (lane < tile)
            a_shared[lane] = __float2half_rn(input[column_start + lane]);
        const int code_blocks_per_row = tile / vector;
        const int tile_code_blocks = tile * code_blocks_per_row;
        for (int item = lane; item < tile_code_blocks; item += 32) {
            // B is KxN and column-major. Each B column is one original
            // output row. Packed VQ is decoded only into shared fragment
            // staging and is never emitted as a global weight tile.
            const int b_column = item / code_blocks_per_row;
            const int local_block = item - b_column * code_blocks_per_row;
            const int output_row = row_start + b_column;
            const int source_block = column_start / vector + local_block;
            if (output_row < rows) {
                const int code = routed_index_value(
                    address,
                    dtype_tag,
                    static_cast<long>(output_row) * blocks + source_block);
                const __half* code_row =
                    codebook + static_cast<long>(code) * vector;
                const int destination =
                    b_column * tile + local_block * vector;
                for (int component = 0; component < vector; ++component)
                    b_shared[destination + component] = code_row[component];
            } else {
                const int destination =
                    b_column * tile + local_block * vector;
                for (int component = 0; component < vector; ++component)
                    b_shared[destination + component] =
                        __float2half_rn(0.0f);
            }
        }
        __syncwarp();
        wmma::load_matrix_sync(a_fragment, a_shared, tile);
        wmma::load_matrix_sync(b_fragment, b_shared, tile);
        wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
        __syncwarp();
    }
    wmma::store_matrix_sync(
        c_shared, accumulator, tile, wmma::mem_row_major);
    __syncwarp();
    if (lane < tile && row_start + lane < rows)
        output[row_start + lane] = c_shared[lane];
}

torch::Tensor dense_vq_mma_packed_m1(
    torch::Tensor input,
    torch::Tensor packed,
    torch::Tensor codebook,
    int64_t rows,
    int64_t blocks,
    int64_t bits)
{
#if defined(__HIP_PLATFORM_AMD__)
    TORCH_CHECK(false, "Dense VQ direct MMA is unavailable on HIP");
#else
    TORCH_CHECK(
        input.is_cuda() && packed.is_cuda() && codebook.is_cuda() &&
        input.scalar_type() == at::kFloat && input.dim() == 2 &&
        input.size(0) == 1 && input.is_contiguous() &&
        packed.scalar_type() == at::kByte && packed.dim() == 1 &&
        packed.is_contiguous() &&
        codebook.scalar_type() == at::kHalf && codebook.dim() == 2 &&
        codebook.is_contiguous(),
        "Dense VQ direct MMA requires CUDA FP32 [1,C], packed uint8, "
        "and FP16 codebook");
    const int tag = dense_vq_dtype_tag(static_cast<int>(bits));
    const int vector = static_cast<int>(codebook.size(1));
    const int64_t columns = blocks * vector;
    const int64_t expected_bits = rows * blocks * bits;
    TORCH_CHECK(
        tag >= 0 && rows > 0 && rows % 16 == 0 && blocks > 0 &&
        vector > 0 && 16 % vector == 0 && columns % 16 == 0 &&
        expected_bits % 8 == 0 &&
        packed.numel() == expected_bits / 8 &&
        input.size(1) == columns &&
        codebook.size(0) <= (int64_t{1} << bits),
        "Dense VQ direct MMA metadata mismatch");
    auto output = torch::empty({1, rows}, input.options());
    dense_vq_mma_packed_m1_kernel<<<
        static_cast<unsigned>((rows + 15) / 16), 32, 0,
        at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<float>(),
            packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(codebook.data_ptr<at::Half>()),
            output.data_ptr<float>(),
            static_cast<int>(rows),
            static_cast<int>(blocks),
            vector,
            tag);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
#endif
}



torch::Tensor dense_vq_dequant_packed(
    torch::Tensor packed,
    torch::Tensor codebook,
    int64_t rows,
    int64_t blocks,
    int64_t bits,
    torch::Tensor row_ids)
{
    TORCH_CHECK(
        packed.is_cuda() && codebook.is_cuda() && row_ids.is_cuda(),
        "dense VQ dequant operands must be CUDA tensors");
    TORCH_CHECK(
        packed.scalar_type() == at::kByte && packed.dim() == 1 &&
        codebook.scalar_type() == at::kFloat && codebook.dim() == 2 &&
        row_ids.scalar_type() == at::kLong && row_ids.dim() == 1,
        "dense VQ dequant operand layout mismatch");
    TORCH_CHECK(
        packed.is_contiguous() && codebook.is_contiguous() &&
        row_ids.is_contiguous(),
        "dense VQ dequant operands must be contiguous");
    const int tag = dense_vq_dtype_tag(static_cast<int>(bits));
    const int vector = static_cast<int>(codebook.size(1));
    const int64_t expected_bits = rows * blocks * bits;
    TORCH_CHECK(
        tag >= 0 && rows > 0 && blocks > 0 && expected_bits % 8 == 0 &&
        packed.numel() == expected_bits / 8,
        "dense VQ dequant metadata mismatch");
    const int64_t selected = row_ids.numel() == 0 ? rows : row_ids.numel();
    // Row ids already originate from the model's validated tokenizer range.
    // A host-side min/max + item() here invalidates CUDA Graph capture and
    // serializes every compact decode token, so selection stays device-only.
    auto output = torch::empty(
        {selected, blocks * vector},
        torch::TensorOptions().dtype(torch::kBFloat16).device(packed.device()));
    const int64_t count = selected * blocks;
    dense_vq_dequant_packed_kernel<<<
        static_cast<unsigned>((count + 255) / 256), 256, 0,
        at::cuda::getCurrentCUDAStream()>>>(
            packed.data_ptr<uint8_t>(), codebook.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            selected, blocks, vector, tag,
            row_ids.numel() ? row_ids.data_ptr<int64_t>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// VQ -> native 8-bit execution image.
//
// Quantization belongs to the codebook, not to every reconstructed weight.
// The caller quantizes the small codebook once (E4M3 on SM89/SM90, symmetric
// INT8 on SM75/SM80/SM86). Runtime conversion consequently consists only of
// packed-index extraction plus one aligned 2/4/8/16-byte vector copy per VQ
// block. The output buffer is supplied by the caller so a bounded execution
// cache can reuse fixed storage without allocator or graph-address churn.
template <int VECTOR>
__global__ void dense_vq_expand_native8_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ quantized_codebook,
    uint8_t* __restrict__ output,
    const long selected_rows,
    const long blocks,
    const int dtype_tag,
    const int64_t* __restrict__ row_ids)
{
    const long item = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long count = selected_rows * blocks;
    if (item >= count) return;
    const long selected_row = item / blocks;
    const long block = item - selected_row * blocks;
    const long source_row = row_ids == nullptr
        ? selected_row
        : row_ids[selected_row];
    const int64_t address = static_cast<int64_t>(
        reinterpret_cast<uintptr_t>(packed));
    const int code = routed_index_value(
        address, dtype_tag, source_row * blocks + block);
    const auto* source = quantized_codebook + static_cast<long>(code) * VECTOR;
    auto* destination = output + item * VECTOR;
    if constexpr (VECTOR == 16) {
        *reinterpret_cast<uint4*>(destination) =
            *reinterpret_cast<const uint4*>(source);
    } else if constexpr (VECTOR == 8) {
        *reinterpret_cast<uint2*>(destination) =
            *reinterpret_cast<const uint2*>(source);
    } else if constexpr (VECTOR == 4) {
        *reinterpret_cast<uint32_t*>(destination) =
            *reinterpret_cast<const uint32_t*>(source);
    } else {
        *reinterpret_cast<uint16_t*>(destination) =
            *reinterpret_cast<const uint16_t*>(source);
    }
}

torch::Tensor dense_vq_expand_native8(
    torch::Tensor packed,
    torch::Tensor quantized_codebook,
    torch::Tensor output,
    int64_t rows,
    int64_t blocks,
    int64_t bits,
    torch::Tensor row_ids)
{
    TORCH_CHECK(
        packed.is_cuda() && quantized_codebook.is_cuda() &&
        output.is_cuda() && row_ids.is_cuda(),
        "VQ native8 conversion operands must be CUDA tensors");
    const auto execution_dtype = quantized_codebook.scalar_type();
    TORCH_CHECK(
        packed.scalar_type() == at::kByte && packed.dim() == 1 &&
        (execution_dtype == at::ScalarType::Float8_e4m3fn ||
         execution_dtype == at::kChar) &&
        quantized_codebook.dim() == 2 &&
        output.scalar_type() == execution_dtype && output.dim() == 2 &&
        row_ids.scalar_type() == at::kLong && row_ids.dim() == 1,
        "VQ native8 conversion requires packed U8, E4M3/INT8 codebook and "
        "matching output");
    TORCH_CHECK(
        packed.is_contiguous() && quantized_codebook.is_contiguous() &&
        output.is_contiguous() && row_ids.is_contiguous() &&
        packed.get_device() == quantized_codebook.get_device() &&
        packed.get_device() == output.get_device() &&
        packed.get_device() == row_ids.get_device(),
        "VQ native8 conversion operands must be contiguous and colocated");
    const int tag = dense_vq_dtype_tag(static_cast<int>(bits));
    const int vector = static_cast<int>(quantized_codebook.size(1));
    const int64_t expected_bits = rows * blocks * bits;
    const int64_t selected = row_ids.numel() == 0 ? rows : row_ids.numel();
    TORCH_CHECK(
        tag >= 0 && rows > 0 && blocks > 0 &&
        (vector == 2 || vector == 4 || vector == 8 || vector == 16) &&
        expected_bits % 8 == 0 && packed.numel() == expected_bits / 8 &&
        quantized_codebook.size(0) <= (int64_t{1} << bits) &&
        output.sizes() == torch::IntArrayRef({selected, blocks * vector}),
        "VQ native8 conversion metadata or output shape mismatch");
    // The owning projection validates row ids before dispatch. Avoid a
    // host-side min/max synchronization here so whole-token CUDA Graph
    // capture remains possible.
    const int64_t count = selected * blocks;
    const int grid = static_cast<int>((count + 255) / 256);
    auto stream = at::cuda::getCurrentCUDAStream(packed.get_device());
    const auto* codebook = static_cast<const uint8_t*>(
        quantized_codebook.data_ptr());
    auto* destination = static_cast<uint8_t*>(output.data_ptr());
    if (vector == 16) {
        dense_vq_expand_native8_kernel<16><<<grid, 256, 0, stream>>>(
            packed.data_ptr<uint8_t>(), codebook, destination, selected,
            blocks, tag, row_ids.numel() ? row_ids.data_ptr<int64_t>() : nullptr);
    } else if (vector == 8) {
        dense_vq_expand_native8_kernel<8><<<grid, 256, 0, stream>>>(
            packed.data_ptr<uint8_t>(), codebook, destination, selected,
            blocks, tag, row_ids.numel() ? row_ids.data_ptr<int64_t>() : nullptr);
    } else if (vector == 4) {
        dense_vq_expand_native8_kernel<4><<<grid, 256, 0, stream>>>(
            packed.data_ptr<uint8_t>(), codebook, destination, selected,
            blocks, tag, row_ids.numel() ? row_ids.data_ptr<int64_t>() : nullptr);
    } else {
        dense_vq_expand_native8_kernel<2><<<grid, 256, 0, stream>>>(
            packed.data_ptr<uint8_t>(), codebook, destination, selected,
            blocks, tag, row_ids.numel() ? row_ids.data_ptr<int64_t>() : nullptr);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// Projection-VQ runtime metadata keeps the original 15 pointer/shape rows
// first, followed by four immutable tile rows per projection:
// bit width, packed row bytes, indices per packed group, blocks per tile.
// This is a Marlin-style *view*: it changes neither the packed payload nor
// its residency and costs only 12 int64 values per selected expert.
constexpr int CCCP_PROJECTION_LEGACY_META_ROWS = 15;
constexpr int CCCP_PROJECTION_TILE_META_ROWS = 27;
constexpr int CCCP_TILE_ROWS_PER_PROJECTION = 4;
constexpr int CCCP_TILE_INDEX_GROUP = 8;

struct RoutedTileMetadata {
    int bits;
    int row_bytes;
    int group;
    int tile_blocks;
    int valid;
};

__device__ __forceinline__ int packed_tile_index_group8(
    const int64_t index_address,
    const int row_bytes,
    const int bits,
    const int row,
    const int block)
{
    const auto* bytes = reinterpret_cast<const volatile uint8_t*>(
        static_cast<uintptr_t>(index_address));
    const int lane_in_group = threadIdx.x & (CCCP_TILE_INDEX_GROUP - 1);
    const int leader = threadIdx.x - lane_in_group;
    const long base =
        static_cast<long>(row) * row_bytes +
        static_cast<long>(block >> 3) * bits;
    unsigned words[4] = {};
    if (lane_in_group == 0) {
        #pragma unroll
        for (int byte = 0; byte < 15; ++byte) {
            if (byte < bits) {
                const unsigned value = static_cast<unsigned>(bytes[base + byte]);
                const int word = byte >> 2;
                const int shift = (byte & 3) * 8;
                words[word] |= value << shift;
            }
        }
    }
    const unsigned active = __activemask();
    #pragma unroll
    for (int word = 0; word < 4; ++word)
        words[word] = __shfl_sync(active, words[word], leader);
    const int bit = lane_in_group * bits;
    const int word = bit >> 5;
    const int shift = bit & 31;
    unsigned long long window = words[word];
    if (word < 3)
        window |= static_cast<unsigned long long>(words[word + 1]) << 32;
    return static_cast<int>(
        (window >> shift) & ((1u << bits) - 1u));
}

__device__ __forceinline__ int routed_tile_index_value(
    const int64_t index_address,
    const RoutedTileMetadata& tile,
    const int dtype_tag,
    const int row,
    const int block,
    const int blocks)
{
    if (
        tile.valid && tile.group == CCCP_TILE_INDEX_GROUP &&
        block < blocks
    )
        return packed_tile_index_group8(
            index_address, tile.row_bytes, tile.bits, row, block);
    return routed_index_value(
        index_address, dtype_tag, static_cast<long>(row) * blocks + block);
}

__device__ __forceinline__ float vq_block_dot4_bf16(
    const __nv_bfloat16* cb,
    const __nv_bfloat16* x)
{
    const auto* cb2 = reinterpret_cast<const __nv_bfloat162*>(cb);
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    const float2 cv0 = __bfloat1622float2(cb2[0]);
    const float2 xv0 = __bfloat1622float2(x2[0]);
    const float2 cv1 = __bfloat1622float2(cb2[1]);
    const float2 xv1 = __bfloat1622float2(x2[1]);
    float part = fmaf(cv0.x, xv0.x, 0.f);
    part = fmaf(cv0.y, xv0.y, part);
    part = fmaf(cv1.x, xv1.x, part);
    part = fmaf(cv1.y, xv1.y, part);
    return part;
}

template <int VECTOR>
__device__ __forceinline__ float vq_block_dot_fixed_bf16(
    const __nv_bfloat16* cb,
    const __nv_bfloat16* x)
{
    static_assert(
        VECTOR == 4 || VECTOR == 8 || VECTOR == 16,
        "registered routed VQ vectors are d4/d8/d16");
    const auto* cb2 = reinterpret_cast<const __nv_bfloat162*>(cb);
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    float value = 0.f;
    #pragma unroll
    for (int pair = 0; pair < VECTOR / 2; ++pair) {
        const float2 code = __bfloat1622float2(cb2[pair]);
        const float2 input = __bfloat1622float2(x2[pair]);
        value = fmaf(code.x, input.x, value);
        value = fmaf(code.y, input.y, value);
    }
    return value;
}

__device__ __forceinline__ float vq_block_dot_routed_bf16(
    const __nv_bfloat16* cb,
    const __nv_bfloat16* x,
    const int vector)
{
    if (vector == 4)
        return vq_block_dot_fixed_bf16<4>(cb, x);
    if (vector == 8)
        return vq_block_dot_fixed_bf16<8>(cb, x);
    if (vector == 16)
        return vq_block_dot_fixed_bf16<16>(cb, x);
    return vq_block_dot(cb, x, vector);
}

template <int VECTOR>
__device__ __forceinline__ void vq_block_dot_pair_fixed_bf16(
    const __nv_bfloat16* gate_cb,
    const __nv_bfloat16* up_cb,
    const __nv_bfloat16* x,
    float& gate_value,
    float& up_value)
{
    static_assert(
        VECTOR == 4 || VECTOR == 8 || VECTOR == 16,
        "registered routed VQ vectors are d4/d8/d16");
    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate_cb);
    const auto* up2 = reinterpret_cast<const __nv_bfloat162*>(up_cb);
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    #pragma unroll
    for (int pair = 0; pair < VECTOR / 2; ++pair) {
        const float2 input = __bfloat1622float2(x2[pair]);
        const float2 gate = __bfloat1622float2(gate2[pair]);
        const float2 up = __bfloat1622float2(up2[pair]);
        gate_value = fmaf(gate.x, input.x, gate_value);
        gate_value = fmaf(gate.y, input.y, gate_value);
        up_value = fmaf(up.x, input.x, up_value);
        up_value = fmaf(up.y, input.y, up_value);
    }
}

__device__ __forceinline__ void vq_block_dot_pair_routed_bf16(
    const __nv_bfloat16* gate_cb,
    const __nv_bfloat16* up_cb,
    const __nv_bfloat16* x,
    const int vector,
    float& gate_value,
    float& up_value)
{
    if (vector == 4) {
        vq_block_dot_pair_fixed_bf16<4>(
            gate_cb, up_cb, x, gate_value, up_value);
    } else if (vector == 8) {
        vq_block_dot_pair_fixed_bf16<8>(
            gate_cb, up_cb, x, gate_value, up_value);
    } else if (vector == 16) {
        vq_block_dot_pair_fixed_bf16<16>(
            gate_cb, up_cb, x, gate_value, up_value);
    } else {
        gate_value += vq_block_dot(gate_cb, x, vector);
        up_value += vq_block_dot(up_cb, x, vector);
    }
}

__device__ __forceinline__ float vq_block_dot_routed_fp8_codebook(
    const uint8_t* codebook,
    const __nv_bfloat16* input,
    const int vector,
    const float scale)
{
    float value = 0.0f;
    #pragma unroll
    for (int component = 0; component < 16; component += 4) {
        if (component >= vector) break;
        __nv_fp8x4_e4m3 packed_code;
        packed_code.__x = __ldg(reinterpret_cast<const uint32_t*>(
            codebook + component));
        const float4 code = static_cast<float4>(packed_code);
        const float2 input01 = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(input + component));
        const float2 input23 = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(input + component + 2));
        value = fmaf(code.x * scale, input01.x, value);
        value = fmaf(code.y * scale, input01.y, value);
        value = fmaf(code.z * scale, input23.x, value);
        value = fmaf(code.w * scale, input23.y, value);
    }
    return value;
}

__device__ __forceinline__ float vq_gemv_routed_row_fp8_codebook(
    const int64_t index_address,
    const uint8_t* __restrict__ codebook,
    const __nv_bfloat16* __restrict__ input,
    const int blocks,
    const int vector,
    const int dtype_tag,
    const long index_row,
    const float scale)
{
    float value = 0.0f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        const int code = routed_index_value(
            index_address, dtype_tag, index_row + block);
        value += vq_block_dot_routed_fp8_codebook(
            codebook + static_cast<long>(code) * vector,
            input + block * vector,
            vector,
            scale);
    }
    return value;
}

__device__ __forceinline__ void vq_gemv_routed_pair_fp8_codebook(
    const int64_t gate_index_address,
    const int64_t up_index_address,
    const uint8_t* __restrict__ gate_codebook,
    const uint8_t* __restrict__ up_codebook,
    const __nv_bfloat16* __restrict__ input,
    const int blocks,
    const int vector,
    const int gate_dtype_tag,
    const int up_dtype_tag,
    const long index_row,
    const float gate_scale,
    const float up_scale,
    float& gate_value,
    float& up_value)
{
    gate_value = 0.0f;
    up_value = 0.0f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        const long offset = index_row + block;
        const int gate_code = routed_index_value(
            gate_index_address, gate_dtype_tag, offset);
        const int up_code = routed_index_value(
            up_index_address, up_dtype_tag, offset);
        const __nv_bfloat16* input_block = input + block * vector;
        gate_value += vq_block_dot_routed_fp8_codebook(
            gate_codebook + static_cast<long>(gate_code) * vector,
            input_block,
            vector,
            gate_scale);
        up_value += vq_block_dot_routed_fp8_codebook(
            up_codebook + static_cast<long>(up_code) * vector,
            input_block,
            vector,
            up_scale);
    }
}

__device__ __forceinline__ void vq_gemv_routed_pair(
    const int64_t gate_index_address,
    const int64_t up_index_address,
    const __nv_bfloat16* __restrict__ gate_codebook,
    const __nv_bfloat16* __restrict__ up_codebook,
    const __nv_bfloat16* __restrict__ input,
    const int blocks,
    const int vector,
    const int dtype_tag,
    const long index_row,
    float& gate_value,
    float& up_value)
{
    gate_value = 0.f;
    up_value = 0.f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        const long offset = index_row + block;
        const int gate_code = routed_index_value(
            gate_index_address, dtype_tag, offset);
        const int up_code = routed_index_value(
            up_index_address, dtype_tag, offset);
        vq_block_dot_pair_routed_bf16(
            gate_codebook + static_cast<long>(gate_code) * vector,
            up_codebook + static_cast<long>(up_code) * vector,
            input + block * vector,
            vector,
            gate_value,
            up_value);
    }
}

constexpr int CCCP_PROJECTION_P10_SHARED_STRIDE = 10;
constexpr int CCCP_PROJECTION_P8_SHARED_STRIDE = 6;
constexpr int CCCP_PROJECTION_P11_CODES = 2048;
constexpr int CCCP_PROJECTION_P11_VECTOR = 8;
constexpr int CCCP_PROJECTION_P11_SHARED_STRIDE = 10;

__device__ __forceinline__ void vq_block_dot8_pair_bf16(
    const __nv_bfloat16* gate_cb,
    const __nv_bfloat16* up_cb,
    const __nv_bfloat16* input,
    float& gate_value,
    float& up_value)
{
    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate_cb);
    const auto* up2 = reinterpret_cast<const __nv_bfloat162*>(up_cb);
    const auto* input2 = reinterpret_cast<const __nv_bfloat162*>(input);
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 x = __bfloat1622float2(input2[pair]);
        const float2 gate = __bfloat1622float2(gate2[pair]);
        const float2 up = __bfloat1622float2(up2[pair]);
        gate_value = fmaf(gate.x, x.x, gate_value);
        gate_value = fmaf(gate.y, x.y, gate_value);
        up_value = fmaf(up.x, x.x, up_value);
        up_value = fmaf(up.y, x.y, up_value);
    }
}

__device__ __forceinline__ void vq_gemv_routed_p10_pair(
    const int64_t gate_index_address,
    const int64_t up_index_address,
    const __nv_bfloat16* __restrict__ gate_codebook,
    const __nv_bfloat16* __restrict__ up_codebook,
    const __nv_bfloat16* __restrict__ input,
    const int blocks,
    const long index_row,
    float& gate_value,
    float& up_value)
{
    const auto* gate_indices = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(gate_index_address));
    const auto* up_indices = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(up_index_address));
    gate_value = 0.f;
    up_value = 0.f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        const unsigned active = __activemask();
        const int sublane = threadIdx.x & 3;
        const long base = ((index_row + block) >> 2) * 5;
        const unsigned gate_byte = gate_indices[base + sublane];
        const unsigned up_byte = up_indices[base + sublane];
        unsigned gate_next = __shfl_down_sync(
            active, gate_byte, 1, 4);
        unsigned up_next = __shfl_down_sync(
            active, up_byte, 1, 4);
        if (sublane == 3) {
            gate_next = gate_indices[base + 4];
            up_next = up_indices[base + 4];
        }
        const int following_bits = 2 * (sublane + 1);
        const unsigned following_mask =
            (1u << following_bits) - 1u;
        const int left_shift = 8 - 2 * sublane;
        const int gate_code = static_cast<int>(
            (gate_byte >> (2 * sublane)) |
            ((gate_next & following_mask) << left_shift));
        const int up_code = static_cast<int>(
            (up_byte >> (2 * sublane)) |
            ((up_next & following_mask) << left_shift));
        vq_block_dot8_pair_bf16(
            gate_codebook +
                static_cast<long>(gate_code) *
                    CCCP_PROJECTION_P10_SHARED_STRIDE,
            up_codebook +
                static_cast<long>(up_code) *
                    CCCP_PROJECTION_P10_SHARED_STRIDE,
            input + block * 8,
            gate_value,
            up_value);
    }
}

__device__ __forceinline__ float vq_gemv_routed_p8_shared(
    const int64_t index_address,
    const __nv_bfloat16* __restrict__ codebook,
    const __nv_bfloat16* __restrict__ input,
    const int blocks,
    const long index_row)
{
    const auto* indices = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(index_address));
    float value = 0.f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        const int code = static_cast<int>(indices[index_row + block]);
        value += vq_block_dot4_bf16(
            codebook +
                static_cast<long>(code) *
                    CCCP_PROJECTION_P8_SHARED_STRIDE,
            input + block * 4);
    }
    return value;
}

__device__ __forceinline__ float vq_gemv_routed_p11_shared(
    const int64_t index_address,
    const __nv_bfloat16* __restrict__ codebook,
    const __nv_bfloat16* __restrict__ input,
    const int blocks,
    const long index_row)
{
    float value = 0.f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        const int code = routed_index_value(
            index_address, 6, index_row + block);
        value += vq_block_dot_fixed_bf16<8>(
            codebook +
                static_cast<long>(code) *
                    CCCP_PROJECTION_P11_SHARED_STRIDE,
            input + block * CCCP_PROJECTION_P11_VECTOR);
    }
    return value;
}

__device__ __forceinline__ float vq_gemv_routed_row(
    const int64_t index_address,
    const __nv_bfloat16* __restrict__ codebook,
    const __nv_bfloat16* __restrict__ input,
    const int blocks,
    const int vector,
    const int dtype_tag,
    const long index_row)
{
    float value = 0.f;
    if (dtype_tag == 0 && vector == 4) {
        const auto* indices = reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(index_address));
        for (int block = threadIdx.x; block < blocks; block += 32) {
            const int code = static_cast<int>(
                indices[index_row + block]);
            value += vq_block_dot4_bf16(
                codebook + (long)code * 4,
                input + block * 4);
        }
    } else if (dtype_tag == 2 && (blocks & 1) == 0) {
        // Packed-12 stores two adjacent indices in three bytes.  One even
        // lane loads each pair and broadcasts the assembled word.
        const auto* indices = reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(index_address));
        for (int block = threadIdx.x; block < blocks; block += 32) {
            const unsigned active = __activemask();
            unsigned packed = 0;
            const int leader = threadIdx.x & ~1;
            if ((threadIdx.x & 1) == 0) {
                const long base = ((index_row + block) >> 1) * 3;
                packed =
                    static_cast<unsigned>(indices[base]) |
                    (static_cast<unsigned>(indices[base + 1]) << 8) |
                    (static_cast<unsigned>(indices[base + 2]) << 16);
            }
            packed = __shfl_sync(active, packed, leader);
            const int code = static_cast<int>(
                (packed >> ((threadIdx.x & 1) * 12)) & 0xfffu);
            value += vq_block_dot_routed_bf16(
                codebook + (long)code * vector,
                input + block * vector,
                vector);
        }
    } else if (dtype_tag == 4 && (blocks & 3) == 0) {
        // Packed-10 stores four adjacent indices in five bytes. One lane
        // loads the 40-bit group and broadcasts it to the four consumers.
        const auto* indices = reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(index_address));
        for (int block = threadIdx.x; block < blocks; block += 32) {
            const unsigned active = __activemask();
            const int leader = threadIdx.x & ~3;
            unsigned low = 0;
            unsigned high = 0;
            if ((threadIdx.x & 3) == 0) {
                const long base = ((index_row + block) >> 2) * 5;
                unsigned long long packed = 0;
                #pragma unroll
                for (int byte = 0; byte < 5; ++byte)
                    packed |=
                        static_cast<unsigned long long>(
                            indices[base + byte])
                        << (8 * byte);
                low = static_cast<unsigned>(packed);
                high = static_cast<unsigned>(packed >> 32);
            }
            low = __shfl_sync(active, low, leader);
            high = __shfl_sync(active, high, leader);
            const unsigned long long packed =
                static_cast<unsigned long long>(low) |
                (static_cast<unsigned long long>(high) << 32);
            const int code = static_cast<int>(
                (packed >> (10 * (threadIdx.x & 3))) & 0x3ffu);
            value += vq_block_dot_routed_bf16(
                codebook + (long)code * vector,
                input + block * vector,
                vector);
        }
    } else if (dtype_tag == 3 && (blocks & 3) == 0) {
        // Packed-14 stores four adjacent indices in seven bytes.  One lane
        // loads each group and broadcasts the same 56-bit word.
        const auto* indices = reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(index_address));
        for (int block = threadIdx.x; block < blocks; block += 32) {
            const unsigned active = __activemask();
            const int leader = threadIdx.x & ~3;
            unsigned low = 0;
            unsigned high = 0;
            if ((threadIdx.x & 3) == 0) {
                const long base = ((index_row + block) >> 2) * 7;
                unsigned long long packed = 0;
                #pragma unroll
                for (int byte = 0; byte < 7; ++byte)
                    packed |=
                        static_cast<unsigned long long>(
                            indices[base + byte])
                        << (8 * byte);
                low = static_cast<unsigned>(packed);
                high = static_cast<unsigned>(packed >> 32);
            }
            low = __shfl_sync(active, low, leader);
            high = __shfl_sync(active, high, leader);
            const unsigned long long packed =
                static_cast<unsigned long long>(low) |
                (static_cast<unsigned long long>(high) << 32);
            const int code = static_cast<int>(
                (packed >> (14 * (threadIdx.x & 3))) & 0x3fffu);
            value += vq_block_dot_routed_bf16(
                codebook + (long)code * vector,
                input + block * vector,
                vector);
        }
    } else {
        for (int block = threadIdx.x; block < blocks; block += 32) {
            const int code = routed_index_value(
                index_address,
                dtype_tag,
                index_row + block);
            value += vq_block_dot_routed_bf16(
                codebook + (long)code * vector,
                input + block * vector,
                vector);
        }
    }
    return value;
}

struct RoutedBlockMetadata {
    int64_t index_address;
    int64_t codebook_address;
    int blocks;
    int vector;
    int dtype_tag;
    int valid;
};

__device__ __forceinline__ RoutedTileMetadata projection_tile_metadata(
    const int64_t* metadata,
    const int metadata_rows,
    const int expert_count,
    const int expert,
    const int projection,
    const int blocks,
    const int dtype_tag)
{
    RoutedTileMetadata tile{};
    if (metadata_rows != CCCP_PROJECTION_TILE_META_ROWS)
        return tile;
    const int base =
        CCCP_PROJECTION_LEGACY_META_ROWS +
        projection * CCCP_TILE_ROWS_PER_PROJECTION;
    tile.bits = static_cast<int>(
        metadata[(long)(base + 0) * expert_count + expert]);
    tile.row_bytes = static_cast<int>(
        metadata[(long)(base + 1) * expert_count + expert]);
    tile.group = static_cast<int>(
        metadata[(long)(base + 2) * expert_count + expert]);
    tile.tile_blocks = static_cast<int>(
        metadata[(long)(base + 3) * expert_count + expert]);
    const int expected_bits = (
        dtype_tag >= 5 && dtype_tag <= 8 ? 2 * dtype_tag - 1 : 0);
    tile.valid = (
        expected_bits == tile.bits &&
        tile.bits >= 9 && tile.bits <= 15 &&
        tile.group == CCCP_TILE_INDEX_GROUP &&
        tile.row_bytes == (blocks * tile.bits + 7) / 8 &&
        blocks % CCCP_TILE_INDEX_GROUP == 0 &&
        tile.tile_blocks == 32);
    return tile;
}

__device__ __forceinline__ float vq_gemv_routed_row_tiled(
    const int64_t index_address,
    const __nv_bfloat16* __restrict__ codebook,
    const __nv_bfloat16* __restrict__ input,
    const int blocks,
    const int vector,
    const int dtype_tag,
    const int row,
    const RoutedTileMetadata& tile)
{
    float value = 0.f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        const int code = routed_tile_index_value(
            index_address, tile, dtype_tag, row, block, blocks);
        value += vq_block_dot_routed_bf16(
            codebook + static_cast<long>(code) * vector,
            input + block * vector,
            vector);
    }
    return value;
}

__device__ __forceinline__ void vq_gemv_routed_pair_tiled(
    const int64_t gate_index_address,
    const int64_t up_index_address,
    const __nv_bfloat16* __restrict__ gate_codebook,
    const __nv_bfloat16* __restrict__ up_codebook,
    const __nv_bfloat16* __restrict__ input,
    const int blocks,
    const int vector,
    const int gate_dtype_tag,
    const int up_dtype_tag,
    const int row,
    const RoutedTileMetadata& gate_tile,
    const RoutedTileMetadata& up_tile,
    float& gate_value,
    float& up_value)
{
    gate_value = 0.f;
    up_value = 0.f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        const int gate_code = routed_tile_index_value(
            gate_index_address,
            gate_tile,
            gate_dtype_tag,
            row,
            block,
            blocks);
        const int up_code = routed_tile_index_value(
            up_index_address,
            up_tile,
            up_dtype_tag,
            row,
            block,
            blocks);
        vq_block_dot_pair_routed_bf16(
            gate_codebook + static_cast<long>(gate_code) * vector,
            up_codebook + static_cast<long>(up_code) * vector,
            input + block * vector,
            vector,
            gate_value,
            up_value);
    }
}

template <int WARPS>
__global__ void vq_gemv_routed_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    __nv_bfloat16* __restrict__ out,
    const int K,
    const int E,
    const int meta_base,
    const int R,
    const int C,
    const long x_stride_n,
    const bool skip_p12,
    const int route_offset,
    const bool vector_input_copy)
{
    const int n = blockIdx.y + route_offset;
    if (n >= K) return;

    const int row =
        blockIdx.x * WARPS + threadIdx.y;
    extern __shared__ unsigned char raw_smem[];
    __shared__ RoutedBlockMetadata route_meta;
    auto* xs = reinterpret_cast<__nv_bfloat16*>(raw_smem);
    const __nv_bfloat16* xrow = x + (long)n * x_stride_n;
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    if (linear_thread == 0) {
        const int expert_id = static_cast<int>(route_ids[n]);
        route_meta.valid = 0;
        if (expert_id >= 0 && expert_id < E) {
            route_meta.index_address =
                metadata[(long)(meta_base + 0) * E + expert_id];
            if (route_meta.index_address != 0) {
                route_meta.codebook_address =
                    metadata[(long)(meta_base + 1) * E + expert_id];
                route_meta.blocks = static_cast<int>(
                    metadata[
                        (long)(meta_base + 2) * E + expert_id]);
                route_meta.vector = static_cast<int>(
                    metadata[
                        (long)(meta_base + 3) * E + expert_id]);
                route_meta.dtype_tag = static_cast<int>(
                    metadata[
                        (long)(meta_base + 4) * E + expert_id]);
                route_meta.valid = !(
                    skip_p12 &&
                    route_meta.dtype_tag == 2 &&
                    (
                        route_meta.vector == 4 ||
                        route_meta.vector == 8
                    )
                );
            }
        }
    }
    if (
        vector_input_copy &&
        (C & 7) == 0 &&
        (
            reinterpret_cast<uintptr_t>(xrow) &
            (alignof(uint4) - 1)
        ) == 0
    ) {
        const auto* x4 = reinterpret_cast<const uint4*>(xrow);
        auto* xs4 = reinterpret_cast<uint4*>(xs);
        for (int i = linear_thread;
             i < C / 8; i += 32 * WARPS)
            xs4[i] = x4[i];
    } else {
        for (int i = linear_thread; i < C; i += 32 * WARPS)
            xs[i] = xrow[i];
    }
    __syncthreads();
    if (!route_meta.valid || row >= R) return;

    const auto* codebook = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(route_meta.codebook_address));
    float value = vq_gemv_routed_row(
        route_meta.index_address,
        codebook,
        xs,
        route_meta.blocks,
        route_meta.vector,
        route_meta.dtype_tag,
        (long)row * route_meta.blocks);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, off);
    if (threadIdx.x == 0)
        out[(long)n * R + row] = __float2bfloat16_rn(value);
}

template <int WARPS>
inline void launch_vq_gemv_routed(
    const __nv_bfloat16* input,
    const int64_t* route_ids,
    const int64_t* metadata,
    __nv_bfloat16* output,
    const int top_k,
    const int expert_count,
    const int metadata_base,
    const int output_rows,
    const int input_cols,
    const long input_stride,
    const bool skip_p12,
    const int route_offset,
    const int active_count,
    const bool vector_input_copy,
    cudaStream_t stream)
{
    vq_gemv_routed_kernel<WARPS><<<
        dim3(
            (unsigned)((output_rows + WARPS - 1) / WARPS),
            (unsigned)active_count),
        dim3(32, WARPS),
        (size_t)input_cols * sizeof(__nv_bfloat16),
        stream>>>(
            input,
            route_ids,
            metadata,
            output,
            top_k,
            expert_count,
            metadata_base,
            output_rows,
            input_cols,
            input_stride,
            skip_p12,
            route_offset,
            vector_input_copy);
}

constexpr int CCCP_PROJECTION_ROWS_PER_WARP = 4;
constexpr int CCCP_PROJECTION_P10_CODES = 1024;
constexpr int CCCP_PROJECTION_P10_VECTOR = 8;
constexpr int CCCP_PROJECTION_P8_CODES = 256;
constexpr int CCCP_PROJECTION_P8_VECTOR = 4;

// Activation is part of the public operator capability, not a model switch.
// 0 = SiTU (Kimi), 1 = clamped SiLU/SwiGLU (DeepSeek/GLM).
__device__ __forceinline__ float projection_gate_up_activation(
    float gate,
    float up,
    const int activation_kind,
    const float beta,
    const float linear_beta,
    const float limit)
{
    if (activation_kind == 1) {
        if (limit > 0.f) {
            gate = fminf(gate, limit);
            up = fminf(fmaxf(up, -limit), limit);
        }
        return (gate / (1.f + expf(-gate))) * up;
    }
    const float nonlinear =
        beta * tanhf(gate / beta) / (1.f + expf(-gate));
    if (linear_beta > 0.f)
        up = linear_beta * tanhf(up / linear_beta);
    return nonlinear * up;
}

// Low-memory native route: compact indices and compact E4M3 codebooks stay
// resident. Gate/Up and Down convert only referenced codewords in registers;
// there is no expanded expert-weight workspace between lookup and dot.
template <int WARPS, int ROWS_PER_WARP>
__global__ void vq_projection_gate_up_compact_fp8_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ activated,
    const int top_k,
    const int expert_count,
    const int output_rows,
    const int input_cols,
    const int activation_kind,
    const float beta,
    const float linear_beta,
    const float limit)
{
    const int position = blockIdx.y;
    if (position >= top_k) return;
    const int expert = static_cast<int>(route_ids[position]);
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    constexpr int block_threads = 32 * WARPS;
    extern __shared__ __nv_bfloat16 compact_fp8_gate_up_input[];
    __shared__ RoutedBlockMetadata gate_meta;
    __shared__ RoutedBlockMetadata up_meta;
    __shared__ float gate_scale;
    __shared__ float up_scale;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        gate_meta.valid = 0;
        up_meta.valid = 0;
        if (expert >= 0 && expert < expert_count) {
            gate_meta.index_address = metadata[expert];
            gate_meta.codebook_address =
                metadata[(long)expert_count + expert];
            gate_meta.blocks = static_cast<int>(
                metadata[(long)2 * expert_count + expert]);
            gate_meta.vector = static_cast<int>(
                metadata[(long)3 * expert_count + expert]);
            gate_meta.dtype_tag = static_cast<int>(
                metadata[(long)4 * expert_count + expert]);
            up_meta.index_address =
                metadata[(long)5 * expert_count + expert];
            up_meta.codebook_address =
                metadata[(long)6 * expert_count + expert];
            up_meta.blocks = static_cast<int>(
                metadata[(long)7 * expert_count + expert]);
            up_meta.vector = static_cast<int>(
                metadata[(long)8 * expert_count + expert]);
            up_meta.dtype_tag = static_cast<int>(
                metadata[(long)9 * expert_count + expert]);
            gate_meta.valid = gate_meta.index_address != 0 &&
                gate_meta.codebook_address != 0 && gate_meta.blocks > 0;
            up_meta.valid = up_meta.index_address != 0 &&
                up_meta.codebook_address != 0 && up_meta.blocks > 0;
            gate_scale = scales[(long)expert * 3];
            up_scale = scales[(long)expert * 3 + 1];
        }
    }
    const auto* input4 = reinterpret_cast<const uint4*>(input);
    auto* shared4 = reinterpret_cast<uint4*>(compact_fp8_gate_up_input);
    for (int item = linear_thread; item < input_cols / 8;
         item += block_threads)
        shared4[item] = input4[item];
    __syncthreads();
    if (!gate_meta.valid || !up_meta.valid) return;

    const auto* gate_codebook = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(gate_meta.codebook_address));
    const auto* up_codebook = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(up_meta.codebook_address));
    float gate_values[ROWS_PER_WARP] = {};
    float up_values[ROWS_PER_WARP] = {};
    #pragma unroll
    for (int item = 0; item < ROWS_PER_WARP; ++item) {
        const int row = blockIdx.x * (WARPS * ROWS_PER_WARP) +
            threadIdx.y + item * WARPS;
        if (row >= output_rows) continue;
        if (gate_meta.blocks == up_meta.blocks &&
            gate_meta.vector == up_meta.vector) {
            vq_gemv_routed_pair_fp8_codebook(
                gate_meta.index_address,
                up_meta.index_address,
                gate_codebook,
                up_codebook,
                compact_fp8_gate_up_input,
                gate_meta.blocks,
                gate_meta.vector,
                gate_meta.dtype_tag,
                up_meta.dtype_tag,
                (long)row * gate_meta.blocks,
                gate_scale,
                up_scale,
                gate_values[item],
                up_values[item]);
        } else {
            gate_values[item] = vq_gemv_routed_row_fp8_codebook(
                gate_meta.index_address,
                gate_codebook,
                compact_fp8_gate_up_input,
                gate_meta.blocks,
                gate_meta.vector,
                gate_meta.dtype_tag,
                (long)row * gate_meta.blocks,
                gate_scale);
            up_values[item] = vq_gemv_routed_row_fp8_codebook(
                up_meta.index_address,
                up_codebook,
                compact_fp8_gate_up_input,
                up_meta.blocks,
                up_meta.vector,
                up_meta.dtype_tag,
                (long)row * up_meta.blocks,
                up_scale);
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int item = 0; item < ROWS_PER_WARP; ++item) {
            gate_values[item] += __shfl_down_sync(
                0xffffffffu, gate_values[item], offset);
            up_values[item] += __shfl_down_sync(
                0xffffffffu, up_values[item], offset);
        }
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int item = 0; item < ROWS_PER_WARP; ++item) {
            const int row = blockIdx.x * (WARPS * ROWS_PER_WARP) +
                threadIdx.y + item * WARPS;
            if (row < output_rows)
                activated[(long)position * output_rows + row] =
                    __float2bfloat16_rn(projection_gate_up_activation(
                        gate_values[item], up_values[item], activation_kind,
                        beta, linear_beta, limit));
        }
    }
}

template <int WARPS, int ROWS_PER_WARP>
__global__ void vq_projection_down_compact_fp8_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ output,
    const int top_k,
    const int expert_count,
    const int output_rows,
    const int input_cols)
{
    const int position = blockIdx.y;
    if (position >= top_k) return;
    const int expert = static_cast<int>(route_ids[position]);
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    constexpr int block_threads = 32 * WARPS;
    extern __shared__ __nv_bfloat16 compact_fp8_down_input[];
    __shared__ RoutedBlockMetadata down_meta;
    __shared__ float down_scale;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        down_meta.valid = 0;
        if (expert >= 0 && expert < expert_count) {
            down_meta.index_address =
                metadata[(long)10 * expert_count + expert];
            down_meta.codebook_address =
                metadata[(long)11 * expert_count + expert];
            down_meta.blocks = static_cast<int>(
                metadata[(long)12 * expert_count + expert]);
            down_meta.vector = static_cast<int>(
                metadata[(long)13 * expert_count + expert]);
            down_meta.dtype_tag = static_cast<int>(
                metadata[(long)14 * expert_count + expert]);
            down_meta.valid = down_meta.index_address != 0 &&
                down_meta.codebook_address != 0 && down_meta.blocks > 0;
            down_scale = scales[(long)expert * 3 + 2];
        }
    }
    const __nv_bfloat16* input_row = input + (long)position * input_cols;
    const auto* input4 = reinterpret_cast<const uint4*>(input_row);
    auto* shared4 = reinterpret_cast<uint4*>(compact_fp8_down_input);
    for (int item = linear_thread; item < input_cols / 8;
         item += block_threads)
        shared4[item] = input4[item];
    __syncthreads();
    if (!down_meta.valid) return;

    const auto* codebook = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(down_meta.codebook_address));
    float values[ROWS_PER_WARP] = {};
    #pragma unroll
    for (int item = 0; item < ROWS_PER_WARP; ++item) {
        const int row = blockIdx.x * (WARPS * ROWS_PER_WARP) +
            threadIdx.y + item * WARPS;
        if (row < output_rows)
            values[item] = vq_gemv_routed_row_fp8_codebook(
                down_meta.index_address,
                codebook,
                compact_fp8_down_input,
                down_meta.blocks,
                down_meta.vector,
                down_meta.dtype_tag,
                (long)row * down_meta.blocks,
                down_scale);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int item = 0; item < ROWS_PER_WARP; ++item)
            values[item] += __shfl_down_sync(
                0xffffffffu, values[item], offset);
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int item = 0; item < ROWS_PER_WARP; ++item) {
            const int row = blockIdx.x * (WARPS * ROWS_PER_WARP) +
                threadIdx.y + item * WARPS;
            if (row < output_rows)
                output[(long)position * output_rows + row] =
                    __float2bfloat16_rn(values[item]);
        }
    }
}

template <int VECTOR, int DTYPE_TAG>
__device__ __forceinline__ float vq_gemv_routed_row_q8_codebook_t(
    const int64_t index_address,
    const int8_t* __restrict__ codebook,
    const int8_t* __restrict__ input,
    const float* __restrict__ input_scales,
    const int blocks,
    const long index_count,
    const long index_row,
    const float codebook_scale)
{
    float value = 0.0f;
    if constexpr (DTYPE_TAG == 4) {
        for (int block = threadIdx.x * 4;
            block < blocks;
            block += 32 * 4) {
            if (block + 3 < blocks) {
                int codes[4];
                routed_p10_index_quad(
                    index_address, index_row + block, index_count, codes);
                #pragma unroll
                for (int item = 0; item < 4; ++item) {
                    value += static_cast<float>(
                        vq_block_dot_routed_q8_codebook_i32_t<VECTOR>(
                            codebook + static_cast<long>(codes[item]) * VECTOR,
                            input + (block + item) * VECTOR));
                }
            } else {
                #pragma unroll
                for (int item = 0; item < 4; ++item) {
                    if (block + item >= blocks) break;
                    const int code = routed_index_value_t<DTYPE_TAG>(
                        index_address, index_row + block + item, index_count);
                    value += static_cast<float>(
                        vq_block_dot_routed_q8_codebook_i32_t<VECTOR>(
                            codebook + static_cast<long>(code) * VECTOR,
                            input + (block + item) * VECTOR));
                }
            }
        }
    } else {
        for (int block = threadIdx.x; block < blocks; block += 32) {
            const int code = routed_index_value_t<DTYPE_TAG>(
                index_address, index_row + block, index_count);
            value += static_cast<float>(
                vq_block_dot_routed_q8_codebook_i32_t<VECTOR>(
                    codebook + static_cast<long>(code) * VECTOR,
                    input + block * VECTOR));
        }
    }
    return value * (codebook_scale * input_scales[0]);
}

template <int VECTOR>
__device__ __forceinline__ float vq_gemv_routed_row_q8_codebook_vector_t(
    const int64_t index_address,
    const int8_t* __restrict__ codebook,
    const int8_t* __restrict__ input,
    const float* __restrict__ input_scales,
    const int blocks,
    const int dtype_tag,
    const long index_count,
    const long index_row,
    const float codebook_scale)
{
    if (dtype_tag == 0)
        return vq_gemv_routed_row_q8_codebook_t<VECTOR, 0>(
            index_address, codebook, input, input_scales, blocks,
            index_count, index_row, codebook_scale);
    if (dtype_tag == 1)
        return vq_gemv_routed_row_q8_codebook_t<VECTOR, 1>(
            index_address, codebook, input, input_scales, blocks,
            index_count, index_row, codebook_scale);
    if (dtype_tag == 2)
        return vq_gemv_routed_row_q8_codebook_t<VECTOR, 2>(
            index_address, codebook, input, input_scales, blocks,
            index_count, index_row, codebook_scale);
    if (dtype_tag == 3)
        return vq_gemv_routed_row_q8_codebook_t<VECTOR, 3>(
            index_address, codebook, input, input_scales, blocks,
            index_count, index_row, codebook_scale);
    if (dtype_tag == 4)
        return vq_gemv_routed_row_q8_codebook_t<VECTOR, 4>(
            index_address, codebook, input, input_scales, blocks,
            index_count, index_row, codebook_scale);
    if (dtype_tag == 5)
        return vq_gemv_routed_row_q8_codebook_t<VECTOR, 5>(
            index_address, codebook, input, input_scales, blocks,
            index_count, index_row, codebook_scale);
    if (dtype_tag == 6)
        return vq_gemv_routed_row_q8_codebook_t<VECTOR, 6>(
            index_address, codebook, input, input_scales, blocks,
            index_count, index_row, codebook_scale);
    if (dtype_tag == 7)
        return vq_gemv_routed_row_q8_codebook_t<VECTOR, 7>(
            index_address, codebook, input, input_scales, blocks,
            index_count, index_row, codebook_scale);
    return vq_gemv_routed_row_q8_codebook_t<VECTOR, 8>(
        index_address, codebook, input, input_scales, blocks,
        index_count, index_row, codebook_scale);
}

__device__ __forceinline__ float vq_gemv_routed_row_q8_codebook(
    const int64_t index_address,
    const int8_t* __restrict__ codebook,
    const int8_t* __restrict__ input,
    const float* __restrict__ input_scales,
    const int blocks,
    const int vector,
    const int dtype_tag,
    const long index_count,
    const long index_row,
    const float codebook_scale)
{
    if (vector == 16)
        return vq_gemv_routed_row_q8_codebook_vector_t<16>(
            index_address, codebook, input, input_scales, blocks,
            dtype_tag, index_count, index_row, codebook_scale);
    if (vector == 8)
        return vq_gemv_routed_row_q8_codebook_vector_t<8>(
            index_address, codebook, input, input_scales, blocks,
            dtype_tag, index_count, index_row, codebook_scale);
    return vq_gemv_routed_row_q8_codebook_vector_t<4>(
        index_address, codebook, input, input_scales, blocks,
        dtype_tag, index_count, index_row, codebook_scale);
}

template <int VECTOR, int DTYPE_TAG>
__device__ __forceinline__ void vq_gemv_routed_pair_q8_codebook_same_t(
    const int64_t gate_index_address,
    const int64_t up_index_address,
    const int8_t* __restrict__ gate_codebook,
    const int8_t* __restrict__ up_codebook,
    const int8_t* __restrict__ input,
    const float* __restrict__ input_scales,
    const int blocks,
    const long index_count,
    const long index_row,
    const float gate_scale,
    const float up_scale,
    float& gate_value,
    float& up_value)
{
    gate_value = 0.0f;
    up_value = 0.0f;
    if constexpr (DTYPE_TAG == 4) {
        for (int block = threadIdx.x * 4;
            block < blocks;
            block += 32 * 4) {
            if (block + 3 < blocks) {
                int gate_codes[4];
                int up_codes[4];
                routed_p10_index_quad(
                    gate_index_address, index_row + block,
                    index_count, gate_codes);
                routed_p10_index_quad(
                    up_index_address, index_row + block,
                    index_count, up_codes);
                #pragma unroll
                for (int item = 0; item < 4; ++item) {
                    const int8_t* input_block =
                        input + (block + item) * VECTOR;
                    gate_value += static_cast<float>(
                        vq_block_dot_routed_q8_codebook_i32_t<VECTOR>(
                            gate_codebook +
                                static_cast<long>(gate_codes[item]) * VECTOR,
                            input_block));
                    up_value += static_cast<float>(
                        vq_block_dot_routed_q8_codebook_i32_t<VECTOR>(
                            up_codebook +
                                static_cast<long>(up_codes[item]) * VECTOR,
                            input_block));
                }
            } else {
                #pragma unroll
                for (int item = 0; item < 4; ++item) {
                    if (block + item >= blocks) break;
                    const long offset = index_row + block + item;
                    const int gate_code = routed_index_value_t<DTYPE_TAG>(
                        gate_index_address, offset, index_count);
                    const int up_code = routed_index_value_t<DTYPE_TAG>(
                        up_index_address, offset, index_count);
                    const int8_t* input_block =
                        input + (block + item) * VECTOR;
                    gate_value += static_cast<float>(
                        vq_block_dot_routed_q8_codebook_i32_t<VECTOR>(
                            gate_codebook +
                                static_cast<long>(gate_code) * VECTOR,
                            input_block));
                    up_value += static_cast<float>(
                        vq_block_dot_routed_q8_codebook_i32_t<VECTOR>(
                            up_codebook +
                                static_cast<long>(up_code) * VECTOR,
                            input_block));
                }
            }
        }
    } else {
        for (int block = threadIdx.x; block < blocks; block += 32) {
            const long offset = index_row + block;
            const int gate_code = routed_index_value_t<DTYPE_TAG>(
                gate_index_address, offset, index_count);
            const int up_code = routed_index_value_t<DTYPE_TAG>(
                up_index_address, offset, index_count);
            const int8_t* input_block = input + block * VECTOR;
            gate_value += static_cast<float>(
                vq_block_dot_routed_q8_codebook_i32_t<VECTOR>(
                    gate_codebook + static_cast<long>(gate_code) * VECTOR,
                    input_block));
            up_value += static_cast<float>(
                vq_block_dot_routed_q8_codebook_i32_t<VECTOR>(
                    up_codebook + static_cast<long>(up_code) * VECTOR,
                    input_block));
        }
    }
    gate_value *= gate_scale * input_scales[0];
    up_value *= up_scale * input_scales[0];
}

template <int VECTOR>
__device__ __forceinline__ void vq_gemv_routed_pair_q8_codebook_t(
    const int64_t gate_index_address,
    const int64_t up_index_address,
    const int8_t* __restrict__ gate_codebook,
    const int8_t* __restrict__ up_codebook,
    const int8_t* __restrict__ input,
    const float* __restrict__ input_scales,
    const int blocks,
    const int gate_dtype_tag,
    const int up_dtype_tag,
    const long index_count,
    const long index_row,
    const float gate_scale,
    const float up_scale,
    float& gate_value,
    float& up_value)
{
    if (gate_dtype_tag == up_dtype_tag) {
        #define CCCP_Q8_PAIR_SAME(TAG) \
            vq_gemv_routed_pair_q8_codebook_same_t<VECTOR, TAG>( \
                gate_index_address, up_index_address, gate_codebook, \
                up_codebook, input, input_scales, blocks, index_count, \
                index_row, \
                gate_scale, up_scale, gate_value, up_value)
        if (gate_dtype_tag == 0) { CCCP_Q8_PAIR_SAME(0); return; }
        if (gate_dtype_tag == 1) { CCCP_Q8_PAIR_SAME(1); return; }
        if (gate_dtype_tag == 2) { CCCP_Q8_PAIR_SAME(2); return; }
        if (gate_dtype_tag == 3) { CCCP_Q8_PAIR_SAME(3); return; }
        if (gate_dtype_tag == 4) { CCCP_Q8_PAIR_SAME(4); return; }
        if (gate_dtype_tag == 5) { CCCP_Q8_PAIR_SAME(5); return; }
        if (gate_dtype_tag == 6) { CCCP_Q8_PAIR_SAME(6); return; }
        if (gate_dtype_tag == 7) { CCCP_Q8_PAIR_SAME(7); return; }
        if (gate_dtype_tag == 8) { CCCP_Q8_PAIR_SAME(8); return; }
        #undef CCCP_Q8_PAIR_SAME
    }
    gate_value = 0.0f;
    up_value = 0.0f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        const long offset = index_row + block;
        const int gate_code = routed_index_value(
            gate_index_address, gate_dtype_tag, offset);
        const int up_code = routed_index_value(
            up_index_address, up_dtype_tag, offset);
        const int8_t* input_block = input + block * VECTOR;
        gate_value += static_cast<float>(
            vq_block_dot_routed_q8_codebook_i32_t<VECTOR>(
                gate_codebook + static_cast<long>(gate_code) * VECTOR,
                input_block));
        up_value += static_cast<float>(
            vq_block_dot_routed_q8_codebook_i32_t<VECTOR>(
                up_codebook + static_cast<long>(up_code) * VECTOR,
                input_block));
    }
    gate_value *= gate_scale * input_scales[0];
    up_value *= up_scale * input_scales[0];
}

__device__ __forceinline__ void vq_gemv_routed_pair_q8_codebook(
    const int64_t gate_index_address,
    const int64_t up_index_address,
    const int8_t* __restrict__ gate_codebook,
    const int8_t* __restrict__ up_codebook,
    const int8_t* __restrict__ input,
    const float* __restrict__ input_scales,
    const int blocks,
    const int vector,
    const int gate_dtype_tag,
    const int up_dtype_tag,
    const long index_count,
    const long index_row,
    const float gate_scale,
    const float up_scale,
    float& gate_value,
    float& up_value)
{
    if (vector == 16) {
        vq_gemv_routed_pair_q8_codebook_t<16>(
            gate_index_address, up_index_address, gate_codebook, up_codebook,
            input, input_scales, blocks, gate_dtype_tag, up_dtype_tag,
            index_count, index_row, gate_scale, up_scale, gate_value,
            up_value);
        return;
    }
    if (vector == 8) {
        vq_gemv_routed_pair_q8_codebook_t<8>(
            gate_index_address, up_index_address, gate_codebook, up_codebook,
            input, input_scales, blocks, gate_dtype_tag, up_dtype_tag,
            index_count, index_row, gate_scale, up_scale, gate_value,
            up_value);
        return;
    }
    vq_gemv_routed_pair_q8_codebook_t<4>(
        gate_index_address, up_index_address, gate_codebook, up_codebook,
        input, input_scales, blocks, gate_dtype_tag, up_dtype_tag,
        index_count, index_row, gate_scale, up_scale, gate_value, up_value);
}

constexpr int CCCP_Q8_P10_CODES = 1 << 10;
constexpr int CCCP_Q8_P10_SHARED_PERMUTATION = 5;
constexpr int CCCP_Q8_P11_CODES = 1 << 11;
constexpr int CCCP_Q8_P11_SHARED_PERMUTATION = 5;
constexpr int CCCP_Q8_P12_CODES = 1 << 12;
constexpr int CCCP_Q8_P12_SHARED_PERMUTATION = 5;
constexpr int CCCP_Q8_P13_CODES = 1 << 13;
constexpr int CCCP_Q8_P13_SHARED_PERMUTATION = 5;

__device__ __forceinline__ int cccp_q8_p10_shared_slot(const int code)
{
    return (code * CCCP_Q8_P10_SHARED_PERMUTATION) &
        (CCCP_Q8_P10_CODES - 1);
}

__device__ __forceinline__ int cccp_q8_p11_shared_slot(const int code)
{
    return (code * CCCP_Q8_P11_SHARED_PERMUTATION) &
        (CCCP_Q8_P11_CODES - 1);
}

__device__ __forceinline__ int cccp_q8_p12_shared_slot(const int code)
{
    return (code * CCCP_Q8_P12_SHARED_PERMUTATION) &
        (CCCP_Q8_P12_CODES - 1);
}

__device__ __forceinline__ int cccp_q8_p13_shared_slot(const int code)
{
    return (code * CCCP_Q8_P13_SHARED_PERMUTATION) &
        (CCCP_Q8_P13_CODES - 1);
}

__device__ __forceinline__ float vq_gemv_routed_row_q8_p11_d4_shared(
    const int64_t index_address,
    const int* __restrict__ codebook,
    const int8_t* __restrict__ input,
    const float input_scale,
    const int blocks,
    const long index_count,
    const long index_row,
    const float codebook_scale)
{
    float value = 0.0f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        const int code = routed_index_value_t<6>(
            index_address, index_row + block, index_count);
        const int input_value = *reinterpret_cast<const int*>(input + block * 4);
        value += static_cast<float>(__dp4a(
            codebook[cccp_q8_p11_shared_slot(code)],
            input_value,
            0));
    }
    return value * (codebook_scale * input_scale);
}

// P10/d4 Gate and Up each use a 4 KiB Q8 codebook.  A decode CTA reuses
// those random lookup rows across sixteen output rows, so stage both tables
// once and permute their bank index by five.  The compact VQ indices remain
// in HBM and no expanded expert-weight image is created.
__device__ __forceinline__ void vq_gemv_routed_pair_q8_p10_d4_shared(
    const int64_t gate_index_address,
    const int64_t up_index_address,
    const int* __restrict__ gate_codebook,
    const int* __restrict__ up_codebook,
    const int8_t* __restrict__ input,
    const float input_scale,
    const int blocks,
    const long index_count,
    const long index_row,
    const float gate_scale,
    const float up_scale,
    float& gate_value,
    float& up_value)
{
    gate_value = 0.0f;
    up_value = 0.0f;
    for (int block = threadIdx.x * 4;
         block < blocks;
         block += 32 * 4) {
        int gate_codes[4];
        int up_codes[4];
        routed_p10_index_quad(
            gate_index_address, index_row + block,
            index_count, gate_codes);
        routed_p10_index_quad(
            up_index_address, index_row + block,
            index_count, up_codes);
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            if (block + item >= blocks) break;
            const int input_value = *reinterpret_cast<const int*>(
                input + (block + item) * 4);
            gate_value += static_cast<float>(__dp4a(
                gate_codebook[cccp_q8_p10_shared_slot(gate_codes[item])],
                input_value,
                0));
            up_value += static_cast<float>(__dp4a(
                up_codebook[cccp_q8_p10_shared_slot(up_codes[item])],
                input_value,
                0));
        }
    }
    gate_value *= gate_scale * input_scale;
    up_value *= up_scale * input_scale;
}

// P11/d4 has twice as many entries as P10 but still fits both Gate and Up Q8
// codebooks in 16 KiB of shared memory.  Keep the compact 11-bit indices in
// HBM and reuse the staged codebook across every output row in this CTA.
__device__ __forceinline__ void vq_gemv_routed_pair_q8_p11_d4_shared(
    const int64_t gate_index_address,
    const int64_t up_index_address,
    const int* __restrict__ gate_codebook,
    const int* __restrict__ up_codebook,
    const int8_t* __restrict__ input,
    const float input_scale,
    const int blocks,
    const long index_count,
    const long index_row,
    const float gate_scale,
    const float up_scale,
    float& gate_value,
    float& up_value)
{
    gate_value = 0.0f;
    up_value = 0.0f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        const long offset = index_row + block;
        const int gate_code = routed_index_value_t<6>(
            gate_index_address, offset, index_count);
        const int up_code = routed_index_value_t<6>(
            up_index_address, offset, index_count);
        const int input_value = *reinterpret_cast<const int*>(input + block * 4);
        gate_value += static_cast<float>(__dp4a(
            gate_codebook[cccp_q8_p11_shared_slot(gate_code)],
            input_value,
            0));
        up_value += static_cast<float>(__dp4a(
            up_codebook[cccp_q8_p11_shared_slot(up_code)],
            input_value,
            0));
    }
    gate_value *= gate_scale * input_scale;
    up_value *= up_scale * input_scale;
}

// A P12/d4 Down codebook is 16 KiB.  Stage it once per CTA and reuse it for
// all output rows while leaving the packed indices in HBM.
__device__ __forceinline__ float vq_gemv_routed_row_q8_p12_d4_shared(
    const int64_t index_address,
    const int* __restrict__ codebook,
    const int8_t* __restrict__ input,
    const float input_scale,
    const int blocks,
    const long index_count,
    const long index_row,
    const float codebook_scale)
{
    float value = 0.0f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        const int code = routed_index_value_t<2>(
            index_address, index_row + block, index_count);
        const int input_value = *reinterpret_cast<const int*>(input + block * 4);
        value += static_cast<float>(__dp4a(
            codebook[cccp_q8_p12_shared_slot(code)],
            input_value,
            0));
    }
    return value * (codebook_scale * input_scale);
}

// P13/d4 is the most common high-precision Up layout.  Its 32 KiB Q8
// codebook is staged alone; Gate keeps its original compact HBM lookup so
// mixed-precision pairs do not require a second large shared allocation.
__device__ __forceinline__ float vq_gemv_routed_row_q8_p13_d4_shared(
    const int64_t index_address,
    const int* __restrict__ codebook,
    const int8_t* __restrict__ input,
    const float input_scale,
    const int blocks,
    const long index_count,
    const long index_row,
    const float codebook_scale)
{
    float value = 0.0f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        const int code = routed_index_value_t<7>(
            index_address, index_row + block, index_count);
        const int input_value = *reinterpret_cast<const int*>(input + block * 4);
        value += static_cast<float>(__dp4a(
            codebook[cccp_q8_p13_shared_slot(code)],
            input_value,
            0));
    }
    return value * (codebook_scale * input_scale);
}

// Quantize one activation row with one tensor scale.  The old compact path
// used one scale per tiny VQ block (d4/d8/d16), forcing every output-row warp
// to load and multiply hundreds of FP32 scales.  A route-global scale keeps
// the DP4A accumulation integer until the final multiply.  Gate and Up share
// the same input row, so their quantized activation is written only once.
__global__ void compact_q8_quantize_rows_global_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ quant_workspace,
    const int rows,
    const int input_cols,
    const int quantized_span,
    const int workspace_stride)
{
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    __shared__ float warp_maxima[8];
    __shared__ float row_scale;
    float absolute_max = 0.0f;
    const __nv_bfloat16* source = input + (long)row * input_cols;
    for (int column = threadIdx.x;
         column < input_cols;
         column += blockDim.x)
        absolute_max = fmaxf(
            absolute_max,
            fabsf(__bfloat162float(source[column])));
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        absolute_max = fmaxf(
            absolute_max,
            __shfl_down_sync(0xffffffffu, absolute_max, offset));
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0)
        warp_maxima[warp] = absolute_max;
    __syncthreads();
    if (warp == 0) {
        float block_max = threadIdx.x < 8
            ? warp_maxima[lane]
            : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            block_max = fmaxf(
                block_max,
                __shfl_down_sync(0xffffffffu, block_max, offset));
        if (lane == 0)
            row_scale = fmaxf(block_max, 1.0e-12f) * (1.0f / 127.0f);
    }
    uint8_t* route_workspace = quant_workspace +
        (long)row * workspace_stride;
    auto* quantized_input = reinterpret_cast<int8_t*>(route_workspace);
    auto* input_scales = reinterpret_cast<float*>(
        route_workspace + quantized_span);
    __syncthreads();
    if (threadIdx.x == 0)
        input_scales[0] = row_scale;
    const float inverse_scale = 1.0f / row_scale;
    for (int column = threadIdx.x;
         column < input_cols;
         column += blockDim.x) {
        const float value = __bfloat162float(source[column]) * inverse_scale;
        quantized_input[column] = static_cast<int8_t>(__float2int_rn(
            fminf(fmaxf(value, -127.0f), 127.0f)));
    }
}

template <int WARPS, int ROWS_PER_WARP>
__launch_bounds__(32 * WARPS, 3)
__global__ void vq_projection_gate_up_compact_q8_kernel(
    const uint8_t* __restrict__ quant_workspace,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ activated,
    const int top_k,
    const int expert_count,
    const int output_rows,
    const int input_cols,
    const int activation_kind,
    const float beta,
    const float linear_beta,
    const float limit)
{
    const int position = blockIdx.y;
    if (position >= top_k) return;
    const int expert = static_cast<int>(route_ids[position]);
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    const int quantized_span = (input_cols + 15) & ~15;
    const uint8_t* route_workspace = quant_workspace;
    const auto* quantized_input = reinterpret_cast<const int8_t*>(
        route_workspace);
    const auto* input_scales = reinterpret_cast<const float*>(
        route_workspace + quantized_span);
    __shared__ RoutedBlockMetadata gate_meta;
    __shared__ RoutedBlockMetadata up_meta;
    __shared__ float gate_scale;
    __shared__ float up_scale;
    __shared__ int shared_q8_d4_tag;
    __shared__ int shared_p13_gate_tag;
    __shared__ int shared_q8_words[CCCP_Q8_P13_CODES + CCCP_Q8_P11_CODES];
    if (linear_thread == 0) {
        gate_meta.valid = 0;
        up_meta.valid = 0;
        shared_q8_d4_tag = -1;
        shared_p13_gate_tag = -1;
        if (expert >= 0 && expert < expert_count) {
            gate_meta.index_address = metadata[expert];
            gate_meta.codebook_address =
                metadata[(long)expert_count + expert];
            gate_meta.blocks = static_cast<int>(
                metadata[(long)2 * expert_count + expert]);
            gate_meta.vector = static_cast<int>(
                metadata[(long)3 * expert_count + expert]);
            gate_meta.dtype_tag = static_cast<int>(
                metadata[(long)4 * expert_count + expert]);
            up_meta.index_address =
                metadata[(long)5 * expert_count + expert];
            up_meta.codebook_address =
                metadata[(long)6 * expert_count + expert];
            up_meta.blocks = static_cast<int>(
                metadata[(long)7 * expert_count + expert]);
            up_meta.vector = static_cast<int>(
                metadata[(long)8 * expert_count + expert]);
            up_meta.dtype_tag = static_cast<int>(
                metadata[(long)9 * expert_count + expert]);
            gate_meta.valid = gate_meta.index_address != 0 &&
                gate_meta.codebook_address != 0 && gate_meta.blocks > 0;
            up_meta.valid = up_meta.index_address != 0 &&
                up_meta.codebook_address != 0 && up_meta.blocks > 0;
            gate_scale = scales[(long)expert * 3];
            up_scale = scales[(long)expert * 3 + 1];
            const bool shared_q8_d4 =
                gate_meta.valid && up_meta.valid &&
                gate_meta.dtype_tag == up_meta.dtype_tag &&
                (gate_meta.dtype_tag == 4 || gate_meta.dtype_tag == 6) &&
                gate_meta.vector == 4 && up_meta.vector == 4 &&
                gate_meta.blocks == input_cols / 4 &&
                up_meta.blocks == input_cols / 4;
            if (shared_q8_d4)
                shared_q8_d4_tag = gate_meta.dtype_tag;
            else if (
                gate_meta.valid && up_meta.valid &&
                up_meta.dtype_tag == 7 && up_meta.vector == 4 &&
                up_meta.blocks == input_cols / 4) {
                shared_q8_d4_tag = 7;
                if (
                    (gate_meta.dtype_tag == 2 || gate_meta.dtype_tag == 6) &&
                    gate_meta.vector == 4 &&
                    gate_meta.blocks == input_cols / 4)
                    shared_p13_gate_tag = gate_meta.dtype_tag;
            }
        }
    }
    __syncthreads();
    if (!gate_meta.valid || !up_meta.valid) return;
    const bool shared_quantized_layout =
        up_meta.blocks == gate_meta.blocks &&
        up_meta.vector == gate_meta.vector;

    const auto* gate_codebook = reinterpret_cast<const int8_t*>(
        static_cast<uintptr_t>(gate_meta.codebook_address));
    const auto* up_codebook = reinterpret_cast<const int8_t*>(
        static_cast<uintptr_t>(up_meta.codebook_address));
    int* shared_gate_q8_d4 = shared_q8_words;
    int* shared_up_q8_d4 = shared_q8_words;
    if (shared_q8_d4_tag == 4 || shared_q8_d4_tag == 6) {
        const auto* gate_words = reinterpret_cast<const int*>(gate_codebook);
        const auto* up_words = reinterpret_cast<const int*>(up_codebook);
        const int code_count = shared_q8_d4_tag == 4 ?
            CCCP_Q8_P10_CODES : CCCP_Q8_P11_CODES;
        shared_up_q8_d4 = shared_q8_words + code_count;
        for (int code = linear_thread;
             code < code_count;
             code += 32 * WARPS) {
            const int slot = shared_q8_d4_tag == 4 ?
                cccp_q8_p10_shared_slot(code) :
                cccp_q8_p11_shared_slot(code);
            shared_gate_q8_d4[slot] = __ldg(gate_words + code);
            shared_up_q8_d4[slot] = __ldg(up_words + code);
        }
        __syncthreads();
    } else if (shared_q8_d4_tag == 7) {
        shared_up_q8_d4 = shared_q8_words;
        if (shared_p13_gate_tag == 2) {
            shared_gate_q8_d4 = shared_q8_words;
            const auto* gate_words = reinterpret_cast<const int*>(gate_codebook);
            for (int code = linear_thread;
                 code < CCCP_Q8_P12_CODES;
                 code += 32 * WARPS) {
                const int slot = cccp_q8_p12_shared_slot(code);
                shared_gate_q8_d4[slot] = __ldg(gate_words + code);
            }
        } else {
            const auto* up_words = reinterpret_cast<const int*>(up_codebook);
            shared_gate_q8_d4 = shared_q8_words + CCCP_Q8_P13_CODES;
            for (int code = linear_thread;
                 code < CCCP_Q8_P13_CODES;
                 code += 32 * WARPS) {
                const int slot = cccp_q8_p13_shared_slot(code);
                shared_up_q8_d4[slot] = __ldg(up_words + code);
            }
            if (shared_p13_gate_tag == 6) {
                const auto* gate_words = reinterpret_cast<const int*>(gate_codebook);
                for (int code = linear_thread;
                     code < CCCP_Q8_P11_CODES;
                     code += 32 * WARPS) {
                    const int slot = cccp_q8_p11_shared_slot(code);
                    shared_gate_q8_d4[slot] = __ldg(gate_words + code);
                }
            }
        }
        __syncthreads();
    }
    float gate_values[ROWS_PER_WARP] = {};
    float up_values[ROWS_PER_WARP] = {};
    if (shared_q8_d4_tag == 7 && shared_p13_gate_tag == 2) {
        // Two phases stage P12 Gate before replacing the buffer with P13 Up.
        #pragma unroll
        for (int item = 0; item < ROWS_PER_WARP; ++item) {
            const int row = blockIdx.x * (WARPS * ROWS_PER_WARP) +
                threadIdx.y + item * WARPS;
            if (row < output_rows)
                gate_values[item] = vq_gemv_routed_row_q8_p12_d4_shared(
                    gate_meta.index_address,
                    shared_gate_q8_d4,
                    quantized_input,
                    input_scales[0],
                    gate_meta.blocks,
                    static_cast<long>(output_rows) * gate_meta.blocks,
                    (long)row * gate_meta.blocks,
                    gate_scale);
        }
        __syncthreads();
        const auto* up_words = reinterpret_cast<const int*>(up_codebook);
        shared_up_q8_d4 = shared_q8_words;
        for (int code = linear_thread;
             code < CCCP_Q8_P13_CODES;
             code += 32 * WARPS) {
            const int slot = cccp_q8_p13_shared_slot(code);
            shared_up_q8_d4[slot] = __ldg(up_words + code);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int item = 0; item < ROWS_PER_WARP; ++item) {
        const int row = blockIdx.x * (WARPS * ROWS_PER_WARP) +
            threadIdx.y + item * WARPS;
        if (row >= output_rows) continue;
        if (shared_q8_d4_tag == 4) {
            vq_gemv_routed_pair_q8_p10_d4_shared(
                gate_meta.index_address,
                up_meta.index_address,
                shared_gate_q8_d4,
                shared_up_q8_d4,
                quantized_input,
                input_scales[0],
                gate_meta.blocks,
                static_cast<long>(output_rows) * gate_meta.blocks,
                (long)row * gate_meta.blocks,
                gate_scale,
                up_scale,
                gate_values[item],
                up_values[item]);
        } else if (shared_q8_d4_tag == 6) {
            vq_gemv_routed_pair_q8_p11_d4_shared(
                gate_meta.index_address,
                up_meta.index_address,
                shared_gate_q8_d4,
                shared_up_q8_d4,
                quantized_input,
                input_scales[0],
                gate_meta.blocks,
                static_cast<long>(output_rows) * gate_meta.blocks,
                (long)row * gate_meta.blocks,
                gate_scale,
                up_scale,
                gate_values[item],
                up_values[item]);
        } else if (shared_q8_d4_tag == 7) {
            if (shared_p13_gate_tag == 2) {
                // Gate was completed before P13 Up replaced the shared buffer.
            } else if (shared_p13_gate_tag == 6)
                gate_values[item] = vq_gemv_routed_row_q8_p11_d4_shared(
                    gate_meta.index_address,
                    shared_gate_q8_d4,
                    quantized_input,
                    input_scales[0],
                    gate_meta.blocks,
                    static_cast<long>(output_rows) * gate_meta.blocks,
                    (long)row * gate_meta.blocks,
                    gate_scale);
            else
                gate_values[item] = vq_gemv_routed_row_q8_codebook(
                    gate_meta.index_address,
                    gate_codebook,
                    quantized_input,
                    input_scales,
                    gate_meta.blocks,
                    gate_meta.vector,
                    gate_meta.dtype_tag,
                    static_cast<long>(output_rows) * gate_meta.blocks,
                    (long)row * gate_meta.blocks,
                    gate_scale);
            up_values[item] = vq_gemv_routed_row_q8_p13_d4_shared(
                up_meta.index_address,
                shared_up_q8_d4,
                quantized_input,
                input_scales[0],
                up_meta.blocks,
                static_cast<long>(output_rows) * up_meta.blocks,
                (long)row * up_meta.blocks,
                up_scale);
        } else if (shared_quantized_layout) {
            vq_gemv_routed_pair_q8_codebook(
                gate_meta.index_address,
                up_meta.index_address,
                gate_codebook,
                up_codebook,
                quantized_input,
                input_scales,
                gate_meta.blocks,
                gate_meta.vector,
                gate_meta.dtype_tag,
                up_meta.dtype_tag,
                static_cast<long>(output_rows) * gate_meta.blocks,
                (long)row * gate_meta.blocks,
                gate_scale,
                up_scale,
                gate_values[item],
                up_values[item]);
        } else {
            gate_values[item] = vq_gemv_routed_row_q8_codebook(
                gate_meta.index_address,
                gate_codebook,
                quantized_input,
                input_scales,
                gate_meta.blocks,
                gate_meta.vector,
                gate_meta.dtype_tag,
                static_cast<long>(output_rows) * gate_meta.blocks,
                (long)row * gate_meta.blocks,
                gate_scale);
            up_values[item] = vq_gemv_routed_row_q8_codebook(
                up_meta.index_address,
                up_codebook,
                quantized_input,
                input_scales,
                up_meta.blocks,
                up_meta.vector,
                up_meta.dtype_tag,
                static_cast<long>(output_rows) * up_meta.blocks,
                (long)row * up_meta.blocks,
                up_scale);
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int item = 0; item < ROWS_PER_WARP; ++item) {
            gate_values[item] += __shfl_down_sync(
                0xffffffffu, gate_values[item], offset);
            up_values[item] += __shfl_down_sync(
                0xffffffffu, up_values[item], offset);
        }
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int item = 0; item < ROWS_PER_WARP; ++item) {
            const int row = blockIdx.x * (WARPS * ROWS_PER_WARP) +
                threadIdx.y + item * WARPS;
            if (row < output_rows)
                activated[(long)position * output_rows + row] =
                    __float2bfloat16_rn(projection_gate_up_activation(
                        gate_values[item], up_values[item], activation_kind,
                        beta, linear_beta, limit));
        }
    }
}

template <int WARPS, int ROWS_PER_WARP>
__global__ void vq_projection_down_compact_q8_kernel(
    const uint8_t* __restrict__ quant_workspace,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ output,
    const int top_k,
    const int expert_count,
    const int output_rows,
    const int input_cols)
{
    const int position = blockIdx.y;
    if (position >= top_k) return;
    const int expert = static_cast<int>(route_ids[position]);
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    const int quantized_span = (input_cols + 15) & ~15;
    const uint8_t* route_workspace = quant_workspace +
        (long)position * 2 * quantized_span;
    const auto* quantized_input = reinterpret_cast<const int8_t*>(
        route_workspace);
    const auto* input_scales = reinterpret_cast<const float*>(
        route_workspace + quantized_span);
    __shared__ RoutedBlockMetadata down_meta;
    __shared__ float down_scale;
    __shared__ int shared_p12_d4;
    __shared__ int shared_down_q8_d4[CCCP_Q8_P12_CODES];
    if (linear_thread == 0) {
        down_meta.valid = 0;
        shared_p12_d4 = 0;
        if (expert >= 0 && expert < expert_count) {
            down_meta.index_address =
                metadata[(long)10 * expert_count + expert];
            down_meta.codebook_address =
                metadata[(long)11 * expert_count + expert];
            down_meta.blocks = static_cast<int>(
                metadata[(long)12 * expert_count + expert]);
            down_meta.vector = static_cast<int>(
                metadata[(long)13 * expert_count + expert]);
            down_meta.dtype_tag = static_cast<int>(
                metadata[(long)14 * expert_count + expert]);
            down_meta.valid = down_meta.index_address != 0 &&
                down_meta.codebook_address != 0 && down_meta.blocks > 0;
            down_scale = scales[(long)expert * 3 + 2];
            shared_p12_d4 = down_meta.valid &&
                down_meta.dtype_tag == 2 && down_meta.vector == 4 &&
                down_meta.blocks == input_cols / 4;
        }
    }
    __syncthreads();
    if (!down_meta.valid) return;
    const auto* codebook = reinterpret_cast<const int8_t*>(
        static_cast<uintptr_t>(down_meta.codebook_address));
    if (shared_p12_d4) {
        const auto* codebook_words = reinterpret_cast<const int*>(codebook);
        for (int code = linear_thread;
             code < CCCP_Q8_P12_CODES;
             code += 32 * WARPS) {
            const int slot = cccp_q8_p12_shared_slot(code);
            shared_down_q8_d4[slot] = __ldg(codebook_words + code);
        }
        __syncthreads();
    }
    float values[ROWS_PER_WARP] = {};
    #pragma unroll
    for (int item = 0; item < ROWS_PER_WARP; ++item) {
        const int row = blockIdx.x * (WARPS * ROWS_PER_WARP) +
            threadIdx.y + item * WARPS;
        if (row < output_rows && shared_p12_d4)
            values[item] = vq_gemv_routed_row_q8_p12_d4_shared(
                down_meta.index_address,
                shared_down_q8_d4,
                quantized_input,
                input_scales[0],
                down_meta.blocks,
                static_cast<long>(output_rows) * down_meta.blocks,
                (long)row * down_meta.blocks,
                down_scale);
        else if (row < output_rows)
            values[item] = vq_gemv_routed_row_q8_codebook(
                down_meta.index_address,
                codebook,
                quantized_input,
                input_scales,
                down_meta.blocks,
                down_meta.vector,
                down_meta.dtype_tag,
                static_cast<long>(output_rows) * down_meta.blocks,
                (long)row * down_meta.blocks,
                down_scale);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int item = 0; item < ROWS_PER_WARP; ++item)
            values[item] += __shfl_down_sync(
                0xffffffffu, values[item], offset);
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int item = 0; item < ROWS_PER_WARP; ++item) {
            const int row = blockIdx.x * (WARPS * ROWS_PER_WARP) +
                threadIdx.y + item * WARPS;
            if (row < output_rows)
                output[(long)position * output_rows + row] =
                    __float2bfloat16_rn(values[item]);
        }
    }
}

// Common three-projection decode path. Gate and Up share the same input, so
// compute both in one CTA and apply the registered gated activation before
// writing the BF16 workspace.
// For p10/d8-k1024, each 16 KiB codebook is staged with a two-BF16 pad per
// entry. The resulting 20-byte stride distributes random lookup rows across
// all shared-memory banks instead of only eight starting banks. Other layouts
// remain correct through the generic packed row decoder, without a
// model-specific kernel or a dequantized matrix.
template <int WARPS, int ROWS_PER_WARP, bool TILE_VIEW>
__global__ void vq_projection_gate_up_situ_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    __nv_bfloat16* __restrict__ activated,
    const int top_k,
    const int batch_size,
    const int expert_count,
    const int output_rows,
    const int input_cols,
    const int activation_kind,
    const float beta,
    const float linear_beta,
    const float limit,
    const bool stage_p10,
    const int metadata_rows)
{
    const int position = blockIdx.z * top_k + blockIdx.y;
    if (position >= batch_size * top_k)
        return;
    const int input_row_index = position / top_k;
    const int expert = static_cast<int>(route_ids[position]);
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    constexpr int block_threads = 32 * WARPS;
    extern __shared__ unsigned char raw_shared[];
    auto* shared_input =
        reinterpret_cast<__nv_bfloat16*>(raw_shared);
    auto* shared_gate_codebook = shared_input + input_cols;
    auto* shared_up_codebook =
        shared_gate_codebook +
        CCCP_PROJECTION_P10_CODES *
            CCCP_PROJECTION_P10_SHARED_STRIDE;
    __shared__ RoutedBlockMetadata gate_meta;
    __shared__ RoutedBlockMetadata up_meta;
    __shared__ RoutedTileMetadata gate_tile;
    __shared__ RoutedTileMetadata up_tile;
    __shared__ int shared_p10;

    if (linear_thread == 0) {
        gate_meta.valid = 0;
        up_meta.valid = 0;
        shared_p10 = 0;
        if (expert >= 0 && expert < expert_count) {
            gate_meta.index_address =
                metadata[(long)0 * expert_count + expert];
            gate_meta.codebook_address =
                metadata[(long)1 * expert_count + expert];
            gate_meta.blocks = static_cast<int>(
                metadata[(long)2 * expert_count + expert]);
            gate_meta.vector = static_cast<int>(
                metadata[(long)3 * expert_count + expert]);
            gate_meta.dtype_tag = static_cast<int>(
                metadata[(long)4 * expert_count + expert]);
            gate_meta.valid = (
                gate_meta.index_address != 0 &&
                gate_meta.codebook_address != 0 &&
                gate_meta.blocks > 0
            );
            up_meta.index_address =
                metadata[(long)5 * expert_count + expert];
            up_meta.codebook_address =
                metadata[(long)6 * expert_count + expert];
            up_meta.blocks = static_cast<int>(
                metadata[(long)7 * expert_count + expert]);
            up_meta.vector = static_cast<int>(
                metadata[(long)8 * expert_count + expert]);
            up_meta.dtype_tag = static_cast<int>(
                metadata[(long)9 * expert_count + expert]);
            up_meta.valid = (
                up_meta.index_address != 0 &&
                up_meta.codebook_address != 0 &&
                up_meta.blocks > 0
            );
            shared_p10 = (
                stage_p10 &&
                gate_meta.valid &&
                up_meta.valid &&
                gate_meta.dtype_tag == 4 &&
                up_meta.dtype_tag == 4 &&
                gate_meta.vector == CCCP_PROJECTION_P10_VECTOR &&
                up_meta.vector == CCCP_PROJECTION_P10_VECTOR &&
                gate_meta.blocks * gate_meta.vector == input_cols &&
                up_meta.blocks * up_meta.vector == input_cols
            );
            if constexpr (TILE_VIEW) {
                gate_tile = projection_tile_metadata(
                    metadata,
                    metadata_rows,
                    expert_count,
                    expert,
                    0,
                    gate_meta.blocks,
                    gate_meta.dtype_tag);
                up_tile = projection_tile_metadata(
                    metadata,
                    metadata_rows,
                    expert_count,
                    expert,
                    1,
                    up_meta.blocks,
                    up_meta.dtype_tag);
            }
        }
    }
    const auto* input4 = reinterpret_cast<const uint4*>(
        input + static_cast<long>(input_row_index) * input_cols);
    auto* shared_input4 = reinterpret_cast<uint4*>(shared_input);
    for (
        int item = linear_thread;
        item < input_cols / 8;
        item += block_threads
    )
        shared_input4[item] = input4[item];
    __syncthreads();
    if (!gate_meta.valid || !up_meta.valid)
        return;

    const auto* gate_global =
        reinterpret_cast<const __nv_bfloat16*>(
            static_cast<uintptr_t>(gate_meta.codebook_address));
    const auto* up_global =
        reinterpret_cast<const __nv_bfloat16*>(
            static_cast<uintptr_t>(up_meta.codebook_address));
    if (shared_p10) {
        for (
            int item = linear_thread;
            item < CCCP_PROJECTION_P10_CODES *
                CCCP_PROJECTION_P10_VECTOR;
            item += block_threads
        ) {
            const int code = item / CCCP_PROJECTION_P10_VECTOR;
            const int component = item % CCCP_PROJECTION_P10_VECTOR;
            const int target =
                code * CCCP_PROJECTION_P10_SHARED_STRIDE + component;
            shared_gate_codebook[target] = gate_global[item];
            shared_up_codebook[target] = up_global[item];
        }
    }
    __syncthreads();

    const auto* gate_codebook = (
        shared_p10 ? shared_gate_codebook : gate_global
    );
    const auto* up_codebook = (
        shared_p10 ? shared_up_codebook : up_global
    );
    float gate_values[ROWS_PER_WARP] = {};
    float up_values[ROWS_PER_WARP] = {};
    #pragma unroll
    for (
        int item = 0;
        item < ROWS_PER_WARP;
        ++item
    ) {
        const int row =
            blockIdx.x *
                (WARPS * ROWS_PER_WARP) +
            threadIdx.y +
            item * WARPS;
        if (row < output_rows) {
            if (shared_p10) {
                vq_gemv_routed_p10_pair(
                    gate_meta.index_address,
                    up_meta.index_address,
                    gate_codebook,
                    up_codebook,
                    shared_input,
                    gate_meta.blocks,
                    (long)row * gate_meta.blocks,
                    gate_values[item],
                    up_values[item]);
            } else if (
                TILE_VIEW &&
                gate_meta.blocks == up_meta.blocks &&
                gate_meta.vector == up_meta.vector &&
                gate_tile.valid && up_tile.valid
            ) {
                vq_gemv_routed_pair_tiled(
                    gate_meta.index_address,
                    up_meta.index_address,
                    gate_codebook,
                    up_codebook,
                    shared_input,
                    gate_meta.blocks,
                    gate_meta.vector,
                    gate_meta.dtype_tag,
                    up_meta.dtype_tag,
                    row,
                    gate_tile,
                    up_tile,
                    gate_values[item],
                    up_values[item]);
            } else if (
                gate_meta.blocks == up_meta.blocks &&
                gate_meta.vector == up_meta.vector &&
                gate_meta.dtype_tag == up_meta.dtype_tag &&
                (
                    gate_meta.dtype_tag == 6 ||
                    gate_meta.dtype_tag == 7 ||
                    gate_meta.dtype_tag == 8
                )
            ) {
                vq_gemv_routed_pair(
                    gate_meta.index_address,
                    up_meta.index_address,
                    gate_codebook,
                    up_codebook,
                    shared_input,
                    gate_meta.blocks,
                    gate_meta.vector,
                    gate_meta.dtype_tag,
                    (long)row * gate_meta.blocks,
                    gate_values[item],
                    up_values[item]);
            } else {
                if constexpr (TILE_VIEW) {
                    gate_values[item] = (
                        gate_tile.valid
                        ? vq_gemv_routed_row_tiled(
                            gate_meta.index_address,
                            gate_codebook,
                            shared_input,
                            gate_meta.blocks,
                            gate_meta.vector,
                            gate_meta.dtype_tag,
                            row,
                            gate_tile)
                        : vq_gemv_routed_row(
                            gate_meta.index_address,
                            gate_codebook,
                            shared_input,
                            gate_meta.blocks,
                            gate_meta.vector,
                            gate_meta.dtype_tag,
                            (long)row * gate_meta.blocks)
                    );
                    up_values[item] = (
                        up_tile.valid
                        ? vq_gemv_routed_row_tiled(
                            up_meta.index_address,
                            up_codebook,
                            shared_input,
                            up_meta.blocks,
                            up_meta.vector,
                            up_meta.dtype_tag,
                            row,
                            up_tile)
                        : vq_gemv_routed_row(
                            up_meta.index_address,
                            up_codebook,
                            shared_input,
                            up_meta.blocks,
                            up_meta.vector,
                            up_meta.dtype_tag,
                            (long)row * up_meta.blocks)
                    );
                } else {
                    gate_values[item] = vq_gemv_routed_row(
                        gate_meta.index_address,
                        gate_codebook,
                        shared_input,
                        gate_meta.blocks,
                        gate_meta.vector,
                        gate_meta.dtype_tag,
                        (long)row * gate_meta.blocks);
                    up_values[item] = vq_gemv_routed_row(
                        up_meta.index_address,
                        up_codebook,
                        shared_input,
                        up_meta.blocks,
                        up_meta.vector,
                        up_meta.dtype_tag,
                        (long)row * up_meta.blocks);
                }
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (
            int item = 0;
            item < ROWS_PER_WARP;
            ++item
        ) {
            gate_values[item] += __shfl_down_sync(
                0xffffffffu,
                gate_values[item],
                offset);
            up_values[item] += __shfl_down_sync(
                0xffffffffu,
                up_values[item],
                offset);
        }
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (
            int item = 0;
            item < ROWS_PER_WARP;
            ++item
        ) {
            const int row =
                blockIdx.x *
                    (WARPS * ROWS_PER_WARP) +
                threadIdx.y +
                item * WARPS;
            if (row < output_rows) {
                const float gate = __bfloat162float(
                    __float2bfloat16_rn(gate_values[item]));
                const float up = __bfloat162float(
                    __float2bfloat16_rn(up_values[item]));
                activated[(long)position * output_rows + row] =
                    __float2bfloat16_rn(projection_gate_up_activation(
                        gate,
                        up,
                        activation_kind,
                        beta,
                        linear_beta,
                        limit));
            }
        }
    }
}

template <int WARPS, int ROWS_PER_WARP, bool TILE_VIEW>
inline void launch_vq_projection_gate_up_situ_impl(
    const __nv_bfloat16* input,
    const int64_t* route_ids,
    const int64_t* metadata,
    __nv_bfloat16* activated,
    const int top_k,
    const int batch_size,
    const int expert_count,
    const int output_rows,
    const int input_cols,
    const int activation_kind,
    const float beta,
    const float linear_beta,
    const float limit,
    const bool stage_p10,
    const int metadata_rows,
    cudaStream_t stream)
{
    const size_t shared_bytes = static_cast<size_t>(
        input_cols +
        (
            stage_p10
                ? 2 * CCCP_PROJECTION_P10_CODES *
                    CCCP_PROJECTION_P10_SHARED_STRIDE
                : 0
        )
    ) * sizeof(__nv_bfloat16);
    // p10 requests exactly 48 KiB of dynamic shared memory for a 4096-wide
    // projection, in addition to the kernel's static metadata.  It therefore
    // still has to opt in to the device's larger shared-memory limit even
    // though the dynamic portion alone is not greater than 48 KiB.
    if (stage_p10) {
        const auto status = cccp_gpu_func_set_attribute(
            vq_projection_gate_up_situ_kernel<
                WARPS, ROWS_PER_WARP, TILE_VIEW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        TORCH_CHECK(
            status == cudaSuccess,
            "failed to configure p10 projection shared memory: ",
            cudaGetErrorString(status));
    }
    vq_projection_gate_up_situ_kernel<
        WARPS, ROWS_PER_WARP, TILE_VIEW><<<
        dim3(
            (unsigned)(
                (
                    output_rows +
                    WARPS * ROWS_PER_WARP - 1
                ) /
                (WARPS * ROWS_PER_WARP)
            ),
            (unsigned)top_k,
            (unsigned)batch_size),
        dim3(32, WARPS),
        shared_bytes,
        stream>>>(
            input,
            route_ids,
            metadata,
            activated,
            top_k,
            batch_size,
            expert_count,
            output_rows,
            input_cols,
            activation_kind,
            beta,
            linear_beta,
            limit,
            stage_p10,
            metadata_rows);
}

template <int WARPS>
inline void launch_vq_projection_gate_up_situ(
    const __nv_bfloat16* input,
    const int64_t* route_ids,
    const int64_t* metadata,
    __nv_bfloat16* activated,
    const int top_k,
    const int batch_size,
    const int expert_count,
    const int output_rows,
    const int input_cols,
    const int activation_kind,
    const float beta,
    const float linear_beta,
    const float limit,
    const bool stage_p10,
    const int metadata_rows,
    const int p10_rows_per_warp,
    const int generic_rows_per_warp,
    cudaStream_t stream)
{
    // Large heterogeneous codebooks are latency-bound random gathers.  One
    // row per warp exposes four times as many independent CTAs on H20.  The
    // p10 specialization keeps four rows per warp because staging its small
    // paired codebooks is the dominant reusable work.  A public tuner lets
    // one CTA reuse that staged pair for more rows without materialising an
    // expanded weight tensor.
    if (stage_p10) {
        const int rows = (
            p10_rows_per_warp == 16 || p10_rows_per_warp == 8
                ? p10_rows_per_warp
                : 4);
        if (metadata_rows == CCCP_PROJECTION_TILE_META_ROWS) {
            if (rows == 16)
                launch_vq_projection_gate_up_situ_impl<WARPS, 16, true>(
                    input, route_ids, metadata, activated, top_k, batch_size, expert_count,
                    output_rows, input_cols, activation_kind, beta, linear_beta,
                    limit, true, metadata_rows, stream);
            else if (rows == 8)
                launch_vq_projection_gate_up_situ_impl<WARPS, 8, true>(
                    input, route_ids, metadata, activated, top_k, batch_size, expert_count,
                    output_rows, input_cols, activation_kind, beta, linear_beta,
                    limit, true, metadata_rows, stream);
            else
                launch_vq_projection_gate_up_situ_impl<WARPS, 4, true>(
                    input, route_ids, metadata, activated, top_k, batch_size, expert_count,
                    output_rows, input_cols, activation_kind, beta, linear_beta,
                    limit, true, metadata_rows, stream);
        } else if (rows == 16) {
            launch_vq_projection_gate_up_situ_impl<WARPS, 16, false>(
                input, route_ids, metadata, activated, top_k, batch_size, expert_count,
                output_rows, input_cols, activation_kind, beta, linear_beta,
                limit, true, metadata_rows, stream);
        } else if (rows == 8) {
            launch_vq_projection_gate_up_situ_impl<WARPS, 8, false>(
                input, route_ids, metadata, activated, top_k, batch_size, expert_count,
                output_rows, input_cols, activation_kind, beta, linear_beta,
                limit, true, metadata_rows, stream);
        } else {
            launch_vq_projection_gate_up_situ_impl<WARPS, 4, false>(
                input, route_ids, metadata, activated, top_k, batch_size, expert_count,
                output_rows, input_cols, activation_kind, beta, linear_beta,
                limit, true, metadata_rows, stream);
        }
    } else {
        // p16/p14 codebooks are not staged in shared memory.  The generic
        // path can safely compute multiple output rows per warp: the input
        // tile is loaded once per CTA and the extra rows only reduce grid
        // scheduling overhead.  Keep the historical one-row default, while
        // allowing a public runtime tuner to select 2 or 4 rows on GPUs
        // where the larger register footprint wins.
        const int rows = (
            generic_rows_per_warp == 4 || generic_rows_per_warp == 2
                ? generic_rows_per_warp
                : 1);
        if (metadata_rows == CCCP_PROJECTION_TILE_META_ROWS) {
            if (rows == 4)
                launch_vq_projection_gate_up_situ_impl<WARPS, 4, true>(
                    input, route_ids, metadata, activated, top_k, batch_size, expert_count,
                    output_rows, input_cols, activation_kind, beta, linear_beta,
                    limit, false, metadata_rows, stream);
            else if (rows == 2)
                launch_vq_projection_gate_up_situ_impl<WARPS, 2, true>(
                    input, route_ids, metadata, activated, top_k, batch_size, expert_count,
                    output_rows, input_cols, activation_kind, beta, linear_beta,
                    limit, false, metadata_rows, stream);
            else
                launch_vq_projection_gate_up_situ_impl<WARPS, 1, true>(
                    input, route_ids, metadata, activated, top_k, batch_size, expert_count,
                    output_rows, input_cols, activation_kind, beta, linear_beta,
                    limit, false, metadata_rows, stream);
        } else if (rows == 4) {
            launch_vq_projection_gate_up_situ_impl<WARPS, 4, false>(
                input, route_ids, metadata, activated, top_k, batch_size, expert_count,
                output_rows, input_cols, activation_kind, beta, linear_beta,
                limit, false, metadata_rows, stream);
        } else if (rows == 2) {
            launch_vq_projection_gate_up_situ_impl<WARPS, 2, false>(
                input, route_ids, metadata, activated, top_k, batch_size, expert_count,
                output_rows, input_cols, activation_kind, beta, linear_beta,
                limit, false, metadata_rows, stream);
        } else {
            launch_vq_projection_gate_up_situ_impl<WARPS, 1, false>(
                input, route_ids, metadata, activated, top_k, batch_size, expert_count,
                output_rows, input_cols, activation_kind, beta, linear_beta,
                limit, false, metadata_rows, stream);
        }
    }
}

// Batch routes by expert before the projection.  The legacy prefill kernel
// assigns one CTA to every (token, expert, output-tile), which rereads the
// same p16 index/codebook bytes once per token.  This public grouped variant
// assigns a CTA to one expert and a small tile of token rows; each warp reads
// the expert index/codebook entry once and accumulates all token dots.  It
// keeps the packed representation intact and accepts both the legacy fused
// Gate+Up/Down directory [10,E] and the three-projection directory [15,E].
template <int WARPS, int TOKENS>
__global__ void vq_projection_gate_up_grouped_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int64_t* __restrict__ token_ids,
    const int64_t* __restrict__ group_experts,
    const int32_t* __restrict__ group_offsets,
    const int64_t* __restrict__ metadata,
    __nv_bfloat16* __restrict__ activated,
    const int groups,
    const int expert_count,
    const int output_rows,
    const int input_cols,
    const int activation_kind,
    const float beta,
    const float linear_beta,
    const float limit,
    const int metadata_rows,
    const int top_k)
{
    const int group = blockIdx.z;
    if (group >= groups)
        return;
    const int begin = group_offsets[group];
    const int end = group_offsets[group + 1];
    const int tile_begin = begin + blockIdx.y * TOKENS;
    if (tile_begin >= end || blockIdx.y >= (end - begin + TOKENS - 1) / TOKENS)
        return;
    const int tile_count = min(TOKENS, end - tile_begin);
    const int expert = static_cast<int>(group_experts[group]);
    const int row = blockIdx.x * WARPS + threadIdx.y;
    extern __shared__ __nv_bfloat16 grouped_input[];
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    for (int item = linear_thread; item < tile_count * input_cols;
         item += WARPS * 32)
        grouped_input[item] = input[
            (long)static_cast<int>(token_ids[tile_begin + item / input_cols]) *
                input_cols + item % input_cols];
    __shared__ RoutedBlockMetadata gate_meta;
    __shared__ RoutedBlockMetadata up_meta;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        gate_meta.index_address = metadata[(long)0 * expert_count + expert];
        gate_meta.codebook_address = metadata[(long)1 * expert_count + expert];
        gate_meta.blocks = static_cast<int>(metadata[(long)2 * expert_count + expert]);
        gate_meta.vector = static_cast<int>(metadata[(long)3 * expert_count + expert]);
        gate_meta.dtype_tag = static_cast<int>(metadata[(long)4 * expert_count + expert]);
        gate_meta.valid = gate_meta.index_address != 0 &&
            gate_meta.codebook_address != 0 && gate_meta.blocks > 0;
        const int up_base = metadata_rows == 10 ? 0 : 5;
        up_meta.index_address = metadata[(long)(up_base + 0) * expert_count + expert];
        up_meta.codebook_address = metadata[(long)(up_base + 1) * expert_count + expert];
        up_meta.blocks = static_cast<int>(metadata[(long)(up_base + 2) * expert_count + expert]);
        up_meta.vector = static_cast<int>(metadata[(long)(up_base + 3) * expert_count + expert]);
        up_meta.dtype_tag = static_cast<int>(metadata[(long)(up_base + 4) * expert_count + expert]);
        up_meta.valid = up_meta.index_address != 0 &&
            up_meta.codebook_address != 0 && up_meta.blocks > 0;
    }
    __syncthreads();
    if (expert < 0 || row >= output_rows)
        return;
    if (!gate_meta.valid || !up_meta.valid)
        return;
    const auto* gate_cb = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(gate_meta.codebook_address));
    const auto* up_cb = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(up_meta.codebook_address));
    float gate_values[TOKENS] = {};
    float up_values[TOKENS] = {};
    const bool same_layout =
        gate_meta.blocks == up_meta.blocks &&
        gate_meta.vector == up_meta.vector &&
        gate_meta.dtype_tag == up_meta.dtype_tag;
    if (same_layout) {
        for (int block = threadIdx.x; block < gate_meta.blocks; block += 32) {
            const long gate_index = (long)row * gate_meta.blocks + block;
            const long up_index = (
                metadata_rows == 10
                    ? (long)(row + output_rows) * up_meta.blocks + block
                    : (long)row * up_meta.blocks + block);
            const int gate_code = routed_index_value(
                gate_meta.index_address, gate_meta.dtype_tag, gate_index);
            const int up_code = routed_index_value(
                up_meta.index_address, up_meta.dtype_tag, up_index);
            const auto* gate_row = gate_cb +
                (long)gate_code * gate_meta.vector;
            const auto* up_row = up_cb +
                (long)up_code * up_meta.vector;
            for (int item = 0; item < TOKENS; ++item) {
                if (item >= tile_count)
                    break;
                const auto* x = grouped_input +
                    item * input_cols + block * gate_meta.vector;
                gate_values[item] += vq_block_dot_routed_bf16(
                    gate_row, x, gate_meta.vector);
                up_values[item] += vq_block_dot_routed_bf16(
                    up_row, x, up_meta.vector);
            }
        }
    } else {
        // Projection-VQ chooses Gate and Up formats independently.  A p11/d8
        // Gate beside a p13/d8 Up is valid and common; treating that pair as
        // absent leaves the activation workspace uninitialized and poisons
        // the entire Prefill with NaNs.  Decode each projection with its own
        // block/vector/tag while retaining the combined loop above for the
        // homogeneous fast path.
        for (int block = threadIdx.x; block < gate_meta.blocks; block += 32) {
            const long index = (long)row * gate_meta.blocks + block;
            const int code = routed_index_value(
                gate_meta.index_address, gate_meta.dtype_tag, index);
            const auto* codebook_row = gate_cb +
                (long)code * gate_meta.vector;
            for (int item = 0; item < TOKENS; ++item) {
                if (item >= tile_count)
                    break;
                const auto* x = grouped_input +
                    item * input_cols + block * gate_meta.vector;
                gate_values[item] += vq_block_dot_routed_bf16(
                    codebook_row, x, gate_meta.vector);
            }
        }
        for (int block = threadIdx.x; block < up_meta.blocks; block += 32) {
            const long index = (
                metadata_rows == 10
                    ? (long)(row + output_rows) * up_meta.blocks + block
                    : (long)row * up_meta.blocks + block);
            const int code = routed_index_value(
                up_meta.index_address, up_meta.dtype_tag, index);
            const auto* codebook_row = up_cb +
                (long)code * up_meta.vector;
            for (int item = 0; item < TOKENS; ++item) {
                if (item >= tile_count)
                    break;
                const auto* x = grouped_input +
                    item * input_cols + block * up_meta.vector;
                up_values[item] += vq_block_dot_routed_bf16(
                    codebook_row, x, up_meta.vector);
            }
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int item = 0; item < TOKENS; ++item) {
            gate_values[item] += __shfl_down_sync(0xffffffffu, gate_values[item], offset);
            up_values[item] += __shfl_down_sync(0xffffffffu, up_values[item], offset);
        }
    }
    if (threadIdx.x == 0) {
        for (int item = 0; item < TOKENS; ++item) {
            if (item >= tile_count)
                break;
            const int sorted = tile_begin + item;
            const float gate = __bfloat162float(__float2bfloat16_rn(gate_values[item]));
            const float up = __bfloat162float(__float2bfloat16_rn(up_values[item]));
            activated[(long)sorted * output_rows + row] =
                __float2bfloat16_rn(projection_gate_up_activation(
                    gate, up, activation_kind, beta, linear_beta, limit));
        }
    }
}

template <int WARPS, int TOKENS>
__global__ void vq_projection_down_grouped_kernel(
    const __nv_bfloat16* __restrict__ activated,
    const int64_t* __restrict__ token_ids,
    const int64_t* __restrict__ group_experts,
    const int32_t* __restrict__ group_offsets,
    const float* __restrict__ weights,
    const int64_t* __restrict__ metadata,
    float* __restrict__ result,
    const int groups,
    const int expert_count,
    const int output_rows,
    const int input_cols,
    const int metadata_rows)
{
    const int group = blockIdx.z;
    if (group >= groups)
        return;
    const int begin = group_offsets[group];
    const int end = group_offsets[group + 1];
    const int tile_begin = begin + blockIdx.y * TOKENS;
    if (tile_begin >= end || blockIdx.y >= (end - begin + TOKENS - 1) / TOKENS)
        return;
    const int tile_count = min(TOKENS, end - tile_begin);
    const int expert = static_cast<int>(group_experts[group]);
    const int row = blockIdx.x * WARPS + threadIdx.y;
    extern __shared__ __nv_bfloat16 grouped_input[];
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    for (int item = linear_thread; item < tile_count * input_cols;
         item += WARPS * 32)
        grouped_input[item] = activated[
            (long)tile_begin * input_cols + item / input_cols * input_cols +
            item % input_cols];
    __shared__ RoutedBlockMetadata down_meta;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        const int down_base = metadata_rows == 10 ? 5 : 10;
        down_meta.index_address = metadata[(long)(down_base + 0) * expert_count + expert];
        down_meta.codebook_address = metadata[(long)(down_base + 1) * expert_count + expert];
        down_meta.blocks = static_cast<int>(metadata[(long)(down_base + 2) * expert_count + expert]);
        down_meta.vector = static_cast<int>(metadata[(long)(down_base + 3) * expert_count + expert]);
        down_meta.dtype_tag = static_cast<int>(metadata[(long)(down_base + 4) * expert_count + expert]);
        down_meta.valid = down_meta.index_address != 0 &&
            down_meta.codebook_address != 0 && down_meta.blocks > 0;
    }
    __syncthreads();
    if (expert < 0 || row >= output_rows)
        return;
    if (!down_meta.valid)
        return;
    const auto* codebook = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(down_meta.codebook_address));
    float values[TOKENS] = {};
    for (int block = threadIdx.x; block < down_meta.blocks; block += 32) {
        const int code = routed_index_value(
            down_meta.index_address, down_meta.dtype_tag,
            (long)row * down_meta.blocks + block);
        const auto* cb = codebook + (long)code * down_meta.vector;
        for (int item = 0; item < TOKENS; ++item) {
            if (item >= tile_count)
                break;
            const int sorted = tile_begin + item;
            const auto* x = grouped_input + item * input_cols +
                block * down_meta.vector;
            values[item] += vq_block_dot_routed_bf16(
                cb, x, down_meta.vector);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        #pragma unroll
        for (int item = 0; item < TOKENS; ++item)
            values[item] += __shfl_down_sync(0xffffffffu, values[item], offset);
    if (threadIdx.x == 0) {
        for (int item = 0; item < TOKENS; ++item) {
            if (item >= tile_count)
                break;
            const int sorted = tile_begin + item;
            const int token = static_cast<int>(token_ids[sorted]);
            const float rounded = __bfloat162float(__float2bfloat16_rn(values[item]));
            atomicAdd(result + (long)token * output_rows + row,
                      rounded * weights[sorted]);
        }
    }
}


// ---------------------------------------------------------------------------
// Public prefill dequant: expand one packed two- or three-projection matrix per
// expert into dense BF16 [E, rows, cols] so the row-batched MoE can run as
// public grouped GEMM (torch._grouped_mm) instead of per-route GEMV kernels.
// One CTA covers blockDim.y output rows of one expert; threads stride the
// packed block dimension, decode each code with the common routed decoder,
// and copy the codebook row verbatim.
__global__ void vq_projection_dequant_kernel(
    const int64_t* __restrict__ metadata,
    const int expert_count,
    const int meta_base,
    __nv_bfloat16* __restrict__ output,
    const long expert_stride,
    const int rows,
    const int cols)
{
    const int expert = blockIdx.z;
    if (expert >= expert_count) return;
    const int64_t index_address =
        metadata[(long)(meta_base + 0) * expert_count + expert];
    const int64_t codebook_address =
        metadata[(long)(meta_base + 1) * expert_count + expert];
    const int blocks = static_cast<int>(
        metadata[(long)(meta_base + 2) * expert_count + expert]);
    const int vector = static_cast<int>(
        metadata[(long)(meta_base + 3) * expert_count + expert]);
    const int tag = static_cast<int>(
        metadata[(long)(meta_base + 4) * expert_count + expert]);
    if (index_address == 0 || codebook_address == 0 || blocks <= 0) return;
    if (blocks * vector != cols) return;
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const __nv_bfloat16* codebook =
        reinterpret_cast<const __nv_bfloat16*>(
            static_cast<uintptr_t>(codebook_address));
    __nv_bfloat16* dst =
        output + (long)expert * expert_stride + (long)row * cols;
    for (int block = threadIdx.x; block < blocks; block += blockDim.x) {
        const int code = routed_index_value(
            index_address, tag, (long)row * blocks + block);
        const __nv_bfloat16* src = codebook + (long)code * vector;
        __nv_bfloat16* out = dst + block * vector;
        if (vector == 16) {
            const uint4 first = *reinterpret_cast<const uint4*>(src);
            const uint4 second = *reinterpret_cast<const uint4*>(src + 8);
            *reinterpret_cast<uint4*>(out) = first;
            *reinterpret_cast<uint4*>(out + 8) = second;
        } else {
            for (int item = 0; item < vector; ++item)
                out[item] = src[item];
        }
    }
}

torch::Tensor vq_projection_dequant(
    torch::Tensor metadata,
    torch::Tensor output_gu,
    torch::Tensor output_down)
{
    TORCH_CHECK(
        metadata.is_cuda() && metadata.scalar_type() == at::kLong &&
        metadata.is_contiguous() && metadata.dim() == 2 &&
        (metadata.size(0) == 10 || metadata.size(0) == 15),
        "projection dequant requires contiguous CUDA int64 [10,E] or [15,E] metadata");
    TORCH_CHECK(
        output_gu.is_cuda() && output_gu.scalar_type() == at::kBFloat16 &&
        output_gu.is_contiguous() && output_gu.dim() == 3 &&
        output_gu.size(1) % 2 == 0,
        "projection dequant gate/up output must be CUDA BF16 [E,2I,C]");
    TORCH_CHECK(
        output_down.is_cuda() &&
        output_down.scalar_type() == at::kBFloat16 &&
        output_down.is_contiguous() && output_down.dim() == 3,
        "projection dequant down output must be CUDA BF16 [E,C,I]");
    const int expert_count = static_cast<int>(metadata.size(1));
    TORCH_CHECK(
        output_gu.size(0) == expert_count &&
        output_down.size(0) == expert_count,
        "projection dequant outputs must cover every metadata expert");
    const int inter = static_cast<int>(output_gu.size(1) / 2);
    const int input_cols = static_cast<int>(output_gu.size(2));
    TORCH_CHECK(
        output_down.size(1) == input_cols &&
        output_down.size(2) == inter,
        "projection dequant down output must be [E,C,I] of gate/up");
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 block(32, 8);
    __nv_bfloat16* gu_pointer =
        reinterpret_cast<__nv_bfloat16*>(output_gu.data_ptr<at::BFloat16>());
    const long gu_stride = (long)2 * inter * input_cols;
    if (metadata.size(0) == 10) {
        // Legacy GLM archives store Gate+Up as one [2I,C] VQ matrix.
        const dim3 gu_grid(
            (2 * inter + block.y - 1) / block.y, 1, expert_count);
        vq_projection_dequant_kernel<<<gu_grid, block, 0, stream>>>(
            metadata.data_ptr<int64_t>(), expert_count, 0,
            gu_pointer, gu_stride, 2 * inter, input_cols);
    } else {
        const dim3 gate_grid(
            (inter + block.y - 1) / block.y, 1, expert_count);
        vq_projection_dequant_kernel<<<gate_grid, block, 0, stream>>>(
            metadata.data_ptr<int64_t>(), expert_count, 0,
            gu_pointer, gu_stride, inter, input_cols);
        vq_projection_dequant_kernel<<<gate_grid, block, 0, stream>>>(
            metadata.data_ptr<int64_t>(), expert_count, 5,
            gu_pointer + (long)inter * input_cols, gu_stride, inter, input_cols);
    }
    const dim3 down_grid(
        (input_cols + block.y - 1) / block.y, 1, expert_count);
    vq_projection_dequant_kernel<<<down_grid, block, 0, stream>>>(
        metadata.data_ptr<int64_t>(), expert_count,
        metadata.size(0) == 10 ? 5 : 10,
        reinterpret_cast<__nv_bfloat16*>(output_down.data_ptr<at::BFloat16>()),
        (long)input_cols * inter, input_cols, inter);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output_gu;
}

// Batched VQ -> native 8-bit projection expansion for Tensor Core execution.
//
// Metadata has the same [10,E]/[15,E] layout as vq_projection_dequant, except
// every codebook pointer already addresses E4M3 or INT8 bytes. Quantizing the
// tiny shared codebooks is deliberately outside this kernel: the hot path only
// unpacks one index and copies one aligned 4/8/16-byte codeword.
//
// One launch reconstructs every Gate/Up/Down projection.  The former version
// launched one specialization for each possible vector width and projection
// (six or nine launches per layer), even though vector width is uniform inside
// a CUDA block.  Selecting the aligned copy width once per block removes that
// launch tax and writes Down directly in either grouped-Prefill or flattened
// two-GEMM Decode layout.
__global__ void vq_projection_expand_native8_kernel(
    const int64_t* __restrict__ metadata,
    const int expert_count,
    const int metadata_rows,
    uint8_t* __restrict__ output_gu,
    uint8_t* __restrict__ output_down,
    const int intermediate,
    const int hidden,
    const bool output_transposed)
{
    const int expert = blockIdx.z;
    if (expert >= expert_count) return;
    const bool legacy_gate_up = metadata_rows == 10;
    const int logical_row = blockIdx.x * blockDim.y + threadIdx.y;
    if (logical_row >= 2 * intermediate + hidden) return;
    const bool is_down = logical_row >= 2 * intermediate;
    const bool is_up = !legacy_gate_up &&
        logical_row >= intermediate && !is_down;
    const int meta_base = is_down ? (legacy_gate_up ? 5 : 10) :
        (is_up ? 5 : 0);
    const int row = is_down
        ? logical_row - 2 * intermediate
        : (is_up ? logical_row - intermediate : logical_row);
    const int cols = is_down ? intermediate : hidden;
    const int64_t index_address =
        metadata[(long)(meta_base + 0) * expert_count + expert];
    const int64_t codebook_address =
        metadata[(long)(meta_base + 1) * expert_count + expert];
    const int blocks = static_cast<int>(
        metadata[(long)(meta_base + 2) * expert_count + expert]);
    const int vector = static_cast<int>(
        metadata[(long)(meta_base + 3) * expert_count + expert]);
    const int tag = static_cast<int>(
        metadata[(long)(meta_base + 4) * expert_count + expert]);
    if (index_address == 0 || codebook_address == 0 || blocks <= 0 ||
        (vector != 4 && vector != 8 && vector != 16) ||
        blocks * vector != cols)
        return;
    const auto* codebook = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(codebook_address));
    uint8_t* destination;
    if (is_down) {
        destination = output_transposed
            ? output_down +
                (long)row * expert_count * intermediate +
                (long)expert * intermediate
            : output_down +
                (long)expert * hidden * intermediate +
                (long)row * intermediate;
    } else {
        const long projection_offset =
            is_up
                ? (long)intermediate * hidden
                : 0;
        destination = output_gu +
            (long)expert * 2 * intermediate * hidden +
            projection_offset + (long)row * hidden;
    }
    for (int block = threadIdx.x; block < blocks; block += blockDim.x) {
        const int code = routed_index_value(
            index_address, tag, (long)row * blocks + block);
        const auto* source = codebook + (long)code * vector;
        auto* target = destination + block * vector;
        if (vector == 16) {
            *reinterpret_cast<uint4*>(target) =
                *reinterpret_cast<const uint4*>(source);
        } else if (vector == 8) {
            *reinterpret_cast<uint2*>(target) =
                *reinterpret_cast<const uint2*>(source);
        } else {
            *reinterpret_cast<uint32_t*>(target) =
                *reinterpret_cast<const uint32_t*>(source);
        }
    }
}

torch::Tensor vq_projection_expand_native8(
    torch::Tensor metadata,
    torch::Tensor output_gu,
    torch::Tensor output_down)
{
    TORCH_CHECK(
        metadata.is_cuda() && metadata.scalar_type() == at::kLong &&
        metadata.is_contiguous() && metadata.dim() == 2 &&
        (metadata.size(0) == 10 || metadata.size(0) == 15),
        "native8 projection expansion requires CUDA int64 [10,E]/[15,E]");
    const auto dtype = output_gu.scalar_type();
    TORCH_CHECK(
        dtype == at::ScalarType::Float8_e4m3fn || dtype == at::kChar,
        "native8 projection output must be E4M3 or INT8");
    TORCH_CHECK(
        output_gu.is_cuda() && output_gu.scalar_type() == dtype &&
        output_gu.is_contiguous() && output_gu.dim() == 3 &&
        output_gu.size(1) % 2 == 0 && output_down.is_cuda() &&
        output_down.scalar_type() == dtype && output_down.is_contiguous() &&
        (output_down.dim() == 2 || output_down.dim() == 3),
        "native8 projection outputs must be contiguous CUDA "
        "[E,2I,C] plus [E,C,I] or Tensor-Core [C,E*I]");
    const int expert_count = static_cast<int>(metadata.size(1));
    const int inter = static_cast<int>(output_gu.size(1) / 2);
    const int hidden = static_cast<int>(output_gu.size(2));
    const bool down_transposed = output_down.dim() == 2;
    const bool valid_down = down_transposed
        ? output_down.size(0) == hidden &&
            output_down.size(1) == (long)expert_count * inter
        : output_down.size(0) == expert_count &&
            output_down.size(1) == hidden && output_down.size(2) == inter;
    TORCH_CHECK(
        output_gu.size(0) == expert_count &&
        valid_down,
        "native8 projection output shape mismatch");
    auto stream = at::cuda::getCurrentCUDAStream(metadata.get_device());
    const dim3 block(32, 8);
    const dim3 grid(
        (2 * inter + hidden + block.y - 1) / block.y,
        1,
        expert_count);
    vq_projection_expand_native8_kernel<<<grid, block, 0, stream>>>(
        metadata.data_ptr<int64_t>(),
        expert_count,
        static_cast<int>(metadata.size(0)),
        static_cast<uint8_t*>(output_gu.data_ptr()),
        static_cast<uint8_t*>(output_down.data_ptr()),
        inter,
        hidden,
        down_transposed);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output_gu;
}

torch::Tensor packed_moe_topk_grouped(
    torch::Tensor input,
    torch::Tensor token_ids,
    torch::Tensor group_experts,
    torch::Tensor group_offsets,
    torch::Tensor weights,
    torch::Tensor metadata,
    int64_t activation_kind_value,
    double beta,
    double linear_beta,
    double limit,
    torch::Tensor hidden_workspace,
    torch::Tensor result,
    int64_t projection_layout_tag_value,
    int64_t max_group_tiles_value)
{
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kBFloat16 &&
        input.is_contiguous() && input.dim() == 2,
        "grouped packed MoE input must be CUDA BF16 [N,D]");
    TORCH_CHECK(token_ids.is_cuda() && token_ids.scalar_type() == at::kLong &&
        token_ids.is_contiguous() && token_ids.dim() == 1,
        "grouped packed MoE token IDs must be CUDA int64 [M]");
    TORCH_CHECK(group_experts.is_cuda() && group_experts.scalar_type() == at::kLong &&
        group_experts.is_contiguous() && group_experts.dim() == 1,
        "grouped packed MoE experts must be CUDA int64 [G]");
    TORCH_CHECK(group_offsets.is_cuda() && group_offsets.scalar_type() == at::kInt &&
        group_offsets.is_contiguous() && group_offsets.dim() == 1 &&
        group_offsets.numel() == group_experts.numel() + 1,
        "grouped packed MoE offsets must be CUDA int32 [G+1]");
    TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == at::kFloat &&
        weights.is_contiguous() && weights.numel() == token_ids.numel(),
        "grouped packed MoE weights must be CUDA float [M]");
    TORCH_CHECK(metadata.is_cuda() && metadata.scalar_type() == at::kLong &&
        metadata.is_contiguous() && metadata.dim() == 2 &&
        (metadata.size(0) == 10 || metadata.size(0) == 15),
        "grouped packed MoE requires [10,E] or [15,E] metadata");
    TORCH_CHECK(hidden_workspace.is_cuda() && hidden_workspace.scalar_type() == at::kBFloat16 &&
        hidden_workspace.is_contiguous() && hidden_workspace.dim() == 2 &&
        hidden_workspace.size(0) == token_ids.numel() &&
        hidden_workspace.size(1) % 2 == 0,
        "grouped packed MoE activation workspace must be [M,2I]");
    TORCH_CHECK(result.is_cuda() && result.scalar_type() == at::kFloat &&
        result.is_contiguous() && result.dim() == 2 &&
        result.size(0) == input.size(0),
        "grouped packed MoE result must be float [N,D]");
    TORCH_CHECK(input.get_device() == metadata.get_device() &&
        input.get_device() == result.get_device(),
        "grouped packed MoE tensors must share one CUDA device");
    TORCH_CHECK(group_experts.numel() > 0 && token_ids.numel() > 0,
        "grouped packed MoE requires non-empty routes");
    const int groups = static_cast<int>(group_experts.numel());
    const int output_rows = static_cast<int>(hidden_workspace.size(1) / 2);
    const int input_cols = static_cast<int>(input.size(1));
    auto stream = at::cuda::getCurrentCUDAStream();
    result.zero_();
    constexpr int warps = 16;
    constexpr int tokens = 4;
    const dim3 block(32, warps);
    const int max_group_tiles = std::max(
        1, static_cast<int>(max_group_tiles_value));
    const dim3 gate_grid(
        (output_rows + warps - 1) / warps,
        max_group_tiles,
        groups);
    vq_projection_gate_up_grouped_kernel<warps, tokens><<<
        gate_grid, block,
        static_cast<size_t>(tokens) * input_cols * sizeof(__nv_bfloat16),
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            token_ids.data_ptr<int64_t>(), group_experts.data_ptr<int64_t>(),
            group_offsets.data_ptr<int32_t>(), metadata.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(hidden_workspace.data_ptr<at::BFloat16>()),
            groups, static_cast<int>(metadata.size(1)), output_rows, input_cols,
            static_cast<int>(activation_kind_value),
            static_cast<float>(beta), static_cast<float>(linear_beta),
            static_cast<float>(limit), static_cast<int>(metadata.size(0)),
            0);
    const dim3 down_grid(
        (input_cols + warps - 1) / warps,
        max_group_tiles,
        groups);
    vq_projection_down_grouped_kernel<warps, tokens><<<
        down_grid, block,
        static_cast<size_t>(tokens) * output_rows * sizeof(__nv_bfloat16),
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(hidden_workspace.data_ptr<at::BFloat16>()),
            token_ids.data_ptr<int64_t>(), group_experts.data_ptr<int64_t>(),
            group_offsets.data_ptr<int32_t>(), weights.data_ptr<float>(),
            metadata.data_ptr<int64_t>(), result.data_ptr<float>(), groups,
            static_cast<int>(metadata.size(1)),
            input_cols, output_rows, static_cast<int>(metadata.size(0)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

// Down keeps the same registered output workspace, but p8/d4-k256 stages its
// complete padded codebook and computes four rows per warp. Padding d4 rows
// from four to six BF16 values changes the shared-memory row stride from
// eight to twelve bytes, distributing random lookup starts over all 32 banks.
// Other formats use the common packed row decoder in the same kernel.
template <int WARPS>
__global__ void vq_projection_down_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    __nv_bfloat16* __restrict__ output,
    const int top_k,
    const int batch_size,
    const int expert_count,
    const int output_rows,
    const int input_cols)
{
    const int position = blockIdx.z * top_k + blockIdx.y;
    if (position >= batch_size * top_k)
        return;
    const int expert = static_cast<int>(route_ids[position]);
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    constexpr int block_threads = 32 * WARPS;
    extern __shared__ unsigned char raw_shared[];
    auto* shared_input =
        reinterpret_cast<__nv_bfloat16*>(raw_shared);
    auto* shared_codebook = shared_input + input_cols;
    __shared__ RoutedBlockMetadata down_meta;
    __shared__ int shared_codebook_kind;

    if (linear_thread == 0) {
        down_meta.valid = 0;
        shared_codebook_kind = 0;
        if (expert >= 0 && expert < expert_count) {
            down_meta.index_address =
                metadata[(long)10 * expert_count + expert];
            down_meta.codebook_address =
                metadata[(long)11 * expert_count + expert];
            down_meta.blocks = static_cast<int>(
                metadata[(long)12 * expert_count + expert]);
            down_meta.vector = static_cast<int>(
                metadata[(long)13 * expert_count + expert]);
            down_meta.dtype_tag = static_cast<int>(
                metadata[(long)14 * expert_count + expert]);
            down_meta.valid = (
                down_meta.index_address != 0 &&
                down_meta.codebook_address != 0 &&
                down_meta.blocks > 0
            );
            const bool shared_p8 = (
                down_meta.valid &&
                down_meta.dtype_tag == 0 &&
                down_meta.vector == CCCP_PROJECTION_P8_VECTOR &&
                down_meta.blocks * down_meta.vector == input_cols
            );
            const bool shared_p11 = (
                down_meta.valid &&
                down_meta.dtype_tag == 6 &&
                down_meta.vector == CCCP_PROJECTION_P11_VECTOR &&
                down_meta.blocks * down_meta.vector == input_cols
            );
            shared_codebook_kind = shared_p8 ? 1 : (shared_p11 ? 2 : 0);
        }
    }
    const __nv_bfloat16* input_row =
        input + (long)position * input_cols;
    const auto* input4 =
        reinterpret_cast<const uint4*>(input_row);
    auto* shared_input4 =
        reinterpret_cast<uint4*>(shared_input);
    for (
        int item = linear_thread;
        item < input_cols / 8;
        item += block_threads
    )
        shared_input4[item] = input4[item];
    __syncthreads();
    if (!down_meta.valid)
        return;

    const auto* global_codebook =
        reinterpret_cast<const __nv_bfloat16*>(
            static_cast<uintptr_t>(down_meta.codebook_address));
    if (shared_codebook_kind != 0) {
        const int codebook_codes = (
            shared_codebook_kind == 1
            ? CCCP_PROJECTION_P8_CODES
            : CCCP_PROJECTION_P11_CODES
        );
        const int codebook_vector = (
            shared_codebook_kind == 1
            ? CCCP_PROJECTION_P8_VECTOR
            : CCCP_PROJECTION_P11_VECTOR
        );
        const int shared_stride = (
            shared_codebook_kind == 1
            ? CCCP_PROJECTION_P8_SHARED_STRIDE
            : CCCP_PROJECTION_P11_SHARED_STRIDE
        );
        const int codebook_items = codebook_codes * codebook_vector;
        for (
            int item = linear_thread;
            item < codebook_items;
            item += block_threads
        ) {
            const int code = item / codebook_vector;
            const int component = item % codebook_vector;
            shared_codebook[
                code * shared_stride + component
            ] = global_codebook[item];
        }
    }
    __syncthreads();
    const auto* codebook = (
        shared_codebook_kind != 0 ? shared_codebook : global_codebook
    );
    float values[CCCP_PROJECTION_ROWS_PER_WARP] = {};
    #pragma unroll
    for (
        int item = 0;
        item < CCCP_PROJECTION_ROWS_PER_WARP;
        ++item
    ) {
        const int row =
            blockIdx.x *
                (WARPS * CCCP_PROJECTION_ROWS_PER_WARP) +
            threadIdx.y +
            item * WARPS;
        if (row < output_rows) {
            values[item] = (
                shared_codebook_kind == 1
                ? vq_gemv_routed_p8_shared(
                    down_meta.index_address,
                    codebook,
                    shared_input,
                    down_meta.blocks,
                    (long)row * down_meta.blocks)
                : shared_codebook_kind == 2
                ? vq_gemv_routed_p11_shared(
                    down_meta.index_address,
                    codebook,
                    shared_input,
                    down_meta.blocks,
                    (long)row * down_meta.blocks)
                : vq_gemv_routed_row(
                    down_meta.index_address,
                    codebook,
                    shared_input,
                    down_meta.blocks,
                    down_meta.vector,
                    down_meta.dtype_tag,
                    (long)row * down_meta.blocks)
            );
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (
            int item = 0;
            item < CCCP_PROJECTION_ROWS_PER_WARP;
            ++item
        )
            values[item] += __shfl_down_sync(
                0xffffffffu,
                values[item],
                offset);
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (
            int item = 0;
            item < CCCP_PROJECTION_ROWS_PER_WARP;
            ++item
        ) {
            const int row =
                blockIdx.x *
                    (WARPS * CCCP_PROJECTION_ROWS_PER_WARP) +
                threadIdx.y +
                item * WARPS;
            if (row < output_rows)
                output[(long)position * output_rows + row] =
                    __float2bfloat16_rn(values[item]);
        }
    }
}

template <int WARPS>
inline void launch_vq_projection_down(
    const __nv_bfloat16* input,
    const int64_t* route_ids,
    const int64_t* metadata,
    __nv_bfloat16* output,
    const int top_k,
    const int batch_size,
    const int expert_count,
    const int output_rows,
    const int input_cols,
    cudaStream_t stream)
{
    const size_t shared_bytes = static_cast<size_t>(
        input_cols +
        CCCP_PROJECTION_P11_CODES * CCCP_PROJECTION_P11_SHARED_STRIDE
    ) * sizeof(__nv_bfloat16);
    vq_projection_down_kernel<WARPS><<<
        dim3(
            (unsigned)(
                (
                    output_rows +
                    WARPS * CCCP_PROJECTION_ROWS_PER_WARP - 1
                ) /
                (WARPS * CCCP_PROJECTION_ROWS_PER_WARP)
            ),
            (unsigned)top_k,
            (unsigned)batch_size),
        dim3(32, WARPS),
        shared_bytes,
        stream>>>(
            input,
            route_ids,
            metadata,
            output,
            top_k,
            batch_size,
            expert_count,
            output_rows,
            input_cols);
}

// Static-tile Top-K Down.  One CTA owns one expert and a 512-row output tile:
// the activated row is staged once, 32 warps compute 16 rows each, and lane 0
// atomically publishes the route-weighted BF16-rounded contribution.  Thus
// the global [Top-K, hidden] workspace and its reduction kernel disappear.
// GU+activation is the first fixed kernel; this is the second and final one.
constexpr int CCCP_DOWN_TILE_WARPS = 16;
constexpr int CCCP_DOWN_TILE_ROWS_PER_WARP = 16;
constexpr int CCCP_DOWN_TILE_ROWS =
    CCCP_DOWN_TILE_WARPS * CCCP_DOWN_TILE_ROWS_PER_WARP;

__global__ void vq_projection_down_topk_tiled_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int64_t* __restrict__ route_ids,
    const float* __restrict__ route_weights,
    const int64_t* __restrict__ metadata,
    float* __restrict__ result,
    const int top_k,
    const int batch_size,
    const int expert_count,
    const int output_rows,
    const int input_cols,
    const int metadata_rows)
{
    const int position = blockIdx.z * top_k + blockIdx.y;
    if (position >= batch_size * top_k)
        return;
    const int input_row_index = position / top_k;
    const int expert = static_cast<int>(route_ids[position]);
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    extern __shared__ __nv_bfloat16 down_tile_input[];
    __shared__ RoutedBlockMetadata down_meta;
    __shared__ RoutedTileMetadata down_tile;
    __shared__ float route_weight;

    if (linear_thread == 0) {
        down_meta.valid = 0;
        route_weight = route_weights[position];
        if (expert >= 0 && expert < expert_count) {
            down_meta.index_address =
                metadata[(long)10 * expert_count + expert];
            down_meta.codebook_address =
                metadata[(long)11 * expert_count + expert];
            down_meta.blocks = static_cast<int>(
                metadata[(long)12 * expert_count + expert]);
            down_meta.vector = static_cast<int>(
                metadata[(long)13 * expert_count + expert]);
            down_meta.dtype_tag = static_cast<int>(
                metadata[(long)14 * expert_count + expert]);
            down_meta.valid = (
                down_meta.index_address != 0 &&
                down_meta.codebook_address != 0 &&
                down_meta.blocks > 0 &&
                down_meta.blocks * down_meta.vector == input_cols);
            down_tile = projection_tile_metadata(
                metadata,
                metadata_rows,
                expert_count,
                expert,
                2,
                down_meta.blocks,
                down_meta.dtype_tag);
        }
    }
    const __nv_bfloat16* input_row =
        input + static_cast<long>(position) * input_cols;
    const auto* input4 = reinterpret_cast<const uint4*>(input_row);
    auto* shared4 = reinterpret_cast<uint4*>(down_tile_input);
    for (
        int item = linear_thread;
        item < input_cols / 8;
        item += CCCP_DOWN_TILE_WARPS * 32)
        shared4[item] = input4[item];
    __syncthreads();
    if (!down_meta.valid)
        return;

    const auto* codebook = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(down_meta.codebook_address));
    float values[CCCP_DOWN_TILE_ROWS_PER_WARP] = {};
    #pragma unroll
    for (int item = 0; item < CCCP_DOWN_TILE_ROWS_PER_WARP; ++item) {
        const int row =
            blockIdx.x * CCCP_DOWN_TILE_ROWS +
            threadIdx.y + item * CCCP_DOWN_TILE_WARPS;
        if (row < output_rows) {
            values[item] = (
                down_tile.valid
                ? vq_gemv_routed_row_tiled(
                    down_meta.index_address,
                    codebook,
                    down_tile_input,
                    down_meta.blocks,
                    down_meta.vector,
                    down_meta.dtype_tag,
                    row,
                    down_tile)
                : vq_gemv_routed_row(
                    down_meta.index_address,
                    codebook,
                    down_tile_input,
                    down_meta.blocks,
                    down_meta.vector,
                    down_meta.dtype_tag,
                    static_cast<long>(row) * down_meta.blocks)
            );
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int item = 0; item < CCCP_DOWN_TILE_ROWS_PER_WARP; ++item)
            values[item] += __shfl_down_sync(
                0xffffffffu, values[item], offset);
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int item = 0; item < CCCP_DOWN_TILE_ROWS_PER_WARP; ++item) {
            const int row =
                blockIdx.x * CCCP_DOWN_TILE_ROWS +
                threadIdx.y + item * CCCP_DOWN_TILE_WARPS;
            if (row < output_rows) {
                const float rounded = __bfloat162float(
                    __float2bfloat16_rn(values[item]));
                atomicAdd(
                    result + static_cast<long>(input_row_index) * output_rows + row,
                    rounded * route_weight);
            }
        }
    }
}

inline void launch_vq_projection_down_topk_tiled(
    const __nv_bfloat16* input,
    const int64_t* route_ids,
    const float* route_weights,
    const int64_t* metadata,
    float* result,
    const int top_k,
    const int batch_size,
    const int expert_count,
    const int output_rows,
    const int input_cols,
    const int metadata_rows,
    cudaStream_t stream)
{
    C10_CUDA_CHECK(cudaMemsetAsync(
        result,
        0,
        static_cast<size_t>(batch_size) * output_rows * sizeof(float),
        stream));
    vq_projection_down_topk_tiled_kernel<<<
        dim3(
            (output_rows + CCCP_DOWN_TILE_ROWS - 1) /
                CCCP_DOWN_TILE_ROWS,
            top_k,
            batch_size),
        dim3(32, CCCP_DOWN_TILE_WARPS),
        static_cast<size_t>(input_cols) * sizeof(__nv_bfloat16),
        stream>>>(
            input,
            route_ids,
            route_weights,
            metadata,
            result,
            top_k,
            batch_size,
            expert_count,
            output_rows,
            input_cols,
            metadata_rows);
}

constexpr int CCCP_P12_CODES = 4096;
constexpr int CCCP_P12_WARPS = 32;
constexpr int CCCP_P12_ROWS_PER_WARP = 4;
constexpr int CCCP_P12_ROWS_PER_BLOCK =
    CCCP_P12_WARPS * CCCP_P12_ROWS_PER_WARP;
// The registered p12 operator only accepts 4D and 8D codebooks.  Keeping a
// 10-element stride wasted 8 KiB of shared memory per CTA and reduced
// occupancy on decode-sized GEMVs.
constexpr int CCCP_P12_SHARED_STRIDE = 8;

// Kimi x/vv are packed 12-bit K4096 tiers.  Each CTA stages the input and
// one codebook in shared memory, then computes four rows per warp.  This
// replaces repeated input loads and random L2 codebook reads for roughly
// ninety percent of routed traffic while preserving the original p12 bytes.
__global__ void vq_gemv_routed_p12_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    __nv_bfloat16* __restrict__ out,
    const int top_k,
    const int expert_count,
    const int metadata_base,
    const int output_rows,
    const int input_cols,
    const long input_stride,
    const int active_count,
    const int route_offset)
{
    if (blockIdx.y >= active_count)
        return;
    const int position = blockIdx.y + route_offset;
    if (position >= top_k)
        return;
    const int expert = static_cast<int>(route_ids[position]);
    if (expert < 0 || expert >= expert_count)
        return;

    const int64_t index_address =
        metadata[(long)(metadata_base + 0) * expert_count + expert];
    if (index_address == 0)
        return;
    const int64_t codebook_address =
        metadata[(long)(metadata_base + 1) * expert_count + expert];
    const int blocks = static_cast<int>(
        metadata[(long)(metadata_base + 2) * expert_count + expert]);
    const int vector = static_cast<int>(
        metadata[(long)(metadata_base + 3) * expert_count + expert]);
    const int dtype_tag = static_cast<int>(
        metadata[(long)(metadata_base + 4) * expert_count + expert]);
    if (dtype_tag != 2 || (vector != 4 && vector != 8))
        return;

    extern __shared__ __nv_bfloat16 p12_shared[];
    auto* shared_input = p12_shared;
    auto* shared_codebook = p12_shared + input_cols;
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    constexpr int block_threads = 32 * CCCP_P12_WARPS;
    const __nv_bfloat16* input_row =
        x + (long)position * input_stride;
    for (
        int item = linear_thread;
        item < input_cols;
        item += block_threads
    )
        shared_input[item] = input_row[item];
    const auto* codebook = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(codebook_address));
    const int codebook_items = CCCP_P12_CODES * vector;
    for (
        int item = linear_thread;
        item < codebook_items;
        item += block_threads
    ) {
        const int code = item / vector;
        const int component = item - code * vector;
        shared_codebook[
            code * CCCP_P12_SHARED_STRIDE + component
        ] = codebook[item];
    }
    __syncthreads();

    float values[CCCP_P12_ROWS_PER_WARP] = {};
    for (int block = threadIdx.x; block < blocks; block += 32) {
        #pragma unroll
        for (
            int item = 0;
            item < CCCP_P12_ROWS_PER_WARP;
            ++item
        ) {
            const int row =
                blockIdx.x * CCCP_P12_ROWS_PER_BLOCK +
                threadIdx.y +
                item * CCCP_P12_WARPS;
            if (row < output_rows) {
                const int code = routed_index_value(
                    index_address,
                    dtype_tag,
                    (long)row * blocks + block);
                const __nv_bfloat16* code_row =
                    shared_codebook +
                    (long)code * CCCP_P12_SHARED_STRIDE;
                const __nv_bfloat16* input_block =
                    shared_input + block * vector;
                values[item] += (
                    vector == 4
                    ? vq_block_dot4_bf16(code_row, input_block)
                    : vq_block_dot(code_row, input_block, 8)
                );
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (
            int item = 0;
            item < CCCP_P12_ROWS_PER_WARP;
            ++item
        )
            values[item] += __shfl_down_sync(
                0xffffffffu,
                values[item],
                offset);
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (
            int item = 0;
            item < CCCP_P12_ROWS_PER_WARP;
            ++item
        ) {
            const int row =
                blockIdx.x * CCCP_P12_ROWS_PER_BLOCK +
                threadIdx.y +
                item * CCCP_P12_WARPS;
            if (row < output_rows)
                out[(long)position * output_rows + row] =
                    __float2bfloat16_rn(values[item]);
        }
    }
}

template <int WARPS>
__global__ void vq_gemv_routed_p12_l2_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    __nv_bfloat16* __restrict__ out,
    const int top_k,
    const int expert_count,
    const int metadata_base,
    const int output_rows,
    const int input_cols,
    const long input_stride,
    const int active_count,
    const int route_offset)
{
    if (blockIdx.y >= active_count)
        return;
    const int position = blockIdx.y + route_offset;
    if (position >= top_k)
        return;
    const int expert = static_cast<int>(route_ids[position]);
    if (expert < 0 || expert >= expert_count)
        return;

    const int64_t index_address =
        metadata[(long)(metadata_base + 0) * expert_count + expert];
    if (index_address == 0)
        return;
    const int64_t codebook_address =
        metadata[(long)(metadata_base + 1) * expert_count + expert];
    const int blocks = static_cast<int>(
        metadata[(long)(metadata_base + 2) * expert_count + expert]);
    const int vector = static_cast<int>(
        metadata[(long)(metadata_base + 3) * expert_count + expert]);
    const int dtype_tag = static_cast<int>(
        metadata[(long)(metadata_base + 4) * expert_count + expert]);
    if (dtype_tag != 2 || (vector != 4 && vector != 8))
        return;

    extern __shared__ __nv_bfloat16 p12_l2_input[];
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    const __nv_bfloat16* input_row =
        x + (long)position * input_stride;
    for (
        int item = linear_thread;
        item < input_cols;
        item += 32 * WARPS
    )
        p12_l2_input[item] = input_row[item];
    __syncthreads();

    const int row = blockIdx.x * WARPS + threadIdx.y;
    if (row >= output_rows)
        return;
    const auto* indices = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(index_address));
    const auto* codebook = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(codebook_address));
    const long row_offset = (long)row * blocks;
    float value = 0.f;
    for (int block = threadIdx.x; block < blocks; block += 32) {
        int code;
        if ((blocks & 1) == 0) {
            unsigned packed = 0;
            if ((threadIdx.x & 1) == 0) {
                const long base = ((row_offset + block) >> 1) * 3;
                packed =
                    static_cast<unsigned>(indices[base]) |
                    (static_cast<unsigned>(indices[base + 1]) << 8) |
                    (static_cast<unsigned>(indices[base + 2]) << 16);
            }
            packed = __shfl_sync(
                __activemask(),
                packed,
                threadIdx.x & ~1);
            code = static_cast<int>(
                (packed >> ((threadIdx.x & 1) * 12)) & 0xfffu);
        } else {
            code = routed_index_value(
                index_address,
                dtype_tag,
                row_offset + block);
        }
        const __nv_bfloat16* code_row =
            codebook + (long)code * vector;
        const __nv_bfloat16* input_block =
            p12_l2_input + block * vector;
        value += (
            vector == 4
            ? vq_block_dot4_bf16(code_row, input_block)
            : vq_block_dot(code_row, input_block, 8)
        );
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(
            0xffffffffu,
            value,
            offset);
    if (threadIdx.x == 0)
        out[(long)position * output_rows + row] =
            __float2bfloat16_rn(value);
}

__global__ void routed_swiglu_bf16_inplace_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    const int K,
    const int E,
    const int inter,
    const float limit)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = K * inter;
    if (i >= total) return;
    const int n = i / inter;
    const int expert_id = static_cast<int>(route_ids[n]);
    if (
        expert_id < 0 || expert_id >= E ||
        metadata[expert_id] == 0
    ) return;
    const int m = i - n * inter;
    const long row = (long)n * 2 * inter;
    float gate = __bfloat162float(hidden[row + m]);
    float up = __bfloat162float(hidden[row + inter + m]);
    if (limit > 0.f) {
        gate = fminf(gate, limit);
        up = fminf(fmaxf(up, -limit), limit);
    }
    const float silu = gate / (1.f + expf(-gate));
    hidden[row + m] = __float2bfloat16_rn(silu * up);
}

__global__ void routed_situ_bf16_inplace_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    const int top_k,
    const int expert_count,
    const int intermediate,
    const float beta,
    const float linear_beta)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = top_k * intermediate;
    if (i >= total) return;
    const int position = i / intermediate;
    const int expert = static_cast<int>(route_ids[position]);
    if (
        expert < 0 || expert >= expert_count ||
        metadata[expert] == 0
    ) return;
    const int column = i - position * intermediate;
    const long row = (long)position * 2 * intermediate;
    const float gate = __bfloat162float(hidden[row + column]);
    float up = __bfloat162float(
        hidden[row + intermediate + column]);
    const float activated =
        beta * tanhf(gate / beta) / (1.f + expf(-gate));
    if (linear_beta > 0.f)
        up = linear_beta * tanhf(up / linear_beta);
    hidden[row + column] =
        __float2bfloat16_rn(activated * up);
}

__global__ void routed_situ_planar_bf16_inplace_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    const int top_k,
    const int expert_count,
    const int intermediate,
    const int activation_kind,
    const float beta,
    const float linear_beta,
    const float limit)
{
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = top_k * intermediate;
    if (item >= total) return;
    const int position = item / intermediate;
    const int expert = static_cast<int>(route_ids[position]);
    if (
        expert < 0 || expert >= expert_count ||
        metadata[expert] == 0
    ) return;
    const float gate = __bfloat162float(hidden[item]);
    const float up = __bfloat162float(hidden[total + item]);
    const float activated = projection_gate_up_activation(
        gate,
        up,
        activation_kind,
        beta,
        linear_beta,
        limit);
    hidden[item] = __float2bfloat16_rn(activated);
}

__global__ void routed_weighted_sum_f32_kernel(
    const __nv_bfloat16* __restrict__ rows,
    const int64_t* __restrict__ route_ids,
    const float* __restrict__ weights,
    const int64_t* __restrict__ metadata,
    float* __restrict__ result,
    const int K,
    const int E,
    const int D,
    const int dtype_filter,
    const bool accumulate);

__global__ void routed_weighted_sum_rows_f32_kernel(
    const __nv_bfloat16* __restrict__ rows,
    const int64_t* __restrict__ route_ids,
    const float* __restrict__ weights,
    const int64_t* __restrict__ metadata,
    float* __restrict__ result,
    const int batch_size,
    const int top_k,
    const int expert_count,
    const int hidden);

constexpr int CCCP_DOWN_REDUCE_ROWS = 1;

// Decode only needs the route-weighted sum of the expert down projections.
// Each expert warp computes a small output-row tile while walking the same
// packed Down metadata. Reducing Top-K inside the same CTA avoids
// materialising [Top-K, hidden] and launching a second reduction kernel. The
// BF16 round before route weighting intentionally matches the unfused path.
template <int ROWS>
__global__ void vq_gemv_routed_down_reduce_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int64_t* __restrict__ route_ids,
    const float* __restrict__ route_weights,
    const int64_t* __restrict__ metadata,
    float* __restrict__ result,
    const int top_k,
    const int batch_size,
    const int expert_count,
    const int metadata_base,
    const int output_rows,
    const int input_cols,
    const long input_stride)
{
    const int input_row_index = blockIdx.y;
    if (input_row_index >= batch_size)
        return;
    const int position = threadIdx.y;
    const int row_base = blockIdx.x * ROWS;
    int expert = -1;
    int64_t index_address = 0;
    int64_t codebook_address = 0;
    int blocks = 0;
    int vector = 0;
    int dtype_tag = 0;
    if (position < top_k) {
        expert = static_cast<int>(
            route_ids[input_row_index * top_k + position]);
        if (expert >= 0 && expert < expert_count) {
            index_address =
                metadata[
                    (long)(metadata_base + 0) * expert_count + expert];
            codebook_address =
                metadata[
                    (long)(metadata_base + 1) * expert_count + expert];
            blocks = static_cast<int>(
                metadata[
                    (long)(metadata_base + 2) * expert_count + expert]);
            vector = static_cast<int>(
                metadata[
                    (long)(metadata_base + 3) * expert_count + expert]);
            dtype_tag = static_cast<int>(
                metadata[
                    (long)(metadata_base + 4) * expert_count + expert]);
        }
    }
    const bool valid = (
        index_address != 0 &&
        codebook_address != 0 &&
        blocks > 0 &&
        (vector == 4 || vector == 8 || vector == 16)
    );
    const auto* codebook = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(codebook_address));
    const __nv_bfloat16* input_row =
        input +
        (static_cast<long>(input_row_index) * top_k + position) *
            input_stride;
    float values[ROWS] = {};
    if (valid) {
        for (int block = threadIdx.x; block < blocks; block += 32) {
            const __nv_bfloat16* input_block =
                input_row + block * vector;
            #pragma unroll
            for (int item = 0; item < ROWS; ++item) {
                const int row = row_base + item;
                if (row < output_rows) {
                    const int code = routed_index_value(
                        index_address,
                        dtype_tag,
                        (long)row * blocks + block);
                    const __nv_bfloat16* code_row =
                        codebook + (long)code * vector;
                    values[item] += (
                        vector == 4
                        ? vq_block_dot4_bf16(code_row, input_block)
                        : vq_block_dot(code_row, input_block, vector)
                    );
                }
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int item = 0; item < ROWS; ++item)
            values[item] += __shfl_down_sync(
                0xffffffffu,
                values[item],
                offset);
    }
    __shared__ float partial[ROWS][MAX_SLOT_EXPERTS];
    if (threadIdx.x == 0) {
        const float weight = (
            position < top_k
            ? route_weights[input_row_index * top_k + position]
            : 0.0f);
        #pragma unroll
        for (int item = 0; item < ROWS; ++item) {
            const float rounded = __bfloat162float(
                __float2bfloat16_rn(values[item]));
            partial[item][position] = rounded * weight;
        }
    }
    __syncthreads();
    // Warp 0 publishes the route-weighted row without a second kernel or a
    // global [Top-K, hidden] buffer.
    if (threadIdx.y == 0 && threadIdx.x < ROWS) {
        const int row_item = threadIdx.x;
        const int row = row_base + row_item;
        if (row < output_rows) {
            float value = 0.0f;
            #pragma unroll
            for (int item = 0; item < MAX_SLOT_EXPERTS; ++item) {
                if (item < top_k)
                    value += partial[row_item][item];
            }
            result[static_cast<long>(input_row_index) * output_rows + row] =
                value;
        }
    }
}

inline void launch_vq_gemv_routed_down_reduce(
    const __nv_bfloat16* input,
    const int64_t* route_ids,
    const float* route_weights,
    const int64_t* metadata,
    float* result,
    const int top_k,
    const int batch_size,
    const int expert_count,
    const int metadata_base,
    const int output_rows,
    const int input_cols,
    const long input_stride,
    const int rows_per_block,
    cudaStream_t stream)
{
    const int rows = (rows_per_block == 4 || rows_per_block == 2)
        ? rows_per_block : 1;
    if (rows == 4) {
        vq_gemv_routed_down_reduce_kernel<4><<<
            dim3((output_rows + 3) / 4, batch_size),
            dim3(32, MAX_SLOT_EXPERTS), 0, stream>>>(
                input, route_ids, route_weights, metadata, result, top_k,
                batch_size, expert_count, metadata_base, output_rows,
                input_cols, input_stride);
    } else if (rows == 2) {
        vq_gemv_routed_down_reduce_kernel<2><<<
            dim3((output_rows + 1) / 2, batch_size),
            dim3(32, MAX_SLOT_EXPERTS), 0, stream>>>(
                input, route_ids, route_weights, metadata, result, top_k,
                batch_size, expert_count, metadata_base, output_rows,
                input_cols, input_stride);
    } else {
        vq_gemv_routed_down_reduce_kernel<1><<<
            dim3(output_rows, batch_size),
            dim3(32, MAX_SLOT_EXPERTS), 0, stream>>>(
                input, route_ids, route_weights, metadata, result, top_k,
                batch_size, expert_count, metadata_base, output_rows,
                input_cols, input_stride);
    }
}

__device__ __forceinline__ int packed_bits_from_tag(const int dtype_tag)
{
    if (dtype_tag == 0) return 8;
    if (dtype_tag == 1) return 16;
    if (dtype_tag == 2) return 12;
    if (dtype_tag == 3) return 14;
    if (dtype_tag == 4) return 10;
    if (dtype_tag == 5) return 9;
    if (dtype_tag >= 6 && dtype_tag <= 8)
        return 2 * dtype_tag - 1;
    return 0;
}

__global__ void packed_stage_topk_blob_copy_kernel(
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    const int64_t* __restrict__ uva_metadata,
    uint8_t* __restrict__ stage,
    const int top_k,
    const int expert_count,
    const int hidden,
    const int expert_stride)
{
    const int position = blockIdx.y;
    if (position >= top_k)
        return;
    const int expert = static_cast<int>(route_ids[position]);
    if (expert < 0 || expert >= expert_count)
        return;
    const int64_t source_address = metadata[expert];
    const int64_t host_address = uva_metadata[expert];
    const int64_t host_down =
        uva_metadata[(long)10 * expert_count + expert];
    const int blocks = static_cast<int>(
        metadata[(long)12 * expert_count + expert]);
    const int dtype_tag = static_cast<int>(
        metadata[(long)14 * expert_count + expert]);
    const int bits = packed_bits_from_tag(dtype_tag);
    const long down_bytes = (
        static_cast<long>(hidden) * blocks * bits + 7) / 8;
    const long byte_count = host_down - host_address + down_bytes;
    const bool valid = (
        source_address != 0 && host_address != 0 &&
        host_down >= host_address && blocks > 0 && bits > 0 &&
        byte_count > 0 && byte_count <= expert_stride);
    const bool cache_hit = source_address != host_address;
    if (!valid || cache_hit)
        return;
    const auto* source = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(source_address));
    auto* destination =
        stage + static_cast<long>(position) * expert_stride;
    const long vectors = byte_count / 16;
    const auto* source4 = reinterpret_cast<const uint4*>(source);
    auto* destination4 = reinterpret_cast<uint4*>(destination);
    for (
        long item =
            static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
        item < vectors;
        item += static_cast<long>(gridDim.x) * blockDim.x
    )
        destination4[item] = source4[item];
    if (blockIdx.x == 0) {
        for (
            long byte = vectors * 16 + threadIdx.x;
            byte < byte_count;
            byte += blockDim.x
        )
            destination[byte] = source[byte];
    }
}

__global__ void packed_stage_topk_blob_metadata_kernel(
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    const int64_t* __restrict__ uva_metadata,
    const uint8_t* __restrict__ stage,
    int64_t* __restrict__ staged_metadata,
    int top_k,
    int experts,
    int metadata_rows,
    int stage_stride);

torch::Tensor packed_stage_topk_three_projection(
    torch::Tensor route_ids,
    torch::Tensor metadata,
    torch::Tensor uva_metadata,
    torch::Tensor stage,
    torch::Tensor staged_metadata,
    torch::Tensor staged_route_ids,
    int64_t hidden_value,
    int64_t intermediate_value,
    int64_t gate_stride_value,
    int64_t up_stride_value,
    int64_t down_stride_value)
{
    TORCH_CHECK(
        route_ids.is_cuda() && route_ids.scalar_type() == at::kLong &&
        route_ids.is_contiguous() && route_ids.dim() == 1 &&
        route_ids.numel() > 0 && route_ids.numel() <= MAX_SLOT_EXPERTS,
        "mapped packed stage route IDs must be CUDA int64 [Top-K]");
    TORCH_CHECK(
        metadata.is_cuda() && metadata.scalar_type() == at::kLong &&
        metadata.is_contiguous() && metadata.dim() == 2 &&
        metadata.size(0) == CCCP_PROJECTION_LEGACY_META_ROWS &&
        uva_metadata.is_cuda() &&
        uva_metadata.scalar_type() == at::kLong &&
        uva_metadata.sizes() == metadata.sizes() &&
        uva_metadata.is_contiguous(),
        "mapped packed stage requires current/UVA int64 [15,E] metadata");
    const int top_k = static_cast<int>(route_ids.numel());
    TORCH_CHECK(
        stage.is_cuda() && stage.scalar_type() == at::kByte &&
        stage.is_contiguous() && stage.dim() == 1 &&
        staged_metadata.is_cuda() &&
        staged_metadata.scalar_type() == at::kLong &&
        staged_metadata.is_contiguous() &&
        staged_metadata.sizes() == torch::IntArrayRef({15, top_k}) &&
        staged_route_ids.is_cuda() &&
        staged_route_ids.scalar_type() == at::kLong &&
        staged_route_ids.is_contiguous() &&
        staged_route_ids.numel() == top_k,
        "mapped packed stage workspaces have an invalid layout");
    TORCH_CHECK(
        hidden_value > 0 && intermediate_value > 0 &&
        gate_stride_value > 0 && up_stride_value > 0 &&
        down_stride_value > 0,
        "mapped packed stage dimensions must be positive");
    const int expert_stride = static_cast<int>(
        gate_stride_value + up_stride_value + down_stride_value);
    TORCH_CHECK(
        stage.numel() >= static_cast<long>(top_k) * expert_stride,
        "mapped packed stage workspace is too small");
    const int device = route_ids.get_device();
    TORCH_CHECK(
        metadata.get_device() == device && uva_metadata.get_device() == device &&
        stage.get_device() == device && staged_metadata.get_device() == device &&
        staged_route_ids.get_device() == device,
        "mapped packed stage tensors must share one CUDA device");
    const int expert_count = static_cast<int>(metadata.size(1));
    auto stream = at::cuda::getCurrentCUDAStream();
    // A cold expert projection is several MiB.  The legacy four-block copy
    // occupied only four SMs on Hopper and left PCIe/host bandwidth mostly
    // idle.  Spread each independent compact projection across the full GPU;
    // cache-hit blocks exit before touching payload bytes.
    int copy_blocks = 64;
    if (const char* configured = std::getenv("CCCP_MAPPED_COPY_BLOCKS")) {
        const int parsed = std::atoi(configured);
        if (parsed > 0)
            copy_blocks = std::max(4, std::min(parsed, 128));
    }
    const dim3 grid(copy_blocks, top_k);
    packed_stage_topk_blob_copy_kernel<<<grid, 256, 0, stream>>>(
        route_ids.data_ptr<int64_t>(),
        metadata.data_ptr<int64_t>(),
        uva_metadata.data_ptr<int64_t>(),
        stage.data_ptr<uint8_t>(),
        top_k,
        expert_count,
        static_cast<int>(hidden_value),
        expert_stride);
    const int metadata_items = 15 * top_k;
    packed_stage_topk_blob_metadata_kernel<<<
        (metadata_items + 127) / 128, 128, 0, stream>>>(
        route_ids.data_ptr<int64_t>(),
        metadata.data_ptr<int64_t>(),
        uva_metadata.data_ptr<int64_t>(),
        stage.data_ptr<uint8_t>(),
        staged_metadata.data_ptr<int64_t>(),
        top_k,
        expert_count,
        15,
        expert_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return staged_metadata;
}

__global__ void packed_stage_topk_blob_metadata_kernel(
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    const int64_t* __restrict__ uva_metadata,
    const uint8_t* __restrict__ stage,
    int64_t* __restrict__ staged_metadata,
    const int top_k,
    const int experts,
    const int metadata_rows,
    const int stage_stride)
{
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = top_k * metadata_rows;
    if (item >= total)
        return;
    const int row = item / top_k;
    const int slot = item - row * top_k;
    const int64_t expert64 = route_ids[slot];
    if (expert64 < 0 || expert64 >= experts) {
        staged_metadata[item] = 0;
        return;
    }
    const int expert = static_cast<int>(expert64);
    const int64_t current_base = metadata[expert];
    const int64_t host_base = uva_metadata[expert];
    int64_t value = metadata[static_cast<long>(row) * experts + expert];
    if (
        current_base == host_base &&
        (row == 0 || row == 5 || row == 10)) {
        const int64_t host_projection =
            uva_metadata[static_cast<long>(row) * experts + expert];
        value = static_cast<int64_t>(reinterpret_cast<uintptr_t>(stage)) +
            static_cast<int64_t>(slot) * stage_stride +
            (host_projection - host_base);
    }
    staged_metadata[item] = value;
}

torch::Tensor packed_moe_topk_compact_fp8_codebook(
    torch::Tensor input,
    torch::Tensor route_ids,
    torch::Tensor weights,
    torch::Tensor metadata,
    torch::Tensor scales,
    int64_t activation_kind_value,
    double beta,
    double linear_beta,
    double limit,
    torch::Tensor hidden_workspace,
    torch::Tensor out_workspace,
    torch::Tensor result)
{
#if defined(__HIP_PLATFORM_AMD__)
    TORCH_CHECK(false, "compact E4M3 codebook MoE is unavailable on HIP");
#else
    TORCH_CHECK(
        input.is_cuda() && input.scalar_type() == at::kBFloat16 &&
        input.is_contiguous() && input.dim() == 2 && input.size(0) == 1,
        "compact E4M3 MoE input must be CUDA BF16 [1,D]");
    TORCH_CHECK(
        route_ids.is_cuda() && route_ids.scalar_type() == at::kLong &&
        route_ids.is_contiguous() && route_ids.dim() == 1 &&
        route_ids.numel() > 0 && route_ids.numel() <= MAX_SLOT_EXPERTS,
        "compact E4M3 MoE route IDs must be CUDA int64 [TopK]");
    TORCH_CHECK(
        weights.is_cuda() && weights.scalar_type() == at::kFloat &&
        weights.is_contiguous() && weights.sizes() == route_ids.sizes(),
        "compact E4M3 MoE weights must match TopK");
    TORCH_CHECK(
        metadata.is_cuda() && metadata.scalar_type() == at::kLong &&
        metadata.is_contiguous() && metadata.dim() == 2 &&
        metadata.size(0) == CCCP_PROJECTION_LEGACY_META_ROWS &&
        metadata.size(1) == route_ids.numel(),
        "compact E4M3 MoE metadata must be int64 [15,TopK]");
    TORCH_CHECK(
        scales.is_cuda() && scales.scalar_type() == at::kFloat &&
        scales.is_contiguous() && scales.dim() == 2 &&
        scales.size(0) == route_ids.numel() && scales.size(1) == 3,
        "compact E4M3 MoE scales must be float32 [TopK,3]");
    const int top_k = static_cast<int>(route_ids.numel());
    const int hidden = static_cast<int>(input.size(1));
    TORCH_CHECK(
        hidden_workspace.is_cuda() &&
        hidden_workspace.scalar_type() == at::kBFloat16 &&
        hidden_workspace.is_contiguous() &&
        hidden_workspace.dim() == 2 &&
        hidden_workspace.size(0) == top_k &&
        hidden_workspace.size(1) % 2 == 0,
        "compact E4M3 MoE hidden workspace must be BF16 [TopK,2I]");
    const int intermediate =
        static_cast<int>(hidden_workspace.size(1) / 2);
    TORCH_CHECK(
        out_workspace.is_cuda() &&
        out_workspace.scalar_type() == at::kBFloat16 &&
        out_workspace.is_contiguous() &&
        out_workspace.sizes() == torch::IntArrayRef({top_k, hidden}) &&
        result.is_cuda() && result.scalar_type() == at::kFloat &&
        result.is_contiguous() && result.numel() == hidden,
        "compact E4M3 MoE output workspaces are invalid");
    const int device = input.get_device();
    TORCH_CHECK(
        route_ids.get_device() == device && weights.get_device() == device &&
        metadata.get_device() == device && scales.get_device() == device &&
        hidden_workspace.get_device() == device &&
        out_workspace.get_device() == device && result.get_device() == device,
        "compact E4M3 MoE tensors must share one CUDA device");
    TORCH_CHECK(
        activation_kind_value >= 0 && activation_kind_value <= 1 &&
        (activation_kind_value != 0 || beta > 0.0),
        "compact E4M3 MoE activation metadata is invalid");

    constexpr int warps = 16;
    constexpr int rows_per_warp = 4;
    const dim3 block(32, warps);
    auto stream = at::cuda::getCurrentCUDAStream(device);
    vq_projection_gate_up_compact_fp8_kernel<
        warps, rows_per_warp><<<
            dim3(
                (intermediate + warps * rows_per_warp - 1) /
                    (warps * rows_per_warp),
                top_k),
            block,
            static_cast<size_t>(hidden) * sizeof(__nv_bfloat16),
            stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                route_ids.data_ptr<int64_t>(),
                metadata.data_ptr<int64_t>(),
                scales.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(
                    hidden_workspace.data_ptr()),
                top_k,
                top_k,
                intermediate,
                hidden,
                static_cast<int>(activation_kind_value),
                static_cast<float>(beta),
                static_cast<float>(linear_beta),
                static_cast<float>(limit));
    vq_projection_down_compact_fp8_kernel<
        warps, rows_per_warp><<<
            dim3(
                (hidden + warps * rows_per_warp - 1) /
                    (warps * rows_per_warp),
                top_k),
            block,
            static_cast<size_t>(intermediate) * sizeof(__nv_bfloat16),
            stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    hidden_workspace.data_ptr()),
                route_ids.data_ptr<int64_t>(),
                metadata.data_ptr<int64_t>(),
                scales.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(out_workspace.data_ptr()),
                top_k,
                top_k,
                hidden,
                intermediate);
    routed_weighted_sum_f32_kernel<<<
        (hidden + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                out_workspace.data_ptr()),
            route_ids.data_ptr<int64_t>(),
            weights.data_ptr<float>(),
            metadata.data_ptr<int64_t>() + (long)5 * top_k,
            result.data_ptr<float>(),
            top_k,
            top_k,
            hidden,
            -1,
            false);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
#endif
}

torch::Tensor packed_moe_topk_compact_q8_codebook(
    torch::Tensor input,
    torch::Tensor route_ids,
    torch::Tensor weights,
    torch::Tensor metadata,
    torch::Tensor scales,
    int64_t activation_kind_value,
    double beta,
    double linear_beta,
    double limit,
    torch::Tensor gate_quant_workspace,
    torch::Tensor down_quant_workspace,
    torch::Tensor hidden_workspace,
    torch::Tensor out_workspace,
    torch::Tensor result)
{
#if defined(__HIP_PLATFORM_AMD__)
    TORCH_CHECK(false, "compact Q8 codebook MoE is unavailable on HIP");
#else
    TORCH_CHECK(
        input.is_cuda() && input.scalar_type() == at::kBFloat16 &&
        input.is_contiguous() && input.dim() == 2 && input.size(0) == 1,
        "compact Q8 MoE input must be CUDA BF16 [1,D]");
    TORCH_CHECK(
        route_ids.is_cuda() && route_ids.scalar_type() == at::kLong &&
        route_ids.is_contiguous() && route_ids.dim() == 1 &&
        route_ids.numel() > 0 && route_ids.numel() <= MAX_SLOT_EXPERTS,
        "compact Q8 MoE route IDs must be CUDA int64 [TopK]");
    TORCH_CHECK(
        weights.is_cuda() && weights.scalar_type() == at::kFloat &&
        weights.is_contiguous() && weights.sizes() == route_ids.sizes(),
        "compact Q8 MoE weights must match TopK");
    TORCH_CHECK(
        metadata.is_cuda() && metadata.scalar_type() == at::kLong &&
        metadata.is_contiguous() && metadata.dim() == 2 &&
        metadata.size(0) == CCCP_PROJECTION_LEGACY_META_ROWS &&
        metadata.size(1) == route_ids.numel(),
        "compact Q8 MoE metadata must be int64 [15,TopK]");
    TORCH_CHECK(
        scales.is_cuda() && scales.scalar_type() == at::kFloat &&
        scales.is_contiguous() && scales.dim() == 2 &&
        scales.size(0) == route_ids.numel() && scales.size(1) == 3,
        "compact Q8 MoE scales must be float32 [TopK,3]");
    const int top_k = static_cast<int>(route_ids.numel());
    const int hidden = static_cast<int>(input.size(1));
    TORCH_CHECK(
        hidden_workspace.is_cuda() &&
        hidden_workspace.scalar_type() == at::kBFloat16 &&
        hidden_workspace.is_contiguous() &&
        hidden_workspace.dim() == 2 &&
        hidden_workspace.size(0) == top_k &&
        hidden_workspace.size(1) % 2 == 0,
        "compact Q8 MoE hidden workspace must be BF16 [TopK,2I]");
    const int intermediate =
        static_cast<int>(hidden_workspace.size(1) / 2);
    const int gate_quantized_span = (hidden + 15) & ~15;
    const int down_quantized_span = (intermediate + 15) & ~15;
    TORCH_CHECK(
        gate_quant_workspace.is_cuda() &&
        gate_quant_workspace.scalar_type() == at::kByte &&
        gate_quant_workspace.is_contiguous() &&
        gate_quant_workspace.dim() == 2 &&
        gate_quant_workspace.size(0) == top_k &&
        gate_quant_workspace.size(1) >= 4 * gate_quantized_span,
        "compact Q8 Gate/Up quant workspace must be uint8 [TopK,4*align(D)]");
    TORCH_CHECK(
        down_quant_workspace.is_cuda() &&
        down_quant_workspace.scalar_type() == at::kByte &&
        down_quant_workspace.is_contiguous() &&
        down_quant_workspace.dim() == 2 &&
        down_quant_workspace.size(0) == top_k &&
        down_quant_workspace.size(1) >= 2 * down_quantized_span,
        "compact Q8 Down quant workspace must be uint8 [TopK,2*align(I)]");
    TORCH_CHECK(
        out_workspace.is_cuda() &&
        out_workspace.scalar_type() == at::kBFloat16 &&
        out_workspace.is_contiguous() &&
        out_workspace.sizes() == torch::IntArrayRef({top_k, hidden}) &&
        result.is_cuda() && result.scalar_type() == at::kFloat &&
        result.is_contiguous() && result.numel() == hidden,
        "compact Q8 MoE output workspaces are invalid");
    const int device = input.get_device();
    TORCH_CHECK(
        route_ids.get_device() == device && weights.get_device() == device &&
        metadata.get_device() == device && scales.get_device() == device &&
        gate_quant_workspace.get_device() == device &&
        down_quant_workspace.get_device() == device &&
        hidden_workspace.get_device() == device &&
        out_workspace.get_device() == device && result.get_device() == device,
        "compact Q8 MoE tensors must share one CUDA device");
    TORCH_CHECK(
        activation_kind_value >= 0 && activation_kind_value <= 1 &&
        (activation_kind_value != 0 || beta > 0.0),
        "compact Q8 MoE activation metadata is invalid");

    constexpr int warps = 16;
    // H20 A/B: 16 warps is faster than 8 or 32 for the mixed compact expert
    // signatures used here. One output row per warp also wins on the real
    // mixed expert distribution, so release intentionally keeps one path.
    const dim3 block(32, warps);
    auto stream = at::cuda::getCurrentCUDAStream(device);
    constexpr int quantize_threads = 256;
    compact_q8_quantize_rows_global_kernel<<<
        1,
        quantize_threads,
        0,
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            gate_quant_workspace.data_ptr<uint8_t>(),
            1,
            hidden,
            gate_quantized_span,
            4 * gate_quantized_span);
    vq_projection_gate_up_compact_q8_kernel<warps, 1><<<
        dim3((intermediate + warps - 1) / warps, top_k),
        block,
        0,
        stream>>>(
            gate_quant_workspace.data_ptr<uint8_t>(),
            route_ids.data_ptr<int64_t>(),
            metadata.data_ptr<int64_t>(),
            scales.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(hidden_workspace.data_ptr()),
            top_k,
            top_k,
            intermediate,
            hidden,
            static_cast<int>(activation_kind_value),
            static_cast<float>(beta),
            static_cast<float>(linear_beta),
            static_cast<float>(limit));
    compact_q8_quantize_rows_global_kernel<<<
        top_k,
        quantize_threads,
        0,
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                hidden_workspace.data_ptr()),
            down_quant_workspace.data_ptr<uint8_t>(),
            top_k,
            intermediate,
            down_quantized_span,
            2 * down_quantized_span);
    vq_projection_down_compact_q8_kernel<warps, 1><<<
        dim3((hidden + warps - 1) / warps, top_k),
        block,
        0,
        stream>>>(
            down_quant_workspace.data_ptr<uint8_t>(),
            route_ids.data_ptr<int64_t>(),
            metadata.data_ptr<int64_t>(),
            scales.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out_workspace.data_ptr()),
            top_k,
            top_k,
            hidden,
            intermediate);
    routed_weighted_sum_f32_kernel<<<
        (hidden + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                out_workspace.data_ptr()),
            route_ids.data_ptr<int64_t>(),
            weights.data_ptr<float>(),
            metadata.data_ptr<int64_t>() + (long)5 * top_k,
            result.data_ptr<float>(),
            top_k,
            top_k,
            hidden,
            -1,
            false);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
#endif
}

bool compact_q8_codebook_l2_prefetch(torch::Tensor metadata)
{
#if defined(__HIP_PLATFORM_AMD__)
    return false;
#else
    if (
        !metadata.is_cuda() || metadata.scalar_type() != at::kLong ||
        !metadata.is_contiguous() || metadata.dim() != 2 ||
        metadata.size(0) != CCCP_PROJECTION_LEGACY_META_ROWS ||
        metadata.size(1) <= 0 || metadata.size(1) > MAX_SLOT_EXPERTS)
        return false;
    const int expert_count = static_cast<int>(metadata.size(1));
    auto stream = at::cuda::getCurrentCUDAStream(metadata.get_device());
    compact_q8_codebook_l2_prefetch_kernel<<<
        3 * expert_count, 128, 0, stream>>>(
            metadata.data_ptr<int64_t>(), expert_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
#endif
}

torch::Tensor packed_moe_topk(
    torch::Tensor input,
    torch::Tensor route_ids,
    torch::Tensor weights,
    torch::Tensor metadata,
    int64_t activation_kind_value,
    double beta,
    double linear_beta,
      double limit,
      torch::Tensor hidden_workspace,
      torch::Tensor out_workspace,
      torch::Tensor result,
      int64_t p12_count_value,
      int64_t projection_layout_tag_value)
{
    TORCH_CHECK(
        input.is_cuda() && input.scalar_type() == at::kBFloat16 &&
        input.is_contiguous() && input.dim() == 2 && input.size(0) == 1,
        "packed MoE decode input must be contiguous CUDA BF16 [1,D]");
    TORCH_CHECK(
        route_ids.is_cuda() && route_ids.scalar_type() == at::kLong &&
        route_ids.is_contiguous() && route_ids.dim() == 1,
        "packed MoE decode route IDs must be CUDA int64 [K]");
      constexpr int batch_size = 1;
      const int top_k = static_cast<int>(route_ids.numel());
      const int total_routes = top_k;
    TORCH_CHECK(
          activation_kind_value >= 0 && activation_kind_value <= 1,
          "packed MoE activation kind must be 0 (SiTU) or 1 (SwiGLU)");
      const int activation_kind = static_cast<int>(activation_kind_value);
    TORCH_CHECK(
          top_k > 0 && top_k <= MAX_SLOT_EXPERTS,
          "packed MoE Top-K must be in [1,16]");
      TORCH_CHECK(
          p12_count_value >= -1 && p12_count_value <= top_k,
          "packed MoE p12 count must be -1 or in [0,Top-K]");
      TORCH_CHECK(
          projection_layout_tag_value >= 0 &&
          projection_layout_tag_value <= 2,
          "packed MoE projection layout tag must be 0, 1 or 2");
      const int p12_count = static_cast<int>(p12_count_value);
      const bool p12_grouped = p12_count >= 0;
      const char* p12_setting = std::getenv("CCCP_P12_SHARED");
      const std::string p12_mode = (
          p12_setting == nullptr ? "direct" : std::string(p12_setting)
      );
      const bool use_p12_shared = (
          p12_mode == "1" || p12_mode == "shared"
      );
      const bool use_p12_l2 = p12_mode == "l2";
      const bool use_p12_specialized =
          use_p12_shared || use_p12_l2;
      const char* p12_warps_setting =
          std::getenv("CCCP_P12_L2_WARPS");
      int p12_l2_warps = (
          p12_warps_setting == nullptr
          ? 16
          : std::atoi(p12_warps_setting)
      );
      if (
          p12_l2_warps != 8 &&
          p12_l2_warps != 16 &&
          p12_l2_warps != 32
      )
          p12_l2_warps = 16;
      const char* routed_warps_setting =
          std::getenv("CCCP_ROUTED_WARPS");
      cudaDeviceProp routed_device_properties{};
      C10_CUDA_CHECK(cudaGetDeviceProperties(
          &routed_device_properties,
          input.get_device()));
      const bool h20_routed_kernel = (
          routed_device_properties.major == 9 &&
          routed_device_properties.minor == 0 &&
          std::string(routed_device_properties.name).find("H20") !=
              std::string::npos
      );
      const int default_routed_warps = (
          h20_routed_kernel ? 16 : ROWS_PER_BLOCK
      );
      int routed_warps = (
          routed_warps_setting == nullptr
          ? default_routed_warps
          : std::atoi(routed_warps_setting)
      );
      if (
          routed_warps != 8 &&
          routed_warps != 16 &&
          routed_warps != 32
      )
          routed_warps = default_routed_warps;
      const char* vector_copy_setting =
          std::getenv("CCCP_ROUTED_VECTOR_COPY");
      const bool vector_input_copy = (
          vector_copy_setting == nullptr ||
          std::string(vector_copy_setting) != "0"
      );
      const int generic_count = (
          use_p12_specialized
          ? (p12_grouped ? top_k - p12_count : top_k)
          : top_k
      );
      const int generic_offset = (
          use_p12_specialized && p12_grouped ? p12_count : 0
      );
      const int p12_active = (
          use_p12_specialized
          ? (p12_grouped ? p12_count : top_k)
          : 0
      );
    TORCH_CHECK(
        weights.is_cuda() && weights.scalar_type() == at::kFloat &&
        weights.is_contiguous() && weights.sizes() == route_ids.sizes(),
        "packed MoE route weights must match [K] or [N,K]");
    TORCH_CHECK(
        metadata.is_cuda() && metadata.scalar_type() == at::kLong &&
        metadata.is_contiguous() && metadata.dim() == 2 &&
        (
            metadata.size(0) == ROUTED_META_ROWS ||
            metadata.size(0) == CCCP_PROJECTION_LEGACY_META_ROWS ||
            metadata.size(0) == CCCP_PROJECTION_TILE_META_ROWS
        ),
        "packed MoE metadata must be CUDA int64 [10,E], [15,E] or [27,E]");
    const bool projection_vq =
        metadata.size(0) == CCCP_PROJECTION_LEGACY_META_ROWS ||
        metadata.size(0) == CCCP_PROJECTION_TILE_META_ROWS;
    TORCH_CHECK(
        batch_size == 1 || projection_vq,
        "batched packed MoE currently requires three projection metadata");
    TORCH_CHECK(
        input.get_device() == route_ids.get_device() &&
        input.get_device() == weights.get_device() &&
        input.get_device() == metadata.get_device() &&
        input.get_device() == hidden_workspace.get_device() &&
        input.get_device() == out_workspace.get_device(),
        "packed MoE inputs and workspaces must share one CUDA device");
    const int expert_count = static_cast<int>(metadata.size(1));
    const int hidden = static_cast<int>(input.size(1));
    TORCH_CHECK(
        hidden_workspace.scalar_type() == at::kBFloat16 &&
        hidden_workspace.is_contiguous() &&
        hidden_workspace.dim() == 2 &&
        hidden_workspace.size(0) == total_routes &&
        hidden_workspace.size(1) % 2 == 0,
        "packed MoE hidden workspace must be BF16 [N*K,2I]");
    const int intermediate =
        static_cast<int>(hidden_workspace.size(1) / 2);
    TORCH_CHECK(
        out_workspace.scalar_type() == at::kBFloat16 &&
        out_workspace.is_contiguous() &&
        out_workspace.sizes() ==
            torch::IntArrayRef({total_routes, hidden}),
        "packed MoE output workspace must be BF16 [N*K,D]");
    TORCH_CHECK(
        result.scalar_type() == at::kFloat &&
        result.is_contiguous() &&
        result.numel() == hidden,
        "packed MoE decode result must be float32 [D]");
    TORCH_CHECK(
        activation_kind != 0 || beta > 0.0,
        "packed MoE SiTU beta must be positive");

    int current = -1;
    C10_CUDA_CHECK(cudaGetDevice(&current));
    TORCH_CHECK(
        current == input.get_device(),
        "packed MoE kernel must run under the input device");
      if (result.get_device() != current)
          ensure_peer_access(
              current,
              result.get_device(),
              "packed MoE direct result");
      dim3 p12_block(32, CCCP_P12_WARPS);
      auto stream = at::cuda::getCurrentCUDAStream();
      if (projection_vq) {
          const auto* input_pointer =
              reinterpret_cast<const __nv_bfloat16*>(
                  input.data_ptr<at::BFloat16>());
          auto* hidden_pointer =
              reinterpret_cast<__nv_bfloat16*>(
                  hidden_workspace.data_ptr<at::BFloat16>());
          auto* output_pointer =
              reinterpret_cast<__nv_bfloat16*>(
                  out_workspace.data_ptr<at::BFloat16>());
          const char* projection_fused_setting =
              std::getenv("CCCP_PROJECTION_FUSED");
          const bool use_projection_fused = (
              projection_layout_tag_value >= 1 &&
              (
                  projection_fused_setting == nullptr ||
                  std::string(projection_fused_setting) != "0"
              )
          );
          TORCH_CHECK(
              batch_size == 1 || use_projection_fused,
              "batched packed MoE requires the fused three-projection path");
          const char* projection_warps_setting =
              std::getenv("CCCP_PROJECTION_WARPS");
          const char* p10_shared_setting =
              std::getenv("CCCP_P10_SHARED");
          const bool p10_shared = (
              projection_layout_tag_value == 2 &&
              (
                  p10_shared_setting == nullptr ||
                  std::atoi(p10_shared_setting) != 0
              )
          );
          int projection_warps = (
              projection_warps_setting == nullptr
              ? (h20_routed_kernel ? 16 : 32)
              : std::atoi(projection_warps_setting)
          );
          if (
              projection_warps != 8 &&
              projection_warps != 16 &&
              projection_warps != 32
          )
              projection_warps = h20_routed_kernel ? 16 : 32;
          const char* projection_rows_setting =
              std::getenv("CCCP_PROJECTION_ROWS");
          const char* p10_rows_setting =
              std::getenv("CCCP_P10_ROWS");
          int p10_rows = (
              p10_rows_setting == nullptr
              ? 4
              : std::atoi(p10_rows_setting));
          if (
              p10_rows != 4 &&
              p10_rows != 8 &&
              p10_rows != 16
          )
              p10_rows = 4;
          int projection_rows = (
              projection_rows_setting == nullptr
              ? 1
              : std::atoi(projection_rows_setting));
          if (
              projection_rows != 1 &&
              projection_rows != 2 &&
              projection_rows != 4
          )
              projection_rows = 1;
          if (use_projection_fused) {
              if (projection_warps == 8) {
                  launch_vq_projection_gate_up_situ<8>(
                      input_pointer,
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      hidden_pointer,
                      top_k,
                      batch_size,
                      expert_count,
                      intermediate,
                      hidden,
                      activation_kind,
                      static_cast<float>(beta),
                      static_cast<float>(linear_beta),
                      static_cast<float>(limit),
                      p10_shared,
                      static_cast<int>(metadata.size(0)),
                      p10_rows,
                      projection_rows,
                      stream);
              } else if (projection_warps == 32) {
                  launch_vq_projection_gate_up_situ<32>(
                      input_pointer,
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      hidden_pointer,
                      top_k,
                      batch_size,
                      expert_count,
                      intermediate,
                      hidden,
                      activation_kind,
                      static_cast<float>(beta),
                      static_cast<float>(linear_beta),
                      static_cast<float>(limit),
                      p10_shared,
                      static_cast<int>(metadata.size(0)),
                      p10_rows,
                      projection_rows,
                      stream);
              } else {
                  launch_vq_projection_gate_up_situ<16>(
                      input_pointer,
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      hidden_pointer,
                      top_k,
                      batch_size,
                      expert_count,
                      intermediate,
                      hidden,
                      activation_kind,
                      static_cast<float>(beta),
                      static_cast<float>(linear_beta),
                      static_cast<float>(limit),
                      p10_shared,
                      static_cast<int>(metadata.size(0)),
                      p10_rows,
                      projection_rows,
                      stream);
              }
              const char* projection_tile_setting =
                  std::getenv("CCCP_PROJECTION_TILE_FUSED");
              const bool projection_tile_fused = (
                  metadata.size(0) == CCCP_PROJECTION_TILE_META_ROWS &&
                  projection_tile_setting != nullptr &&
                  std::string(projection_tile_setting) == "1"
              );
              if (projection_tile_fused) {
                  launch_vq_projection_down_topk_tiled(
                      hidden_pointer,
                      route_ids.data_ptr<int64_t>(),
                      weights.data_ptr<float>(),
                      metadata.data_ptr<int64_t>(),
                      result.data_ptr<float>(),
                      top_k,
                      batch_size,
                      expert_count,
                      hidden,
                      intermediate,
                      static_cast<int>(metadata.size(0)),
                      stream);
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                  return result;
              }
              const char* projection_down_setting =
                  std::getenv("CCCP_PROJECTION_DOWN_REDUCE");
              // The direct projection-down kernel is the common default.
              // Full-model, all-signature tuning on H20 found the extra
              // reduction workspace 15% slower for the 4096x2048 Top-K=6
              // routed shape.  Other architectures never selected it by
              // default.  Keep the experimental reduction path available
              // only through the explicit capability override below.
              const bool projection_down_default = false;
              const bool projection_down_reduce = (
                  projection_down_setting == nullptr
                  ? projection_down_default
                  : std::string(projection_down_setting) != "0"
              );
              if (projection_down_reduce) {
                  const char* down_rows_setting =
                      std::getenv("CCCP_PROJECTION_DOWN_ROWS");
                  int down_rows = (
                      down_rows_setting == nullptr
                      ? 1
                      : std::atoi(down_rows_setting));
                  if (
                      down_rows != 1 &&
                      down_rows != 2 &&
                      down_rows != 4
                  )
                      down_rows = 1;
                  launch_vq_gemv_routed_down_reduce(
                      hidden_pointer,
                      route_ids.data_ptr<int64_t>(),
                      weights.data_ptr<float>(),
                      metadata.data_ptr<int64_t>(),
                      result.data_ptr<float>(),
                      top_k,
                      batch_size,
                      expert_count,
                      10,
                      hidden,
                      intermediate,
                      intermediate,
                      down_rows,
                      stream);
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                  return result;
              }
              if (projection_warps == 8) {
                  launch_vq_projection_down<8>(
                      hidden_pointer,
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      output_pointer,
                      top_k,
                      batch_size,
                      expert_count,
                      hidden,
                      intermediate,
                      stream);
              } else if (projection_warps == 32) {
                  launch_vq_projection_down<32>(
                      hidden_pointer,
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      output_pointer,
                      top_k,
                      batch_size,
                      expert_count,
                      hidden,
                      intermediate,
                      stream);
              } else {
                  launch_vq_projection_down<16>(
                      hidden_pointer,
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      output_pointer,
                      top_k,
                      batch_size,
                      expert_count,
                      hidden,
                      intermediate,
                      stream);
              }
              if (batch_size == 1) {
                  routed_weighted_sum_f32_kernel<<<
                      (hidden + 255) / 256,
                      256,
                      0,
                      stream>>>(
                          output_pointer,
                          route_ids.data_ptr<int64_t>(),
                          weights.data_ptr<float>(),
                          metadata.data_ptr<int64_t>() +
                              (long)5 * expert_count,
                          result.data_ptr<float>(),
                          top_k,
                          expert_count,
                          hidden,
                          -1,
                          false);
              } else {
                  routed_weighted_sum_rows_f32_kernel<<<
                      dim3((hidden + 255) / 256, batch_size),
                      256,
                      0,
                      stream>>>(
                          output_pointer,
                          route_ids.data_ptr<int64_t>(),
                          weights.data_ptr<float>(),
                          metadata.data_ptr<int64_t>() +
                              (long)5 * expert_count,
                          result.data_ptr<float>(),
                          batch_size,
                          top_k,
                          expert_count,
                          hidden);
              }
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              return result;
          }
          const auto launch_projection = [&](
              const __nv_bfloat16* projection_input,
              __nv_bfloat16* projection_output,
              const int metadata_base,
              const int output_rows,
              const int input_columns,
              const long input_stride) {
              if (routed_warps == 8)
                  launch_vq_gemv_routed<8>(
                      projection_input,
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      projection_output,
                      top_k,
                      expert_count,
                      metadata_base,
                      output_rows,
                      input_columns,
                      input_stride,
                      false,
                      0,
                      top_k,
                      vector_input_copy,
                      stream);
              else if (routed_warps == 16)
                  launch_vq_gemv_routed<16>(
                      projection_input,
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      projection_output,
                      top_k,
                      expert_count,
                      metadata_base,
                      output_rows,
                      input_columns,
                      input_stride,
                      false,
                      0,
                      top_k,
                      vector_input_copy,
                      stream);
              else
                  launch_vq_gemv_routed<32>(
                      projection_input,
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      projection_output,
                      top_k,
                      expert_count,
                      metadata_base,
                      output_rows,
                      input_columns,
                      input_stride,
                      false,
                      0,
                      top_k,
                      vector_input_copy,
                      stream);
          };
          // Planar workspace: [all gate rows][all up rows].  This lets both
          // independent codebooks use the common contiguous VQ GEMV kernel.
          launch_projection(
              input_pointer,
              hidden_pointer,
              0,
              intermediate,
              hidden,
              0);
          launch_projection(
              input_pointer,
              hidden_pointer + (long)top_k * intermediate,
              5,
              intermediate,
              hidden,
              0);
          routed_situ_planar_bf16_inplace_kernel<<<
              (top_k * intermediate + 255) / 256,
              256,
              0,
              stream>>>(
                  hidden_pointer,
                  route_ids.data_ptr<int64_t>(),
                  metadata.data_ptr<int64_t>(),
                  top_k,
                  expert_count,
                  intermediate,
                  activation_kind,
                  static_cast<float>(beta),
                  static_cast<float>(linear_beta),
                  static_cast<float>(limit));
          launch_projection(
              hidden_pointer,
              output_pointer,
              10,
              hidden,
              intermediate,
              intermediate);
          // The existing reducer expects down metadata at rows 5..9.  Shift
          // the base by one projection so it sees rows 10..14.
          routed_weighted_sum_f32_kernel<<<
              (hidden + 255) / 256,
              256,
              0,
              stream>>>(
                  output_pointer,
                  route_ids.data_ptr<int64_t>(),
                  weights.data_ptr<float>(),
                  metadata.data_ptr<int64_t>() +
                      (long)5 * expert_count,
                  result.data_ptr<float>(),
                  top_k,
                  expert_count,
                  hidden,
                  -1,
                  false);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
          return result;
      }
      const size_t gu_p12_shared = static_cast<size_t>(
          hidden + (
              use_p12_shared
              ? CCCP_P12_CODES * CCCP_P12_SHARED_STRIDE
              : 0
          )
      ) * sizeof(__nv_bfloat16);
      const size_t down_p12_shared = static_cast<size_t>(
          intermediate + (
              use_p12_shared
              ? CCCP_P12_CODES * CCCP_P12_SHARED_STRIDE
              : 0
          )
      ) * sizeof(__nv_bfloat16);
      const size_t max_p12_shared =
          gu_p12_shared > down_p12_shared
          ? gu_p12_shared
          : down_p12_shared;
      if (use_p12_shared && max_p12_shared > 48 * 1024) {
          static size_t configured_p12_shared[16] = {};
          if (configured_p12_shared[current] < max_p12_shared) {
              const auto attribute_status = cccp_gpu_func_set_attribute(
                  vq_gemv_routed_p12_kernel,
                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                  static_cast<int>(max_p12_shared));
              TORCH_CHECK(
                  attribute_status == cudaSuccess,
                  "failed to configure Kimi p12 shared memory: ",
                  cudaGetErrorString(attribute_status));
              configured_p12_shared[current] = max_p12_shared;
          }
      }
      if (generic_count > 0) {
          const auto* input_pointer =
              reinterpret_cast<const __nv_bfloat16*>(
                  input.data_ptr<at::BFloat16>());
          auto* output_pointer =
              reinterpret_cast<__nv_bfloat16*>(
                  hidden_workspace.data_ptr<at::BFloat16>());
          if (routed_warps == 8)
              launch_vq_gemv_routed<8>(
                  input_pointer, route_ids.data_ptr<int64_t>(),
                  metadata.data_ptr<int64_t>(), output_pointer,
                  top_k, expert_count, 0, 2 * intermediate, hidden, 0,
                  use_p12_specialized, generic_offset, generic_count,
                  vector_input_copy, stream);
          else if (routed_warps == 16)
              launch_vq_gemv_routed<16>(
                  input_pointer, route_ids.data_ptr<int64_t>(),
                  metadata.data_ptr<int64_t>(), output_pointer,
                  top_k, expert_count, 0, 2 * intermediate, hidden, 0,
                  use_p12_specialized, generic_offset, generic_count,
                  vector_input_copy, stream);
          else
              launch_vq_gemv_routed<32>(
                  input_pointer, route_ids.data_ptr<int64_t>(),
                  metadata.data_ptr<int64_t>(), output_pointer,
                  top_k, expert_count, 0, 2 * intermediate, hidden, 0,
                  use_p12_specialized, generic_offset, generic_count,
                  vector_input_copy, stream);
      }
      if (p12_active > 0) {
          if (use_p12_shared) {
              vq_gemv_routed_p12_kernel<<<
                  dim3(
                      (unsigned)(
                          (
                              2 * intermediate +
                              CCCP_P12_ROWS_PER_BLOCK - 1
                          ) / CCCP_P12_ROWS_PER_BLOCK),
                      (unsigned)p12_active),
                  p12_block,
                  gu_p12_shared,
                  stream>>>(
                      reinterpret_cast<const __nv_bfloat16*>(
                          input.data_ptr<at::BFloat16>()),
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      reinterpret_cast<__nv_bfloat16*>(
                          hidden_workspace.data_ptr<at::BFloat16>()),
                      top_k, expert_count, 0, 2 * intermediate, hidden,
                      0, p12_active, 0);
          } else if (p12_l2_warps == 8) {
              vq_gemv_routed_p12_l2_kernel<8><<<
                  dim3(
                      (unsigned)((2 * intermediate + 7) / 8),
                      (unsigned)p12_active),
                  dim3(32, 8),
                  (size_t)hidden * sizeof(__nv_bfloat16),
                  stream>>>(
                      reinterpret_cast<const __nv_bfloat16*>(
                          input.data_ptr<at::BFloat16>()),
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      reinterpret_cast<__nv_bfloat16*>(
                          hidden_workspace.data_ptr<at::BFloat16>()),
                      top_k, expert_count, 0, 2 * intermediate, hidden,
                      0, p12_active, 0);
          } else if (p12_l2_warps == 16) {
              vq_gemv_routed_p12_l2_kernel<16><<<
                  dim3(
                      (unsigned)((2 * intermediate + 15) / 16),
                      (unsigned)p12_active),
                  dim3(32, 16),
                  (size_t)hidden * sizeof(__nv_bfloat16),
                  stream>>>(
                      reinterpret_cast<const __nv_bfloat16*>(
                          input.data_ptr<at::BFloat16>()),
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      reinterpret_cast<__nv_bfloat16*>(
                          hidden_workspace.data_ptr<at::BFloat16>()),
                      top_k, expert_count, 0, 2 * intermediate, hidden,
                      0, p12_active, 0);
          } else {
              vq_gemv_routed_p12_l2_kernel<32><<<
                  dim3(
                      (unsigned)((2 * intermediate + 31) / 32),
                      (unsigned)p12_active),
                  dim3(32, 32),
                  (size_t)hidden * sizeof(__nv_bfloat16),
                  stream>>>(
                      reinterpret_cast<const __nv_bfloat16*>(
                          input.data_ptr<at::BFloat16>()),
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      reinterpret_cast<__nv_bfloat16*>(
                          hidden_workspace.data_ptr<at::BFloat16>()),
                      top_k, expert_count, 0, 2 * intermediate, hidden,
                      0, p12_active, 0);
          }
      }
    routed_situ_bf16_inplace_kernel<<<
        (top_k * intermediate + 255) / 256,
        256,
        0,
        stream>>>(
            reinterpret_cast<__nv_bfloat16*>(
                hidden_workspace.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            metadata.data_ptr<int64_t>(),
            top_k,
            expert_count,
            intermediate,
            static_cast<float>(beta),
            static_cast<float>(linear_beta));
      const char* fused_down_setting =
          std::getenv("CCCP_FUSED_DOWN_REDUCE");
      const bool fused_down_forced = (
          fused_down_setting != nullptr &&
          fused_down_setting[0] == '1' &&
          fused_down_setting[1] == '\0'
      );
      const bool fused_down_disabled = (
          fused_down_setting != nullptr &&
          fused_down_setting[0] == '0' &&
          fused_down_setting[1] == '\0'
      );
      const bool fused_down_reduce = (
          !fused_down_disabled &&
          (
              fused_down_forced ||
              (
                  // Kimi TP8 uses 448 routed-intermediate columns per
                  // rank.  The fused kernel was faster there in the v63
                  // H20 A/B; the current multi-row form keeps the same
                  // capability rule while amortising each CTA barrier.
                  intermediate <= 448 &&
                  result.get_device() == current
              )
          )
      );
      if (fused_down_reduce) {
          const char* down_rows_setting =
              std::getenv("CCCP_PROJECTION_DOWN_ROWS");
          int down_rows = (
              down_rows_setting == nullptr
              ? 1
              : std::atoi(down_rows_setting));
          if (down_rows != 1 && down_rows != 2 && down_rows != 4)
              down_rows = 1;
          launch_vq_gemv_routed_down_reduce(
              reinterpret_cast<const __nv_bfloat16*>(
                  hidden_workspace.data_ptr<at::BFloat16>()),
              route_ids.data_ptr<int64_t>(),
              weights.data_ptr<float>(),
              metadata.data_ptr<int64_t>(),
              result.data_ptr<float>(),
              top_k,
              1,
              expert_count,
              5,
              hidden,
              intermediate,
              2 * intermediate,
              down_rows,
              stream);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
          return result;
      }
      if (generic_count > 0) {
          const auto* input_pointer =
              reinterpret_cast<const __nv_bfloat16*>(
                  hidden_workspace.data_ptr<at::BFloat16>());
          auto* output_pointer =
              reinterpret_cast<__nv_bfloat16*>(
                  out_workspace.data_ptr<at::BFloat16>());
          if (routed_warps == 8)
              launch_vq_gemv_routed<8>(
                  input_pointer, route_ids.data_ptr<int64_t>(),
                  metadata.data_ptr<int64_t>(), output_pointer,
                  top_k, expert_count, 5, hidden, intermediate,
                  2 * intermediate, use_p12_specialized, generic_offset,
                  generic_count, vector_input_copy, stream);
          else if (routed_warps == 16)
              launch_vq_gemv_routed<16>(
                  input_pointer, route_ids.data_ptr<int64_t>(),
                  metadata.data_ptr<int64_t>(), output_pointer,
                  top_k, expert_count, 5, hidden, intermediate,
                  2 * intermediate, use_p12_specialized, generic_offset,
                  generic_count, vector_input_copy, stream);
          else
              launch_vq_gemv_routed<32>(
                  input_pointer, route_ids.data_ptr<int64_t>(),
                  metadata.data_ptr<int64_t>(), output_pointer,
                  top_k, expert_count, 5, hidden, intermediate,
                  2 * intermediate, use_p12_specialized, generic_offset,
                  generic_count, vector_input_copy, stream);
      }
      if (p12_active > 0) {
          if (use_p12_shared) {
              vq_gemv_routed_p12_kernel<<<
                  dim3(
                      (unsigned)(
                          (hidden + CCCP_P12_ROWS_PER_BLOCK - 1) /
                          CCCP_P12_ROWS_PER_BLOCK),
                      (unsigned)p12_active),
                  p12_block,
                  down_p12_shared,
                  stream>>>(
                      reinterpret_cast<const __nv_bfloat16*>(
                          hidden_workspace.data_ptr<at::BFloat16>()),
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      reinterpret_cast<__nv_bfloat16*>(
                          out_workspace.data_ptr<at::BFloat16>()),
                      top_k, expert_count, 5, hidden, intermediate,
                      2 * intermediate, p12_active, 0);
          } else if (p12_l2_warps == 8) {
              vq_gemv_routed_p12_l2_kernel<8><<<
                  dim3(
                      (unsigned)((hidden + 7) / 8),
                      (unsigned)p12_active),
                  dim3(32, 8),
                  (size_t)intermediate * sizeof(__nv_bfloat16),
                  stream>>>(
                      reinterpret_cast<const __nv_bfloat16*>(
                          hidden_workspace.data_ptr<at::BFloat16>()),
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      reinterpret_cast<__nv_bfloat16*>(
                          out_workspace.data_ptr<at::BFloat16>()),
                      top_k, expert_count, 5, hidden, intermediate,
                      2 * intermediate, p12_active, 0);
          } else if (p12_l2_warps == 16) {
              vq_gemv_routed_p12_l2_kernel<16><<<
                  dim3(
                      (unsigned)((hidden + 15) / 16),
                      (unsigned)p12_active),
                  dim3(32, 16),
                  (size_t)intermediate * sizeof(__nv_bfloat16),
                  stream>>>(
                      reinterpret_cast<const __nv_bfloat16*>(
                          hidden_workspace.data_ptr<at::BFloat16>()),
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      reinterpret_cast<__nv_bfloat16*>(
                          out_workspace.data_ptr<at::BFloat16>()),
                      top_k, expert_count, 5, hidden, intermediate,
                      2 * intermediate, p12_active, 0);
          } else {
              vq_gemv_routed_p12_l2_kernel<32><<<
                  dim3(
                      (unsigned)((hidden + 31) / 32),
                      (unsigned)p12_active),
                  dim3(32, 32),
                  (size_t)intermediate * sizeof(__nv_bfloat16),
                  stream>>>(
                      reinterpret_cast<const __nv_bfloat16*>(
                          hidden_workspace.data_ptr<at::BFloat16>()),
                      route_ids.data_ptr<int64_t>(),
                      metadata.data_ptr<int64_t>(),
                      reinterpret_cast<__nv_bfloat16*>(
                          out_workspace.data_ptr<at::BFloat16>()),
                      top_k, expert_count, 5, hidden, intermediate,
                      2 * intermediate, p12_active, 0);
          }
      }
    routed_weighted_sum_f32_kernel<<<
        (hidden + 255) / 256,
        256,
        0,
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                out_workspace.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            weights.data_ptr<float>(),
            metadata.data_ptr<int64_t>(),
            result.data_ptr<float>(),
            top_k,
            expert_count,
            hidden,
            -1,
            false);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

__global__ void routed_weighted_sum_f32_kernel(
    const __nv_bfloat16* __restrict__ rows,
    const int64_t* __restrict__ route_ids,
    const float* __restrict__ weights,
    const int64_t* __restrict__ metadata,
    float* __restrict__ result,
    const int K,
    const int E,
    const int D,
    const int dtype_filter,
    const bool accumulate)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    #pragma unroll
    for (int n = 0; n < MAX_SLOT_EXPERTS; ++n) {
        if (n >= K) continue;
        const int expert_id = static_cast<int>(route_ids[n]);
        if (
            expert_id >= 0 && expert_id < E &&
            metadata[(long)5 * E + expert_id] != 0 &&
            (
                dtype_filter < 0 ||
                metadata[(long)9 * E + expert_id] == dtype_filter
            )
        ) {
            acc = fmaf(
                __bfloat162float(rows[(long)n * D + d]),
                weights[n],
                acc);
        }
    }
    result[d] = accumulate ? result[d] + acc : acc;
}

// Prefill counterpart of the decode weighted reduction.  One CUDA launch
// publishes all rows while preserving the exact BF16-rounded expert output
// and FP32 route accumulation used by the single-token path.
__global__ void routed_weighted_sum_rows_f32_kernel(
    const __nv_bfloat16* __restrict__ rows,
    const int64_t* __restrict__ route_ids,
    const float* __restrict__ weights,
    const int64_t* __restrict__ metadata,
    float* __restrict__ result,
    const int batch_size,
    const int top_k,
    const int expert_count,
    const int hidden)
{
    const int input_row = blockIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (input_row >= batch_size || column >= hidden)
        return;
    const long route_base = static_cast<long>(input_row) * top_k;
    float acc = 0.0f;
    #pragma unroll
    for (int position = 0; position < MAX_SLOT_EXPERTS; ++position) {
        if (position >= top_k)
            continue;
        const int expert = static_cast<int>(
            route_ids[route_base + position]);
        if (
            expert >= 0 && expert < expert_count &&
            metadata[static_cast<long>(5) * expert_count + expert] != 0
        ) {
            acc = fmaf(
                __bfloat162float(
                    rows[(route_base + position) * hidden + column]),
                weights[route_base + position],
                acc);
        }
    }
    result[static_cast<long>(input_row) * hidden + column] = acc;
}

torch::Tensor moe_mlp_routed_slots(
    torch::Tensor x,
    torch::Tensor route_ids,
    torch::Tensor weights,
    torch::Tensor metadata,
    double limit,
    torch::Tensor hidden_workspace,
    torch::Tensor out_workspace,
    torch::Tensor result,
    bool d4_specialized,
    bool accumulate)
{
    TORCH_CHECK(
        x.is_cuda() && x.scalar_type() == at::kBFloat16 &&
        x.is_contiguous() && x.dim() == 2 && x.size(0) == 1,
        "routed slot input must be contiguous CUDA BF16 [1,D]");
    TORCH_CHECK(
        route_ids.is_cuda() && route_ids.scalar_type() == at::kLong &&
        route_ids.is_contiguous() && route_ids.dim() == 1,
        "route IDs must be contiguous CUDA int64 [K]");
    const int K = static_cast<int>(route_ids.numel());
    TORCH_CHECK(
        K > 0 && K <= MAX_SLOT_EXPERTS,
        "routed slot Top-K must be in [1,8]");
    TORCH_CHECK(
        weights.is_cuda() && weights.scalar_type() == at::kFloat &&
        weights.is_contiguous() && weights.dim() == 1 &&
        weights.numel() == K,
        "route weights must be contiguous CUDA float32 [K]");
    TORCH_CHECK(
        metadata.is_cuda() && metadata.scalar_type() == at::kLong &&
        metadata.is_contiguous() && metadata.dim() == 2 &&
        metadata.size(0) == ROUTED_META_ROWS,
        "routed metadata must be contiguous CUDA int64 [10,E]");
    TORCH_CHECK(
        x.get_device() == route_ids.get_device() &&
        x.get_device() == weights.get_device() &&
        x.get_device() == metadata.get_device() &&
        x.get_device() == hidden_workspace.get_device() &&
        x.get_device() == out_workspace.get_device(),
        "routed slot compute tensors must be on one CUDA device");
    TORCH_CHECK(
        result.is_cuda(),
        "routed partial result must be a CUDA tensor");
    int current = -1;
    const auto current_status = cudaGetDevice(&current);
    TORCH_CHECK(
        current_status == cudaSuccess &&
        current == x.get_device(),
        "routed slot kernel must run on its input CUDA device");
    ensure_peer_access(
        current,
        result.get_device(),
        "expert direct return");
    const int E = static_cast<int>(metadata.size(1));
    const int hidden = static_cast<int>(x.size(1));
    TORCH_CHECK(
        hidden_workspace.scalar_type() == at::kBFloat16 &&
        hidden_workspace.is_contiguous() &&
        hidden_workspace.dim() == 2 &&
        hidden_workspace.size(0) == K &&
        hidden_workspace.size(1) % 2 == 0,
        "hidden workspace must be contiguous BF16 [K,2I]");
    const int inter = static_cast<int>(hidden_workspace.size(1) / 2);
    TORCH_CHECK(
        out_workspace.scalar_type() == at::kBFloat16 &&
        out_workspace.is_contiguous() &&
        out_workspace.dim() == 2 &&
        out_workspace.size(0) == K &&
        out_workspace.size(1) == hidden,
        "output workspace must be contiguous BF16 [K,D]");
    TORCH_CHECK(
        result.scalar_type() == at::kFloat &&
        result.is_contiguous() && result.dim() == 1 &&
        result.numel() == hidden,
        "routed partial result must be contiguous float32 [D]");

    dim3 block(32, ROWS_PER_BLOCK);
    auto stream = at::cuda::getCurrentCUDAStream();
    vq_gemv_routed_kernel<ROWS_PER_BLOCK><<<
        dim3(
            (unsigned)((2 * inter + ROWS_PER_BLOCK - 1) /
                       ROWS_PER_BLOCK),
            (unsigned)K),
        block,
        (size_t)hidden * sizeof(__nv_bfloat16),
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                x.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            metadata.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(
                hidden_workspace.data_ptr<at::BFloat16>()),
            K, E, 0, 2 * inter, hidden, 0,
            d4_specialized, 0, true);
    routed_swiglu_bf16_inplace_kernel<<<
        (K * inter + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(
                hidden_workspace.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            metadata.data_ptr<int64_t>(),
            K, E, inter, static_cast<float>(limit));
    vq_gemv_routed_kernel<ROWS_PER_BLOCK><<<
        dim3(
            (unsigned)((hidden + ROWS_PER_BLOCK - 1) /
                       ROWS_PER_BLOCK),
            (unsigned)K),
        block,
        (size_t)inter * sizeof(__nv_bfloat16),
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                hidden_workspace.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            metadata.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(
                out_workspace.data_ptr<at::BFloat16>()),
            K, E, 5, hidden, inter, 2 * inter,
            d4_specialized, 0, true);
    routed_weighted_sum_f32_kernel<<<
        (hidden + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                out_workspace.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            weights.data_ptr<float>(),
            metadata.data_ptr<int64_t>(),
            result.data_ptr<float>(),
            K, E, hidden, -1, accumulate);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

// ---- Dedicated D4/K4096 routed VQ path ---------------------------------
//
// An independent ``vv`` expert has a 4096x4 BF16 codebook and uint16
// indices.  The generic kernel re-reads four random BF16 values for every
// matrix index.  The complete codebook is only 32 KiB, so one CTA can stage
// it once in shared memory and reuse it across 32 output rows.  v256 experts
// remain on the CodeGEMM Psumbook path; metadata for those experts is zero in
// this kernel and returns before the first barrier.

constexpr int CCCP_VV_CODES = 4096;
constexpr int CCCP_VV_VECTOR = 4;
// Six BF16 slots per code make consecutive code starts advance by three
// shared-memory banks instead of two. Random K4096 lookups can then use all
// 32 banks rather than only the 16 even/odd start banks.
constexpr int CCCP_VV_SHARED_STRIDE = 6;
constexpr int CCCP_VV_WARPS_PER_BLOCK = 32;
constexpr int CCCP_VV_ROWS_PER_WARP = 2;
constexpr int CCCP_VV_ROWS_PER_BLOCK =
    CCCP_VV_WARPS_PER_BLOCK * CCCP_VV_ROWS_PER_WARP;

__global__ void vq_gemv_routed_vv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    __nv_bfloat16* __restrict__ out,
    const int top_k,
    const int expert_count,
    const int metadata_base,
    const int output_rows,
    const int input_cols,
    const long input_stride)
{
    const int position = blockIdx.y;
    if (position >= top_k)
        return;
    const int expert = static_cast<int>(route_ids[position]);
    if (expert < 0 || expert >= expert_count)
        return;

    const int64_t index_address =
        metadata[(long)(metadata_base + 0) * expert_count + expert];
    if (index_address == 0)
        return;
    const int64_t codebook_address =
        metadata[(long)(metadata_base + 1) * expert_count + expert];
    const int blocks = static_cast<int>(
        metadata[(long)(metadata_base + 2) * expert_count + expert]);
    const int vector = static_cast<int>(
        metadata[(long)(metadata_base + 3) * expert_count + expert]);
    const int dtype_tag = static_cast<int>(
        metadata[(long)(metadata_base + 4) * expert_count + expert]);
    if (vector != CCCP_VV_VECTOR || dtype_tag != 1)
        return;

    extern __shared__ __nv_bfloat16 vv_shared[];
    auto* shared_input = vv_shared;
    auto* shared_codebook = vv_shared + input_cols;
    const int linear_thread = threadIdx.y * 32 + threadIdx.x;
    const int block_threads = 32 * CCCP_VV_WARPS_PER_BLOCK;
    const __nv_bfloat16* input_row =
        x + (long)position * input_stride;
    for (
        int item = linear_thread;
        item < input_cols;
        item += block_threads
    )
        shared_input[item] = input_row[item];
    const auto* codebook = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(codebook_address));
    constexpr int codebook_items =
        CCCP_VV_CODES * CCCP_VV_VECTOR;
    for (
        int item = linear_thread;
        item < codebook_items;
        item += block_threads
    )
    {
        const int code = item / CCCP_VV_VECTOR;
        const int component = item - code * CCCP_VV_VECTOR;
        shared_codebook[
            code * CCCP_VV_SHARED_STRIDE + component
        ] = codebook[item];
    }
    __syncthreads();

    const auto* indices = reinterpret_cast<const uint16_t*>(
        static_cast<uintptr_t>(index_address));
    float values[CCCP_VV_ROWS_PER_WARP] = {};
    for (int block = threadIdx.x; block < blocks; block += 32) {
        #pragma unroll
        for (int item = 0; item < CCCP_VV_ROWS_PER_WARP; ++item) {
            const int row =
                blockIdx.x * CCCP_VV_ROWS_PER_BLOCK +
                threadIdx.y +
                item * CCCP_VV_WARPS_PER_BLOCK;
            if (row < output_rows) {
                const int code = static_cast<int>(
                    indices[(long)row * blocks + block]);
                values[item] += vq_block_dot4_bf16(
                    shared_codebook +
                        (long)code * CCCP_VV_SHARED_STRIDE,
                    shared_input + block * CCCP_VV_VECTOR);
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int item = 0; item < CCCP_VV_ROWS_PER_WARP; ++item)
            values[item] += __shfl_down_sync(
                0xffffffffu,
                values[item],
                offset);
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int item = 0; item < CCCP_VV_ROWS_PER_WARP; ++item) {
            const int row =
                blockIdx.x * CCCP_VV_ROWS_PER_BLOCK +
                threadIdx.y +
                item * CCCP_VV_WARPS_PER_BLOCK;
            if (row < output_rows)
                out[(long)position * output_rows + row] =
                    __float2bfloat16_rn(values[item]);
        }
    }
}

torch::Tensor moe_mlp_routed_vv(
    torch::Tensor input,
    torch::Tensor route_ids,
    torch::Tensor weights,
    torch::Tensor metadata,
    double limit,
    torch::Tensor hidden_workspace,
    torch::Tensor out_workspace,
    torch::Tensor result,
    bool accumulate)
{
    TORCH_CHECK(
        input.is_cuda() && input.scalar_type() == at::kBFloat16 &&
        input.is_contiguous() && input.dim() == 2 && input.size(0) == 1,
        "vv input must be contiguous CUDA BF16 [1,D]");
    TORCH_CHECK(
        route_ids.is_cuda() && route_ids.scalar_type() == at::kLong &&
        route_ids.is_contiguous() && route_ids.dim() == 1,
        "vv route IDs must be contiguous CUDA int64 [K]");
    const int top_k = static_cast<int>(route_ids.numel());
    TORCH_CHECK(
        top_k > 0 && top_k <= MAX_SLOT_EXPERTS,
        "vv Top-K must be in [1,8]");
    TORCH_CHECK(
        weights.is_cuda() && weights.scalar_type() == at::kFloat &&
        weights.is_contiguous() && weights.sizes() == route_ids.sizes(),
        "vv weights must be contiguous CUDA float32 [K]");
    TORCH_CHECK(
        metadata.is_cuda() && metadata.scalar_type() == at::kLong &&
        metadata.is_contiguous() && metadata.dim() == 2 &&
        metadata.size(0) == ROUTED_META_ROWS,
        "vv metadata must be contiguous CUDA int64 [10,E]");
    TORCH_CHECK(
        input.get_device() == route_ids.get_device() &&
        input.get_device() == weights.get_device() &&
        input.get_device() == metadata.get_device() &&
        input.get_device() == hidden_workspace.get_device() &&
        input.get_device() == out_workspace.get_device(),
        "vv compute tensors must share one device");
    TORCH_CHECK(
        result.is_cuda() && result.scalar_type() == at::kFloat &&
        result.is_contiguous() && result.dim() == 1,
        "vv result must be contiguous CUDA float32 [D]");

    const int hidden = static_cast<int>(input.size(1));
    TORCH_CHECK(
        hidden_workspace.scalar_type() == at::kBFloat16 &&
        hidden_workspace.is_contiguous() &&
        hidden_workspace.dim() == 2 &&
        hidden_workspace.size(0) == top_k &&
        hidden_workspace.size(1) % 2 == 0,
        "vv hidden workspace must be BF16 [K,2I]");
    const int intermediate =
        static_cast<int>(hidden_workspace.size(1) / 2);
    TORCH_CHECK(
        out_workspace.scalar_type() == at::kBFloat16 &&
        out_workspace.is_contiguous() &&
        out_workspace.sizes() ==
            torch::IntArrayRef({top_k, hidden}),
        "vv output workspace must be BF16 [K,D]");
    TORCH_CHECK(
        result.numel() == hidden,
        "vv result width mismatch");

    int current = -1;
    const auto status = cudaGetDevice(&current);
    TORCH_CHECK(
        status == cudaSuccess && current == input.get_device(),
        "vv kernel must run on its input CUDA device");
    ensure_peer_access(current, result.get_device(), "vv direct return");

    const int expert_count = static_cast<int>(metadata.size(1));
    dim3 block(32, CCCP_VV_WARPS_PER_BLOCK);
    auto stream = at::cuda::getCurrentCUDAStream();
    const size_t gu_shared = static_cast<size_t>(
        hidden + CCCP_VV_CODES * CCCP_VV_SHARED_STRIDE
    ) * sizeof(__nv_bfloat16);
    const size_t dn_shared = static_cast<size_t>(
        intermediate + CCCP_VV_CODES * CCCP_VV_SHARED_STRIDE
    ) * sizeof(__nv_bfloat16);
    const size_t max_shared =
        gu_shared > dn_shared ? gu_shared : dn_shared;
    if (max_shared > 48 * 1024) {
        static size_t configured_shared[16] = {};
        if (configured_shared[current] < max_shared) {
            const auto attribute_status = cccp_gpu_func_set_attribute(
                vq_gemv_routed_vv_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(max_shared));
            TORCH_CHECK(
                attribute_status == cudaSuccess,
                "failed to configure vv shared memory: ",
                cudaGetErrorString(attribute_status));
            configured_shared[current] = max_shared;
        }
    }
    vq_gemv_routed_vv_kernel<<<
        dim3(
            (2 * intermediate + CCCP_VV_ROWS_PER_BLOCK - 1) /
                CCCP_VV_ROWS_PER_BLOCK,
            top_k),
        block,
        gu_shared,
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                input.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            metadata.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(
                hidden_workspace.data_ptr<at::BFloat16>()),
            top_k,
            expert_count,
            0,
            2 * intermediate,
            hidden,
            0);
    routed_swiglu_bf16_inplace_kernel<<<
        (top_k * intermediate + 255) / 256,
        256,
        0,
        stream>>>(
            reinterpret_cast<__nv_bfloat16*>(
                hidden_workspace.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            metadata.data_ptr<int64_t>(),
            top_k,
            expert_count,
            intermediate,
            static_cast<float>(limit));
    vq_gemv_routed_vv_kernel<<<
        dim3(
            (hidden + CCCP_VV_ROWS_PER_BLOCK - 1) /
                CCCP_VV_ROWS_PER_BLOCK,
            top_k),
        block,
        dn_shared,
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                hidden_workspace.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            metadata.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(
                out_workspace.data_ptr<at::BFloat16>()),
            top_k,
            expert_count,
            5,
            hidden,
            intermediate,
            2 * intermediate);
    routed_weighted_sum_f32_kernel<<<
        (hidden + 255) / 256,
        256,
        0,
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                out_workspace.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            weights.data_ptr<float>(),
            metadata.data_ptr<int64_t>(),
            result.data_ptr<float>(),
            top_k,
            expert_count,
            hidden,
            1,
            accumulate);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

// ---- HC sinkhorn (hc = 4 fixed: DSV4 Hyper-Connections) ----

__global__ void hc_sinkhorn_kernel(
    const float* __restrict__ mixes,  // [N, 24]
    const float* __restrict__ scale,  // [3]
    const float* __restrict__ base,   // [24]
    float* __restrict__ out,          // [N, 24]: pre[4] | post[4] | comb[16]
    const int N, const int iters, const float eps)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const float* m = mixes + (long)row * 24;
    float* o = out + (long)row * 24;
    const float s0 = scale[0], s1 = scale[1], s2 = scale[2];

    #pragma unroll
    for (int j = 0; j < 4; ++j)
        o[j] = 1.f / (1.f + expf(-(m[j] * s0 + base[j]))) + eps;
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        o[4 + j] = 2.f / (1.f + expf(-(m[4 + j] * s1 + base[4 + j])));

    float c[4][4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float mx = -INFINITY;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            c[j][k] = m[8 + 4 * j + k] * s2 + base[8 + 4 * j + k];
            mx = fmaxf(mx, c[j][k]);
        }
        float sum = 0.f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) { c[j][k] = expf(c[j][k] - mx); sum += c[j][k]; }
        #pragma unroll
        for (int k = 0; k < 4; ++k) c[j][k] = c[j][k] / sum + eps;
    }
    // first column normalize (after softmax), then iters-1 rounds of row+col
    for (int it = 0; it < iters; ++it) {
        if (it > 0) {  // row normalize (skip on round 0: softmax already row-stochastic)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float rs = c[j][0] + c[j][1] + c[j][2] + c[j][3];
                const float inv = 1.f / (rs + eps);
                #pragma unroll
                for (int k = 0; k < 4; ++k) c[j][k] *= inv;
            }
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) {  // column normalize
            float cs = c[0][k] + c[1][k] + c[2][k] + c[3][k];
            const float inv = 1.f / (cs + eps);
            #pragma unroll
            for (int j = 0; j < 4; ++j) c[j][k] *= inv;
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        #pragma unroll
        for (int k = 0; k < 4; ++k)
            o[8 + 4 * j + k] = c[j][k];
}

torch::Tensor hc_sinkhorn(torch::Tensor mixes, torch::Tensor scale,
                          torch::Tensor base, long iters, double eps) {
    TORCH_CHECK(mixes.is_cuda() && scale.is_cuda() && base.is_cuda(),
                "tensors must be CUDA");
    TORCH_CHECK(mixes.scalar_type() == at::kFloat, "mixes must be float32");
    TORCH_CHECK(mixes.size(-1) == 24, "mixes last dim must be 24 (hc=4)");
    auto m2 = mixes.contiguous().view({-1, 24});
    const int N = (int)m2.size(0);
    auto out = torch::empty_like(m2);
    const int threads = 128;
    const int blocks = (N + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();
    hc_sinkhorn_kernel<<<blocks, threads, 0, stream>>>(
        m2.data_ptr<float>(), scale.contiguous().data_ptr<float>(),
        base.contiguous().data_ptr<float>(), out.data_ptr<float>(),
        N, (int)iters, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.view(mixes.sizes());
}

// ---- RMSNorm (fused: one block per row, f32) ----
// out[r, i] = w[i] * x[r, i] * rsqrt(mean_i(x[r]^2) + eps)

__global__ void rmsnorm_kernel(
    const float* __restrict__ x,   // [N, D]
    const float* __restrict__ w,   // [D]
    float* __restrict__ out,       // [N, D]
    const int D, const float eps)
{
    const int r = blockIdx.x;
    const float* xr = x + (long)r * D;
    float* orow = out + (long)r * D;
    float acc = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = xr[i];
        acc += v * v;
    }
    __shared__ float red[32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? red[threadIdx.x] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float scale = rsqrtf(red[0] / (float)D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        orow[i] = w[i] * (xr[i] * scale);
}

torch::Tensor rmsnorm(
    torch::Tensor x,
    torch::Tensor w,
    double eps,
    c10::optional<torch::Tensor> output_buffer)
{
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat && w.scalar_type() == at::kFloat,
                "x/w must be float32");
    auto xc = x.contiguous();
    const int D = (int)xc.size(-1);
    auto x2 = xc.view({-1, D});
    const int N = (int)x2.size(0);
    auto out = output_buffer.has_value()
        ? output_buffer.value().view({-1, D})
        : torch::empty_like(x2);
    TORCH_CHECK(
        out.is_cuda() &&
        out.scalar_type() == at::kFloat &&
        out.is_contiguous() &&
        out.sizes() == x2.sizes() &&
        out.get_device() == x.get_device(),
        "RMSNorm output buffer must be contiguous float32 and match input");
    auto stream = at::cuda::getCurrentCUDAStream();
    // 空批守卫:缓存全命中的续算路径会产生 0 行输入,grid 维度为 0
    // 是 cudaErrorInvalidConfiguration(GLM 双 generate 实证)。
    if (N > 0) {
        rmsnorm_kernel<<<N, 256, 0, stream>>>(
            x2.data_ptr<float>(), w.contiguous().data_ptr<float>(),
            out.data_ptr<float>(), D, (float)eps);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out.view(xc.sizes());
}

template <typename weight_t>
__device__ __forceinline__ float rmsnorm_weight_float(weight_t value)
{
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float rmsnorm_weight_float(
    __nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <typename weight_t>
__global__ void rmsnorm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const weight_t* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    const int D,
    const float eps)
{
    const int row = blockIdx.x;
    const auto* input = x + static_cast<long>(row) * D;
    auto* output = out + static_cast<long>(row) * D;
    float sum = 0.0f;
    for (int item = threadIdx.x; item < D; item += blockDim.x) {
        const float value = __bfloat162float(input[item]);
        sum += value * value;
    }
    __shared__ float reduction[32];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if ((threadIdx.x & 31) == 0)
        reduction[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float value = (
            threadIdx.x < (blockDim.x + 31) / 32
            ? reduction[threadIdx.x]
            : 0.0f);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(
                0xffffffffu,
                value,
                offset);
        if (threadIdx.x == 0)
            reduction[0] = value;
    }
    __syncthreads();
    const float scale = rsqrtf(reduction[0] / static_cast<float>(D) + eps);
    for (int item = threadIdx.x; item < D; item += blockDim.x) {
        // Match the reference's two BF16 boundaries:
        // weight.to(BF16) * normalized.to(BF16).
        const __nv_bfloat16 normalized = __float2bfloat16_rn(
            __bfloat162float(input[item]) * scale);
        const __nv_bfloat16 weight = __float2bfloat16_rn(
            rmsnorm_weight_float(w[item]));
        output[item] = __float2bfloat16_rn(
            __bfloat162float(normalized)
            * __bfloat162float(weight));
    }
}

torch::Tensor rmsnorm_bf16(
    torch::Tensor x,
    torch::Tensor w,
    double eps,
    c10::optional<torch::Tensor> output_buffer)
{
    TORCH_CHECK(
        x.is_cuda() && w.is_cuda(),
        "BF16 RMSNorm tensors must be CUDA");
    TORCH_CHECK(
        x.scalar_type() == at::kBFloat16 &&
        (
            w.scalar_type() == at::kBFloat16 ||
            w.scalar_type() == at::kFloat
        ),
        "BF16 RMSNorm requires BF16 input and BF16/FP32 weight");
    auto input = x.contiguous();
    auto weight = w.contiguous();
    const int width = static_cast<int>(input.size(-1));
    auto rows = input.view({-1, width});
    auto output = output_buffer.has_value()
        ? output_buffer.value().view({-1, width})
        : torch::empty_like(rows);
    TORCH_CHECK(
        weight.dim() == 1 &&
        weight.numel() == width &&
        output.is_cuda() &&
        output.scalar_type() == at::kBFloat16 &&
        output.is_contiguous() &&
        output.sizes() == rows.sizes() &&
        output.get_device() == input.get_device() &&
        weight.get_device() == input.get_device(),
        "BF16 RMSNorm shapes/devices do not match");
    auto stream = at::cuda::getCurrentCUDAStream();
    if (weight.scalar_type() == at::kBFloat16) {
        rmsnorm_bf16_kernel<<<rows.size(0), 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                rows.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            width,
            static_cast<float>(eps));
    } else {
        rmsnorm_bf16_kernel<<<rows.size(0), 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                rows.data_ptr<at::BFloat16>()),
            weight.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            width,
            static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output.view(input.sizes());
}

constexpr int CCCP_RESIDUAL_MAX_ROWS = 16;
constexpr int CCCP_RESIDUAL_STAGED_MAX_ROWS = 32;
constexpr int CCCP_RESIDUAL_THREADS = 256;
constexpr int CCCP_RESIDUAL_WARPS = CCCP_RESIDUAL_THREADS / 32;

__global__ void attention_residual_bf16_kernel(
    const __nv_bfloat16* __restrict__ prefix,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ projection,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ post_norm_weight,
    float* __restrict__ residual_inverse,
    __nv_bfloat16* __restrict__ output,
    const int batch_size,
    const int residual_rows,
    const int width,
    const float eps)
{
    const int rows = residual_rows + 1;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float partial[
        CCCP_RESIDUAL_MAX_ROWS * CCCP_RESIDUAL_WARPS];
    __shared__ float row_values[CCCP_RESIDUAL_MAX_ROWS];

    const int batch = blockIdx.x;
    if (batch >= batch_size)
        return;
    const long residual_base = static_cast<long>(batch) * residual_rows * width;
    const long prefix_base = static_cast<long>(batch) * width;
    const long inverse_base = static_cast<long>(batch) * residual_rows;
    if (threadIdx.x < rows) {
        const int row = threadIdx.x;
        row_values[row] = (
            row < residual_rows && residual_inverse != nullptr
            ? residual_inverse[inverse_base + row]
            : 0.0f);
    }
    __syncthreads();
    for (int row = 0; row < rows; ++row) {
        if (row_values[row] > 0.0f)
            continue;
            const auto* source = (
            row < residual_rows
            ? residual + residual_base + static_cast<long>(row) * width
            : prefix + prefix_base);
        float sum = 0.0f;
        for (
            int item = threadIdx.x;
            item < width;
            item += blockDim.x
        ) {
            const float value = __bfloat162float(source[item]);
            sum += value * value;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0)
            partial[row * CCCP_RESIDUAL_WARPS + warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        for (int row = lane; row < rows; row += 32) {
            if (row_values[row] > 0.0f)
                continue;
            float sum = 0.0f;
            #pragma unroll
            for (int item = 0; item < CCCP_RESIDUAL_WARPS; ++item)
                sum += partial[row * CCCP_RESIDUAL_WARPS + item];
            row_values[row] = rsqrtf(
                sum / static_cast<float>(width) + eps);
            if (row < residual_rows && residual_inverse != nullptr)
                residual_inverse[inverse_base + row] = row_values[row];
        }
    }
    __syncthreads();

    for (int row = 0; row < rows; ++row) {
        const auto* source = (
            row < residual_rows
            ? residual + residual_base + static_cast<long>(row) * width
            : prefix + prefix_base);
        float score = 0.0f;
        const float inverse = row_values[row];
        for (
            int item = threadIdx.x;
            item < width;
            item += blockDim.x
        ) {
            score += (
                __bfloat162float(source[item])
                * inverse
                * __bfloat162float(norm_weight[item])
                * __bfloat162float(projection[item]));
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            score += __shfl_down_sync(
                0xffffffffu,
                score,
                offset);
        if (lane == 0)
            partial[row * CCCP_RESIDUAL_WARPS + warp] = score;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float maximum = -1.0e30f;
        for (int row = 0; row < rows; ++row) {
            float score = 0.0f;
            #pragma unroll
            for (int item = 0; item < CCCP_RESIDUAL_WARPS; ++item)
                score += partial[row * CCCP_RESIDUAL_WARPS + item];
            row_values[row] = score;
            maximum = fmaxf(maximum, score);
        }
        float denominator = 0.0f;
        for (int row = 0; row < rows; ++row) {
            const float value = expf(row_values[row] - maximum);
            row_values[row] = value;
            denominator += value;
        }
        for (int row = 0; row < rows; ++row)
            row_values[row] /= denominator;
    }
    __syncthreads();
    for (
        int item = threadIdx.x;
        item < width;
        item += blockDim.x
    ) {
        float mixed = 0.0f;
        for (int row = 0; row < rows; ++row) {
            const auto* source = (
                row < residual_rows
                ? residual + residual_base + static_cast<long>(row) * width
                : prefix + prefix_base);
            mixed += row_values[row] * __bfloat162float(source[item]);
        }
        output[static_cast<long>(batch) * width + item] =
            __float2bfloat16_rn(mixed);
    }
    if (post_norm_weight == nullptr)
        return;
    __syncthreads();
    float sum = 0.0f;
    for (
        int item = threadIdx.x;
        item < width;
        item += blockDim.x
    ) {
        const float value = __bfloat162float(
            output[static_cast<long>(batch) * width + item]);
        sum += value * value;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float value = (
            lane < CCCP_RESIDUAL_WARPS ? partial[lane] : 0.0f);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(
                0xffffffffu,
                value,
                offset);
        if (lane == 0)
            row_values[0] = value;
    }
    __syncthreads();
    const float scale = rsqrtf(
        row_values[0] / static_cast<float>(width) + eps);
    for (
        int item = threadIdx.x;
        item < width;
        item += blockDim.x
    ) {
        const __nv_bfloat16 normalized = __float2bfloat16_rn(
            __bfloat162float(
                output[static_cast<long>(batch) * width + item]) * scale);
        output[static_cast<long>(batch) * width + item] = __float2bfloat16_rn(
            __bfloat162float(normalized)
            * __bfloat162float(post_norm_weight[item]));
    }
}

// The one-CTA kernel above is fastest for short residual lists, but it
// serializes every row.  Deep residual blocks first calculate their row
// scores with one CTA per row, then use one deterministic CTA for softmax,
// weighted mixing and the optional following RMSNorm.  The reduction and row
// accumulation orders match the short kernel.
__global__ void attention_residual_scores_bf16_kernel(
    const __nv_bfloat16* __restrict__ prefix,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ projection,
    const __nv_bfloat16* __restrict__ norm_weight,
    float* __restrict__ residual_inverse,
    float* __restrict__ scores,
    const int batch_size,
    const int residual_rows,
    const int width,
    const float eps)
{
    const int rows = residual_rows + 1;
    const int batch = blockIdx.x / rows;
    const int row = blockIdx.x % rows;
    if (batch >= batch_size)
        return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const auto* source = (
        row < residual_rows
        ? residual + static_cast<long>(batch) * residual_rows * width
            + static_cast<long>(row) * width
        : prefix + static_cast<long>(batch) * width);
    __shared__ float partial[CCCP_RESIDUAL_WARPS];
    __shared__ float inverse;

    if (threadIdx.x == 0) {
        inverse = (
            row < residual_rows && residual_inverse != nullptr
            ? residual_inverse[static_cast<long>(batch) * residual_rows + row]
            : 0.0f);
    }
    __syncthreads();
    if (inverse <= 0.0f) {
        float sum = 0.0f;
        for (
            int item = threadIdx.x;
            item < width;
            item += blockDim.x
        ) {
            const float value = __bfloat162float(source[item]);
            sum += value * value;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0)
            partial[warp] = sum;
        __syncthreads();
        if (threadIdx.x == 0) {
            float total = 0.0f;
            #pragma unroll
            for (int item = 0; item < CCCP_RESIDUAL_WARPS; ++item)
                total += partial[item];
            inverse = rsqrtf(
                total / static_cast<float>(width) + eps);
            if (row < residual_rows && residual_inverse != nullptr)
                residual_inverse[static_cast<long>(batch) * residual_rows + row]
                    = inverse;
        }
        __syncthreads();
    }

    float score = 0.0f;
    for (
        int item = threadIdx.x;
        item < width;
        item += blockDim.x
    ) {
        score += (
            __bfloat162float(source[item])
            * inverse
            * __bfloat162float(norm_weight[item])
            * __bfloat162float(projection[item]));
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        score += __shfl_down_sync(
            0xffffffffu,
            score,
            offset);
    if (lane == 0)
        partial[warp] = score;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int item = 0; item < CCCP_RESIDUAL_WARPS; ++item)
            total += partial[item];
        scores[static_cast<long>(batch) * rows + row] = total;
    }
}

__global__ void attention_residual_mix_bf16_kernel(
    const __nv_bfloat16* __restrict__ prefix,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ post_norm_weight,
    float* __restrict__ scores,
    __nv_bfloat16* __restrict__ output,
    const int batch_size,
    const int residual_rows,
    const int width,
    const float eps)
{
    const int batch = blockIdx.x;
    if (batch >= batch_size)
        return;
    const int rows = residual_rows + 1;
    const long residual_base = static_cast<long>(batch) * residual_rows * width;
    const long prefix_base = static_cast<long>(batch) * width;
    const long score_base = static_cast<long>(batch) * rows;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float probabilities[CCCP_RESIDUAL_STAGED_MAX_ROWS];
    __shared__ float partial[CCCP_RESIDUAL_WARPS];

    if (threadIdx.x == 0) {
        float maximum = -1.0e30f;
        for (int row = 0; row < rows; ++row)
            maximum = fmaxf(maximum, scores[score_base + row]);
        float denominator = 0.0f;
        for (int row = 0; row < rows; ++row) {
            const float value = expf(scores[score_base + row] - maximum);
            probabilities[row] = value;
            denominator += value;
        }
        for (int row = 0; row < rows; ++row)
            probabilities[row] /= denominator;
    }
    __syncthreads();
    for (
        int item = threadIdx.x;
        item < width;
        item += blockDim.x
    ) {
        float mixed = 0.0f;
        for (int row = 0; row < rows; ++row) {
            const auto* source = (
                row < residual_rows
                ? residual + residual_base + static_cast<long>(row) * width
                : prefix + prefix_base);
            mixed += (
                probabilities[row]
                * __bfloat162float(source[item]));
        }
        output[static_cast<long>(batch) * width + item] =
            __float2bfloat16_rn(mixed);
    }
    if (post_norm_weight == nullptr)
        return;
    __syncthreads();
    float sum = 0.0f;
    for (
        int item = threadIdx.x;
        item < width;
        item += blockDim.x
    ) {
        const float value = __bfloat162float(
            output[static_cast<long>(batch) * width + item]);
        sum += value * value;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float value = (
            lane < CCCP_RESIDUAL_WARPS ? partial[lane] : 0.0f);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(
                0xffffffffu,
                value,
                offset);
        if (lane == 0)
            scores[score_base] = value;
    }
    __syncthreads();
    const float scale = rsqrtf(
        scores[score_base] / static_cast<float>(width) + eps);
    for (
        int item = threadIdx.x;
        item < width;
        item += blockDim.x
    ) {
        const __nv_bfloat16 normalized = __float2bfloat16_rn(
            __bfloat162float(output[static_cast<long>(batch) * width + item])
            * scale);
        output[static_cast<long>(batch) * width + item] = __float2bfloat16_rn(
            __bfloat162float(normalized)
            * __bfloat162float(post_norm_weight[item]));
    }
}

torch::Tensor attention_residual_bf16(
    torch::Tensor prefix,
    torch::Tensor residual,
    torch::Tensor projection,
    torch::Tensor norm_weight,
    c10::optional<torch::Tensor> post_norm_weight,
    double eps,
    c10::optional<torch::Tensor> output_buffer,
    c10::optional<torch::Tensor> score_workspace,
    long single_cta_max_rows,
    c10::optional<torch::Tensor> residual_inverse)
{
    TORCH_CHECK(
        prefix.is_cuda() && residual.is_cuda() &&
        projection.is_cuda() && norm_weight.is_cuda(),
        "Attention residual tensors must be CUDA");
    TORCH_CHECK(
        prefix.scalar_type() == at::kBFloat16 &&
        residual.scalar_type() == at::kBFloat16 &&
        projection.scalar_type() == at::kBFloat16 &&
        norm_weight.scalar_type() == at::kBFloat16,
        "Attention residual currently requires BF16");
    TORCH_CHECK(
        prefix.dim() == 2 && prefix.size(0) > 0 &&
        residual.dim() == 3 && residual.size(0) == prefix.size(0) &&
        residual.size(2) == prefix.size(1) &&
        projection.numel() == prefix.size(1) &&
        norm_weight.numel() == prefix.size(1) &&
        residual.size(1) > 0 &&
        residual.size(1) + 1 <= CCCP_RESIDUAL_STAGED_MAX_ROWS,
        "Attention residual shapes do not match");
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty_like(prefix);
    TORCH_CHECK(
        prefix.is_contiguous() && residual.is_contiguous() &&
        projection.is_contiguous() && norm_weight.is_contiguous() &&
        output.is_contiguous() &&
        output.scalar_type() == at::kBFloat16 &&
        output.sizes() == prefix.sizes() &&
        output.get_device() == prefix.get_device() &&
        residual.get_device() == prefix.get_device() &&
        projection.get_device() == prefix.get_device() &&
        norm_weight.get_device() == prefix.get_device(),
        "Attention residual buffers must be contiguous and colocated");
    const __nv_bfloat16* post_norm_ptr = nullptr;
    if (post_norm_weight.has_value()) {
        const auto post = post_norm_weight.value();
        TORCH_CHECK(
            post.is_cuda() &&
            post.scalar_type() == at::kBFloat16 &&
            post.is_contiguous() &&
            post.numel() == prefix.size(1) &&
            post.get_device() == prefix.get_device(),
            "Attention residual post-norm weight must be colocated BF16");
        post_norm_ptr = reinterpret_cast<const __nv_bfloat16*>(
            post.data_ptr<at::BFloat16>());
    }
    float* residual_inverse_ptr = nullptr;
    if (residual_inverse.has_value()) {
        const auto inverse = residual_inverse.value();
        TORCH_CHECK(
            inverse.is_cuda() &&
            inverse.scalar_type() == at::kFloat &&
            inverse.is_contiguous() &&
            inverse.numel() >= prefix.size(0) * residual.size(1) &&
            inverse.get_device() == prefix.get_device(),
            "Attention residual inverse cache must be colocated "
            "contiguous float32[>=residual_rows]");
        residual_inverse_ptr = inverse.data_ptr<float>();
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    const int rows = static_cast<int>(residual.size(1)) + 1;
    TORCH_CHECK(
        single_cta_max_rows >= 1 &&
        single_cta_max_rows <= CCCP_RESIDUAL_MAX_ROWS,
        "Attention residual single-CTA threshold must be in [1,16]");
    if (rows <= single_cta_max_rows) {
        attention_residual_bf16_kernel<<<
            prefix.size(0),
            CCCP_RESIDUAL_THREADS,
            0,
            stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    prefix.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    residual.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    projection.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    norm_weight.data_ptr<at::BFloat16>()),
                post_norm_ptr,
                residual_inverse_ptr,
                reinterpret_cast<__nv_bfloat16*>(
                    output.data_ptr<at::BFloat16>()),
                static_cast<int>(prefix.size(0)),
                static_cast<int>(residual.size(1)),
                static_cast<int>(prefix.size(1)),
                static_cast<float>(eps));
    } else {
        TORCH_CHECK(
            score_workspace.has_value(),
            "deep Attention residual requires a score workspace");
        const auto workspace = score_workspace.value();
        TORCH_CHECK(
            workspace.is_cuda() &&
            workspace.scalar_type() == at::kFloat &&
            workspace.is_contiguous() &&
            workspace.numel() >= prefix.size(0) * CCCP_RESIDUAL_STAGED_MAX_ROWS &&
            workspace.get_device() == prefix.get_device(),
            "Attention residual score workspace must be colocated "
            "contiguous float32[>=32]");
        attention_residual_scores_bf16_kernel<<<
            prefix.size(0) * rows,
            CCCP_RESIDUAL_THREADS,
            0,
            stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    prefix.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    residual.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    projection.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    norm_weight.data_ptr<at::BFloat16>()),
                residual_inverse_ptr,
                workspace.data_ptr<float>(),
                static_cast<int>(prefix.size(0)),
                static_cast<int>(residual.size(1)),
                static_cast<int>(prefix.size(1)),
                static_cast<float>(eps));
        attention_residual_mix_bf16_kernel<<<
            prefix.size(0),
            CCCP_RESIDUAL_THREADS,
            0,
            stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    prefix.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    residual.data_ptr<at::BFloat16>()),
                post_norm_ptr,
                workspace.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(
                    output.data_ptr<at::BFloat16>()),
                static_cast<int>(prefix.size(0)),
                static_cast<int>(residual.size(1)),
                static_cast<int>(prefix.size(1)),
                static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void tp_hidden_add_bf16_kernel(
    const __nv_bfloat16* __restrict__ left,
    const __nv_bfloat16* __restrict__ right,
    __nv_bfloat16* __restrict__ output,
    const int count)
{
    for (
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < count;
        index += blockDim.x * gridDim.x
    ) {
        output[index] = __float2bfloat16_rn(
            __bfloat162float(left[index])
            + __bfloat162float(right[index]));
    }
}

std::vector<torch::Tensor> tp_hidden_add_batch(
    std::vector<torch::Tensor> left,
    std::vector<int64_t> left_events,
    std::vector<torch::Tensor> right,
    std::vector<int64_t> right_events,
    std::vector<torch::Tensor> outputs,
    std::vector<int64_t> output_events)
{
    const size_t count = outputs.size();
    TORCH_CHECK(
        count > 0 &&
        left.size() == count &&
        left_events.size() == count &&
        right.size() == count &&
        right_events.size() == count &&
        output_events.size() == count,
        "TPHidden add vectors must be non-empty and size-equal");
    int original_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&original_device));
    for (size_t rank = 0; rank < count; ++rank) {
        const int target = outputs[rank].get_device();
        TORCH_CHECK(
            left[rank].is_cuda() &&
            right[rank].is_cuda() &&
            outputs[rank].is_cuda() &&
            left[rank].get_device() == target &&
            right[rank].get_device() == target &&
            left[rank].scalar_type() == at::kBFloat16 &&
            right[rank].scalar_type() == at::kBFloat16 &&
            outputs[rank].scalar_type() == at::kBFloat16 &&
            left[rank].is_contiguous() &&
            right[rank].is_contiguous() &&
            outputs[rank].is_contiguous() &&
            left[rank].sizes() == outputs[rank].sizes() &&
            right[rank].sizes() == outputs[rank].sizes(),
            "TPHidden add requires matching colocated BF16 tensors");
        C10_CUDA_CHECK(cudaSetDevice(target));
        const auto stream = at::cuda::getCurrentCUDAStream(target);
        C10_CUDA_CHECK(cudaStreamWaitEvent(
            stream,
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(left_events[rank])),
            0));
        C10_CUDA_CHECK(cudaStreamWaitEvent(
            stream,
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(right_events[rank])),
            0));
        const int items = static_cast<int>(outputs[rank].numel());
        const int blocks = std::min(32, (items + 255) / 256);
        tp_hidden_add_bf16_kernel<<<blocks, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                left[rank].data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                right[rank].data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                outputs[rank].data_ptr<at::BFloat16>()),
            items);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(cudaEventRecord(
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(output_events[rank])),
            stream));
    }
    C10_CUDA_CHECK(cudaSetDevice(original_device));
    return outputs;
}

std::vector<torch::Tensor> tp_hidden_rmsnorm_batch(
    std::vector<torch::Tensor> inputs,
    std::vector<int64_t> input_events,
    std::vector<torch::Tensor> weights,
    double eps,
    std::vector<torch::Tensor> outputs,
    std::vector<int64_t> output_events)
{
    const size_t count = outputs.size();
    TORCH_CHECK(
        count > 0 &&
        inputs.size() == count &&
        input_events.size() == count &&
        weights.size() == count &&
        output_events.size() == count,
        "TPHidden RMSNorm vectors must be non-empty and size-equal");
    int original_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&original_device));
    for (size_t rank = 0; rank < count; ++rank) {
        const int target = outputs[rank].get_device();
        TORCH_CHECK(
            inputs[rank].get_device() == target &&
            weights[rank].get_device() == target,
            "TPHidden RMSNorm tensors must be colocated");
        C10_CUDA_CHECK(cudaSetDevice(target));
        const auto stream = at::cuda::getCurrentCUDAStream(target);
        C10_CUDA_CHECK(cudaStreamWaitEvent(
            stream,
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(input_events[rank])),
            0));
        rmsnorm_bf16(
            inputs[rank],
            weights[rank],
            eps,
            outputs[rank]);
        C10_CUDA_CHECK(cudaEventRecord(
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(output_events[rank])),
            stream));
    }
    C10_CUDA_CHECK(cudaSetDevice(original_device));
    return outputs;
}

std::vector<torch::Tensor> tp_hidden_residual_mix_batch(
    std::vector<torch::Tensor> prefixes,
    std::vector<int64_t> prefix_events,
    std::vector<torch::Tensor> residuals,
    std::vector<int64_t> residual_events,
    std::vector<torch::Tensor> projections,
    std::vector<torch::Tensor> norm_weights,
    std::vector<torch::Tensor> post_norm_weights,
    std::vector<torch::Tensor> workspaces,
    std::vector<torch::Tensor> residual_inverses,
    double eps,
    long single_cta_max_rows,
    std::vector<torch::Tensor> outputs,
    std::vector<int64_t> output_events)
{
    const size_t count = outputs.size();
    TORCH_CHECK(
        count > 0 &&
        prefixes.size() == count &&
        prefix_events.size() == count &&
        residuals.size() == count &&
        residual_events.size() == count &&
        projections.size() == count &&
        norm_weights.size() == count &&
        post_norm_weights.size() == count &&
        workspaces.size() == count &&
        residual_inverses.size() == count &&
        output_events.size() == count,
        "TPHidden residual vectors must be non-empty and size-equal");
    int original_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&original_device));
    for (size_t rank = 0; rank < count; ++rank) {
        const int target = outputs[rank].get_device();
        TORCH_CHECK(
            prefixes[rank].get_device() == target &&
            residuals[rank].get_device() == target &&
            projections[rank].get_device() == target &&
            norm_weights[rank].get_device() == target &&
            post_norm_weights[rank].get_device() == target &&
            workspaces[rank].get_device() == target &&
            residual_inverses[rank].get_device() == target,
            "TPHidden residual tensors must be colocated");
        C10_CUDA_CHECK(cudaSetDevice(target));
        const auto stream = at::cuda::getCurrentCUDAStream(target);
        C10_CUDA_CHECK(cudaStreamWaitEvent(
            stream,
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(prefix_events[rank])),
            0));
        C10_CUDA_CHECK(cudaStreamWaitEvent(
            stream,
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(residual_events[rank])),
            0));
        attention_residual_bf16(
            prefixes[rank],
            residuals[rank],
            projections[rank],
            norm_weights[rank],
            post_norm_weights[rank],
            eps,
            outputs[rank],
            workspaces[rank],
            single_cta_max_rows,
            residual_inverses[rank]);
        C10_CUDA_CHECK(cudaEventRecord(
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(output_events[rank])),
            stream));
    }
    C10_CUDA_CHECK(cudaSetDevice(original_device));
    return outputs;
}

torch::Tensor glm_mla_bmm_decode(
    torch::Tensor input,
    torch::Tensor weight,
    bool transpose_weight,
    c10::optional<torch::Tensor> output_buffer)
{
    TORCH_CHECK(
        input.is_cuda() && weight.is_cuda(),
        "MLA decode GEMM inputs must be CUDA");
    TORCH_CHECK(
        input.scalar_type() == at::kBFloat16 &&
        weight.scalar_type() == at::kBFloat16,
        "MLA decode GEMM inputs must be BF16");
    TORCH_CHECK(
        input.dim() == 3 &&
        input.size(1) == 1 &&
        input.stride(2) == 1 &&
        weight.dim() == 3 &&
        weight.is_contiguous() &&
        input.size(0) == weight.size(0),
        "MLA decode GEMM expects input[H,1,K] and contiguous weight");
    const int heads = static_cast<int>(input.size(0));
    const int inner = static_cast<int>(input.size(2));
    const int output_width = static_cast<int>(
        transpose_weight ? weight.size(1) : weight.size(2));
    TORCH_CHECK(
        (
            transpose_weight
            ? weight.size(2) == inner
            : weight.size(1) == inner
        ) &&
        input.get_device() == weight.get_device(),
        "MLA decode GEMM shapes/devices do not match");
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty(
            {heads, 1, output_width},
            input.options());
    TORCH_CHECK(
        output.is_cuda() &&
        output.scalar_type() == at::kBFloat16 &&
        output.is_contiguous() &&
        output.sizes() == torch::IntArrayRef(
            {heads, 1, output_width}) &&
        output.get_device() == input.get_device(),
        "MLA decode GEMM output must be contiguous BF16 [H,1,N]");

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasOperation_t weight_op = transpose_weight
        ? CUBLAS_OP_T
        : CUBLAS_OP_N;
    const int lda = transpose_weight
        ? inner
        : output_width;
    TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
        handle,
        weight_op,
        CUBLAS_OP_N,
        output_width,
        1,
        inner,
        &alpha,
        weight.data_ptr<at::BFloat16>(),
        CUDA_R_16BF,
        lda,
        static_cast<long long>(weight.stride(0)),
        input.data_ptr<at::BFloat16>(),
        CUDA_R_16BF,
        inner,
        static_cast<long long>(input.stride(0)),
        &beta,
        output.data_ptr<at::BFloat16>(),
        CUDA_R_16BF,
        output_width,
        static_cast<long long>(output.stride(0)),
        heads,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return output;
}

// ---- RoPE (interleaved pairs, decode T=1 fast path) ----
// rows share one (cos, sin) phase: out[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
//                                  out[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
// inverse=true: conjugate (sin negated).

__global__ void rope1_kernel(
    const float* __restrict__ x,    // [N, rd]
    const float* __restrict__ cs,   // [rd/2]
    const float* __restrict__ sn,   // [rd/2]
    float* __restrict__ out,        // [N, rd]
    const int rd2, const int inverse)
{
    const int r = blockIdx.x;
    const float* xr = x + (long)r * rd2 * 2;
    float* orow = out + (long)r * rd2 * 2;
    for (int i = threadIdx.x; i < rd2; i += blockDim.x) {
        const float c = cs[i], s = inverse ? -sn[i] : sn[i];
        const float x1 = xr[2 * i], x2 = xr[2 * i + 1];
        orow[2 * i] = x1 * c - x2 * s;
        orow[2 * i + 1] = x1 * s + x2 * c;
    }
}

torch::Tensor rope1(torch::Tensor x, torch::Tensor cs, torch::Tensor sn,
                    bool inverse) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat, "x must be CUDA f32");
    auto xc = x.contiguous();
    const int rd = (int)xc.size(-1);
    auto x2 = xc.view({-1, rd});
    const int N = (int)x2.size(0);
    auto out = torch::empty_like(x2);
    auto stream = at::cuda::getCurrentCUDAStream();
    rope1_kernel<<<N, 64, 0, stream>>>(
        x2.data_ptr<float>(), cs.contiguous().data_ptr<float>(),
        sn.contiguous().data_ptr<float>(), out.data_ptr<float>(),
        rd / 2, inverse ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.view(xc.sizes());
}

// ---- GLM MLA RoPE (Q and shared K in one launch, HF cat layout) ----
// The reference first computes two rounded multiplies and then a rounded
// add/sub in separate ATen kernels.  Explicit round-to-nearest intrinsics keep
// this fused implementation from contracting the expression into an FMA.

__global__ void glm_rope_qk_kernel(
    const float* __restrict__ q,     // [H*T, rd], interleaved input
    const float* __restrict__ k,     // [T, rd], interleaved input
    const float* __restrict__ cs,    // [T, rd/2]
    const float* __restrict__ sn,    // [T, rd/2]
    float* __restrict__ qo,          // [H*T, rd], cat output
    float* __restrict__ ko,          // [T, rd], cat output
    const int nq, const int T, const int rd2)
{
    const int row = blockIdx.x;
    const bool is_q = row < nq;
    const int local_row = is_q ? row : row - nq;
    const int phase = is_q ? local_row % T : local_row;
    const float* xr = (is_q ? q : k) + (long)local_row * rd2 * 2;
    float* yr = (is_q ? qo : ko) + (long)local_row * rd2 * 2;
    const float* cr = cs + (long)phase * rd2;
    const float* sr = sn + (long)phase * rd2;
    for (int i = threadIdx.x; i < rd2; i += blockDim.x) {
        const float x1 = xr[2 * i];
        const float x2 = xr[2 * i + 1];
        const float a = __fmul_rn(x1, cr[i]);
        const float b = __fmul_rn(x2, sr[i]);
        const float c = __fmul_rn(x2, cr[i]);
        const float d = __fmul_rn(x1, sr[i]);
        yr[i] = __fsub_rn(a, b);
        yr[i + rd2] = __fadd_rn(c, d);
    }
}

std::vector<torch::Tensor> glm_rope_qk(
    torch::Tensor q, torch::Tensor k, torch::Tensor cs, torch::Tensor sn) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && cs.is_cuda() && sn.is_cuda(),
                "GLM RoPE tensors must be CUDA");
    TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat &&
                cs.scalar_type() == at::kFloat && sn.scalar_type() == at::kFloat,
                "GLM RoPE tensors must be float32");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && cs.dim() == 2 && sn.dim() == 2,
                "GLM RoPE expects q[H,T,D], k[1,T,D], cos/sin[T,D/2]");
    const int T = (int)q.size(1);
    const int rd = (int)q.size(2);
    TORCH_CHECK(k.size(0) == 1 && k.size(1) == T && k.size(2) == rd,
                "GLM RoPE q/k shape mismatch");
    TORCH_CHECK(rd % 2 == 0 && cs.size(0) == T && sn.size(0) == T &&
                cs.size(1) * 2 == rd && sn.size(1) * 2 == rd,
                "GLM RoPE phase shape mismatch");
    auto qc = q.contiguous();
    auto kc = k.contiguous();
    auto cc = cs.contiguous();
    auto sc = sn.contiguous();
    auto qo = torch::empty_like(qc);
    auto ko = torch::empty_like(kc);
    const int nq = (int)(q.size(0) * T);
    auto stream = at::cuda::getCurrentCUDAStream();
    glm_rope_qk_kernel<<<nq + T, 64, 0, stream>>>(
        qc.data_ptr<float>(), kc.data_ptr<float>(),
        cc.data_ptr<float>(), sc.data_ptr<float>(),
        qo.data_ptr<float>(), ko.data_ptr<float>(),
        nq, T, rd / 2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {qo, ko};
}

// GLM decode-only latent KV preparation.  The reference path launches:
// RMSNorm(c_kv), a BF16 KV copy, a strided Q copy, RoPE(Q/K), a BF16 Q
// conversion and a BF16 K copy.  Decode has T=1 and fixed destination rows,
// so two kernels can preserve the same FP32 arithmetic and BF16 boundaries.

__global__ void glm_ckv_rms_write_bf16_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    __nv_bfloat16* __restrict__ output_base,
    const int64_t* __restrict__ position_ptr,
    const int capacity,
    const int width,
    const float eps)
{
    float acc = 0.f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float value = x[i];
        acc += value * value;
    }
    __shared__ float reduction[32];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    if ((threadIdx.x & 31) == 0)
        reduction[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float value = threadIdx.x < (blockDim.x + 31) / 32
            ? reduction[threadIdx.x]
            : 0.f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(
                0xffffffffu, value, offset);
        if (threadIdx.x == 0)
            reduction[0] = value;
    }
    __syncthreads();
    const float scale = rsqrtf(
        reduction[0] / static_cast<float>(width) + eps);
    const int64_t position = position_ptr[0];
    if (position < 0 || position >= capacity)
        return;
    __nv_bfloat16* output =
        output_base + position * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float value = w[i] * (x[i] * scale);
        output[i] = __float2bfloat16_rn(value);
    }
}

__global__ void glm_rope_qk_write_bf16_kernel(
    const float* __restrict__ q,
    const long q_row_stride,
    const float* __restrict__ k,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    __nv_bfloat16* __restrict__ q_output,
    __nv_bfloat16* __restrict__ k_output_base,
    const int64_t* __restrict__ position_ptr,
    const int capacity,
    const int heads,
    const int rope_half)
{
    const int64_t position = position_ptr[0];
    if (position < 0 || position >= capacity)
        return;
    const float* cs =
        cos_cache + position * rope_half;
    const float* sn =
        sin_cache + position * rope_half;
    const int row = blockIdx.x;
    const bool is_q = row < heads;
    const float* input = is_q
        ? q + static_cast<long>(row) * q_row_stride
        : k;
    __nv_bfloat16* output = is_q
        ? q_output + static_cast<long>(row) * rope_half * 2
        : k_output_base + position * rope_half * 2;
    for (
        int index = threadIdx.x;
        index < rope_half;
        index += blockDim.x
    ) {
        const float x1 = input[2 * index];
        const float x2 = input[2 * index + 1];
        const float a = __fmul_rn(x1, cs[index]);
        const float b = __fmul_rn(x2, sn[index]);
        const float c = __fmul_rn(x2, cs[index]);
        const float d = __fmul_rn(x1, sn[index]);
        output[index] = __float2bfloat16_rn(
            __fsub_rn(a, b));
        output[index + rope_half] = __float2bfloat16_rn(
            __fadd_rn(c, d));
    }
}

torch::Tensor glm_latent_kv_decode_prepare(
    torch::Tensor c_raw,
    torch::Tensor c_weight,
    torch::Tensor q_rot,
    torch::Tensor k_rot,
    torch::Tensor cos_cache,
    torch::Tensor sin_cache,
    torch::Tensor ckv_buffer,
    torch::Tensor krot_buffer,
    torch::Tensor position,
    double eps,
    c10::optional<torch::Tensor> q_output_buffer)
{
    TORCH_CHECK(
        c_raw.is_cuda() && c_weight.is_cuda() &&
        q_rot.is_cuda() && k_rot.is_cuda() &&
        cos_cache.is_cuda() && sin_cache.is_cuda() &&
        ckv_buffer.is_cuda() && krot_buffer.is_cuda() &&
        position.is_cuda(),
        "GLM latent decode tensors must be CUDA");
    TORCH_CHECK(
        c_raw.scalar_type() == at::kFloat &&
        c_weight.scalar_type() == at::kFloat &&
        q_rot.scalar_type() == at::kFloat &&
        k_rot.scalar_type() == at::kFloat &&
        cos_cache.scalar_type() == at::kFloat &&
        sin_cache.scalar_type() == at::kFloat,
        "GLM latent decode inputs must be float32");
    TORCH_CHECK(
        ckv_buffer.scalar_type() == at::kBFloat16 &&
        krot_buffer.scalar_type() == at::kBFloat16 &&
        position.scalar_type() == at::kLong &&
        position.numel() == 1 &&
        position.is_contiguous(),
        "GLM latent KV buffers/position have invalid dtypes");
    TORCH_CHECK(
        c_raw.dim() == 2 && c_raw.size(0) == 1 &&
        c_weight.dim() == 1 &&
        c_raw.size(1) == c_weight.size(0),
        "GLM latent C shapes do not match");
    TORCH_CHECK(
        q_rot.dim() == 3 && q_rot.size(1) == 1 &&
        k_rot.dim() == 3 && k_rot.size(0) == 1 &&
        k_rot.size(1) == 1 &&
        q_rot.size(2) == k_rot.size(2) &&
        q_rot.size(2) % 2 == 0,
        "GLM decode RoPE expects Q[H,1,D] and K[1,1,D]");
    const int latent = static_cast<int>(c_raw.size(1));
    const int heads = static_cast<int>(q_rot.size(0));
    const int rope = static_cast<int>(q_rot.size(2));
    TORCH_CHECK(
        ckv_buffer.dim() == 2 &&
        ckv_buffer.size(1) == latent &&
        krot_buffer.dim() == 2 &&
        krot_buffer.size(1) == rope &&
        ckv_buffer.size(0) == krot_buffer.size(0),
        "GLM latent KV destination shape mismatch");
    TORCH_CHECK(
        cos_cache.dim() == 2 && sin_cache.dim() == 2 &&
        cos_cache.sizes() == sin_cache.sizes() &&
        cos_cache.size(1) * 2 == rope &&
        cos_cache.size(0) > 0,
        "GLM RoPE cache shape/position mismatch");
    const int device = c_raw.get_device();
    TORCH_CHECK(
        c_weight.get_device() == device &&
        q_rot.get_device() == device &&
        k_rot.get_device() == device &&
        cos_cache.get_device() == device &&
        sin_cache.get_device() == device &&
        ckv_buffer.get_device() == device &&
        krot_buffer.get_device() == device &&
        position.get_device() == device,
        "GLM latent decode tensors must share one device");

    auto q_output = q_output_buffer.has_value()
        ? q_output_buffer.value()
        : torch::empty(
            q_rot.sizes(),
            q_rot.options().dtype(at::kBFloat16));
    TORCH_CHECK(
        q_output.is_cuda() &&
        q_output.scalar_type() == at::kBFloat16 &&
        q_output.is_contiguous() &&
        q_output.sizes() == q_rot.sizes() &&
        q_output.get_device() == device,
        "GLM latent Q output must be contiguous BF16 and match Q shape");
    auto stream = at::cuda::getCurrentCUDAStream();
    glm_ckv_rms_write_bf16_kernel<<<1, 256, 0, stream>>>(
        c_raw.data_ptr<float>(),
        c_weight.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(
            ckv_buffer.data_ptr<at::BFloat16>()),
        position.data_ptr<int64_t>(),
        static_cast<int>(std::min(
            ckv_buffer.size(0),
            cos_cache.size(0))),
        latent,
        static_cast<float>(eps));
    glm_rope_qk_write_bf16_kernel<<<
        heads + 1,
        64,
        0,
        stream>>>(
            q_rot.data_ptr<float>(),
            q_rot.stride(0),
            k_rot.data_ptr<float>(),
            cos_cache.data_ptr<float>(),
            sin_cache.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(
                q_output.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                krot_buffer.data_ptr<at::BFloat16>()),
            position.data_ptr<int64_t>(),
            static_cast<int>(std::min(
                krot_buffer.size(0),
                cos_cache.size(0))),
            heads,
            rope / 2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return q_output;
}

// Merge the two latent-MLA score components and their scale in one launch.
// Inputs are BF16 GEMM outputs; the explicit operations mirror
// a.float()/scale + b.float()/scale without changing either GEMM.
__global__ void glm_merge_scores_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    float* __restrict__ out,
    const long n, const float scale)
{
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (long)blockDim.x * gridDim.x) {
        const float av = __fdiv_rn(__bfloat162float(a[i]), scale);
        const float bv = __fdiv_rn(__bfloat162float(b[i]), scale);
        out[i] = __fadd_rn(av, bv);
    }
}

torch::Tensor glm_merge_scores(
    torch::Tensor a, torch::Tensor b, double scale) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "GLM scores must be CUDA");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 &&
                b.scalar_type() == at::kBFloat16,
                "GLM score merge currently requires BF16");
    TORCH_CHECK(a.sizes() == b.sizes(), "GLM score shapes must match");
    auto ac = a.contiguous();
    auto bc = b.contiguous();
    auto out = torch::empty(ac.sizes(), ac.options().dtype(at::kFloat));
    const long n = ac.numel();
    const int blocks = (int)std::min<long>((n + 255) / 256, 4096);
    auto stream = at::cuda::getCurrentCUDAStream();
    glm_merge_scores_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            ac.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            bc.data_ptr<at::BFloat16>()),
        out.data_ptr<float>(), n, (float)scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ---- DSV4 decode attention core (B=1, T=1, f32) ----
// Window and compressed KV are separate contiguous views. Scores are kept in
// shared memory so score matvec, sink-softmax and value reduction need one
// launch instead of a chain of tiny ATen kernels.

__device__ __forceinline__ float warp_max_f32(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ float warp_sum_f32(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// ---- Latent MLA decode core (BF16, dynamic device-side length) ----
//
// Q-B, latent preparation, this core, Wuv and O-projection are captured in
// one rank-local CUDA Graph. Reading the device position inside the kernel
// keeps the graph valid for every context length without a host-side plan.

__global__ void latent_mla_attention_scores_kernel(
    const __nv_bfloat16* __restrict__ qa,
    const __nv_bfloat16* __restrict__ qrot,
    const __nv_bfloat16* __restrict__ ckv,
    const __nv_bfloat16* __restrict__ krot,
    const int64_t* __restrict__ position,
    float* __restrict__ scores,
    const int heads,
    const int latent,
    const int rope,
    const int capacity,
    const float scale,
    const bool scores_only)
{
    const int head = blockIdx.x;
    if (head >= heads) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = blockDim.x >> 5;
    const int length = min(
        max(static_cast<int>(position[0]) + 1, 1),
        capacity);
    __shared__ float reduced[8];
    __shared__ float score_max;
    __shared__ float denominator;

    const __nv_bfloat16* qah =
        qa + static_cast<long>(head) * latent;
    const __nv_bfloat16* qrh =
        qrot + static_cast<long>(head) * rope;
    float local_max = -INFINITY;
    for (int token = tid; token < length; token += blockDim.x) {
        const __nv_bfloat16* ck =
            ckv + static_cast<long>(token) * latent;
        const __nv_bfloat16* kr =
            krot + static_cast<long>(token) * rope;
        float nope_score = 0.0f;
        float rope_score = 0.0f;
        for (int dim = 0; dim < latent; ++dim) {
            nope_score = fmaf(
                __bfloat162float(qah[dim]),
                __bfloat162float(ck[dim]),
                nope_score);
        }
        for (int dim = 0; dim < rope; ++dim) {
            rope_score = fmaf(
                __bfloat162float(qrh[dim]),
                __bfloat162float(kr[dim]),
                rope_score);
        }
        // Match eager decode exactly: both GEMMs produce BF16, their sum and
        // scalar multiply are rounded to BF16, and only then softmax promotes
        // the score tensor to FP32.
        nope_score = __bfloat162float(
            __float2bfloat16_rn(nope_score));
        rope_score = __bfloat162float(
            __float2bfloat16_rn(rope_score));
        const float merged = __bfloat162float(
            __float2bfloat16_rn(nope_score + rope_score));
        const float score = __bfloat162float(
            __float2bfloat16_rn(merged / scale));
        scores[static_cast<long>(head) * capacity + token] = score;
        local_max = fmaxf(local_max, score);
    }
    if (scores_only)
        return;
    local_max = warp_max_f32(local_max);
    if (lane == 0) reduced[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float value = lane < nwarps ? reduced[lane] : -INFINITY;
        value = warp_max_f32(value);
        if (lane == 0) score_max = value;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int token = tid; token < length; token += blockDim.x) {
        float value = expf(
            scores[static_cast<long>(head) * capacity + token]
            - score_max);
        scores[static_cast<long>(head) * capacity + token] = value;
        local_sum += value;
    }
    local_sum = warp_sum_f32(local_sum);
    if (lane == 0) reduced[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float value = lane < nwarps ? reduced[lane] : 0.0f;
        value = warp_sum_f32(value);
        if (lane == 0) denominator = value;
    }
    __syncthreads();
    for (int token = tid; token < length; token += blockDim.x) {
        scores[static_cast<long>(head) * capacity + token] =
            __bfloat162float(
                __float2bfloat16_rn(
                    scores[
                        static_cast<long>(head) * capacity + token
                    ] / denominator));
    }
}

__global__ void latent_mla_attention_value_kernel(
    const float* __restrict__ scores,
    const __nv_bfloat16* __restrict__ ckv,
    const int64_t* __restrict__ position,
    __nv_bfloat16* __restrict__ output,
    const int heads,
    const int latent,
    const int capacity)
{
    const int head = blockIdx.x;
    const int dim = blockIdx.y * blockDim.x + threadIdx.x;
    if (head >= heads || dim >= latent) return;
    const int length = min(
        max(static_cast<int>(position[0]) + 1, 1),
        capacity);
    const float* weights =
        scores + static_cast<long>(head) * capacity;
    float value = 0.0f;
    for (int token = 0; token < length; ++token) {
        value = fmaf(
            weights[token],
            __bfloat162float(
                ckv[static_cast<long>(token) * latent + dim]),
            value);
    }
    output[static_cast<long>(head) * latent + dim] =
        __float2bfloat16_rn(value);
}

torch::Tensor latent_mla_attention_decode(
    torch::Tensor qa,
    torch::Tensor qrot,
    torch::Tensor ckv,
    torch::Tensor krot,
    torch::Tensor position,
    double scale,
    torch::Tensor score_workspace,
    c10::optional<torch::Tensor> output_buffer)
{
    TORCH_CHECK(
        qa.is_cuda() && qrot.is_cuda() && ckv.is_cuda() &&
        krot.is_cuda() && position.is_cuda() &&
        score_workspace.is_cuda(),
        "latent MLA tensors must be CUDA");
    TORCH_CHECK(
        qa.scalar_type() == at::kBFloat16 &&
        qrot.scalar_type() == at::kBFloat16 &&
        ckv.scalar_type() == at::kBFloat16 &&
        krot.scalar_type() == at::kBFloat16 &&
        position.scalar_type() == at::kLong &&
        score_workspace.scalar_type() == at::kFloat,
        "latent MLA requires BF16 state, int64 position and FP32 scores");
    TORCH_CHECK(
        qa.is_contiguous() && qrot.is_contiguous() &&
        ckv.is_contiguous() && krot.is_contiguous() &&
        position.is_contiguous() && score_workspace.is_contiguous(),
        "latent MLA tensors must be contiguous");
    TORCH_CHECK(
        qa.dim() == 3 && qa.size(1) == 1 &&
        qrot.dim() == 3 && qrot.size(1) == 1 &&
        qa.size(0) == qrot.size(0) &&
        ckv.dim() == 2 && krot.dim() == 2 &&
        ckv.size(0) == krot.size(0) &&
        ckv.size(1) == qa.size(2) &&
        krot.size(1) == qrot.size(2) &&
        score_workspace.sizes() == torch::IntArrayRef(
            {qa.size(0), ckv.size(0)}) &&
        position.numel() == 1 && scale > 0.0,
        "latent MLA shapes do not match");
    const auto device = qa.get_device();
    TORCH_CHECK(
        qrot.get_device() == device &&
        ckv.get_device() == device &&
        krot.get_device() == device &&
        position.get_device() == device &&
        score_workspace.get_device() == device,
        "latent MLA tensors must share one device");
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty_like(qa);
    TORCH_CHECK(
        output.is_cuda() &&
        output.scalar_type() == at::kBFloat16 &&
        output.is_contiguous() &&
        output.sizes() == qa.sizes() &&
        output.get_device() == device,
        "latent MLA output must be contiguous BF16 and match Q-A");
    const int heads = static_cast<int>(qa.size(0));
    const int latent = static_cast<int>(qa.size(2));
    const int rope = static_cast<int>(qrot.size(2));
    const int capacity = static_cast<int>(ckv.size(0));
    auto stream = at::cuda::getCurrentCUDAStream();
    latent_mla_attention_scores_kernel<<<heads, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            qa.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            qrot.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            ckv.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            krot.data_ptr<at::BFloat16>()),
        position.data_ptr<int64_t>(),
        score_workspace.data_ptr<float>(),
        heads,
        latent,
        rope,
        capacity,
        static_cast<float>(scale),
        false);
    const dim3 value_grid(
        heads,
        (latent + 255) / 256);
    latent_mla_attention_value_kernel<<<
        value_grid,
        256,
        0,
        stream>>>(
            score_workspace.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(
                ckv.data_ptr<at::BFloat16>()),
            position.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            heads,
            latent,
            capacity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor latent_mla_attention_scores(
    torch::Tensor qa,
    torch::Tensor qrot,
    torch::Tensor ckv,
    torch::Tensor krot,
    torch::Tensor position,
    double scale,
    torch::Tensor score_workspace)
{
    TORCH_CHECK(
        qa.is_cuda() && qrot.is_cuda() && ckv.is_cuda() &&
        krot.is_cuda() && position.is_cuda() &&
        score_workspace.is_cuda() &&
        qa.scalar_type() == at::kBFloat16 &&
        qrot.scalar_type() == at::kBFloat16 &&
        ckv.scalar_type() == at::kBFloat16 &&
        krot.scalar_type() == at::kBFloat16 &&
        position.scalar_type() == at::kLong &&
        score_workspace.scalar_type() == at::kFloat,
        "latent MLA score tensors have invalid device or dtype");
    TORCH_CHECK(
        qa.is_contiguous() && qrot.is_contiguous() &&
        ckv.is_contiguous() && krot.is_contiguous() &&
        position.is_contiguous() && score_workspace.is_contiguous() &&
        qa.dim() == 3 && qa.size(1) == 1 &&
        qrot.dim() == 3 && qrot.size(1) == 1 &&
        qa.size(0) == qrot.size(0) &&
        ckv.dim() == 2 && krot.dim() == 2 &&
        ckv.size(0) == krot.size(0) &&
        ckv.size(1) == qa.size(2) &&
        krot.size(1) == qrot.size(2) &&
        score_workspace.sizes() == torch::IntArrayRef(
            {qa.size(0), ckv.size(0)}) &&
        position.numel() == 1 && scale > 0.0,
        "latent MLA score shapes do not match");
    const int device = qa.get_device();
    TORCH_CHECK(
        qrot.get_device() == device &&
        ckv.get_device() == device &&
        krot.get_device() == device &&
        position.get_device() == device &&
        score_workspace.get_device() == device,
        "latent MLA score tensors must share one device");
    auto stream = at::cuda::getCurrentCUDAStream();
    latent_mla_attention_scores_kernel<<<
        qa.size(0),
        256,
        0,
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                qa.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                qrot.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                ckv.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                krot.data_ptr<at::BFloat16>()),
            position.data_ptr<int64_t>(),
            score_workspace.data_ptr<float>(),
            static_cast<int>(qa.size(0)),
            static_cast<int>(qa.size(2)),
            static_cast<int>(qrot.size(2)),
            static_cast<int>(ckv.size(0)),
            static_cast<float>(scale),
            true);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return score_workspace;
}

__global__ void dsv4_attn_decode_kernel(
    const float* __restrict__ q,          // [H,D]
    const float* __restrict__ win_kv,     // [W,D]
    const int64_t* __restrict__ win_pos,  // [W], negative means invalid
    const float* __restrict__ comp_kv,    // [C,D]
    const float* __restrict__ sink,       // [H]
    const float* __restrict__ cs,         // [rd/2]
    const float* __restrict__ sn,         // [rd/2]
    float* __restrict__ out,              // [H,D]
    const int H, const int D, const int W, const int C, const int rd,
    const float scale)
{
    const int h = blockIdx.x;
    if (h >= H) return;
    const int tid = threadIdx.x;
    const int S = W + C;
    extern __shared__ float smem[];
    float* qsh = smem;             // D
    float* scores = qsh + D;       // S
    float* osh = scores + S;       // D
    float* red = osh + D;          // one value per warp
    __shared__ float score_max;
    __shared__ float denom;

    const float* qh = q + (long)h * D;
    for (int d = tid; d < D; d += blockDim.x)
        qsh[d] = qh[d];
    __syncthreads();

    for (int s = tid; s < S; s += blockDim.x) {
        const bool valid = s >= W || win_pos[s] >= 0;
        const float* kv = s < W ? win_kv + (long)s * D
                                : comp_kv + (long)(s - W) * D;
        float acc = 0.f;
        if (valid) {
            for (int d = 0; d < D; ++d)
                acc = fmaf(qsh[d], kv[d], acc);
        }
        scores[s] = valid ? acc * scale : -INFINITY;
    }
    __syncthreads();

    float mx = -INFINITY;
    for (int s = tid; s < S; s += blockDim.x)
        mx = fmaxf(mx, scores[s]);
    mx = warp_max_f32(mx);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    if (warp == 0) {
        float v = lane < nwarps ? red[lane] : -INFINITY;
        v = warp_max_f32(v);
        if (lane == 0) score_max = v;
    }
    __syncthreads();

    float z = 0.f;
    for (int s = tid; s < S; s += blockDim.x) {
        const float e = expf(scores[s] - score_max);
        scores[s] = e;
        z += e;
    }
    z = warp_sum_f32(z);
    if (lane == 0) red[warp] = z;
    __syncthreads();
    if (warp == 0) {
        float v = lane < nwarps ? red[lane] : 0.f;
        v = warp_sum_f32(v);
        if (lane == 0)
            denom = v + expf(sink[h] - score_max);
    }
    __syncthreads();

    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < S; ++s) {
            const float* kv = s < W ? win_kv + (long)s * D
                                    : comp_kv + (long)(s - W) * D;
            acc = fmaf(scores[s], kv[d], acc);
        }
        osh[d] = acc / denom;
    }
    __syncthreads();

    const int plain = D - rd;
    for (int d = tid; d < D; d += blockDim.x) {
        float v;
        if (d < plain) {
            v = osh[d];
        } else {
            const int r = d - plain;
            const int pair = r >> 1;
            const float x0 = osh[plain + 2 * pair];
            const float x1 = osh[plain + 2 * pair + 1];
            v = (r & 1) ? (-x0 * sn[pair] + x1 * cs[pair])
                        : ( x0 * cs[pair] + x1 * sn[pair]);
        }
        out[(long)h * D + d] = v;
    }
}

torch::Tensor dsv4_attn_decode(
    torch::Tensor q, torch::Tensor win_kv, torch::Tensor win_pos,
    torch::Tensor comp_kv, torch::Tensor sink, torch::Tensor cs,
    torch::Tensor sn, double scale) {
    TORCH_CHECK(q.is_cuda() && win_kv.is_cuda() && win_pos.is_cuda()
                && comp_kv.is_cuda() && sink.is_cuda() && cs.is_cuda() && sn.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(q.scalar_type() == at::kFloat && win_kv.scalar_type() == at::kFloat
                && comp_kv.scalar_type() == at::kFloat && sink.scalar_type() == at::kFloat
                && cs.scalar_type() == at::kFloat && sn.scalar_type() == at::kFloat,
                "attention tensors must be float32");
    TORCH_CHECK(win_pos.scalar_type() == at::kLong, "win_pos must be int64");
    TORCH_CHECK(q.dim() == 3 && q.size(0) == 1, "q must be [1,H,D]");
    TORCH_CHECK(win_kv.dim() == 3 && win_kv.size(0) == 1, "win_kv must be [1,W,D]");
    TORCH_CHECK(comp_kv.dim() == 3 && comp_kv.size(0) == 1, "comp_kv must be [1,C,D]");
    const int H = (int)q.size(1), D = (int)q.size(2);
    const int W = (int)win_kv.size(1), C = (int)comp_kv.size(1);
    const int rd = (int)cs.numel() * 2;
    TORCH_CHECK(win_kv.size(2) == D && comp_kv.size(2) == D, "KV head dim mismatch");
    TORCH_CHECK(win_pos.numel() == W, "win_pos size mismatch");
    TORCH_CHECK(sink.numel() == H, "sink size mismatch");
    TORCH_CHECK(sn.numel() * 2 == rd && rd <= D, "RoPE size mismatch");
    TORCH_CHECK(W + C > 0 && W + C <= 4096, "fused sequence length out of range");

    auto qc = q.contiguous();
    auto wc = win_kv.contiguous();
    auto pc = win_pos.contiguous();
    auto cc = comp_kv.contiguous();
    auto sk = sink.contiguous();
    auto csc = cs.contiguous();
    auto snc = sn.contiguous();
    auto out = torch::empty_like(qc);
    const int threads = 128;
    const size_t smem = (size_t)(2 * D + W + C + 4) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    dsv4_attn_decode_kernel<<<H, threads, smem, stream>>>(
        qc.data_ptr<float>(), wc.data_ptr<float>(), pc.data_ptr<int64_t>(),
        cc.data_ptr<float>(), sk.data_ptr<float>(), csc.data_ptr<float>(),
        snc.data_ptr<float>(), out.data_ptr<float>(),
        H, D, W, C, rd, (float)scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

template <typename scalar_t>
__device__ __forceinline__ float fp8_cache_value(
    const scalar_t* source,
    const int index)
{
    return static_cast<float>(source[index]);
}

template <>
__device__ __forceinline__ float fp8_cache_value<__nv_bfloat16>(
    const __nv_bfloat16* source,
    const int index)
{
    return __bfloat162float(source[index]);
}

template <typename scalar_t>
__device__ __forceinline__ void store_model1_fp8_cache(
    const scalar_t* source,
    uint8_t* cache,
    const int page_items,
    const int item,
    float* shared_scales)
{
    constexpr int nope_width = 448;
    constexpr int rope_width = 64;
    constexpr int payload_bytes = 576;
    constexpr int scale_stride = 8;
    constexpr int storage_bytes = payload_bytes + scale_stride;
    const int page = item / page_items;
    const int slot = item % page_items;
    uint8_t* page_base = cache +
        static_cast<long>(page) * page_items * storage_bytes;
    uint8_t* payload = page_base + static_cast<long>(slot) * payload_bytes;
    uint8_t* scales = page_base +
        static_cast<long>(page_items) * payload_bytes +
        static_cast<long>(slot) * scale_stride;
    if (threadIdx.x < 7) {
        const int tile = threadIdx.x;
        float maximum = 0.0f;
        for (int offset = 0; offset < 64; ++offset)
            maximum = fmaxf(
                maximum,
                fabsf(fp8_cache_value(source, tile * 64 + offset)));
        float scale = fmaxf(maximum / 448.0f, 1.0e-4f);
        scale = exp2f(ceilf(log2f(scale)));
        shared_scales[tile] = scale;
        scales[tile] = static_cast<uint8_t>(
            (__float_as_uint(scale) >> 23) & 0xffu);
    }
    if (threadIdx.x == 7)
        scales[7] = 0;
    __syncthreads();
    for (int index = threadIdx.x; index < nope_width; index += blockDim.x) {
        const float scale = shared_scales[index / 64];
        __nv_fp8_e4m3 quantized(fp8_cache_value(source, index) / scale);
        payload[index] = quantized.__x;
    }
    auto* rope = reinterpret_cast<__nv_bfloat16*>(payload + nope_width);
    for (int index = threadIdx.x; index < rope_width; index += blockDim.x)
        rope[index] = __float2bfloat16_rn(
            fp8_cache_value(source, nope_width + index));
    __syncthreads();
}

__global__ void dsv4_kv_commit_controlled_kernel(
    const float* __restrict__ kv,
    float* __restrict__ window,
    int64_t* __restrict__ window_positions,
    const int64_t* __restrict__ control,
    uint8_t* __restrict__ fp8_cache,
    const int width,
    const int window_size)
{
    __shared__ float fp8_scales[7];
    const int64_t position = control[1];
    const int slot = static_cast<int>(position % window_size);
    for (int index = threadIdx.x; index < width; index += blockDim.x)
        window[static_cast<long>(slot) * width + index] = kv[index];
    if (threadIdx.x == 0)
        window_positions[slot] = position;
    if (fp8_cache != nullptr && width == 512)
        store_model1_fp8_cache(
            kv, fp8_cache, window_size, slot, fp8_scales);
}

void dsv4_kv_commit_controlled(
    torch::Tensor kv,
    torch::Tensor window,
    torch::Tensor window_positions,
    torch::Tensor control,
    c10::optional<torch::Tensor> fp8_cache)
{
    TORCH_CHECK(
        kv.is_cuda() && window.is_cuda() &&
        window_positions.is_cuda() && control.is_cuda() &&
        kv.scalar_type() == at::kFloat &&
        window.scalar_type() == at::kFloat &&
        window_positions.scalar_type() == at::kLong &&
        control.scalar_type() == at::kLong &&
        kv.is_contiguous() && window.is_contiguous() &&
        window_positions.is_contiguous() && control.is_contiguous() &&
        kv.numel() == window.size(-1) &&
        window.dim() == 3 && window.size(0) == 1 &&
        window_positions.numel() == window.size(1) &&
        control.numel() >= 2,
        "controlled DSV4 KV commit tensors are inconsistent");
    uint8_t* fp8_pointer = nullptr;
    if (fp8_cache.has_value()) {
        auto cache = fp8_cache.value();
        TORCH_CHECK(
            cache.is_cuda() && cache.is_contiguous() &&
            cache.element_size() == 1 && cache.dim() == 4 &&
            cache.size(0) == 1 && cache.size(1) == window.size(1) &&
            cache.size(2) == 1 && cache.size(3) == 584,
            "controlled window FP8 cache layout is inconsistent");
        fp8_pointer = static_cast<uint8_t*>(cache.data_ptr());
    }
    dsv4_kv_commit_controlled_kernel<<<
        1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            kv.data_ptr<float>(),
            window.data_ptr<float>(),
            window_positions.data_ptr<int64_t>(),
            control.data_ptr<int64_t>(),
            fp8_pointer,
            static_cast<int>(window.size(2)),
            static_cast<int>(window.size(1)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename ape_t>
__global__ void dsv4_compressor_step_controlled_kernel(
    const __nv_bfloat16* __restrict__ projected,
    const ape_t* __restrict__ ape,
    __nv_bfloat16* __restrict__ ckv,
    float* __restrict__ cscore,
    const __nv_bfloat16* __restrict__ norm,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    const int64_t* __restrict__ page_ptrs,
    const int64_t* __restrict__ control,
    uint8_t* __restrict__ model1_cache,
    uint8_t* __restrict__ indexer_cache,
    float* __restrict__ indexer_scales,
    const int model1_page_items,
    const int ratio,
    const int kv_rows,
    const int score_rows,
    const int width,
    const int rope_width,
    const int page_items,
    const bool overlap,
    const bool hadamard,
    const float eps)
{
    extern __shared__ float shared[];
    float* pooled = shared;
    float* reduction = pooled + width;
    const int tid = threadIdx.x;
    const int64_t position = control[1];
    const int phase = static_cast<int>(position % ratio);
    const int state_width = overlap ? 2 * width : width;
    const int write_slot = phase + (overlap ? ratio : 0);
    for (int index = tid; index < kv_rows; index += blockDim.x)
        ckv[static_cast<long>(write_slot) * state_width + index] =
            projected[index];
    for (int index = tid; index < score_rows; index += blockDim.x) {
        float bias;
        if constexpr (std::is_same_v<ape_t, float>)
            bias = ape[static_cast<long>(phase) * score_rows + index];
        else
            bias = __bfloat162float(
                ape[static_cast<long>(phase) * score_rows + index]);
        cscore[static_cast<long>(write_slot) * state_width + index] =
            __bfloat162float(projected[kv_rows + index]) + bias;
    }
    __syncthreads();
    if ((position + 1) % ratio != 0)
        return;

    for (int dimension = tid; dimension < width; dimension += blockDim.x) {
        const int samples = overlap ? 2 * ratio : ratio;
        float maximum = -INFINITY;
        for (int sample = 0; sample < samples; ++sample) {
            const int slot = sample < ratio ? sample : ratio + sample - ratio;
            const int half = overlap && sample >= ratio ? width : 0;
            const int score_index = min(dimension, score_rows - 1);
            maximum = fmaxf(
                maximum,
                cscore[static_cast<long>(slot) * state_width +
                       half + score_index]);
        }
        float denominator = 0.0f;
        float value = 0.0f;
        for (int sample = 0; sample < samples; ++sample) {
            const int slot = sample < ratio ? sample : ratio + sample - ratio;
            const int half = overlap && sample >= ratio ? width : 0;
            const int score_index = min(dimension, score_rows - 1);
            const float weight = expf(
                cscore[static_cast<long>(slot) * state_width +
                       half + score_index] - maximum);
            denominator += weight;
            value += weight * __bfloat162float(
                ckv[static_cast<long>(slot) * state_width + half + dimension]);
        }
        pooled[dimension] = value / denominator;
    }
    __syncthreads();

    if (overlap) {
        for (
            int index = tid;
            index < ratio * state_width;
            index += blockDim.x
        ) {
            ckv[index] = ckv[static_cast<long>(ratio) * state_width + index];
            cscore[index] =
                cscore[static_cast<long>(ratio) * state_width + index];
        }
    }

    float square = 0.0f;
    for (int index = tid; index < width; index += blockDim.x)
        square = fmaf(pooled[index], pooled[index], square);
    reduction[tid] = square;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            reduction[tid] += reduction[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(reduction[0] / width + eps);
    for (int index = tid; index < width; index += blockDim.x)
        pooled[index] *= inv * __bfloat162float(norm[index]);
    __syncthreads();

    const int rope_start = width - rope_width;
    for (int pair = tid; pair < rope_width / 2; pair += blockDim.x) {
        const int index = rope_start + 2 * pair;
        const float first = pooled[index];
        const float second = pooled[index + 1];
        const float c = rope_cos[pair];
        const float s = rope_sin[pair];
        pooled[index] = first * c - second * s;
        pooled[index + 1] = first * s + second * c;
    }
    __syncthreads();

    if (hadamard) {
        for (int block = 1; block < width; block <<= 1) {
            for (int index = tid; index < width / 2; index += blockDim.x) {
                const int group = index / block;
                const int offset = index % block;
                const int left = group * 2 * block + offset;
                const int right = left + block;
                const float a = pooled[left];
                const float b = pooled[right];
                pooled[left] = a + b;
                pooled[right] = a - b;
            }
            __syncthreads();
        }
    }

    const int64_t item = position / ratio;
    const int page = static_cast<int>(item / page_items);
    const int offset = static_cast<int>(item % page_items);
    auto* destination = reinterpret_cast<__nv_bfloat16*>(
        static_cast<uintptr_t>(page_ptrs[page]));
    const float output_scale = hadamard ? rsqrtf(static_cast<float>(width)) : 1.0f;
    for (int index = tid; index < width; index += blockDim.x)
        destination[static_cast<long>(offset) * width + index] =
            __float2bfloat16_rn(pooled[index] * output_scale);
    __syncthreads();
    if (model1_cache != nullptr && width == 512 && !hadamard) {
        store_model1_fp8_cache(
            pooled,
            model1_cache,
            model1_page_items,
            static_cast<int>(item),
            reduction);
    }
    if (
        indexer_cache != nullptr && indexer_scales != nullptr &&
        width == 128 && hadamard
    ) {
        if (tid == 0) {
            float maximum = 0.0f;
            for (int index = 0; index < width; ++index)
                maximum = fmaxf(
                    maximum,
                    fabsf(pooled[index] * output_scale));
            reduction[0] = fmaxf(maximum / 448.0f, 1.0e-8f);
            indexer_scales[item] = reduction[0];
        }
        __syncthreads();
        const float inverse_scale = 1.0f / reduction[0];
        for (int index = tid; index < width; index += blockDim.x) {
            __nv_fp8_e4m3 quantized(
                pooled[index] * output_scale * inverse_scale);
            indexer_cache[static_cast<long>(item) * width + index] =
                quantized.__x;
        }
    }
}

void dsv4_compressor_step_controlled(
    torch::Tensor projected,
    torch::Tensor ape,
    torch::Tensor ckv,
    torch::Tensor cscore,
    torch::Tensor norm,
    torch::Tensor rope_cos,
    torch::Tensor rope_sin,
    torch::Tensor page_ptrs,
    torch::Tensor control,
    c10::optional<torch::Tensor> model1_cache,
    c10::optional<torch::Tensor> indexer_cache,
    c10::optional<torch::Tensor> indexer_scales,
    int64_t ratio,
    int64_t kv_rows,
    int64_t width,
    int64_t rope_width,
    int64_t page_items,
    bool overlap,
    bool hadamard,
    double eps)
{
    TORCH_CHECK(
        projected.is_cuda() && ape.is_cuda() && ckv.is_cuda() &&
        cscore.is_cuda() && norm.is_cuda() && rope_cos.is_cuda() &&
        rope_sin.is_cuda() && page_ptrs.is_cuda() && control.is_cuda() &&
        projected.scalar_type() == at::kBFloat16 &&
        ckv.scalar_type() == at::kBFloat16 &&
        cscore.scalar_type() == at::kFloat &&
        norm.scalar_type() == at::kBFloat16 &&
        rope_cos.scalar_type() == at::kFloat &&
        rope_sin.scalar_type() == at::kFloat &&
        page_ptrs.scalar_type() == at::kLong &&
        control.scalar_type() == at::kLong &&
        (ape.scalar_type() == at::kFloat || ape.scalar_type() == at::kBFloat16),
        "controlled compressor dtypes/devices are unsupported");
    TORCH_CHECK(
        ratio > 0 && width > 0 && width <= 1024 &&
        rope_width > 0 && rope_width <= width && rope_width % 2 == 0 &&
        kv_rows >= width && kv_rows <= 2 * width &&
        projected.numel() > kv_rows &&
        ckv.dim() == 3 && cscore.sizes() == ckv.sizes() &&
        ckv.size(0) == 1 && ckv.size(1) == ratio * (overlap ? 2 : 1) &&
        ckv.size(2) == width * (overlap ? 2 : 1) &&
        norm.numel() == width && rope_cos.numel() * 2 == rope_width &&
        rope_sin.numel() == rope_cos.numel() &&
        page_items > 0 && page_ptrs.numel() > 0 && control.numel() >= 2,
        "controlled compressor shapes are inconsistent");
    uint8_t* model1_pointer = nullptr;
    int model1_page_items = 0;
    uint8_t* indexer_pointer = nullptr;
    float* indexer_scale_pointer = nullptr;
    if (model1_cache.has_value()) {
        auto cache = model1_cache.value();
        TORCH_CHECK(
            cache.is_cuda() && cache.is_contiguous() &&
            cache.element_size() == 1 && cache.dim() == 4 &&
            cache.size(1) > 0 && cache.size(2) == 1 &&
            cache.size(3) == 584 && width == 512 && !hadamard,
            "controlled compressor Model1 cache layout is inconsistent");
        model1_pointer = static_cast<uint8_t*>(cache.data_ptr());
        model1_page_items = static_cast<int>(cache.size(1));
    }
    if (indexer_cache.has_value() || indexer_scales.has_value()) {
        TORCH_CHECK(
            indexer_cache.has_value() && indexer_scales.has_value(),
            "Indexer FP8 values/scales must be supplied together");
        auto cache = indexer_cache.value();
        auto scales = indexer_scales.value();
        TORCH_CHECK(
            cache.is_cuda() && scales.is_cuda() && cache.is_contiguous() &&
            scales.is_contiguous() && cache.element_size() == 1 &&
            scales.scalar_type() == at::kFloat && cache.dim() == 2 &&
            cache.size(1) == width && scales.numel() == cache.size(0) &&
            width == 128 && hadamard,
            "controlled compressor Indexer FP8 cache layout is inconsistent");
        indexer_pointer = static_cast<uint8_t*>(cache.data_ptr());
        indexer_scale_pointer = scales.data_ptr<float>();
    }
    const int score_rows = static_cast<int>(projected.numel() - kv_rows);
    const auto stream = at::cuda::getCurrentCUDAStream();
    const size_t shared_bytes = static_cast<size_t>(width + 256) * sizeof(float);
    if (ape.scalar_type() == at::kFloat) {
        dsv4_compressor_step_controlled_kernel<float><<<
            1, 256, shared_bytes, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(projected.data_ptr<at::BFloat16>()),
                ape.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(ckv.data_ptr<at::BFloat16>()),
                cscore.data_ptr<float>(),
                reinterpret_cast<const __nv_bfloat16*>(norm.data_ptr<at::BFloat16>()),
                rope_cos.data_ptr<float>(), rope_sin.data_ptr<float>(),
                page_ptrs.data_ptr<int64_t>(), control.data_ptr<int64_t>(),
                model1_pointer, indexer_pointer, indexer_scale_pointer,
                model1_page_items,
                static_cast<int>(ratio), static_cast<int>(kv_rows), score_rows,
                static_cast<int>(width), static_cast<int>(rope_width),
                static_cast<int>(page_items), overlap, hadamard,
                static_cast<float>(eps));
    } else {
        dsv4_compressor_step_controlled_kernel<__nv_bfloat16><<<
            1, 256, shared_bytes, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(projected.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(ape.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(ckv.data_ptr<at::BFloat16>()),
                cscore.data_ptr<float>(),
                reinterpret_cast<const __nv_bfloat16*>(norm.data_ptr<at::BFloat16>()),
                rope_cos.data_ptr<float>(), rope_sin.data_ptr<float>(),
                page_ptrs.data_ptr<int64_t>(), control.data_ptr<int64_t>(),
                model1_pointer, indexer_pointer, indexer_scale_pointer,
                model1_page_items,
                static_cast<int>(ratio), static_cast<int>(kv_rows), score_rows,
                static_cast<int>(width), static_cast<int>(rope_width),
                static_cast<int>(page_items), overlap, hadamard,
                static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void paged_indexer_query_fp8_kernel(
    const __nv_bfloat16* __restrict__ query,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    const int heads,
    const int width,
    const int rope_width)
{
    const int head = blockIdx.x;
    if (head >= heads)
        return;
    extern __shared__ float values[];
    const int tid = threadIdx.x;
    if (tid < width)
        values[tid] = __bfloat162float(
            query[static_cast<long>(head) * width + tid]);
    __syncthreads();
    const int rope_start = width - rope_width;
    if (tid < rope_width / 2) {
        const int index = rope_start + 2 * tid;
        const float first = values[index];
        const float second = values[index + 1];
        const float c = rope_cos[tid];
        const float s = rope_sin[tid];
        values[index] = first * c - second * s;
        values[index + 1] = first * s + second * c;
    }
    __syncthreads();
    for (int block = 1; block < width; block <<= 1) {
        if (tid < width / 2) {
            const int group = tid / block;
            const int offset = tid % block;
            const int left = group * 2 * block + offset;
            const int right = left + block;
            const float a = values[left];
            const float b = values[right];
            values[left] = a + b;
            values[right] = a - b;
        }
        __syncthreads();
    }
    const float normalization = rsqrtf(static_cast<float>(width));
    if (tid == 0) {
        float maximum = 0.0f;
        for (int index = 0; index < width; ++index)
            maximum = fmaxf(
                maximum,
                fabsf(values[index] * normalization));
        scales[head] = fmaxf(maximum / 448.0f, 1.0e-8f);
    }
    __syncthreads();
    if (tid < width) {
        __nv_fp8_e4m3 quantized(
            values[tid] * normalization / scales[head]);
        output[static_cast<long>(head) * width + tid] = quantized.__x;
    }
}

void paged_indexer_query_fp8(
    torch::Tensor query,
    torch::Tensor rope_cos,
    torch::Tensor rope_sin,
    torch::Tensor output,
    torch::Tensor scales)
{
    TORCH_CHECK(
        query.is_cuda() && rope_cos.is_cuda() && rope_sin.is_cuda() &&
        output.is_cuda() && scales.is_cuda() &&
        query.scalar_type() == at::kBFloat16 &&
        rope_cos.scalar_type() == at::kFloat &&
        rope_sin.scalar_type() == at::kFloat &&
        output.element_size() == 1 && scales.scalar_type() == at::kFloat &&
        query.is_contiguous() && output.is_contiguous() &&
        scales.is_contiguous() && query.dim() == 4 && query.size(0) == 1 &&
        query.size(1) == 1 && output.dim() == 2 &&
        output.size(0) == query.size(2) && output.size(1) == query.size(3) &&
        scales.numel() == query.size(2) &&
        rope_cos.numel() == rope_sin.numel() &&
        rope_cos.numel() * 2 <= query.size(3) && query.size(3) <= 256,
        "paged Indexer query FP8 tensors are inconsistent");
    const int heads = static_cast<int>(query.size(2));
    const int width = static_cast<int>(query.size(3));
    paged_indexer_query_fp8_kernel<<<
        heads,
        width,
        static_cast<size_t>(width) * sizeof(float),
        at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                query.data_ptr<at::BFloat16>()),
            rope_cos.data_ptr<float>(), rope_sin.data_ptr<float>(),
            static_cast<uint8_t*>(output.data_ptr()),
            scales.data_ptr<float>(), heads, width,
            static_cast<int>(rope_cos.numel() * 2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void paged_indexer_reduce_logits_kernel(
    const __nv_bfloat16* __restrict__ head_logits,
    const float* __restrict__ head_weights,
    const int64_t* __restrict__ control,
    float* __restrict__ output,
    const int heads,
    const int candidates,
    const int compression_ratio)
{
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidates)
        return;
    const int valid = min(
        candidates,
        static_cast<int>((control[1] + 1) / compression_ratio));
    if (candidate >= valid) {
        output[candidate] = -INFINITY;
        return;
    }
    float score = 0.0f;
    for (int head = 0; head < heads; ++head) {
        const float value = __bfloat162float(
            head_logits[static_cast<long>(head) * candidates + candidate]);
        score = fmaf(fmaxf(value, 0.0f), head_weights[head], score);
    }
    output[candidate] = score;
}

void paged_indexer_reduce_logits(
    torch::Tensor head_logits,
    torch::Tensor head_weights,
    torch::Tensor control,
    torch::Tensor output,
    int64_t compression_ratio)
{
    TORCH_CHECK(
        head_logits.is_cuda() && head_weights.is_cuda() &&
        control.is_cuda() && output.is_cuda() &&
        head_logits.scalar_type() == at::kBFloat16 &&
        head_weights.scalar_type() == at::kFloat &&
        control.scalar_type() == at::kLong &&
        output.scalar_type() == at::kFloat &&
        head_logits.is_contiguous() && head_weights.is_contiguous() &&
        control.is_contiguous() && output.is_contiguous() &&
        head_logits.dim() == 2 &&
        head_weights.numel() == head_logits.size(0) &&
        output.numel() == head_logits.size(1) && control.numel() >= 2 &&
        compression_ratio > 0,
        "paged Indexer reduction tensors are inconsistent");
    const int candidates = static_cast<int>(head_logits.size(1));
    paged_indexer_reduce_logits_kernel<<<
        (candidates + 255) / 256,
        256,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                head_logits.data_ptr<at::BFloat16>()),
            head_weights.data_ptr<float>(), control.data_ptr<int64_t>(),
            output.data_ptr<float>(), static_cast<int>(head_logits.size(0)),
            candidates, static_cast<int>(compression_ratio));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void sparse_attention_inverse_rope_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    float* __restrict__ output,
    const int heads,
    const int width,
    const int rope_width)
{
    const int head = blockIdx.x;
    const int dimension = threadIdx.x;
    if (head >= heads || dimension >= width)
        return;
    const int plain = width - rope_width;
    float value;
    if (dimension < plain) {
        value = __bfloat162float(
            input[static_cast<long>(head) * width + dimension]);
    } else {
        const int relative = dimension - plain;
        const int pair = relative / 2;
        const float first = __bfloat162float(
            input[static_cast<long>(head) * width + plain + 2 * pair]);
        const float second = __bfloat162float(
            input[static_cast<long>(head) * width + plain + 2 * pair + 1]);
        value = relative & 1
            ? -first * rope_sin[pair] + second * rope_cos[pair]
            : first * rope_cos[pair] + second * rope_sin[pair];
    }
    output[static_cast<long>(head) * width + dimension] = value;
}

torch::Tensor sparse_attention_inverse_rope(
    torch::Tensor input,
    torch::Tensor rope_cos,
    torch::Tensor rope_sin)
{
    TORCH_CHECK(
        input.is_cuda() && rope_cos.is_cuda() && rope_sin.is_cuda() &&
        input.scalar_type() == at::kBFloat16 &&
        rope_cos.scalar_type() == at::kFloat &&
        rope_sin.scalar_type() == at::kFloat && input.is_contiguous() &&
        input.dim() == 4 && input.size(0) == 1 && input.size(1) == 1 &&
        rope_cos.numel() == rope_sin.numel() &&
        rope_cos.numel() * 2 <= input.size(3) && input.size(3) <= 1024,
        "sparse Attention inverse RoPE tensors are inconsistent");
    auto output = torch::empty(
        input.sizes(),
        input.options().dtype(torch::kFloat32));
    sparse_attention_inverse_rope_kernel<<<
        static_cast<int>(input.size(2)),
        static_cast<int>(input.size(3)),
        0,
        at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                input.data_ptr<at::BFloat16>()),
            rope_cos.data_ptr<float>(), rope_sin.data_ptr<float>(),
            output.data_ptr<float>(), static_cast<int>(input.size(2)),
            static_cast<int>(input.size(3)),
            static_cast<int>(rope_cos.numel() * 2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void dsv4_attn_decode_controlled_kernel(
    const float* __restrict__ q,
    const float* __restrict__ win_kv,
    const int64_t* __restrict__ win_pos,
    const __nv_bfloat16* __restrict__ comp_kv,
    const float* __restrict__ sink,
    const float* __restrict__ cs,
    const float* __restrict__ sn,
    const int64_t* __restrict__ control,
    float* __restrict__ out,
    const int H, const int D, const int W, const int C, const int rd,
    const int ratio, const bool selected_topk, const float scale)
{
    const int h = blockIdx.x;
    if (h >= H) return;
    const int tid = threadIdx.x;
    const int64_t position = control[1];
    const int available = ratio > 0
        ? min(C, static_cast<int>((position + 1) / ratio))
        : 0;
    const int valid_c = selected_topk && available >= C ? C : available;
    const int S = W + valid_c;
    extern __shared__ float smem[];
    float* qsh = smem;
    float* scores = qsh + D;
    const int score_span = (S + 3) & ~3;
    float* osh = scores + score_span;
    float* red = osh + D;
    __shared__ float score_max;
    __shared__ float denom;
    const float* qh = q + static_cast<long>(h) * D;
    for (int d = tid; d < D; d += blockDim.x) qsh[d] = qh[d];
    __syncthreads();
    for (int s = tid; s < S; s += blockDim.x) {
        const bool window_item = s < W;
        const bool valid = window_item
            ? (win_pos[s] >= 0 && win_pos[s] <= position &&
               win_pos[s] > position - W)
            : true;
        float acc = 0.0f;
        if (valid) {
            if (window_item) {
                const float* kv = win_kv + static_cast<long>(s) * D;
                const auto* query4 =
                    reinterpret_cast<const float4*>(qsh);
                const auto* key4 =
                    reinterpret_cast<const float4*>(kv);
                const int vectors = D >> 2;
                for (int vector = 0; vector < vectors; ++vector) {
                    const float4 query_value = query4[vector];
                    const float4 key_value = key4[vector];
                    acc = fmaf(query_value.x, key_value.x, acc);
                    acc = fmaf(query_value.y, key_value.y, acc);
                    acc = fmaf(query_value.z, key_value.z, acc);
                    acc = fmaf(query_value.w, key_value.w, acc);
                }
                for (int d = vectors << 2; d < D; ++d)
                    acc = fmaf(qsh[d], kv[d], acc);
            } else {
                const __nv_bfloat16* kv = comp_kv + static_cast<long>(s - W) * D;
                const auto* query4 =
                    reinterpret_cast<const float4*>(qsh);
                const auto* key2 =
                    reinterpret_cast<const __nv_bfloat162*>(kv);
                const int vectors = D >> 2;
                for (int vector = 0; vector < vectors; ++vector) {
                    const float4 query_value = query4[vector];
                    const float2 first = __bfloat1622float2(
                        key2[2 * vector]);
                    const float2 second = __bfloat1622float2(
                        key2[2 * vector + 1]);
                    acc = fmaf(query_value.x, first.x, acc);
                    acc = fmaf(query_value.y, first.y, acc);
                    acc = fmaf(query_value.z, second.x, acc);
                    acc = fmaf(query_value.w, second.y, acc);
                }
                for (int d = vectors << 2; d < D; ++d)
                    acc = fmaf(qsh[d], __bfloat162float(kv[d]), acc);
            }
        }
        scores[s] = valid ? acc * scale : -INFINITY;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int s = tid; s < S; s += blockDim.x) mx = fmaxf(mx, scores[s]);
    mx = warp_max_f32(mx);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    if (warp == 0) {
        float value = lane < nwarps ? red[lane] : -INFINITY;
        value = warp_max_f32(value);
        if (lane == 0) score_max = value;
    }
    __syncthreads();
    float z = 0.0f;
    for (int s = tid; s < S; s += blockDim.x) {
        const float value = expf(scores[s] - score_max);
        scores[s] = value;
        z += value;
    }
    z = warp_sum_f32(z);
    if (lane == 0) red[warp] = z;
    __syncthreads();
    if (warp == 0) {
        float value = lane < nwarps ? red[lane] : 0.0f;
        value = warp_sum_f32(value);
        if (lane == 0) denom = value + expf(sink[h] - score_max);
    }
    __syncthreads();
    const int plain = D - rd;
    const bool vector_rope =
        (D & 3) == 0 && (plain & 3) == 0 && (rd & 3) == 0;
    auto* output4 = reinterpret_cast<float4*>(osh);
    auto* final_output4 = reinterpret_cast<float4*>(
        out + static_cast<long>(h) * D);
    const int output_vectors = D >> 2;
    for (int vector = tid; vector < output_vectors; vector += blockDim.x) {
        const int d = vector << 2;
        float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int s = 0; s < W; ++s) {
            const float weight = scores[s];
            const float4 kv = reinterpret_cast<const float4*>(
                win_kv + static_cast<long>(s) * D)[vector];
            value.x = fmaf(weight, kv.x, value.x);
            value.y = fmaf(weight, kv.y, value.y);
            value.z = fmaf(weight, kv.z, value.z);
            value.w = fmaf(weight, kv.w, value.w);
        }
        for (int s = 0; s < valid_c; ++s) {
            const float weight = scores[W + s];
            const auto* kv = reinterpret_cast<const __nv_bfloat162*>(
                comp_kv + static_cast<long>(s) * D + d);
            const float2 first = __bfloat1622float2(kv[0]);
            const float2 second = __bfloat1622float2(kv[1]);
            value.x = fmaf(weight, first.x, value.x);
            value.y = fmaf(weight, first.y, value.y);
            value.z = fmaf(weight, second.x, value.z);
            value.w = fmaf(weight, second.y, value.w);
        }
        const float4 normalized = make_float4(
            value.x / denom,
            value.y / denom,
            value.z / denom,
            value.w / denom);
        if (vector_rope) {
            if (d < plain) {
                final_output4[vector] = normalized;
            } else {
                const int pair = (d - plain) >> 1;
                const float first_cos = cs[pair];
                const float first_sin = sn[pair];
                const float second_cos = cs[pair + 1];
                const float second_sin = sn[pair + 1];
                final_output4[vector] = make_float4(
                    normalized.x * first_cos +
                        normalized.y * first_sin,
                    -normalized.x * first_sin +
                        normalized.y * first_cos,
                    normalized.z * second_cos +
                        normalized.w * second_sin,
                    -normalized.z * second_sin +
                        normalized.w * second_cos);
            }
        } else {
            output4[vector] = normalized;
        }
    }
    for (int d = (output_vectors << 2) + tid;
         d < D;
         d += blockDim.x) {
        float value = 0.0f;
        for (int s = 0; s < W; ++s)
            value = fmaf(scores[s], win_kv[static_cast<long>(s) * D + d], value);
        for (int s = 0; s < valid_c; ++s)
            value = fmaf(
                scores[W + s],
                __bfloat162float(comp_kv[static_cast<long>(s) * D + d]),
                value);
        osh[d] = value / denom;
    }
    if (vector_rope) return;
    __syncthreads();
    for (int d = tid; d < D; d += blockDim.x) {
        float value;
        if (d < plain) value = osh[d];
        else {
            const int r = d - plain;
            const int pair = r >> 1;
            const float first = osh[plain + 2 * pair];
            const float second = osh[plain + 2 * pair + 1];
            value = (r & 1)
                ? (-first * sn[pair] + second * cs[pair])
                : ( first * cs[pair] + second * sn[pair]);
        }
        out[static_cast<long>(h) * D + d] = value;
    }
}

torch::Tensor dsv4_attn_decode_controlled(
    torch::Tensor q, torch::Tensor win_kv, torch::Tensor win_pos,
    torch::Tensor comp_kv, torch::Tensor sink, torch::Tensor cs,
    torch::Tensor sn, torch::Tensor control, double scale,
    int64_t ratio, bool selected_topk)
{
    TORCH_CHECK(
        q.is_cuda() && win_kv.is_cuda() && win_pos.is_cuda() &&
        comp_kv.is_cuda() && sink.is_cuda() && cs.is_cuda() &&
        sn.is_cuda() && control.is_cuda() &&
        q.scalar_type() == at::kFloat && win_kv.scalar_type() == at::kFloat &&
        win_pos.scalar_type() == at::kLong && comp_kv.scalar_type() == at::kBFloat16 &&
        sink.scalar_type() == at::kFloat && cs.scalar_type() == at::kFloat &&
        sn.scalar_type() == at::kFloat && control.scalar_type() == at::kLong &&
        q.dim() == 3 && q.size(0) == 1 && win_kv.dim() == 3 &&
        win_kv.size(0) == 1 && comp_kv.dim() == 3 && comp_kv.size(0) == 1 &&
        control.numel() >= 2,
        "controlled DSV4 attention tensors are inconsistent");
    const int H = static_cast<int>(q.size(1));
    const int D = static_cast<int>(q.size(2));
    const int W = static_cast<int>(win_kv.size(1));
    const int C = static_cast<int>(comp_kv.size(1));
    const int rd = static_cast<int>(cs.numel() * 2);
    TORCH_CHECK(
        win_kv.size(2) == D && comp_kv.size(2) == D &&
        win_pos.numel() == W && sink.numel() == H && sn.numel() == cs.numel() &&
        W + C > 0 && W + C <= 4096 && ratio >= 0,
        "controlled DSV4 attention shapes are inconsistent");
    auto output = torch::empty_like(q);
    const int threads = D >= 512 ? 256 : 128;
    const size_t shared_bytes = static_cast<size_t>(2 * D + W + C + 11) * sizeof(float);
    dsv4_attn_decode_controlled_kernel<<<
        H, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<float>(), win_kv.data_ptr<float>(), win_pos.data_ptr<int64_t>(),
            reinterpret_cast<const __nv_bfloat16*>(comp_kv.data_ptr<at::BFloat16>()),
            sink.data_ptr<float>(), cs.data_ptr<float>(), sn.data_ptr<float>(),
            control.data_ptr<int64_t>(), output.data_ptr<float>(),
            H, D, W, C, rd, static_cast<int>(ratio), selected_topk,
            static_cast<float>(scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// ---- DSV4 HC pre (RMS + 24-row GEMV + sinkhorn + channel reduce) ----

template <typename wt_t>
__device__ __forceinline__ float hc_weight(const wt_t* p) {
    return (float)(*p);
}

template <>
__device__ __forceinline__ float hc_weight<__nv_bfloat16>(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

// ---- DSV4 HC post (4-channel residual mix, BF16 state) ----
//
// result[n,k,d] = post[n,k] * out[n,d]
//               + sum_j comb[n,j,k] * residual[n,j,d]
//
// Decode has N=1.  Four blocks run the output channels in parallel while a
// single launch replaces the casts, broadcasts, multiply, reduction and add
// sequence emitted by the PyTorch reference.

template <typename out_t>
__global__ void dsv4_hc_post_bf16_kernel(
    const out_t* __restrict__ out,                 // [N,D]
    const __nv_bfloat16* __restrict__ residual,    // [N,4,D]
    const __nv_bfloat16* __restrict__ post,        // [N,4]
    const __nv_bfloat16* __restrict__ comb,        // [N,4,4]
    __nv_bfloat16* __restrict__ result,            // [N,4,D]
    const int D)
{
    const int n = blockIdx.x >> 2;
    const int k = blockIdx.x & 3;
    __shared__ float coeff[5];
    if (threadIdx.x == 0) {
        coeff[0] = __bfloat162float(post[(long)n * 4 + k]);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            coeff[1 + j] =
                __bfloat162float(comb[(long)n * 16 + 4 * j + k]);
    }
    __syncthreads();

    const out_t* on = out + (long)n * D;
    const __nv_bfloat16* rn = residual + (long)n * 4 * D;
    __nv_bfloat16* dst = result + ((long)n * 4 + k) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = coeff[0] * hc_weight(on + d);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            acc = fmaf(
                coeff[1 + j],
                __bfloat162float(rn[(long)j * D + d]),
                acc);
        dst[d] = __float2bfloat16_rn(acc);
    }
}

void dsv4_hc_post_into(
    torch::Tensor out, torch::Tensor residual,
    torch::Tensor post, torch::Tensor comb, torch::Tensor result)
{
    TORCH_CHECK(
        out.is_cuda() && residual.is_cuda() && post.is_cuda() && comb.is_cuda(),
        "all tensors must be CUDA");
    TORCH_CHECK(
        out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16,
        "out must be float32 or bfloat16");
    TORCH_CHECK(
        residual.scalar_type() == at::kBFloat16 &&
        post.scalar_type() == at::kBFloat16 &&
        comb.scalar_type() == at::kBFloat16,
        "residual/post/comb must be bfloat16");
    TORCH_CHECK(
        residual.dim() >= 2 && residual.size(-2) == 4,
        "residual must end in [4,D]");
    const int D = (int)residual.size(-1);
    const int N = (int)(residual.numel() / (4L * D));
    TORCH_CHECK(out.numel() == (long)N * D, "out must contain N*D values");
    TORCH_CHECK(post.numel() == (long)N * 4, "post must contain N*4 values");
    TORCH_CHECK(comb.numel() == (long)N * 16, "comb must contain N*16 values");
    TORCH_CHECK(
        result.is_cuda() && result.scalar_type() == at::kBFloat16 &&
        result.device() == residual.device() && result.is_contiguous() &&
        result.numel() == residual.numel(),
        "HC result buffer must be contiguous BF16 with residual shape/device");
    TORCH_CHECK(
        result.data_ptr() != residual.data_ptr(),
        "HC result buffer must not alias residual");

    auto oc = out.contiguous();
    auto rc = residual.contiguous();
    auto pc = post.contiguous();
    auto cc = comb.contiguous();
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = N * 4;
    if (out.scalar_type() == at::kBFloat16) {
        dsv4_hc_post_bf16_kernel<__nv_bfloat16><<<blocks, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                oc.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                rc.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                pc.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                cc.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                result.data_ptr<at::BFloat16>()),
            D);
    } else {
        dsv4_hc_post_bf16_kernel<float><<<blocks, 256, 0, stream>>>(
            oc.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(
                rc.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                pc.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                cc.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                result.data_ptr<at::BFloat16>()),
            D);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor dsv4_hc_post(
    torch::Tensor out, torch::Tensor residual,
    torch::Tensor post, torch::Tensor comb)
{
    auto result = torch::empty_like(residual);
    dsv4_hc_post_into(out, residual, post, comb, result);
    return result.view(residual.sizes());
}

torch::Tensor dsv4_hc_post_out(
    torch::Tensor out, torch::Tensor residual,
    torch::Tensor post, torch::Tensor comb, torch::Tensor result)
{
    dsv4_hc_post_into(out, residual, post, comb, result);
    return result.view(residual.sizes());
}

// Batch-1 MoE decode reaches HC post with a FP32 routed result and a BF16
// shared-expert result.  Folding their BF16 merge into HC post removes the
// intermediate cast and add tensors while preserving the reference rounding.
__global__ void dsv4_hc_post_moe_bf16_kernel(
    const float* __restrict__ routed,              // [N,D]
    const __nv_bfloat16* __restrict__ shared,      // [N,D]
    const __nv_bfloat16* __restrict__ residual,    // [N,4,D]
    const __nv_bfloat16* __restrict__ post,        // [N,4]
    const __nv_bfloat16* __restrict__ comb,        // [N,4,4]
    __nv_bfloat16* __restrict__ result,            // [N,4,D]
    const int D)
{
    const int n = blockIdx.x >> 2;
    const int k = blockIdx.x & 3;
    __shared__ float coeff[5];
    if (threadIdx.x == 0) {
        coeff[0] = __bfloat162float(post[(long)n * 4 + k]);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            coeff[1 + j] =
                __bfloat162float(comb[(long)n * 16 + 4 * j + k]);
    }
    __syncthreads();

    const float* routed_n = routed + (long)n * D;
    const __nv_bfloat16* shared_n = shared + (long)n * D;
    const __nv_bfloat16* residual_n = residual + (long)n * 4 * D;
    __nv_bfloat16* dst = result + ((long)n * 4 + k) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const __nv_bfloat16 routed_bf16 =
            __float2bfloat16_rn(routed_n[d]);
        const __nv_bfloat16 merged_bf16 = __float2bfloat16_rn(
            __bfloat162float(routed_bf16) +
            __bfloat162float(shared_n[d]));
        float acc = coeff[0] * __bfloat162float(merged_bf16);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            acc = fmaf(
                coeff[1 + j],
                __bfloat162float(residual_n[(long)j * D + d]),
                acc);
        dst[d] = __float2bfloat16_rn(acc);
    }
}

__global__ void dsv4_hc_post_moe_f32_shared_kernel(
    const float* __restrict__ routed,
    const float* __restrict__ shared,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ post,
    const __nv_bfloat16* __restrict__ comb,
    __nv_bfloat16* __restrict__ result,
    const int D)
{
    const int n = blockIdx.x >> 2;
    const int k = blockIdx.x & 3;
    __shared__ float coeff[5];
    if (threadIdx.x == 0) {
        coeff[0] = __bfloat162float(post[(long)n * 4 + k]);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            coeff[1 + j] =
                __bfloat162float(comb[(long)n * 16 + 4 * j + k]);
    }
    __syncthreads();
    const float* routed_n = routed + (long)n * D;
    const float* shared_n = shared + (long)n * D;
    const __nv_bfloat16* residual_n = residual + (long)n * 4 * D;
    __nv_bfloat16* dst = result + ((long)n * 4 + k) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const __nv_bfloat16 routed_bf16 =
            __float2bfloat16_rn(routed_n[d]);
        const __nv_bfloat16 shared_bf16 =
            __float2bfloat16_rn(shared_n[d]);
        const __nv_bfloat16 merged_bf16 = __float2bfloat16_rn(
            __bfloat162float(routed_bf16) +
            __bfloat162float(shared_bf16));
        float acc = coeff[0] * __bfloat162float(merged_bf16);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            acc = fmaf(
                coeff[1 + j],
                __bfloat162float(residual_n[(long)j * D + d]),
                acc);
        dst[d] = __float2bfloat16_rn(acc);
    }
}

void dsv4_hc_post_moe_into(
    torch::Tensor routed, torch::Tensor shared, torch::Tensor residual,
    torch::Tensor post, torch::Tensor comb, torch::Tensor result)
{
    TORCH_CHECK(
        routed.is_cuda() && shared.is_cuda() && residual.is_cuda() &&
        post.is_cuda() && comb.is_cuda(),
        "all tensors must be CUDA");
    TORCH_CHECK(
        routed.scalar_type() == at::kFloat &&
        (shared.scalar_type() == at::kBFloat16 ||
         shared.scalar_type() == at::kFloat) &&
        residual.scalar_type() == at::kBFloat16 &&
        post.scalar_type() == at::kBFloat16 &&
        comb.scalar_type() == at::kBFloat16,
        "routed must be float32; shared must be float32/bfloat16; "
        "residual/post/comb must be bfloat16");
    TORCH_CHECK(
        residual.dim() >= 2 && residual.size(-2) == 4,
        "residual must end in [4,D]");
    const int D = (int)residual.size(-1);
    const int N = (int)(residual.numel() / (4L * D));
    TORCH_CHECK(
        routed.numel() == (long)N * D &&
        shared.numel() == (long)N * D,
        "routed/shared must contain N*D values");
    TORCH_CHECK(post.numel() == (long)N * 4, "post must contain N*4 values");
    TORCH_CHECK(comb.numel() == (long)N * 16, "comb must contain N*16 values");
    TORCH_CHECK(
        result.is_cuda() && result.scalar_type() == at::kBFloat16 &&
        result.device() == residual.device() && result.is_contiguous() &&
        result.numel() == residual.numel(),
        "HC result buffer must be contiguous BF16 with residual shape/device");
    TORCH_CHECK(
        result.data_ptr() != residual.data_ptr(),
        "HC result buffer must not alias residual");

    auto routed_c = routed.contiguous();
    auto shared_c = shared.contiguous();
    auto residual_c = residual.contiguous();
    auto post_c = post.contiguous();
    auto comb_c = comb.contiguous();
    auto stream = at::cuda::getCurrentCUDAStream();
    if (shared.scalar_type() == at::kFloat) {
        dsv4_hc_post_moe_f32_shared_kernel<<<N * 4, 256, 0, stream>>>(
            routed_c.data_ptr<float>(), shared_c.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(
                residual_c.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                post_c.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                comb_c.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                result.data_ptr<at::BFloat16>()),
            D);
    } else {
        dsv4_hc_post_moe_bf16_kernel<<<N * 4, 256, 0, stream>>>(
            routed_c.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(
                shared_c.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                residual_c.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                post_c.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                comb_c.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                result.data_ptr<at::BFloat16>()),
            D);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor dsv4_hc_post_moe(
    torch::Tensor routed, torch::Tensor shared, torch::Tensor residual,
    torch::Tensor post, torch::Tensor comb)
{
    auto result = torch::empty_like(residual);
    dsv4_hc_post_moe_into(
        routed, shared, residual, post, comb, result);
    return result.view(residual.sizes());
}

torch::Tensor dsv4_hc_post_moe_out(
    torch::Tensor routed, torch::Tensor shared, torch::Tensor residual,
    torch::Tensor post, torch::Tensor comb, torch::Tensor result)
{
    dsv4_hc_post_moe_into(
        routed, shared, residual, post, comb, result);
    return result.view(residual.sizes());
}

template <typename wt_t>
__global__ void dsv4_hc_pre_kernel(
    const float* __restrict__ x,       // [N,4,D]
    const wt_t* __restrict__ fn,       // [24,4D]
    const float* __restrict__ scale,   // [3]
    const float* __restrict__ base,    // [24]
    float* __restrict__ y,             // [N,D]
    float* __restrict__ post_out,      // [N,4]
    float* __restrict__ comb_out,      // [N,16]
    const int D, const int iters, const float eps)
{
    const int n = blockIdx.x;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int flatD = 4 * D;
    const float* xn = x + (long)n * flatD;
    __shared__ float red[8];
    __shared__ float inv_rms;
    __shared__ float mixes[24];
    __shared__ float pre[4];
    __shared__ float post[4];
    __shared__ float comb[16];

    float ss = 0.f;
    for (int i = tid; i < flatD; i += 256) {
        const float v = xn[i];
        ss = fmaf(v, v, ss);
    }
    ss = warp_sum_f32(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = lane < 8 ? red[lane] : 0.f;
        v = warp_sum_f32(v);
        if (lane == 0) inv_rms = rsqrtf(v / (float)flatD + eps);
    }
    __syncthreads();

    #pragma unroll
    for (int batch = 0; batch < 3; ++batch) {
        const int m = batch * 8 + warp;
        const wt_t* fm = fn + (long)m * flatD;
        float acc = 0.f;
        for (int i = lane; i < flatD; i += 32)
            acc = fmaf(xn[i], hc_weight(fm + i), acc);
        acc = warp_sum_f32(acc);
        if (lane == 0) mixes[m] = acc * inv_rms;
    }
    __syncthreads();

    if (tid == 0) {
        const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            pre[j] = 1.f / (1.f + expf(-(mixes[j] * s0 + base[j]))) + eps;
            post[j] = 2.f / (1.f + expf(-(mixes[4 + j] * s1 + base[4 + j])));
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float mx = -INFINITY;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                comb[4 * j + k] = mixes[8 + 4 * j + k] * s2 + base[8 + 4 * j + k];
                mx = fmaxf(mx, comb[4 * j + k]);
            }
            float sum = 0.f;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                comb[4 * j + k] = expf(comb[4 * j + k] - mx);
                sum += comb[4 * j + k];
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                comb[4 * j + k] = comb[4 * j + k] / sum + eps;
        }
        for (int it = 0; it < iters; ++it) {
            if (it > 0) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float sum = comb[4*j] + comb[4*j+1] + comb[4*j+2] + comb[4*j+3];
                    const float inv = 1.f / (sum + eps);
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) comb[4*j+k] *= inv;
                }
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float sum = comb[k] + comb[4+k] + comb[8+k] + comb[12+k];
                const float inv = 1.f / (sum + eps);
                #pragma unroll
                for (int j = 0; j < 4; ++j) comb[4*j+k] *= inv;
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) post_out[(long)n * 4 + j] = post[j];
        #pragma unroll
        for (int j = 0; j < 16; ++j) comb_out[(long)n * 16 + j] = comb[j];
    }
    __syncthreads();

    for (int d = tid; d < D; d += 256) {
        float v = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            v = fmaf(pre[j], xn[(long)j * D + d], v);
        y[(long)n * D + d] = v;
    }
}

template <typename wt_t, typename norm_t>
__global__ void dsv4_hc_pre_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,  // [N,4,D]
    const wt_t* __restrict__ fn,          // [24,4D]
    const float* __restrict__ scale,      // [3]
    const float* __restrict__ base,       // [24]
    const norm_t* __restrict__ norm,      // [D]
    __nv_bfloat16* __restrict__ y,        // [N,D]
    __nv_bfloat16* __restrict__ post_out, // [N,4]
    __nv_bfloat16* __restrict__ comb_out, // [N,16]
    const int D, const int iters, const float eps)
{
    const int n = blockIdx.x;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int flatD = 4 * D;
    const __nv_bfloat16* xn = x + (long)n * flatD;
    __shared__ float red[8];
    __shared__ float inv_rms;
    __shared__ float inv_y_rms;
    __shared__ float mixes[24];
    __shared__ float pre[4];
    __shared__ float post[4];
    __shared__ float comb[16];

    float ss = 0.f;
    for (int i = tid; i < flatD; i += 256) {
        const float v = __bfloat162float(xn[i]);
        ss = fmaf(v, v, ss);
    }
    ss = warp_sum_f32(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = lane < 8 ? red[lane] : 0.f;
        v = warp_sum_f32(v);
        if (lane == 0) inv_rms = rsqrtf(v / (float)flatD + eps);
    }
    __syncthreads();

    #pragma unroll
    for (int batch = 0; batch < 3; ++batch) {
        const int m = batch * 8 + warp;
        const wt_t* fm = fn + (long)m * flatD;
        float acc = 0.f;
        for (int i = lane; i < flatD; i += 32)
            acc = fmaf(__bfloat162float(xn[i]), hc_weight(fm + i), acc);
        acc = warp_sum_f32(acc);
        if (lane == 0) mixes[m] = acc * inv_rms;
    }
    __syncthreads();

    if (tid == 0) {
        const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            pre[j] = 1.f / (1.f + expf(-(mixes[j] * s0 + base[j]))) + eps;
            post[j] = 2.f / (1.f + expf(-(mixes[4 + j] * s1 + base[4 + j])));
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float mx = -INFINITY;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                comb[4 * j + k] =
                    mixes[8 + 4 * j + k] * s2 + base[8 + 4 * j + k];
                mx = fmaxf(mx, comb[4 * j + k]);
            }
            float sum = 0.f;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                comb[4 * j + k] = expf(comb[4 * j + k] - mx);
                sum += comb[4 * j + k];
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                comb[4 * j + k] = comb[4 * j + k] / sum + eps;
        }
        for (int it = 0; it < iters; ++it) {
            if (it > 0) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float sum =
                        comb[4*j] + comb[4*j+1] + comb[4*j+2] + comb[4*j+3];
                    const float inv = 1.f / (sum + eps);
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) comb[4*j+k] *= inv;
                }
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float sum =
                    comb[k] + comb[4+k] + comb[8+k] + comb[12+k];
                const float inv = 1.f / (sum + eps);
                #pragma unroll
                for (int j = 0; j < 4; ++j) comb[4*j+k] *= inv;
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            post_out[(long)n * 4 + j] = __float2bfloat16_rn(post[j]);
        #pragma unroll
        for (int j = 0; j < 16; ++j)
            comb_out[(long)n * 16 + j] = __float2bfloat16_rn(comb[j]);
    }
    __syncthreads();

    float yss = 0.f;
    for (int d = tid; d < D; d += 256) {
        float v = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            v = fmaf(pre[j], __bfloat162float(xn[(long)j * D + d]), v);
        yss = fmaf(v, v, yss);
    }
    yss = warp_sum_f32(yss);
    if (lane == 0) red[warp] = yss;
    __syncthreads();
    if (warp == 0) {
        float v = lane < 8 ? red[lane] : 0.f;
        v = warp_sum_f32(v);
        if (lane == 0) inv_y_rms = rsqrtf(v / (float)D + eps);
    }
    __syncthreads();

    for (int d = tid; d < D; d += 256) {
        float v = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            v = fmaf(pre[j], __bfloat162float(xn[(long)j * D + d]), v);
        y[(long)n * D + d] =
            __float2bfloat16_rn(v * inv_y_rms * hc_weight(norm + d));
    }
}

// Decode fast path.  The original all-in-one HC kernel launches one
// block per token and evaluates the 24-row GEMV in three serial batches of
// eight warps.  For N=1 that leaves almost the whole GPU idle.  Split the
// operation into 24 independent GEMV blocks followed by one finish block.
template <typename wt_t>
__global__ void dsv4_hc_mix_parallel_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,       // [N,4,D]
    const wt_t* __restrict__ fn,                // [24,4D]
    __nv_bfloat16* __restrict__ scratch,       // y [N,D], temporary mixes
    const int D, const float eps)
{
    const int n = blockIdx.x / 24;
    const int m = blockIdx.x - n * 24;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int flatD = 4 * D;
    const __nv_bfloat16* xn = x + (long)n * flatD;
    const wt_t* fm = fn + (long)m * flatD;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(xn);
    const int pairs = flatD / 2;
    float dot = 0.f;
    float ss = 0.f;
    for (int i = tid; i < pairs; i += blockDim.x) {
        const float2 xv = __bfloat1622float2(x2[i]);
        dot = fmaf(xv.x, hc_weight(fm + i * 2), dot);
        dot = fmaf(xv.y, hc_weight(fm + i * 2 + 1), dot);
        if (m == 0) {
            ss = fmaf(xv.x, xv.x, ss);
            ss = fmaf(xv.y, xv.y, ss);
        }
    }
    dot = warp_sum_f32(dot);
    if (m == 0)
        ss = warp_sum_f32(ss);
    __shared__ float dot_warp[8];
    __shared__ float ss_warp[8];
    if (lane == 0) {
        dot_warp[warp] = dot;
        if (m == 0)
            ss_warp[warp] = ss;
    }
    __syncthreads();
    if (warp == 0) {
        float dv = lane < 8 ? dot_warp[lane] : 0.f;
        dv = warp_sum_f32(dv);
        if (lane == 0) {
            float* mixes = reinterpret_cast<float*>(
                scratch + (long)n * D);
            mixes[m] = dv;
        }
        if (m == 0) {
            float sv = lane < 8 ? ss_warp[lane] : 0.f;
            sv = warp_sum_f32(sv);
            if (lane == 0) {
                float* mixes = reinterpret_cast<float*>(
                    scratch + (long)n * D);
                mixes[24] = rsqrtf(sv / (float)flatD + eps);
            }
        }
    }
}

// The 24 raw float mixes plus one shared input RMS value temporarily occupy
// the first 100 bytes of the BF16 y output; the finish kernel loads them into
// shared memory before overwriting y.  Only block zero evaluates the input
// sum-of-squares.  The previous implementation repeated that identical work
// in all 24 GEMV blocks, nearly doubling the hot-path memory/FMA traffic.

constexpr int CCCP_DSV4_HC_FINISH_THREADS = 512;
constexpr int CCCP_DSV4_HC_FINISH_WARPS = 16;

__global__ void dsv4_hc_finish_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,       // [N,4,D]
    const float* __restrict__ scale,            // [3]
    const float* __restrict__ base,             // [24]
    const __nv_bfloat16* __restrict__ norm,     // [D]
    __nv_bfloat16* __restrict__ y,              // [N,D], starts as scratch
    __nv_bfloat16* __restrict__ post_out,       // [N,4]
    __nv_bfloat16* __restrict__ comb_out,       // [N,16]
    const int D, const int iters, const float eps)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const __nv_bfloat16* xn = x + (long)n * 4 * D;
    __nv_bfloat16* yn = y + (long)n * D;
    __shared__ float red[CCCP_DSV4_HC_FINISH_WARPS];
    __shared__ float inv_y_rms;
    __shared__ float mixes[24];
    __shared__ float pre[4];
    __shared__ float post[4];
    __shared__ float comb[16];

    if (tid == 0) {
        const float* mix_scratch = reinterpret_cast<const float*>(yn);
        const float inv_x_rms = mix_scratch[24];
        #pragma unroll
        for (int i = 0; i < 24; ++i)
            mixes[i] = mix_scratch[i] * inv_x_rms;

        const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            pre[j] = 1.f / (1.f + expf(-(mixes[j] * s0 + base[j]))) + eps;
            post[j] = 2.f / (1.f + expf(-(mixes[4 + j] * s1 + base[4 + j])));
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float mx = -INFINITY;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                comb[4 * j + k] =
                    mixes[8 + 4 * j + k] * s2 + base[8 + 4 * j + k];
                mx = fmaxf(mx, comb[4 * j + k]);
            }
            float sum = 0.f;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                comb[4 * j + k] = expf(comb[4 * j + k] - mx);
                sum += comb[4 * j + k];
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                comb[4 * j + k] = comb[4 * j + k] / sum + eps;
        }
        for (int it = 0; it < iters; ++it) {
            if (it > 0) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float sum =
                        comb[4*j] + comb[4*j+1] + comb[4*j+2] + comb[4*j+3];
                    const float inv = 1.f / (sum + eps);
                    #pragma unroll
                    for (int k = 0; k < 4; ++k)
                        comb[4*j+k] *= inv;
                }
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float sum =
                    comb[k] + comb[4+k] + comb[8+k] + comb[12+k];
                const float inv = 1.f / (sum + eps);
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    comb[4*j+k] *= inv;
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            post_out[(long)n * 4 + j] = __float2bfloat16_rn(post[j]);
        #pragma unroll
        for (int j = 0; j < 16; ++j)
            comb_out[(long)n * 16 + j] = __float2bfloat16_rn(comb[j]);
    }
    __syncthreads();

    float yss = 0.f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            v = fmaf(pre[j], __bfloat162float(xn[(long)j * D + d]), v);
        yss = fmaf(v, v, yss);
    }
    yss = warp_sum_f32(yss);
    if (lane == 0) red[warp] = yss;
    __syncthreads();
    if (warp == 0) {
        float v = lane < CCCP_DSV4_HC_FINISH_WARPS ? red[lane] : 0.f;
        v = warp_sum_f32(v);
        if (lane == 0)
            inv_y_rms = rsqrtf(v / (float)D + eps);
    }
    __syncthreads();

    for (int d = tid; d < D; d += blockDim.x) {
        float v = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            v = fmaf(pre[j], __bfloat162float(xn[(long)j * D + d]), v);
        yn[d] = __float2bfloat16_rn(
            v * inv_y_rms * __bfloat162float(norm[d]));
    }
}

template <typename wt_t>
void launch_dsv4_hc_pre_norm_bf16_parallel(
    const torch::Tensor& x, const torch::Tensor& fn,
    const torch::Tensor& scale, const torch::Tensor& base,
    const torch::Tensor& norm, torch::Tensor& y,
    torch::Tensor& post, torch::Tensor& comb,
    int N, int D, int iters, float eps)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    dsv4_hc_mix_parallel_bf16_kernel<wt_t><<<N * 24, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const wt_t*>(fn.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(
            y.data_ptr<at::BFloat16>()),
        D, eps);
    dsv4_hc_finish_norm_bf16_kernel<<<N, CCCP_DSV4_HC_FINISH_THREADS,
        0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            x.data_ptr<at::BFloat16>()),
        scale.data_ptr<float>(),
        base.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(
            norm.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            y.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            post.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            comb.data_ptr<at::BFloat16>()),
        D, iters, eps);
}

template <typename wt_t, typename norm_t>
void launch_dsv4_hc_pre_norm_bf16(
    const torch::Tensor& x, const torch::Tensor& fn,
    const torch::Tensor& scale, const torch::Tensor& base,
    const torch::Tensor& norm, torch::Tensor& y,
    torch::Tensor& post, torch::Tensor& comb,
    int N, int D, int iters, float eps)
{
    dim3 block(32, 8);
    auto stream = at::cuda::getCurrentCUDAStream();
    dsv4_hc_pre_norm_bf16_kernel<wt_t, norm_t><<<N, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const wt_t*>(fn.data_ptr()),
        scale.data_ptr<float>(), base.data_ptr<float>(),
        reinterpret_cast<const norm_t*>(norm.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(post.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(comb.data_ptr<at::BFloat16>()),
        D, iters, eps);
}

void dsv4_hc_pre_norm_into(
    torch::Tensor x, torch::Tensor fn, torch::Tensor scale,
    torch::Tensor base, torch::Tensor norm, torch::Tensor y,
    torch::Tensor post, torch::Tensor comb, long iters, double eps)
{
    TORCH_CHECK(
        x.is_cuda() && fn.is_cuda() && scale.is_cuda() &&
        base.is_cuda() && norm.is_cuda(),
        "all tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(
        fn.scalar_type() == at::kFloat || fn.scalar_type() == at::kBFloat16,
        "fn must be float32 or bfloat16");
    TORCH_CHECK(
        norm.scalar_type() == at::kFloat || norm.scalar_type() == at::kBFloat16,
        "norm must be float32 or bfloat16");
    TORCH_CHECK(
        scale.scalar_type() == at::kFloat && base.scalar_type() == at::kFloat,
        "scale/base must be float32");
    TORCH_CHECK(x.dim() >= 2 && x.size(-2) == 4, "x must end in [4,D]");
    const int D = (int)x.size(-1);
    const int N = (int)(x.numel() / (4L * D));
    TORCH_CHECK(fn.numel() == 24L * 4L * D, "fn must be [24,4D]");
    TORCH_CHECK(norm.numel() == D, "norm must be [D]");
    TORCH_CHECK(scale.numel() == 3 && base.numel() == 24,
                "HC parameter size mismatch");
    TORCH_CHECK(
        y.is_cuda() && post.is_cuda() && comb.is_cuda(),
        "HC output buffers must be CUDA tensors");
    TORCH_CHECK(
        y.device() == x.device() && post.device() == x.device() &&
        comb.device() == x.device(),
        "HC output buffers must share the input device");
    TORCH_CHECK(
        y.scalar_type() == at::kBFloat16 &&
        post.scalar_type() == at::kBFloat16 &&
        comb.scalar_type() == at::kBFloat16,
        "HC output buffers must be bfloat16");
    TORCH_CHECK(
        y.is_contiguous() && post.is_contiguous() && comb.is_contiguous(),
        "HC output buffers must be contiguous");
    TORCH_CHECK(
        y.numel() == (long)N * D && post.numel() == (long)N * 4 &&
        comb.numel() == (long)N * 16,
        "HC output buffer size mismatch");

    auto xc = x.contiguous();
    auto fc = fn.contiguous();
    auto sc = scale.contiguous();
    auto bc = base.contiguous();
    auto nc = norm.contiguous();
    if (norm.scalar_type() == at::kBFloat16 && D >= 50) {
        if (fn.scalar_type() == at::kBFloat16) {
            launch_dsv4_hc_pre_norm_bf16_parallel<__nv_bfloat16>(
                xc, fc, sc, bc, nc, y, post, comb,
                N, D, (int)iters, (float)eps);
        } else {
            launch_dsv4_hc_pre_norm_bf16_parallel<float>(
                xc, fc, sc, bc, nc, y, post, comb,
                N, D, (int)iters, (float)eps);
        }
    } else if (fn.scalar_type() == at::kBFloat16 &&
               norm.scalar_type() == at::kBFloat16) {
        launch_dsv4_hc_pre_norm_bf16<__nv_bfloat16, __nv_bfloat16>(
            xc, fc, sc, bc, nc, y, post, comb,
            N, D, (int)iters, (float)eps);
    } else if (fn.scalar_type() == at::kBFloat16) {
        launch_dsv4_hc_pre_norm_bf16<__nv_bfloat16, float>(
            xc, fc, sc, bc, nc, y, post, comb, N, D, (int)iters, (float)eps);
    } else if (norm.scalar_type() == at::kBFloat16) {
        launch_dsv4_hc_pre_norm_bf16<float, __nv_bfloat16>(
            xc, fc, sc, bc, nc, y, post, comb, N, D, (int)iters, (float)eps);
    } else {
        launch_dsv4_hc_pre_norm_bf16<float, float>(
            xc, fc, sc, bc, nc, y, post, comb, N, D, (int)iters, (float)eps);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> dsv4_hc_pre_norm(
    torch::Tensor x, torch::Tensor fn, torch::Tensor scale,
    torch::Tensor base, torch::Tensor norm, long iters, double eps)
{
    TORCH_CHECK(x.dim() >= 2, "x must end in [4,D]");
    const int D = (int)x.size(-1);
    const int N = (int)(x.numel() / (4L * D));
    auto y = torch::empty({N, D}, x.options());
    auto post = torch::empty({N, 4}, x.options());
    auto comb = torch::empty({N, 16}, x.options());
    dsv4_hc_pre_norm_into(
        x, fn, scale, base, norm, y, post, comb, iters, eps);
    return {y, post, comb};
}

std::vector<torch::Tensor> dsv4_hc_pre_norm_out(
    torch::Tensor x, torch::Tensor fn, torch::Tensor scale,
    torch::Tensor base, torch::Tensor norm, torch::Tensor y,
    torch::Tensor post, torch::Tensor comb, long iters, double eps)
{
    dsv4_hc_pre_norm_into(
        x, fn, scale, base, norm, y, post, comb, iters, eps);
    return {y, post, comb};
}

std::vector<torch::Tensor> dsv4_hc_pre(
    torch::Tensor x, torch::Tensor fn, torch::Tensor scale, torch::Tensor base,
    long iters, double eps) {
    TORCH_CHECK(x.is_cuda() && fn.is_cuda() && scale.is_cuda() && base.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(fn.scalar_type() == at::kFloat || fn.scalar_type() == at::kBFloat16,
                "fn must be float32 or bfloat16");
    TORCH_CHECK(scale.scalar_type() == at::kFloat && base.scalar_type() == at::kFloat,
                "scale/base must be float32");
    TORCH_CHECK(x.dim() >= 2 && x.size(-2) == 4, "x must end in [4,D]");
    const int D = (int)x.size(-1);
    const int N = (int)(x.numel() / (4L * D));
    TORCH_CHECK(fn.numel() == 24L * 4L * D, "fn must be [24,4D]");
    TORCH_CHECK(scale.numel() == 3 && base.numel() == 24, "HC parameter size mismatch");
    auto xc = x.contiguous();
    auto fc = fn.contiguous();
    auto sc = scale.contiguous();
    auto bc = base.contiguous();
    auto y = torch::empty({N, D}, x.options());
    auto post = torch::empty({N, 4}, x.options());
    auto comb = torch::empty({N, 16}, x.options());
    dim3 block(32, 8);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (fn.scalar_type() == at::kFloat) {
        dsv4_hc_pre_kernel<float><<<N, block, 0, stream>>>(
            xc.data_ptr<float>(), fc.data_ptr<float>(), sc.data_ptr<float>(),
            bc.data_ptr<float>(), y.data_ptr<float>(), post.data_ptr<float>(),
            comb.data_ptr<float>(), D, (int)iters, (float)eps);
    } else {
        dsv4_hc_pre_kernel<__nv_bfloat16><<<N, block, 0, stream>>>(
            xc.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(fc.data_ptr<at::BFloat16>()),
            sc.data_ptr<float>(), bc.data_ptr<float>(), y.data_ptr<float>(),
            post.data_ptr<float>(), comb.data_ptr<float>(),
            D, (int)iters, (float)eps);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, post, comb};
}

__global__ void dsv4_route_post_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ bias,
    const bool* __restrict__ mask,
    float* __restrict__ weights,
    int64_t* __restrict__ indices,
    int experts,
    int top_k) {
    extern __shared__ float choices[];
    __shared__ float route_warp_values[32];
    __shared__ int route_warp_indices[32];
    __shared__ int route_selected[16];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = (blockDim.x + 31) >> 5;
    for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        const float corrected = scores[expert] + bias[expert];
        // A restored CUDA graph must never turn a non-finite router value
        // into an unavailable expert.  Keep every allowed expert selectable
        // as a deterministic zero-weight fallback, while masked experts stay
        // outside the candidate set under all IEEE comparison outcomes.
        choices[expert] = mask[expert]
            ? (isfinite(corrected) ? corrected : -FLT_MAX)
            : -INFINITY;
    }
    __syncthreads();

    for (int rank = 0; rank < top_k; ++rank) {
        float best = -INFINITY;
        int best_expert = -1;
        for (int expert = tid; expert < experts; expert += blockDim.x) {
            const float value = choices[expert];
            if (!isfinite(value)) {
                continue;
            }
            if (best_expert < 0 || value > best ||
                (value == best && expert < best_expert)) {
                best = value;
                best_expert = expert;
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_value =
                __shfl_down_sync(0xffffffffu, best, offset);
            const int other_expert =
                __shfl_down_sync(0xffffffffu, best_expert, offset);
            if (other_expert >= 0 &&
                (best_expert < 0 || other_value > best ||
                 (other_value == best && other_expert < best_expert))) {
                best = other_value;
                best_expert = other_expert;
            }
        }
        if (lane == 0) {
            route_warp_values[warp] = best;
            route_warp_indices[warp] = best_expert;
        }
        __syncthreads();

        if (warp == 0) {
            best = lane < warps ? route_warp_values[lane] : -INFINITY;
            best_expert = lane < warps ? route_warp_indices[lane] : -1;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other_value =
                    __shfl_down_sync(0xffffffffu, best, offset);
                const int other_expert =
                    __shfl_down_sync(0xffffffffu, best_expert, offset);
                if (other_expert >= 0 &&
                    (best_expert < 0 || other_value > best ||
                     (other_value == best && other_expert < best_expert))) {
                    best = other_value;
                    best_expert = other_expert;
                }
            }
            if (lane == 0) {
                route_selected[rank] = best_expert;
                indices[rank] = best_expert;
                const float selected_score = scores[best_expert];
                weights[rank] = isfinite(selected_score)
                    ? selected_score
                    : 0.0f;
            }
        }
        __syncthreads();

        const int selected = route_selected[rank];
        for (int expert = tid; expert < experts; expert += blockDim.x) {
            if (expert == selected) {
                choices[expert] = -INFINITY;
            }
        }
        __syncthreads();
    }
}

std::vector<torch::Tensor> dsv4_route_post(
    torch::Tensor scores,
    torch::Tensor bias,
    torch::Tensor mask,
    long top_k) {
    TORCH_CHECK(scores.is_cuda() && bias.is_cuda() && mask.is_cuda(),
                "scores/bias/mask must be CUDA");
    TORCH_CHECK(scores.scalar_type() == at::kFloat &&
                bias.scalar_type() == at::kFloat,
                "scores/bias must be float32");
    TORCH_CHECK(mask.scalar_type() == at::kBool, "mask must be bool");
    TORCH_CHECK(scores.dim() == 2 && scores.size(0) == 1,
                "scores must be [1,E]");
    const int experts = (int)scores.size(1);
    TORCH_CHECK(bias.numel() == experts && mask.numel() == experts,
                "bias/mask size must match experts");
    TORCH_CHECK(experts > 0 && experts <= 1024,
                "experts must be in [1,1024]");
    TORCH_CHECK(top_k > 0 && top_k <= 16 && top_k <= experts,
                "top_k must be in [1,min(16,E)]");

    auto sc = scores.contiguous();
    auto bc = bias.contiguous();
    auto mc = mask.contiguous();
    auto weights = torch::empty({1, top_k}, scores.options());
    auto indices = torch::empty(
        {1, top_k}, scores.options().dtype(at::kLong));
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = experts < 256 ? 128 : 256;
    dsv4_route_post_kernel<<<1, threads, experts * sizeof(float), stream>>>(
        sc.data_ptr<float>(),
        bc.data_ptr<float>(),
        mc.data_ptr<bool>(),
        weights.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        experts,
        (int)top_k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {weights, indices};
}

// ---- GLM sigmoid router + corrected Top-K + normalized route weights ----

__global__ void sigmoid_route_select_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const bool* __restrict__ mask,
    float* __restrict__ weights,
    int64_t* __restrict__ indices,
    int experts,
    int top_k,
    float routed_scaling) {
    extern __shared__ float choices[];
    __shared__ float route_warp_values[32];
    __shared__ int route_warp_indices[32];
    __shared__ int route_selected[16];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = (blockDim.x + 31) >> 5;
    for (int expert = tid; expert < experts; expert += blockDim.x) {
        const float probability =
            1.0f / (1.0f + expf(-logits[expert]));
        const float corrected = probability + bias[expert];
        choices[expert] = mask[expert]
            ? (isfinite(corrected) ? corrected : -FLT_MAX)
            : -INFINITY;
    }
    __syncthreads();

    for (int rank = 0; rank < top_k; ++rank) {
        float best = -INFINITY;
        int best_expert = -1;
        for (int expert = tid; expert < experts; expert += blockDim.x) {
            const float value = choices[expert];
            if (!isfinite(value)) {
                continue;
            }
            if (
                best_expert < 0 ||
                value > best ||
                (value == best && expert < best_expert)
            ) {
                best = value;
                best_expert = expert;
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_value =
                __shfl_down_sync(0xffffffffu, best, offset);
            const int other_expert =
                __shfl_down_sync(0xffffffffu, best_expert, offset);
            if (
                other_expert >= 0 &&
                (
                    best_expert < 0 ||
                    other_value > best ||
                    (
                        other_value == best &&
                        other_expert < best_expert
                    )
                )
            ) {
                best = other_value;
                best_expert = other_expert;
            }
        }
        if (lane == 0) {
            route_warp_values[warp] = best;
            route_warp_indices[warp] = best_expert;
        }
        __syncthreads();
        if (warp == 0) {
            best = lane < warps
                ? route_warp_values[lane]
                : -INFINITY;
            best_expert = lane < warps
                ? route_warp_indices[lane]
                : -1;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other_value =
                    __shfl_down_sync(0xffffffffu, best, offset);
                const int other_expert =
                    __shfl_down_sync(
                        0xffffffffu,
                        best_expert,
                        offset);
                if (
                    other_expert >= 0 &&
                    (
                        best_expert < 0 ||
                        other_value > best ||
                        (
                            other_value == best &&
                            other_expert < best_expert
                        )
                    )
                ) {
                    best = other_value;
                    best_expert = other_expert;
                }
            }
            if (lane == 0) {
                route_selected[rank] = best_expert;
                indices[rank] = best_expert;
                const float probability =
                    1.0f / (1.0f + expf(-logits[best_expert]));
                weights[rank] = isfinite(probability)
                    ? probability
                    : 0.0f;
            }
        }
        __syncthreads();
        const int selected = route_selected[rank];
        for (int expert = tid; expert < experts; expert += blockDim.x) {
            if (expert == selected) {
                choices[expert] = -INFINITY;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        float sum = 1.0e-20f;
        for (int rank = 0; rank < top_k; ++rank) {
            sum += weights[rank];
        }
        const float factor = routed_scaling / sum;
        for (int rank = 0; rank < top_k; ++rank) {
            weights[rank] *= factor;
        }
    }
}

__device__ __forceinline__ uint32_t route_ordered_float(float value)
{
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

// One radix sort replaces Top-K rounds of block-wide reduction and removal.
// The 64-bit key preserves the reference ordering exactly: corrected score
// descending, then expert ID ascending for ties.
__global__ void sigmoid_route_radix_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const bool* __restrict__ mask,
    float* __restrict__ weights,
    int64_t* __restrict__ indices,
    const int experts,
    const int top_k,
    const float routed_scaling)
{
    constexpr int kThreads = 256;
    constexpr int kItems = 4;
    using Sort = cub::BlockRadixSort<
        unsigned long long,
        kThreads,
        kItems,
        int>;
    __shared__ typename Sort::TempStorage sort_storage;
    __shared__ int selected[16];
    unsigned long long keys[kItems];
    int values[kItems];
    #pragma unroll
    for (int item = 0; item < kItems; ++item) {
        const int expert = threadIdx.x * kItems + item;
        float choice = -INFINITY;
        if (expert < experts && mask[expert]) {
            const float probability =
                1.0f / (1.0f + expf(-logits[expert]));
            const float corrected = probability + bias[expert];
            choice = isfinite(corrected) ? corrected : -FLT_MAX;
        }
        const uint32_t score_key = route_ordered_float(choice);
        keys[item] =
            (static_cast<unsigned long long>(score_key) << 32) |
            static_cast<unsigned long long>(
                0xffffffffu - static_cast<uint32_t>(expert));
        values[item] = expert;
    }
    Sort(sort_storage).SortDescending(keys, values);
    #pragma unroll
    for (int item = 0; item < kItems; ++item) {
        const int rank = threadIdx.x * kItems + item;
        if (rank < top_k)
            selected[rank] = values[item];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 1.0e-20f;
        for (int rank = 0; rank < top_k; ++rank) {
            const int expert = selected[rank];
            const float probability =
                1.0f / (1.0f + expf(-logits[expert]));
            indices[rank] = expert;
            weights[rank] = isfinite(probability) ? probability : 0.0f;
            sum += weights[rank];
        }
        const float factor = routed_scaling / sum;
        for (int rank = 0; rank < top_k; ++rank)
            weights[rank] *= factor;
    }
}

void launch_sigmoid_route(
    const float* logits,
    const float* bias,
    const bool* mask,
    float* weights,
    int64_t* indices,
    const int experts,
    const int top_k,
    const float routed_scaling,
    cudaStream_t stream)
{
    const char* radix_setting = std::getenv("CCCP_ROUTE_RADIX");
    const bool use_radix = (
        radix_setting == nullptr ||
        (radix_setting[0] == '1' && radix_setting[1] == '\0'));
    if (use_radix) {
        sigmoid_route_radix_kernel<<<1, 256, 0, stream>>>(
            logits,
            bias,
            mask,
            weights,
            indices,
            experts,
            top_k,
            routed_scaling);
    } else {
        const int threads = experts < 256 ? 128 : 256;
        sigmoid_route_select_kernel<<<
            1,
            threads,
            experts * sizeof(float),
            stream>>>(
                logits,
                bias,
                mask,
                weights,
                indices,
                experts,
                top_k,
                routed_scaling);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> sigmoid_route(
    torch::Tensor logits,
    torch::Tensor bias,
    torch::Tensor mask,
    long top_k,
    double routed_scaling) {
    TORCH_CHECK(
        logits.is_cuda() && bias.is_cuda() && mask.is_cuda(),
        "sigmoid router logits/bias/mask must be CUDA");
    TORCH_CHECK(
        logits.scalar_type() == at::kFloat &&
        bias.scalar_type() == at::kFloat &&
        mask.scalar_type() == at::kBool,
        "sigmoid router logits/bias/mask dtype mismatch");
    TORCH_CHECK(
        logits.dim() == 2 && logits.size(0) == 1,
        "sigmoid router logits must be [1,E]");
    const int experts = static_cast<int>(logits.size(1));
    TORCH_CHECK(
        bias.numel() == experts && mask.numel() == experts,
        "sigmoid router bias/mask size mismatch");
    TORCH_CHECK(
        experts > 0 && experts <= 1024,
        "sigmoid router experts must be in [1,1024]");
    TORCH_CHECK(
        top_k > 0 && top_k <= 16 && top_k <= experts,
        "sigmoid router top_k must be in [1,min(16,E)]");
    TORCH_CHECK(
        logits.get_device() == bias.get_device() &&
        logits.get_device() == mask.get_device(),
        "sigmoid router tensors must be on one device");

    auto lc = logits.contiguous();
    auto bc = bias.contiguous();
    auto mc = mask.contiguous();
    auto weights = torch::empty({1, top_k}, logits.options());
    auto indices = torch::empty(
        {1, top_k},
        logits.options().dtype(at::kLong));
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_sigmoid_route(
        lc.data_ptr<float>(),
        bc.data_ptr<float>(),
        mc.data_ptr<bool>(),
        weights.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        experts,
        static_cast<int>(top_k),
        static_cast<float>(routed_scaling),
        stream);
    return {weights, indices};
}

std::vector<torch::Tensor> sigmoid_route_out(
    torch::Tensor logits,
    torch::Tensor bias,
    torch::Tensor mask,
    long top_k,
    double routed_scaling,
    torch::Tensor weights,
    torch::Tensor indices)
{
    TORCH_CHECK(
        logits.is_cuda() && bias.is_cuda() && mask.is_cuda() &&
        weights.is_cuda() && indices.is_cuda(),
        "sigmoid router tensors must be CUDA");
    TORCH_CHECK(
        logits.scalar_type() == at::kFloat &&
        bias.scalar_type() == at::kFloat &&
        mask.scalar_type() == at::kBool &&
        weights.scalar_type() == at::kFloat &&
        indices.scalar_type() == at::kLong,
        "sigmoid router output-buffer dtype mismatch");
    TORCH_CHECK(
        logits.is_contiguous() && bias.is_contiguous() &&
        mask.is_contiguous() && weights.is_contiguous() &&
        indices.is_contiguous() &&
        logits.dim() == 2 && logits.size(0) == 1,
        "sigmoid router output-buffer tensors must be contiguous");
    const int experts = static_cast<int>(logits.size(1));
    TORCH_CHECK(
        experts > 0 && experts <= 1024 &&
        bias.numel() == experts && mask.numel() == experts &&
        top_k > 0 && top_k <= 16 && top_k <= experts &&
        weights.sizes() == torch::IntArrayRef({1, top_k}) &&
        indices.sizes() == torch::IntArrayRef({1, top_k}),
        "sigmoid router output-buffer shapes do not match");
    const int device = logits.get_device();
    TORCH_CHECK(
        bias.get_device() == device &&
        mask.get_device() == device &&
        weights.get_device() == device &&
        indices.get_device() == device,
        "sigmoid router output-buffer tensors must share one device");
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_sigmoid_route(
        logits.data_ptr<float>(),
        bias.data_ptr<float>(),
        mask.data_ptr<bool>(),
        weights.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        experts,
        static_cast<int>(top_k),
        static_cast<float>(routed_scaling),
        stream);
    return {weights, indices};
}

__device__ __forceinline__ float sqrtsoftplus_route_score(
    const float logit)
{
    const float softplus = logit > 20.0f
        ? logit
        : log1pf(expf(logit));
    return sqrtf(fmaxf(softplus, 0.0f));
}

struct RouteKeyGreater {
    __device__ __forceinline__ bool operator()(
        const unsigned long long& left,
        const unsigned long long& right) const
    {
        return left > right;
    }
};

// DSV4-style sqrt(softplus) routing used to be expressed as eleven small
// ATen kernels (score, mask, Top-K, gather, normalize, and fixed-buffer
// copies) inside every captured layer graph.  One block now performs the
// exact corrected selection and publishes Graph-stable outputs directly.
__global__ void sqrtsoftplus_route_radix_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const bool* __restrict__ mask,
    float* __restrict__ weights,
    int64_t* __restrict__ indices,
    const int experts,
    const int top_k,
    const bool normalize,
    const float routed_scaling)
{
    constexpr int kThreads = 256;
    constexpr int kItems = 4;
    constexpr int kWarps = kThreads / 32;
    constexpr int kWarpTopK = 6;
    using FirstSort = cub::WarpMergeSort<
        unsigned long long, kItems, 32>;
    using FinalSort = cub::WarpMergeSort<
        unsigned long long, 2, 32>;
    __shared__ typename FirstSort::TempStorage first_storage[kWarps];
    __shared__ typename FinalSort::TempStorage final_storage;
    __shared__ unsigned long long warp_candidates[8 * 6];
    __shared__ int selected[16];
    unsigned long long keys[kItems];
    #pragma unroll
    for (int item = 0; item < kItems; ++item) {
        const int expert = threadIdx.x * kItems + item;
        float choice = -INFINITY;
        if (expert < experts && mask[expert]) {
            const float score = sqrtsoftplus_route_score(logits[expert]);
            const float corrected = score + bias[expert];
            choice = isfinite(corrected) ? corrected : -FLT_MAX;
        }
        const uint32_t score_key = route_ordered_float(choice);
        keys[item] =
            (static_cast<unsigned long long>(score_key) << 32) |
            static_cast<unsigned long long>(
                0xffffffffu - static_cast<uint32_t>(expert));
    }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    FirstSort first_stage(first_storage[warp]);
    first_stage.Sort(keys, RouteKeyGreater{});
    #pragma unroll
    for (int item = 0; item < kItems; ++item) {
        const int rank = lane * kItems + item;
        if (rank < kWarpTopK)
            warp_candidates[warp * kWarpTopK + rank] = keys[item];
    }
    __syncthreads();
    if (warp == 0) {
        unsigned long long final_keys[2];
        #pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int rank = lane * 2 + item;
            final_keys[item] = rank < kWarps * kWarpTopK
                ? warp_candidates[rank]
                : 0ull;
        }
        FinalSort final_stage(final_storage);
        final_stage.Sort(final_keys, RouteKeyGreater{});
        #pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int rank = lane * 2 + item;
            if (rank < top_k)
                selected[rank] = static_cast<int>(
                    0xffffffffu - static_cast<uint32_t>(final_keys[item]));
        }
        __syncwarp();
    }
    if (threadIdx.x == 0) {
        float sum = 1.0e-20f;
        for (int rank = 0; rank < top_k; ++rank) {
            const int expert = selected[rank];
            const float score = sqrtsoftplus_route_score(logits[expert]);
            indices[rank] = expert;
            weights[rank] = isfinite(score) ? score : 0.0f;
            sum += weights[rank];
        }
        const float factor = normalize
            ? routed_scaling / sum
            : routed_scaling;
        for (int rank = 0; rank < top_k; ++rank)
            weights[rank] *= factor;
    }
}

std::vector<torch::Tensor> sqrtsoftplus_route_out(
    torch::Tensor logits,
    torch::Tensor bias,
    torch::Tensor mask,
    long top_k,
    bool normalize,
    double routed_scaling,
    torch::Tensor weights,
    torch::Tensor indices)
{
    TORCH_CHECK(
        logits.is_cuda() && bias.is_cuda() && mask.is_cuda() &&
        weights.is_cuda() && indices.is_cuda(),
        "sqrtsoftplus router tensors must be CUDA");
    TORCH_CHECK(
        logits.scalar_type() == at::kFloat &&
        bias.scalar_type() == at::kFloat &&
        mask.scalar_type() == at::kBool &&
        weights.scalar_type() == at::kFloat &&
        indices.scalar_type() == at::kLong,
        "sqrtsoftplus router tensor dtype mismatch");
    TORCH_CHECK(
        logits.is_contiguous() && bias.is_contiguous() &&
        mask.is_contiguous() && weights.is_contiguous() &&
        indices.is_contiguous() &&
        logits.dim() == 2 && logits.size(0) == 1,
        "sqrtsoftplus router tensors must be contiguous and logits [1,E]");
    const int experts = static_cast<int>(logits.size(1));
    TORCH_CHECK(
        experts > 0 && experts <= 1024 &&
        bias.numel() == experts && mask.numel() == experts &&
        top_k > 0 && top_k <= 16 && top_k <= experts &&
        weights.sizes() == torch::IntArrayRef({1, top_k}) &&
        indices.sizes() == torch::IntArrayRef({1, top_k}),
        "sqrtsoftplus router buffer shapes do not match");
    const int device = logits.get_device();
    TORCH_CHECK(
        bias.get_device() == device && mask.get_device() == device &&
        weights.get_device() == device && indices.get_device() == device,
        "sqrtsoftplus router tensors must share one device");
    auto stream = at::cuda::getCurrentCUDAStream();
    sqrtsoftplus_route_radix_kernel<<<1, 256, 0, stream>>>(
        logits.data_ptr<float>(),
        bias.data_ptr<float>(),
        mask.data_ptr<bool>(),
        weights.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        experts,
        static_cast<int>(top_k),
        normalize,
        static_cast<float>(routed_scaling));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {weights, indices};
}

template <typename input_t, int rows_per_block>
__global__ void linear_route_logits_kernel(
    const input_t* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ logits,
    const int experts,
    const int hidden)
{
    extern __shared__ float shared_input[];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    for (
        int column = linear_thread;
        column < hidden;
        column += 32 * rows_per_block
    )
        shared_input[column] = vq_scalar_to_float(input + column);
    __syncthreads();
    const int expert =
        static_cast<int>(blockIdx.x) * rows_per_block + threadIdx.y;
    if (expert >= experts)
        return;
    const float* row =
        weight + static_cast<long>(expert) * hidden;
    float value = 0.0f;
    for (int column = lane; column < hidden; column += 32)
        value = __fmaf_rn(
            shared_input[column],
            __ldg(row + column),
            value);
    value = warp_sum_f32(value);
    if (lane == 0)
        logits[expert] = value;
}

std::vector<torch::Tensor> linear_sigmoid_route_out(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor mask,
    long top_k,
    double routed_scaling,
    torch::Tensor logits,
    torch::Tensor weights,
    torch::Tensor indices)
{
    TORCH_CHECK(
        input.is_cuda() && weight.is_cuda() &&
        bias.is_cuda() && mask.is_cuda() &&
        logits.is_cuda() && weights.is_cuda() && indices.is_cuda(),
        "linear sigmoid router tensors must be CUDA");
    TORCH_CHECK(
        (
            input.scalar_type() == at::kBFloat16 ||
            input.scalar_type() == at::kFloat
        ) &&
        weight.scalar_type() == at::kFloat &&
        bias.scalar_type() == at::kFloat &&
        mask.scalar_type() == at::kBool &&
        logits.scalar_type() == at::kFloat &&
        weights.scalar_type() == at::kFloat &&
        indices.scalar_type() == at::kLong,
        "linear sigmoid router dtype mismatch");
    TORCH_CHECK(
        input.is_contiguous() && weight.is_contiguous() &&
        bias.is_contiguous() && mask.is_contiguous() &&
        logits.is_contiguous() && weights.is_contiguous() &&
        indices.is_contiguous() &&
        input.dim() == 2 && input.size(0) == 1 &&
        weight.dim() == 2,
        "linear sigmoid router tensors must be contiguous matrices");
    const int experts = static_cast<int>(weight.size(0));
    const int hidden = static_cast<int>(weight.size(1));
    TORCH_CHECK(
        input.size(1) == hidden &&
        experts > 0 && experts <= 1024 &&
        bias.numel() == experts && mask.numel() == experts &&
        logits.sizes() == torch::IntArrayRef({1, experts}) &&
        top_k > 0 && top_k <= 16 && top_k <= experts &&
        weights.sizes() == torch::IntArrayRef({1, top_k}) &&
        indices.sizes() == torch::IntArrayRef({1, top_k}),
        "linear sigmoid router shapes do not match");
    const int device = input.get_device();
    TORCH_CHECK(
        weight.get_device() == device &&
        bias.get_device() == device &&
        mask.get_device() == device &&
        logits.get_device() == device &&
        weights.get_device() == device &&
        indices.get_device() == device,
        "linear sigmoid router tensors must share one device");
    constexpr int rows_per_block = 32;
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 block(32, rows_per_block);
    const int grid = (experts + rows_per_block - 1) / rows_per_block;
    const size_t shared_bytes =
        static_cast<size_t>(hidden) * sizeof(float);
    if (input.scalar_type() == at::kBFloat16) {
        linear_route_logits_kernel<
            __nv_bfloat16,
            rows_per_block><<<grid, block, shared_bytes, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    input.data_ptr<at::BFloat16>()),
                weight.data_ptr<float>(),
                logits.data_ptr<float>(),
                experts,
                hidden);
    } else {
        linear_route_logits_kernel<
            float,
            rows_per_block><<<grid, block, shared_bytes, stream>>>(
                input.data_ptr<float>(),
                weight.data_ptr<float>(),
                logits.data_ptr<float>(),
                experts,
                hidden);
    }
    launch_sigmoid_route(
        logits.data_ptr<float>(),
        bias.data_ptr<float>(),
        mask.data_ptr<bool>(),
        weights.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        experts,
        static_cast<int>(top_k),
        static_cast<float>(routed_scaling),
        stream);
    return {weights, indices};
}

__global__ void paged_gather_bf16_kernel(
    const int64_t* __restrict__ page_ptrs,
    const int64_t* __restrict__ indices,
    __nv_bfloat16* __restrict__ output,
    int64_t items,
    int page_items,
    int dim) {
    const int64_t linear =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = items * static_cast<int64_t>(dim);
    if (linear >= total) {
        return;
    }
    const int64_t output_item = linear / dim;
    const int feature = static_cast<int>(linear % dim);
    const int64_t source_item = indices[output_item];
    const int64_t page_index = source_item / page_items;
    const int64_t page_offset = source_item % page_items;
    const auto* page = reinterpret_cast<const __nv_bfloat16*>(
        page_ptrs[page_index]);
    output[linear] = page[
        page_offset * static_cast<int64_t>(dim) + feature
    ];
}

torch::Tensor paged_gather_bf16(
    torch::Tensor page_ptrs,
    torch::Tensor indices,
    long page_items,
    long dim) {
    TORCH_CHECK(page_ptrs.is_cuda() && indices.is_cuda(),
                "page_ptrs and indices must be CUDA");
    TORCH_CHECK(page_ptrs.scalar_type() == at::kLong,
                "page_ptrs must be int64");
    TORCH_CHECK(indices.scalar_type() == at::kLong,
                "indices must be int64");
    TORCH_CHECK(page_ptrs.dim() == 1 && page_ptrs.numel() > 0,
                "page_ptrs must be a non-empty vector");
    TORCH_CHECK(page_items > 0 && dim > 0,
                "page_items and dim must be positive");

    auto pc = page_ptrs.contiguous();
    auto ic = indices.contiguous().view({-1});
    auto output = torch::empty(
        {ic.numel(), dim},
        ic.options().dtype(at::kBFloat16));
    if (ic.numel() == 0) {
        return output;
    }
    constexpr int threads = 256;
    const int64_t total = ic.numel() * dim;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    paged_gather_bf16_kernel<<<blocks, threads, 0, stream>>>(
        pc.data_ptr<int64_t>(),
        ic.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(
            output.data_ptr<at::BFloat16>()),
        ic.numel(),
        static_cast<int>(page_items),
        static_cast<int>(dim));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void hadamard_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int width,
    float scale) {
    extern __shared__ float values[];
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const int64_t base = static_cast<int64_t>(row) * width;
    if (lane < width) {
        values[lane] = __bfloat162float(input[base + lane]);
    }
    __syncthreads();

    for (int span = 1; span < width; span <<= 1) {
        if (lane < width / 2) {
            const int group = lane / span;
            const int offset = lane - group * span;
            const int left = group * (span << 1) + offset;
            const int right = left + span;
            const float a = values[left];
            const float b = values[right];
            values[left] = a + b;
            values[right] = a - b;
        }
        __syncthreads();
    }
    if (lane < width) {
        output[base + lane] = __float2bfloat16_rn(values[lane] * scale);
    }
}

torch::Tensor hadamard_bf16(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16,
                "input must be bfloat16");
    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
    auto x = input.contiguous();
    const int width = static_cast<int>(x.size(-1));
    TORCH_CHECK(
        width > 0 && width <= 256 && (width & (width - 1)) == 0,
        "last dimension must be a power of two up to 256");
    const int64_t rows = x.numel() / width;
    auto output = torch::empty_like(x);
    if (rows == 0) {
        return output.view(input.sizes());
    }
    const float scale = static_cast<float>(
        1.0 / std::sqrt(static_cast<double>(width)));
    auto stream = at::cuda::getCurrentCUDAStream();
    hadamard_bf16_kernel<<<
        static_cast<int>(rows),
        width,
        static_cast<size_t>(width) * sizeof(float),
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            width,
            scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output.view(input.sizes());
}

// ---- Direct packed INT4-G64 GEMV for single-token decode ----
// Mirrors the USSR VQ kernel layout: one CTA stages the activation row once,
// then eight warps consume packed weights directly. No floating-point weight
// matrix is materialized. Each warp loads one G64 scale, broadcasts it, and
// unpacks one coalesced 32-byte group into registers.

constexpr int INT4_ROWS_PER_BLOCK = 32;

__global__ void int4_embedding_lookup_kernel(
    const uint8_t* __restrict__ packed_row,
    const __half* __restrict__ scale_row,
    float* __restrict__ output,
    int cols)
{
    for (
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        col < cols;
        col += blockDim.x * gridDim.x
    ) {
        const uint8_t code = __ldg(packed_row + col / 2);
        const int quantized = (col & 1)
            ? static_cast<int>(code >> 4) - 8
            : static_cast<int>(code & 15) - 8;
        const float scale = __half2float(
            __ldg(scale_row + col / 64));
        output[col] = __fmul_rn(
            static_cast<float>(quantized),
            scale);
    }
}

__global__ void int4_embedding_lookup_device_row_kernel(
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scales,
    const int64_t* __restrict__ row_ptr,
    float* __restrict__ output,
    int rows,
    int packed_cols,
    int scale_cols,
    int cols)
{
    const int64_t row = row_ptr[0];
    if (row < 0 || row >= rows)
        return;
    const uint8_t* packed_row =
        packed + row * packed_cols;
    const __half* scale_row =
        scales + row * scale_cols;
    for (
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        col < cols;
        col += blockDim.x * gridDim.x
    ) {
        const uint8_t code = __ldg(packed_row + col / 2);
        const int quantized = (col & 1)
            ? static_cast<int>(code >> 4) - 8
            : static_cast<int>(code & 15) - 8;
        const float scale = __half2float(
            __ldg(scale_row + col / 64));
        output[col] = __fmul_rn(
            static_cast<float>(quantized),
            scale);
    }
}

torch::Tensor int4_embedding_lookup(
    torch::Tensor packed,
    torch::Tensor scales,
    long row,
    long cols,
    long group_size,
    c10::optional<torch::Tensor> output_buffer)
{
    TORCH_CHECK(
        packed.is_cuda() && scales.is_cuda(),
        "INT4 embedding weights must be CUDA");
    TORCH_CHECK(
        packed.scalar_type() == at::kByte &&
        scales.scalar_type() == at::kHalf &&
        packed.is_contiguous() &&
        scales.is_contiguous() &&
        packed.dim() == 2 &&
        scales.dim() == 2,
        "INT4 embedding weights must be contiguous uint8/float16 matrices");
    TORCH_CHECK(
        group_size == 64 &&
        cols > 0 &&
        cols % 64 == 0 &&
        packed.size(1) * 2 == cols &&
        scales.size(0) == packed.size(0) &&
        scales.size(1) == cols / group_size &&
        row >= 0 &&
        row < packed.size(0),
        "INT4 embedding shape, group size or row is invalid");
    const int device = packed.get_device();
    TORCH_CHECK(
        scales.get_device() == device,
        "INT4 embedding tensors must share one device");
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty(
            {1, cols},
            scales.options().dtype(at::kFloat));
    TORCH_CHECK(
        output.is_cuda() &&
        output.scalar_type() == at::kFloat &&
        output.is_contiguous() &&
        output.sizes() == torch::IntArrayRef({1, cols}) &&
        output.get_device() == device,
        "INT4 embedding output must be contiguous float32 [1,cols]");
    const int threads = 256;
    const int blocks = std::min(
        32,
        (static_cast<int>(cols) + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    int4_embedding_lookup_kernel<<<
        blocks,
        threads,
        0,
        stream>>>(
            packed.data_ptr<uint8_t>() +
                row * packed.size(1),
            reinterpret_cast<const __half*>(
                scales.data_ptr<at::Half>()) +
                row * scales.size(1),
            output.data_ptr<float>(),
            static_cast<int>(cols));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor int4_embedding_lookup_device_row(
    torch::Tensor packed,
    torch::Tensor scales,
    torch::Tensor row,
    long cols,
    long group_size,
    c10::optional<torch::Tensor> output_buffer)
{
    TORCH_CHECK(
        packed.is_cuda() &&
        scales.is_cuda() &&
        row.is_cuda(),
        "device-row INT4 embedding inputs must be CUDA");
    TORCH_CHECK(
        packed.scalar_type() == at::kByte &&
        scales.scalar_type() == at::kHalf &&
        row.scalar_type() == at::kLong &&
        packed.is_contiguous() &&
        scales.is_contiguous() &&
        row.is_contiguous() &&
        packed.dim() == 2 &&
        scales.dim() == 2 &&
        row.numel() == 1,
        "device-row INT4 embedding input layouts do not match");
    TORCH_CHECK(
        group_size == 64 &&
        cols > 0 &&
        cols % 64 == 0 &&
        packed.size(1) * 2 == cols &&
        scales.size(0) == packed.size(0) &&
        scales.size(1) == cols / group_size,
        "device-row INT4 embedding shapes do not match");
    const int device = packed.get_device();
    TORCH_CHECK(
        scales.get_device() == device &&
        row.get_device() == device,
        "device-row INT4 embedding inputs must share one device");
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty(
            {1, cols},
            scales.options().dtype(at::kFloat));
    TORCH_CHECK(
        output.is_cuda() &&
        output.scalar_type() == at::kFloat &&
        output.is_contiguous() &&
        output.sizes() == torch::IntArrayRef({1, cols}) &&
        output.get_device() == device,
        "device-row INT4 embedding output must be float32 [1,cols]");
    const int threads = 256;
    const int blocks = std::min(
        32,
        (static_cast<int>(cols) + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    int4_embedding_lookup_device_row_kernel<<<
        blocks,
        threads,
        0,
        stream>>>(
            packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(
                scales.data_ptr<at::Half>()),
            row.data_ptr<int64_t>(),
            output.data_ptr<float>(),
            static_cast<int>(packed.size(0)),
            static_cast<int>(packed.size(1)),
            static_cast<int>(scales.size(1)),
            static_cast<int>(cols));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


// ---------------------------------------------------------------------------
// INT4 GEMV vector4-segmented — the measured best of both worlds:
// v1 vector4's conflict-free lane layout (columns stride-2 inside 64-col
// groups, 8 lanes per group, one shfl scale per subgroup) but the
// activation is staged per 4096-column segment instead of the whole row,
// cutting shared staging traffic ~4x for wide matrices.  Segments combine
// through the v2 partial/reduce pair.  All-architecture LDG + FMA only.
// ---------------------------------------------------------------------------
__global__ void int4_gemv_v2_reduce_kernel(
    const float* __restrict__ partial,
    float* __restrict__ output,
    const int rows,
    const int segments);

constexpr int kInt4V4SegCols = 4096;  // 64 groups per segment
constexpr int kInt4V4Warps = 8;       // rows per block

template <typename input_t>
__global__ void int4_gemv_packed_f32_v4s_kernel(
    const input_t* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scales,
    float* __restrict__ partial,
    const int rows,
    const int cols,
    const int groups,
    const int segments)
{
    __shared__ float shared_x[kInt4V4SegCols];
    const int lane = threadIdx.x;
    const int row = blockIdx.y * kInt4V4Warps + threadIdx.y;
    const int seg = blockIdx.x;
    const int col_base = seg * kInt4V4SegCols;
    const int seg_cols = min(kInt4V4SegCols, cols - col_base);
    if (seg_cols <= 0) return;
    const int seg_groups = seg_cols >> 6;

    // Stage only this segment's activation (16 KiB), 256 threads.
    for (int c = threadIdx.y * 32 + lane; c < seg_cols; c += 256) {
        shared_x[c] = vq_scalar_to_float(x + (long)(col_base + c));
    }
    __syncthreads();
    if (row >= rows) return;

    const int packed_cols = cols >> 1;
    const uint8_t* packed_row =
        packed + (long)row * packed_cols + (col_base >> 1);
    const __half* scale_row = scales + (long)row * groups;

    // v1 vector4 inner loop restricted to this segment's groups.
    const int group_in_iteration = lane >> 3;
    const int group_lane = lane & 7;
    float acc0 = 0.f, acc1 = 0.f;
    const int group_span = kInt4V4SegCols >> 6;  // 64
    for (int gbase = seg * group_span;
         gbase < seg * group_span + seg_groups; gbase += 8) {
        const int g0 = gbase + group_in_iteration;
        const int g1 = gbase + 4 + group_in_iteration;
        float s0 = 0.f, s1 = 0.f;
        if (group_lane == 0) {
            if (g0 < seg * group_span + seg_groups)
                s0 = __half2float(__ldg(scale_row + g0));
            if (g1 < seg * group_span + seg_groups)
                s1 = __half2float(__ldg(scale_row + g1));
        }
        s0 = __shfl_sync(0xffffffffu, s0, 0, 8);
        s1 = __shfl_sync(0xffffffffu, s1, 0, 8);
        if (g0 < seg * group_span + seg_groups) {
            const uint32_t codes = __ldg(reinterpret_cast<const uint32_t*>(
                packed_row + (g0 - seg * group_span) * 32 + group_lane * 4));
            const int col = g0 * 64 + group_lane * 8 - col_base;
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&codes);
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const uint8_t q = bytes[item];
                acc0 = __fmaf_rn(
                    static_cast<float>((q & 15) - 8) * s0,
                    shared_x[col + item * 2], acc0);
                acc0 = __fmaf_rn(
                    static_cast<float>((q >> 4) - 8) * s0,
                    shared_x[col + item * 2 + 1], acc0);
            }
        }
        if (g1 < seg * group_span + seg_groups) {
            const uint32_t codes = __ldg(reinterpret_cast<const uint32_t*>(
                packed_row + (g1 - seg * group_span) * 32 + group_lane * 4));
            const int col = g1 * 64 + group_lane * 8 - col_base;
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&codes);
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const uint8_t q = bytes[item];
                acc1 = __fmaf_rn(
                    static_cast<float>((q & 15) - 8) * s1,
                    shared_x[col + item * 2], acc1);
                acc1 = __fmaf_rn(
                    static_cast<float>((q >> 4) - 8) * s1,
                    shared_x[col + item * 2 + 1], acc1);
            }
        }
    }
    float acc = acc0 + acc1;
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        partial[(long)row * segments + seg] = acc;
    }
}

template <typename input_t>
torch::Tensor int4_gemv_packed_f32_v4s(
    torch::Tensor x,
    torch::Tensor packed,
    torch::Tensor scales,
    int64_t rows,
    int64_t cols,
    int64_t groups)
{
    TORCH_CHECK(cols % 64 == 0, "v4s requires 64-column G64 groups");
    auto stream = at::cuda::getCurrentCUDAStream();
    const int segments =
        (int)((cols + kInt4V4SegCols - 1) / kInt4V4SegCols);
    static torch::Tensor partial_cache;
    const long needed = (long)rows * segments;
    if (!partial_cache.defined() ||
        partial_cache.device() != x.device() ||
        partial_cache.numel() < needed) {
        partial_cache = torch::empty(
            {needed},
            torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    }
    torch::Tensor partial = partial_cache.narrow(0, 0, needed);
    auto output = torch::empty(
        {rows},
        torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    dim3 block(32, kInt4V4Warps);
    dim3 grid(segments, (unsigned)((rows + kInt4V4Warps - 1) / kInt4V4Warps));
    int4_gemv_packed_f32_v4s_kernel<input_t><<<grid, block, 0, stream>>>(
        reinterpret_cast<const input_t*>(x.data_ptr()),
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr()),
        partial.data_ptr<float>(),
        (int)rows, (int)cols, (int)groups, segments);
    int4_gemv_v2_reduce_kernel<<<(unsigned)rows, 32, 0, stream>>>(
        partial.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)rows, segments);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__device__ __forceinline__ uint32_t half2_bits_m(__half2 v)
{
    return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t h2_m(__half lo, __half hi)
{
    uint32_t v;
    asm("mov.b32 %0, {%1, %2};" : "=r"(v) : "h"(__half_as_ushort(lo)),
        "h"(__half_as_ushort(hi)));
    return v;
}
__device__ __forceinline__ void mma_m16n8k16_m(
    const uint32_t* a, const uint32_t* b, float* c0, float* c1)
{
    float c[4] = {*c0, *c1, 0.f, 0.f};
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    *c0 = c[0];
    *c1 = c[1];
}
__global__ void marlin_reduce_kernel_g(
    const float* __restrict__ partial,
    float* __restrict__ output,
    int rows,
    int slices)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float acc = 0.f;
    for (int s = 0; s < slices; ++s) acc += partial[(long)row * slices + s];
    output[row] = acc;
}

// ---------------------------------------------------------------------------
// INT4 GEMV v17 — marlin superstep tiles + FMA + shared-x (SM75+).
// Same repacked layout as int4_gemv_marlin (one uint32 per lane per
// 32-column superstep), tensor-core path replaced by eight FMAs against
// conflict-free shared-x; per-4-lane-group reduction (one group = one row).
// Measured 731 GB/s @5120x17408, 1000 GB/s @248320x5120 on H20.
// ---------------------------------------------------------------------------
__global__ void int4_repack_v21_kernel(
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

constexpr int V21_SLICE = 2048;
constexpr int V21_SMALL_SLICE = 512;

// SLICE 模板化：小矩阵(k<=5120)用 512 列细切片,块数 x4 改善
// ramp/占用(第二十轮处方,584→750+GB/s);大矩阵维持 2048。
template <int SLICE>
__global__ void int4_gemv_v21_kernel(
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
    const int k0 = slice * SLICE;
    const int here = min(SLICE, cols - k0);
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


__global__ void int4_v21_reduce(
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

torch::Tensor int4_repack_v21(torch::Tensor packed, int64_t rows, int64_t cols)
{
    const long words = (((long)rows + 7) / 8) * (cols / 128) * 128;
    auto dst = torch::empty(
        {words * 4},
        torch::TensorOptions().dtype(torch::kUInt8).device(packed.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (int)((words + 255) / 256 > 4096 ? 4096 : (words + 255) / 256);
    int4_repack_v21_kernel<<<blocks, 256, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<uint32_t*>(dst.data_ptr()),
        (int)rows, (int)cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dst;
}

torch::Tensor int4_gemv_v21(
    torch::Tensor x,
    torch::Tensor repacked,
    torch::Tensor scales,
    int64_t rows,
    int64_t cols,
    int64_t groups)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    // 小矩阵 k 细切片:块数 x4 改善 ramp/占用。auto=按 cols 选,
    // 可用 CCCP_INT4_V21_KSLICE=512/2048 强制(A/B 用)。
    static const int slice_env = [] {
        const char* flag = std::getenv("CCCP_INT4_V21_KSLICE");
        if (flag && flag[0] == '5') return 512;
        if (flag && flag[0] == '2') return 2048;
        return 0;  // auto
    }();
    const bool small = slice_env
        ? (slice_env == 512)
        : (cols <= 5120);
    const int slice_cols = small ? V21_SMALL_SLICE : V21_SLICE;
    const int slices = (int)((cols + slice_cols - 1) / slice_cols);
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
    if (small) {
        int4_gemv_v21_kernel<V21_SMALL_SLICE><<<
            grid, 128, V21_SMALL_SLICE * sizeof(float), stream>>>(
            x.data_ptr<float>(),
            reinterpret_cast<const uint32_t*>(repacked.data_ptr()),
            reinterpret_cast<const __half*>(scales.data_ptr()),
            partial.data_ptr<float>(), (int)rows, (int)cols, (int)groups, slices);
    } else {
        int4_gemv_v21_kernel<V21_SLICE><<<
            grid, 128, V21_SLICE * sizeof(float), stream>>>(
            x.data_ptr<float>(),
            reinterpret_cast<const uint32_t*>(repacked.data_ptr()),
            reinterpret_cast<const __half*>(scales.data_ptr()),
            partial.data_ptr<float>(), (int)rows, (int)cols, (int)groups, slices);
    }
    int4_v21_reduce<<<(rows + 255) / 256, 256, 0, stream>>>(
        partial.data_ptr<float>(), output.data_ptr<float>(), (int)rows, slices);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void int4_repack_v21b_kernel(
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

constexpr int V21B_SLICE = 2048;
constexpr int V21B_MAXB = 6;  // draft5+首token=6;verify 成本对 B 平坦(权重单流)

// 与 v21 相同的 SLICE 模板化(小矩阵 512):v21b 必须与 v21 的 k 分组树
// 完全一致才保持 bit 一等(第二十二轮实证:分组不一致时 7e-7 差异会
// 翻转 MTP fast-top3 接受判定)。
template <int SLICE>
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
    extern __shared__ float sxB[];        // [V21B_MAXB][SLICE]
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int slice = blockIdx.x;
    const int k0 = slice * SLICE;
    const int here = min(SLICE, cols - k0);
    if (here <= 0) return;
    for (int idx = threadIdx.x; idx < B * here; idx += 128) {
        const int b = idx / here;
        const int c = idx % here;
        sxB[b * SLICE + c] = x[(long)b * cols + k0 + c];
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

    float acc[V21B_MAXB][32];
#pragma unroll
    for (int b = 0; b < V21B_MAXB; ++b)
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
            for (int b = 0; b < V21B_MAXB; ++b) {
                if (b >= B) break;
                const float* xb = sxB + b * SLICE;
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
            for (int b = 0; b < V21B_MAXB; ++b) {
                if (b >= B) break;
                const float* xb = sxB + b * SLICE;
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
    for (int b = 0; b < V21B_MAXB; ++b) {
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

__global__ void int4_v21b_reduce(
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

torch::Tensor int4_repack_v21b(torch::Tensor packed, int64_t rows, int64_t cols)
{
    const long words = (((long)rows + 7) / 8) * (cols / 128) * 128;
    auto dst = torch::empty(
        {words * 4},
        torch::TensorOptions().dtype(torch::kUInt8).device(packed.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (int)((words + 255) / 256 > 4096 ? 4096 : (words + 255) / 256);
    int4_repack_v21b_kernel<<<blocks, 256, 0, stream>>>(
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
    TORCH_CHECK(B >= 1 && B <= V21B_MAXB, "batch 1..5");
    auto stream = at::cuda::getCurrentCUDAStream();
    // 与 int4_gemv_v21 完全一致的 slice 选择规则,保证 draft(v21)与
    // verify(v21b)的 k 分组树逐位同构。
    static const int slice_env_b = [] {
        const char* flag = std::getenv("CCCP_INT4_V21_KSLICE");
        if (flag && flag[0] == '5') return 512;
        if (flag && flag[0] == '2') return 2048;
        return 0;  // auto
    }();
    const bool small_b = slice_env_b
        ? (slice_env_b == 512)
        : (cols <= 5120);
    const int slice_cols = small_b ? V21_SMALL_SLICE : V21B_SLICE;
    const int slices = (int)((cols + slice_cols - 1) / slice_cols);
    // B=6 x 2048 x 4B = 48KB 恰在默认上限,显式 opt-in 避免边界拒绝。
    static size_t configured_b = 0;
    const size_t shared_b =
        (size_t)V21B_MAXB * slice_cols * sizeof(float);
    if (configured_b < shared_b) {
        cudaFuncSetAttribute(
            int4_gemv_v21b_kernel<V21_SMALL_SLICE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shared_b);
        cudaFuncSetAttribute(
            int4_gemv_v21b_kernel<V21B_SLICE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shared_b);
        configured_b = shared_b;
    }
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
    const size_t shared =
        (size_t)V21B_MAXB * slice_cols * sizeof(float);
    if (small_b) {
        int4_gemv_v21b_kernel<V21_SMALL_SLICE><<<grid, 128, shared, stream>>>(
            x.data_ptr<float>(),
            reinterpret_cast<const uint32_t*>(repacked.data_ptr()),
            reinterpret_cast<const __half*>(scales.data_ptr()),
            partial.data_ptr<float>(),
            (int)rows, (int)cols, (int)groups, slices, B);
    } else {
        int4_gemv_v21b_kernel<V21B_SLICE><<<grid, 128, shared, stream>>>(
            x.data_ptr<float>(),
            reinterpret_cast<const uint32_t*>(repacked.data_ptr()),
            reinterpret_cast<const __half*>(scales.data_ptr()),
            partial.data_ptr<float>(),
            (int)rows, (int)cols, (int)groups, slices, B);
    }
    int4_v21b_reduce<<<(B * rows + 255) / 256, 256, 0, stream>>>(
        partial.data_ptr<float>(), output.data_ptr<float>(),
        (int)rows, slices, B);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

constexpr int V1B_MAXB = 6;
constexpr int V1B_SLICE = 4096;   // per-slice shared: B*4096*4B <= 80KB

// 命名避开文件头部 #define ROWS_PER_BLOCK 32（宏会让模板参数被展开成字面量）。
// 逐 token 路径实际走 vector4 kernel（groups%4==0 时）：每 lane 负责一个
// group（lane>>3），组内 4 字节连乘，warp_sum_f32 归约。v1b 必须逐位复刻
// 该求和顺序，MTP fast-top3 的严格贪心比较才不被舍入差异翻转。
template <int V1B_RPB>
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
    const int row = blockIdx.x * V1B_RPB + threadIdx.y;
    if (row >= rows) return;
    const int packed_cols = cols >> 1;
    const uint8_t* qrow = packed + (long)row * packed_cols;
    const __half* srow = scales + (long)row * groups;
    const int slice_groups = V1B_SLICE >> 6;   // 64（64%4==0，组对齐保持）
    const int slices = (cols + V1B_SLICE - 1) / V1B_SLICE;
    const int group_in_iteration = lane >> 3;
    const int group_lane = lane & 7;

    float accs[V1B_MAXB];
#pragma unroll
    for (int b = 0; b < V1B_MAXB; ++b) accs[b] = 0.f;

    for (int slice = 0; slice < slices; ++slice) {
        const int g0 = slice * slice_groups;
        const int g1 = min(groups, g0 + slice_groups);
        const int here = (min(cols, (g1 << 6)) - (g0 << 6));
        // Stage this slice's activations for all B rows.  增量式推进
        // b/k，避免每个元素做除法/取模。
        const int stride = 32 * V1B_RPB;
        const int linear = threadIdx.y * 32 + lane;
        for (int c = linear; c < B * here; c += stride) {
            int b = c / here;
            int k = c - b * here;
            sx[b * V1B_SLICE + k] = __ldg(
                x + (long)b * cols + (g0 << 6) + k);
        }
        __syncthreads();
        for (int group_base = g0; group_base < g1; group_base += 4) {
            // 尾部不足 4 组时：越界段 scale=0 且 group/col 夹回有效范围，
            // 使 fma 贡献恰为 0，同时保证 __shfl_sync 全 warp 参与。
            const bool live =
                group_base + group_in_iteration < g1;
            const int group = live
                ? group_base + group_in_iteration
                : g1 - 1;
            float scale = group_lane == 0 && live
                ? __half2float(srow[group])
                : 0.f;
            scale = __shfl_sync(0xffffffffu, scale, 0, 8);
            const uint32_t codes = __ldg(
                reinterpret_cast<const uint32_t*>(
                    qrow + group * 32 + group_lane * 4));
            const int col_begin =
                group * 64 + group_lane * 8 - (g0 << 6);
            // 先预取本组全部 x 向量（10×float4），再进 FMA 链——
            // 打断 LDS→FFMA 的串行依赖。
            float4 xbuf[2][V1B_MAXB];
#pragma unroll
            for (int half4 = 0; half4 < 2; ++half4) {
                const int col = col_begin + half4 * 4;
#pragma unroll
                for (int b = 0; b < V1B_MAXB; ++b) {
                    xbuf[half4][b] = *reinterpret_cast<const float4*>(
                        sx + b * V1B_SLICE + col);
                }
            }
#pragma unroll
            for (int half4 = 0; half4 < 2; ++half4) {
                const uint16_t pair = static_cast<uint16_t>(
                    codes >> (half4 * 16));
                const uint8_t byte_a = static_cast<uint8_t>(pair);
                const uint8_t byte_b = static_cast<uint8_t>(pair >> 8);
                // 乘数顺序与 vector4 的 item 循环一致：c0..c7 逐列推进。
                const float m0 = __fmul_rn(
                    static_cast<float>((byte_a & 15) - 8), scale);
                const float m1 = __fmul_rn(
                    static_cast<float>((byte_a >> 4) - 8), scale);
                const float m2 = __fmul_rn(
                    static_cast<float>((byte_b & 15) - 8), scale);
                const float m3 = __fmul_rn(
                    static_cast<float>((byte_b >> 4) - 8), scale);
                // b 维全程无分支（B<5 时多算的槽位不写回）。
#pragma unroll
                for (int b = 0; b < V1B_MAXB; ++b) {
                    const float4 xv = xbuf[half4][b];
                    accs[b] = __fmaf_rn(m0, xv.x, accs[b]);
                    accs[b] = __fmaf_rn(m1, xv.y, accs[b]);
                    accs[b] = __fmaf_rn(m2, xv.z, accs[b]);
                    accs[b] = __fmaf_rn(m3, xv.w, accs[b]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int b = 0; b < V1B_MAXB; ++b) {
        float acc = accs[b];
        acc += __shfl_down_sync(0xffffffffu, acc, 16);
        acc += __shfl_down_sync(0xffffffffu, acc, 8);
        acc += __shfl_down_sync(0xffffffffu, acc, 4);
        acc += __shfl_down_sync(0xffffffffu, acc, 2);
        acc += __shfl_down_sync(0xffffffffu, acc, 1);
        if (lane == 0 && b < B) {
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
    // 固定按 V1B_MAXB 槽位分配：b>=B 的槽位读未写共享（结果不写回），
    // 换取计算段无分支；80KB 仍允许 2 block/SM。
    const size_t shared = (size_t)V1B_MAXB * V1B_SLICE * sizeof(float);
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

// ── v30: llama.cpp MMVQ 思路移植 ─────────────────────────────────
// Q8 激活(int8+每 64 列组 fp32 尺度)× int4_g64 权重(原生行主序,不
// repack 不拷贝),dp4a 整数点积,组内 int32 精确累加,每 32 列×B 一次
// 浮点尺度合成。对照 v21/vector4 的逐 nibble I2FP+FMUL(计算段 4211
// cycles 的来源),整数路径把每权重字节的算术密度降 ~4x;权重驻留
// 0.5B/权重,显存不增。通道对齐:字节 j 的 lo/hi nibble = 列 2j/2j+1,
// Q8 激活按原生列序输出即天然对齐(第十六轮 dp4a 证伪的错位是 v21
// 交织布局所致,此处不复存在)。
// v30b: uint4 宽载(32 列/lane),尺度合成降频 4x,xs 驻 shared。
constexpr int V30_MAXB = 6;

template <int V30_RPB>
__global__ void int4_gemv_v30_kernel(
    const int32_t* __restrict__ xq,       // [B, cols] int8x4 打包
    const float* __restrict__ xs,         // [B, groups]
    const uint8_t* __restrict__ packed,   // [rows, cols/2]
    const __half* __restrict__ scales,    // [rows, groups]
    float* __restrict__ output,           // [B, rows]
    const int rows,
    const int cols,
    const int groups,
    const int B)
{
    extern __shared__ float sx8[];        // [V30_MAXB][groups]
    const int lane = threadIdx.x;
    const int row = blockIdx.x * V30_RPB + threadIdx.y;
    const int stride = 32 * V30_RPB;
    for (int i = threadIdx.y * 32 + lane; i < B * groups; i += stride) {
        sx8[i] = __ldg(xs + i);
    }
    __syncthreads();
    if (row >= rows) return;
    const uint8_t* qrow = packed + (long)row * (cols >> 1);
    const __half* srow = scales + (long)row * groups;
    // 每_lane 负责一个组的 16 字节半段(32 列):warp 每迭代推进 16 组。
    // pair(lane&~1)共享组 scale,width-2 shuffle 广播。
    const int gsub = lane >> 1;
    const int ghalf = lane & 1;

    float accs[V30_MAXB];
#pragma unroll
    for (int b = 0; b < V30_MAXB; ++b) accs[b] = 0.f;

    for (int g0 = 0; g0 < groups; g0 += 16)
    {
        const int g = g0 + gsub;
        const bool live = g < groups;
        float ws = (ghalf == 0 && live)
            ? __half2float(__ldg(srow + g))
            : 0.f;
        ws = __shfl_sync(0xffffffffu, ws, lane & ~1, 2);
        if (!live) continue;
        const uint4 cw = *reinterpret_cast<const uint4*>(
            qrow + g * 32 + ghalf * 16);
        // 16 字节 → 8 个 int8x4 通道字;字节 k 的 lo/hi = 列 2k/2k+1。
        uint32_t wv[8];
        const uint32_t bytes[4] = {cw.x, cw.y, cw.z, cw.w};
#pragma unroll
        for (int h = 0; h < 4; ++h)
        {
            const uint32_t v = bytes[h];
            const uint32_t k0 = v & 0xFF, k1 = (v >> 8) & 0xFF;
            const uint32_t k2 = (v >> 16) & 0xFF, k3 = (v >> 24) & 0xFF;
            wv[h * 2] =
                (((k0 & 15) - 8) & 0xFF) |
                ((((k0 >> 4) - 8) & 0xFF) << 8) |
                (((k1 & 15) - 8) & 0xFF) << 16 |
                ((((k1 >> 4) - 8) & 0xFF) << 24);
            wv[h * 2 + 1] =
                (((k2 & 15) - 8) & 0xFF) |
                ((((k2 >> 4) - 8) & 0xFF) << 8) |
                (((k3 & 15) - 8) & 0xFF) << 16 |
                ((((k3 >> 4) - 8) & 0xFF) << 24);
        }
        const int cb4 = (g << 6) >> 2;    // 组起始列的 int8x4 槽号
        const int cb8 = ghalf * 8;        // 半段 16 槽中的 8 个
#pragma unroll
        for (int b = 0; b < V30_MAXB; ++b)
        {
            if (b >= B) break;
            const int32_t* xb = xq + (long)b * (cols >> 2);
            const uint4 xv0 = *reinterpret_cast<const uint4*>(
                xb + cb4 + cb8);
            const uint4 xv1 = *reinterpret_cast<const uint4*>(
                xb + cb4 + cb8 + 4);
            const uint32_t xw[8] = {
                xv0.x, xv0.y, xv0.z, xv0.w,
                xv1.x, xv1.y, xv1.z, xv1.w,
            };
            int ai = 0;
#pragma unroll
            for (int h = 0; h < 8; ++h)
            {
                ai = __dp4a((int)wv[h], (int)xw[h], ai);
            }
            accs[b] += (float)ai * (ws * sx8[b * groups + g]);
        }
    }
#pragma unroll
    for (int b = 0; b < V30_MAXB; ++b)
    {
        if (b >= B) break;
        float acc = accs[b];
        acc += __shfl_down_sync(0xffffffffu, acc, 16);
        acc += __shfl_down_sync(0xffffffffu, acc, 8);
        acc += __shfl_down_sync(0xffffffffu, acc, 4);
        acc += __shfl_down_sync(0xffffffffu, acc, 2);
        acc += __shfl_down_sync(0xffffffffu, acc, 1);
        if (lane == 0)
        {
            output[(long)b * rows + row] = acc;
        }
    }
}

// v30 配套:Q8 组量化单 kernel(替代 ~8 次 torch 小算子,省 ~50us/组)。
__global__ void v30_quant_kernel(
    const float* __restrict__ x,          // [B*groups, 64]
    int8_t* __restrict__ q8,              // [B*groups*64]
    float* __restrict__ xs)               // [B*groups]
{
    const long idx = blockIdx.x;
    const int col = threadIdx.x;
    const int lane = col & 31;
    const int warp = col >> 5;
    float a = fabsf(__ldg(x + idx * 64 + col));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    }
    __shared__ float wmax[2];
    if (lane == 0) wmax[warp] = a;
    __syncthreads();
    const float am = fmaxf(wmax[0], wmax[1]);
    const float s = fmaxf(am, 1e-12f) / 127.f;
    if (col == 0) xs[idx] = s;
    const float v = __ldg(x + idx * 64 + col);
    q8[idx * 64 + col] = (int8_t)rintf(
        fminf(fmaxf(v / s, -127.f), 127.f));
}

std::vector<torch::Tensor> int4_v30_quant(torch::Tensor x)
{
    TORCH_CHECK(
        x.device().is_cuda() && x.dtype() == torch::kFloat32 && x.dim() == 2,
        "v30 quant expects 2D fp32 cuda");
    const long total = x.numel();
    TORCH_CHECK(total % 64 == 0, "cols%64");
    const long n_groups = total / 64;
    auto q8 = torch::empty(
        {total}, torch::TensorOptions().dtype(torch::kInt8).device(x.device()));
    auto xs = torch::empty(
        {n_groups}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    v30_quant_kernel<<<(unsigned)n_groups, 64, 0, stream>>>(
        x.data_ptr<float>(),
        reinterpret_cast<int8_t*>(q8.data_ptr()),
        xs.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q8, xs};
}

torch::Tensor int4_gemv_v30(
    torch::Tensor xq,                     // [B, cols] int8
    torch::Tensor xs,                     // [B, groups] fp32
    torch::Tensor packed,
    torch::Tensor scales,
    int64_t rows,
    int64_t cols,
    int64_t groups)
{
    const int B = (int)xq.size(0);
    TORCH_CHECK(B >= 1 && B <= V30_MAXB, "batch 1..6");
    TORCH_CHECK(cols % 64 == 0, "cols%64");
    TORCH_CHECK(xq.dtype() == torch::kInt8 && xs.dtype() == torch::kFloat32,
                "q8 activation dtype");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto output = torch::empty(
        {B, rows}, torch::TensorOptions().dtype(torch::kFloat32).device(xq.device()));
    const size_t shared = (size_t)V30_MAXB * (size_t)groups * sizeof(float);
    static size_t configured30 = 0;
    if (configured30 < shared) {
        auto set30 = [&](auto kernel) {
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)shared);
        };
        set30(int4_gemv_v30_kernel<32>);
        set30(int4_gemv_v30_kernel<16>);
        set30(int4_gemv_v30_kernel<8>);
        configured30 = shared;
    }
    const int rpb = rows >= 4096 ? 32 : (rows >= 2048 ? 16 : 8);
    if (rpb == 32) {
        int4_gemv_v30_kernel<32><<<(rows + 31) / 32, dim3(32, 32), shared, stream>>>(
            reinterpret_cast<const int32_t*>(xq.data_ptr<int8_t>()),
            xs.data_ptr<float>(), packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(scales.data_ptr()),
            output.data_ptr<float>(), (int)rows, (int)cols, (int)groups, B);
    } else if (rpb == 16) {
        int4_gemv_v30_kernel<16><<<(rows + 15) / 16, dim3(32, 16), shared, stream>>>(
            reinterpret_cast<const int32_t*>(xq.data_ptr<int8_t>()),
            xs.data_ptr<float>(), packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(scales.data_ptr()),
            output.data_ptr<float>(), (int)rows, (int)cols, (int)groups, B);
    } else {
        int4_gemv_v30_kernel<8><<<(rows + 7) / 8, dim3(32, 8), shared, stream>>>(
            reinterpret_cast<const int32_t*>(xq.data_ptr<int8_t>()),
            xs.data_ptr<float>(), packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(scales.data_ptr()),
            output.data_ptr<float>(), (int)rows, (int)cols, (int)groups, B);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
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
    float acc = (a0 + a1) + (a2 + a3);
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, off, 4);
    }
    if (row < rows && (lane & 3) == 0) {
        partial[(long)row * slices + slice] = acc;
    }
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
    marlin_reduce_kernel_g<<<(unsigned)((rows + 255) / 256), 256, 0, stream>>>(
        partial.data_ptr<float>(), output.data_ptr<float>(),
        (int)rows, slices);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// ---------------------------------------------------------------------------
// Marlin-style repack + GEMV (all architectures, SM75+).
// Repack: superstep tile = 8 rows x 32 cols; GEMV thread t (j=t/4,i=t%4)
// needs input bytes {i, i+4, i+8, i+12} of row j -> one aligned uint32.
// After repack a superstep is 32 consecutive uint32 (128 B): one fully
// coalesced 4-byte load per lane, then registers-only dequant + mma.
// ---------------------------------------------------------------------------
__global__ void int4_repack_marlin_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const int rows,
    const int cols)
{
    const long words_total =
        (((long)rows + 7) / 8) * (cols / 32) * 32;  // uint32 count
    for (long w = blockIdx.x * (long)blockDim.x + threadIdx.x;
         w < words_total;
         w += (long)gridDim.x * blockDim.x) {
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
            out |= ((uint32_t)srow[i]) | ((uint32_t)srow[i + 4] << 8) |
                   ((uint32_t)srow[i + 8] << 16) | ((uint32_t)srow[i + 12] << 24);
        }
        ((uint32_t*)dst)[w] = out;
    }
}

torch::Tensor int4_repack_marlin(
    torch::Tensor packed,
    int64_t rows,
    int64_t cols)
{
    TORCH_CHECK(cols % 32 == 0, "marlin repack requires cols%32==0");
    const long words = (((long)rows + 7) / 8) * (cols / 32) * 32;
    auto dst = torch::empty(
        {words * 4},
        torch::TensorOptions().dtype(torch::kUInt8).device(packed.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    const long total = words;
    const int blocks = (int)((total + 255) / 256 > 4096 ? 4096 : (total + 255) / 256);
    int4_repack_marlin_kernel<<<blocks, 256, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        dst.data_ptr<uint8_t>(),
        (int)rows, (int)cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dst;
}

constexpr int MARLIN_G_KSPLIT = 2048;

__global__ void int4_gemv_marlin_kernel(
    const __half* __restrict__ x,
    const uint32_t* __restrict__ repacked,
    const __half* __restrict__ scales,
    float* __restrict__ partial,
    const int rows,
    const int cols,
    const int groups,
    const int slices)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    // mma.m16n8k16 需要 SM80+;旧架构编成空 kernel,宿主侧按架构回退。
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row0 = (blockIdx.y * 4 + warp) * 8;
    const int slice = blockIdx.x;
    const int k0 = slice * MARLIN_G_KSPLIT;
    const int k1 = min(cols, k0 + MARLIN_G_KSPLIT);
    if (row0 >= rows) return;
    const int j = lane >> 2;
    const int i = lane & 3;
    const int tiles_k = cols / 32;
    const int row = row0 + j;
    const __half* srow = scales + (long)row * groups;
    float c0 = 0.f, c1 = 0.f;
    for (int k = k0; k < k1; k += 32) {
        const uint32_t word = __ldg(repacked +
            (((long)(row0 >> 3) * tiles_k + (k >> 5)) << 5) + lane);
        const float sc = __half2float(__ldg(srow + ((k + 2 * i) >> 6)));
        const uint32_t bw0 = (word      ) & 0xFF;
        const uint32_t bw1 = (word >>  8) & 0xFF;
        const uint32_t bw2 = (word >> 16) & 0xFF;
        const uint32_t bw3 = (word >> 24) & 0xFF;
        uint32_t a0[4] = {0u, 0u, 0u, 0u}, a1[4] = {0u, 0u, 0u, 0u};
        if (lane < 4) {
            const int ak = 2 * lane;
            a0[0] = half2_bits_m(__halves2half2(x[k + ak], x[k + ak + 1]));
            a0[2] = half2_bits_m(__halves2half2(x[k + ak + 8], x[k + ak + 9]));
            a1[0] = half2_bits_m(__halves2half2(x[k + 16 + ak], x[k + 16 + ak + 1]));
            a1[2] = half2_bits_m(__halves2half2(x[k + 16 + ak + 8], x[k + 16 + ak + 9]));
        }
        const uint32_t xs[2][2] = {{bw0, bw1}, {bw2, bw3}};
#pragma unroll
        for (int hs = 0; hs < 2; ++hs) {
            const float l0 = static_cast<float>(
                static_cast<int>(xs[hs][0] & 15) - 8) * sc;
            const float h0 = static_cast<float>(
                static_cast<int>(xs[hs][0] >> 4) - 8) * sc;
            const float l1 = static_cast<float>(
                static_cast<int>(xs[hs][1] & 15) - 8) * sc;
            const float h1 = static_cast<float>(
                static_cast<int>(xs[hs][1] >> 4) - 8) * sc;
            uint32_t bfrag[2] = {
                h2_m(__float2half_rn(l0), __float2half_rn(h0)),
                h2_m(__float2half_rn(l1), __float2half_rn(h1)),
            };
            mma_m16n8k16_m(hs == 0 ? a0 : a1, bfrag, &c0, &c1);
        }
    }
    // C fragment: lanes 0..3 (m==0) each own two whole output rows,
    // n = 2*lane and 2*lane+1 — no cross-lane reduce.
    if (lane < 4) {
        const int n0 = row0 + 2 * lane;
        if (n0 < rows) partial[(long)n0 * slices + slice] = c0;
        if (n0 + 1 < rows) partial[(long)(n0 + 1) * slices + slice] = c1;
    }
#else
    // SM80 以下:mma 路径不可用;宿主侧(见 int4_gemv_marlin)在旧架
    // 构上拒绝调用,空实现仅为通过编译。
    (void)x; (void)repacked; (void)scales; (void)partial;
    (void)rows; (void)cols; (void)groups; (void)slices;
#endif
}

torch::Tensor int4_gemv_marlin(
    torch::Tensor x,
    torch::Tensor repacked,
    torch::Tensor scales,
    int64_t rows,
    int64_t cols,
    int64_t groups)
{
    // mma.m16n8k16 需要 SM80+;旧架构设备侧为空实现(不写输出),宿主
    // 侧直接返回空 Tensor 让上层走回退,避免静默错误结果。
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, x.get_device());
    if (prop.major < 8) {
        return torch::Tensor();
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    const int slices = (int)((cols + MARLIN_G_KSPLIT - 1) / MARLIN_G_KSPLIT);
    static torch::Tensor partial_cache;
    const long needed = (long)rows * slices;
    if (!partial_cache.defined() || partial_cache.numel() < needed ||
        partial_cache.device() != x.device()) {
        partial_cache = torch::empty(
            {needed}, torch::TensorOptions()
                          .dtype(torch::kFloat32).device(x.device()));
    }
    torch::Tensor partial = partial_cache.narrow(0, 0, needed);
    auto output = torch::empty(
        {rows}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    dim3 block(128);
    dim3 grid(slices, (unsigned)(((rows / 8) + 3) / 4));
    int4_gemv_marlin_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr()),
        reinterpret_cast<const uint32_t*>(repacked.data_ptr()),
        reinterpret_cast<const __half*>(scales.data_ptr()),
        partial.data_ptr<float>(),
        (int)rows, (int)cols, (int)groups, slices);
    marlin_reduce_kernel_g<<<(unsigned)((rows + 255) / 256), 256, 0, stream>>>(
        partial.data_ptr<float>(), output.data_ptr<float>(),
        (int)rows, slices);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// ---------------------------------------------------------------------------
// INT4 GEMV v2 — split-K batch=1 kernel, all architectures (SM70+).
// One warp owns one output row inside a 1024-column segment; each lane loads
// exactly one 16-byte uint4 (32 packed nibbles = 32 columns) plus the single
// G64 group scale, so the whole row streams through the memory system at
// full coalescing width with no serial dependence chain.  Numerics are
// identical to v1: acc += ((nibble - 8) * scale) * x[col].
// ---------------------------------------------------------------------------
constexpr int kInt4V2SegCols = 4096;  // 32 lanes x 4 uint4 steps
constexpr int kInt4V2Warps = 8;       // rows per block

template <typename input_t>
__global__ void int4_gemv_packed_f32_v2_kernel(
    const input_t* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scales,
    float* __restrict__ partial,
    const int rows,
    const int cols,
    const int groups,
    const int segments)
{
    __shared__ float shared_x[kInt4V2SegCols];
    const int lane = threadIdx.x;
    const int row = blockIdx.y * kInt4V2Warps + (threadIdx.y & 7);
    const int seg = blockIdx.x;
    const int col_base = seg * kInt4V2SegCols;
    const int seg_cols = min(kInt4V2SegCols, cols - col_base);
    if (seg_cols <= 0) return;

    for (int c = threadIdx.y * 32 + lane; c < seg_cols; c += 256) {
        shared_x[c] = vq_scalar_to_float(x + (long)(col_base + c));
    }
    __syncthreads();
    if (row >= rows) return;

    const int packed_cols = cols >> 1;
    const uint8_t* packed_row =
        packed + (long)row * packed_cols + (col_base >> 1);
    const __half* scale_row = scales + (long)row * groups;

    // Each lane strides four 16-byte words (128 columns) per step; four
    // independent accumulators keep four loads in flight and break the FMA
    // dependence chain.  seg_cols is a multiple of 64, so any word that
    // starts inside the segment is fully valid — no tail guards needed.
    float accs[4] = {0.f, 0.f, 0.f, 0.f};
    const int words = seg_cols / 32;
    for (int step = lane * 4; step < words; step += 128) {
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int word_index = step + u;
            if (word_index >= words) break;
            const int local = word_index * 32;
            const float scale = __half2float(
                __ldg(scale_row + ((col_base + local) >> 6)));
            const uint4 word = __ldg(reinterpret_cast<const uint4*>(
                packed_row + (local >> 1)));
            const uint8_t* bytes =
                reinterpret_cast<const uint8_t*>(&word);
            float acc = 0.f;
#pragma unroll
            for (int b = 0; b < 16; ++b) {
                const uint8_t q = bytes[b];
                acc = __fmaf_rn(
                    static_cast<float>((q & 15) - 8) * scale,
                    shared_x[local + b * 2],
                    acc);
                acc = __fmaf_rn(
                    static_cast<float>((q >> 4) - 8) * scale,
                    shared_x[local + b * 2 + 1],
                    acc);
            }
            accs[u & 3] += acc;
        }
    }
    float acc = (accs[0] + accs[1]) + (accs[2] + accs[3]);
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        partial[(long)row * segments + seg] = acc;
    }
}

__global__ void int4_gemv_v2_reduce_kernel(
    const float* __restrict__ partial,
    float* __restrict__ output,
    const int rows,
    const int segments)
{
    const int row = blockIdx.x;
    float acc = 0.f;
    for (int s = 0; s < segments; ++s) {
        acc += partial[(long)row * segments + s];
    }
    output[row] = acc;
}

template <typename input_t>
torch::Tensor int4_gemv_packed_f32_v2(
    torch::Tensor x,
    torch::Tensor packed,
    torch::Tensor scales,
    int64_t rows,
    int64_t cols,
    int64_t groups)
{
    TORCH_CHECK(
        cols % 64 == 0,
        "int4 v2 GEMV requires 64-column G64 groups");
    auto stream = at::cuda::getCurrentCUDAStream();
    const int segments =
        (int)((cols + kInt4V2SegCols - 1) / kInt4V2SegCols);
    // Static per-device partial buffer: repeated decode GEMVs reuse one
    // allocation (address stability also keeps CUDA-Graph replays valid).
    static torch::Tensor partial_cache;
    const long needed = (long)rows * segments;
    if (!partial_cache.defined() ||
        partial_cache.device() != x.device() ||
        partial_cache.numel() < needed) {
        partial_cache = torch::empty(
            {needed},
            torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    }
    torch::Tensor partial = partial_cache.narrow(0, 0, needed);
    auto output = torch::empty(
        {rows},
        torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    dim3 block(32, kInt4V2Warps);
    dim3 grid(
        segments,
        (unsigned)((rows + kInt4V2Warps - 1) / kInt4V2Warps));
    int4_gemv_packed_f32_v2_kernel<input_t><<<grid, block, 0, stream>>>(
        reinterpret_cast<const input_t*>(x.data_ptr()),
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr()),
        partial.data_ptr<float>(),
        (int)rows, (int)cols, (int)groups, segments);
    int4_gemv_v2_reduce_kernel<<<rows, 32, 0, stream>>>(
        partial.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)rows, segments);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// ---------------------------------------------------------------------------
// INT4 GEMV vector8 — v1's conflict-free layout, doubled per-lane traffic.
// Keeps the proven pattern (lane owns columns stride-2 inside a 64-column
// group, activation staged in shared memory, one group scale per shfl
// subgroup) but processes TWO adjacent groups per lane step with two
// independent accumulators, halving the iteration count of vector4 while
// keeping two 4-byte loads in flight.  All-architecture: plain LDG + FMA.
// ---------------------------------------------------------------------------
template <typename input_t, int rows_per_block>
__global__ void int4_gemv_packed_f32_vector8_kernel(
    const input_t* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scales,
    float* __restrict__ output,
    int rows,
    int cols,
    int groups)
{
    extern __shared__ float shared_x[];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    for (int col = linear_thread; col < cols; col += 32 * rows_per_block) {
        shared_x[col] = vq_scalar_to_float(x + col);
    }
    __syncthreads();

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= rows) return;
    const int packed_cols = cols >> 1;
    const uint8_t* packed_row =
        packed + (long)row * packed_cols;
    const __half* scale_row = scales + (long)row * groups;

    // 8 lanes cooperate on one 64-column group (2 columns per lane); the
    // warp covers four groups (256 columns) per step.
    const int group_lane = lane & 7;
    const int group_slot = lane >> 3;      // 0..3
    float acc0 = 0.f, acc1 = 0.f;
    for (int group_base = 0; group_base < groups; group_base += 8) {
        // Even step group for acc0, odd for acc1.
        const int g0 = group_base + group_slot;        // 4 groups, stride 1
        const int g1 = group_base + 4 + group_slot;    // next 4 groups
        float s0 = 0.f, s1 = 0.f;
        if (group_lane == 0) {
            if (g0 < groups) s0 = __half2float(__ldg(scale_row + g0));
            if (g1 < groups) s1 = __half2float(__ldg(scale_row + g1));
        }
        s0 = __shfl_sync(0xffffffffu, s0, 0, 8);
        s1 = __shfl_sync(0xffffffffu, s1, 0, 8);
        if (g0 < groups) {
            const uint32_t codes = __ldg(reinterpret_cast<const uint32_t*>(
                packed_row + g0 * 32 + group_lane * 4));
            const int col = g0 * 64 + group_lane * 8;
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&codes);
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const uint8_t q = bytes[item];
                acc0 = __fmaf_rn(
                    static_cast<float>((q & 15) - 8) * s0,
                    shared_x[col + item * 2], acc0);
                acc0 = __fmaf_rn(
                    static_cast<float>((q >> 4) - 8) * s0,
                    shared_x[col + item * 2 + 1], acc0);
            }
        }
        if (g1 < groups) {
            const uint32_t codes = __ldg(reinterpret_cast<const uint32_t*>(
                packed_row + g1 * 32 + group_lane * 4));
            const int col = g1 * 64 + group_lane * 8;
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&codes);
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const uint8_t q = bytes[item];
                acc1 = __fmaf_rn(
                    static_cast<float>((q & 15) - 8) * s1,
                    shared_x[col + item * 2], acc1);
                acc1 = __fmaf_rn(
                    static_cast<float>((q >> 4) - 8) * s1,
                    shared_x[col + item * 2 + 1], acc1);
            }
        }
    }
    float acc = acc0 + acc1;
    acc = warp_sum_f32(acc);
    if (lane == 0) output[row] = acc;
}

template <typename input_t, int rows_per_block>
__global__ void int4_gemv_packed_f32_kernel(
    const input_t* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scales,
    float* __restrict__ output,
    int rows,
    int cols,
    int groups) {
    extern __shared__ float shared_x[];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    for (int col = linear_thread;
         col < cols;
         col += 32 * rows_per_block) {
        shared_x[col] = vq_scalar_to_float(x + col);
    }
    __syncthreads();

    const int row =
        blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= rows) {
        return;
    }
    const int packed_cols = cols / 2;
    const uint8_t* qrow =
        packed + static_cast<int64_t>(row) * packed_cols;
    const __half* srow =
        scales + static_cast<int64_t>(row) * groups;
    float acc = 0.0f;
    for (int group = 0; group < groups; ++group) {
        float scale = lane == 0 ? __half2float(srow[group]) : 0.0f;
        scale = __shfl_sync(0xffffffffu, scale, 0);
        const int byte_index = group * 32 + lane;
        const uint8_t q = __ldg(qrow + byte_index);
        const int col = group * 64 + lane * 2;
        const float low = __fmul_rn(
            static_cast<float>((q & 15) - 8),
            scale);
        const float high = __fmul_rn(
            static_cast<float>((q >> 4) - 8),
            scale);
        acc = __fmaf_rn(low, shared_x[col], acc);
        acc = __fmaf_rn(high, shared_x[col + 1], acc);
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        output[row] = acc;
    }
}

__device__ __forceinline__ float4 fp8x4_scale_to_f32(
    const uint32_t packed,
    const float scale)
{
    __nv_fp8x4_e4m3 fp8_values;
    fp8_values.__x = packed;
    const float4 values = static_cast<float4>(fp8_values);
    const __nv_bfloat162 scale_pair = __float2bfloat162_rn(scale);
    const float2 scaled01 = __bfloat1622float2(__hmul2(
        __floats2bfloat162_rn(values.x, values.y),
        scale_pair));
    const float2 scaled23 = __bfloat1622float2(__hmul2(
        __floats2bfloat162_rn(values.z, values.w),
        scale_pair));
    return make_float4(
        scaled01.x,
        scaled01.y,
        scaled23.x,
        scaled23.y);
}

template <typename input_t, int rows_per_block>
__global__ void block_fp8_gemv_f32_kernel(
    const input_t* __restrict__ input,
    const uint8_t* __restrict__ weights,
    const float* __restrict__ scales,
    float* __restrict__ output,
    const int rows,
    const int cols,
    const int scale_cols)
{
    extern __shared__ unsigned char fp8_shared_raw[];
    auto* fp8_shared_input =
        reinterpret_cast<__nv_bfloat16*>(fp8_shared_raw);
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    for (
        int column = linear_thread;
        column < cols;
        column += 32 * rows_per_block
    ) {
        fp8_shared_input[column] = __float2bfloat16_rn(
            vq_scalar_to_float(input + column));
    }
    __syncthreads();

    const int row =
        blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= rows)
        return;
    const auto* weight_row =
        weights + static_cast<long>(row) * cols;
    const auto* scale_row =
        scales + static_cast<long>(row / 128) * scale_cols;
    float accumulator = 0.f;
    for (
        int column_block = 0;
        column_block < scale_cols;
        ++column_block
    ) {
        float scale = lane == 0
            ? scale_row[column_block]
            : 0.f;
        scale = __shfl_sync(0xffffffffu, scale, 0);
        const float rounded_scale = __bfloat162float(
            __float2bfloat16_rn(scale));
        const int begin = column_block * 128;
        const int end = min(begin + 128, cols);
        const int column = begin + lane * 4;
        if (column + 3 < end) {
            const float4 scaled = fp8x4_scale_to_f32(
                __ldg(reinterpret_cast<const uint32_t*>(
                    weight_row + column)),
                scale);
            accumulator = __fmaf_rn(
                scaled.x,
                __bfloat162float(fp8_shared_input[column]),
                accumulator);
            accumulator = __fmaf_rn(
                scaled.y,
                __bfloat162float(fp8_shared_input[column + 1]),
                accumulator);
            accumulator = __fmaf_rn(
                scaled.z,
                __bfloat162float(fp8_shared_input[column + 2]),
                accumulator);
            accumulator = __fmaf_rn(
                scaled.w,
                __bfloat162float(fp8_shared_input[column + 3]),
                accumulator);
        } else {
            for (int tail = column; tail < end; ++tail) {
                __nv_fp8_e4m3 fp8_value;
                fp8_value.__x = weight_row[tail];
                const float rounded_value = __bfloat162float(
                    __float2bfloat16_rn(
                        static_cast<float>(fp8_value)));
                const float scaled_value = __bfloat162float(
                    __float2bfloat16_rn(
                        rounded_value * rounded_scale));
                accumulator = __fmaf_rn(
                    scaled_value,
                    __bfloat162float(fp8_shared_input[tail]),
                    accumulator);
            }
        }
    }
    accumulator = warp_sum_f32(accumulator);
    if (lane == 0)
        output[row] = accumulator;
}

// Logical row concatenation for several independent block-FP8 projections.
// The pointer/row metadata is persistent device memory owned by the public
// ProjectionGroup wrapper.  This keeps every source tensor compact, shares
// one input staging pass, removes per-projection launches and writes directly
// into the final logical output without torch.cat.
template <typename input_t, int rows_per_block>
__global__ void block_fp8_grouped_gemv_f32_kernel(
    const input_t* __restrict__ input,
    const int64_t* __restrict__ weight_ptrs,
    const int64_t* __restrict__ scale_ptrs,
    const int32_t* __restrict__ row_offsets,
    float* __restrict__ output,
    const int groups,
    const int input_rows,
    const int total_rows,
    const int cols,
    const int scale_cols)
{
    extern __shared__ unsigned char fp8_grouped_shared_raw[];
    auto* shared_input =
        reinterpret_cast<__nv_bfloat16*>(fp8_grouped_shared_raw);
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    const int block_row = blockIdx.x * rows_per_block;
    int input_group = 0;
    while (
        input_group + 1 < groups &&
        block_row >= row_offsets[input_group + 1]
    ) {
        ++input_group;
    }
    const input_t* block_input = input +
        static_cast<long>(input_rows == 1 ? 0 : input_group) * cols;
    for (
        int column = linear_thread;
        column < cols;
        column += 32 * rows_per_block
    ) {
        shared_input[column] = __float2bfloat16_rn(
            vq_scalar_to_float(block_input + column));
    }
    __syncthreads();

    const int output_row =
        blockIdx.x * rows_per_block + threadIdx.y;
    if (output_row >= total_rows)
        return;
    int group = 0;
    while (
        group + 1 < groups &&
        output_row >= row_offsets[group + 1]
    ) {
        ++group;
    }
    const int row = output_row - row_offsets[group];
    const auto* weights = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(weight_ptrs[group]));
    const auto* scales = reinterpret_cast<const float*>(
        static_cast<uintptr_t>(scale_ptrs[group]));
    const auto* weight_row =
        weights + static_cast<long>(row) * cols;
    const auto* scale_row =
        scales + static_cast<long>(row / 128) * scale_cols;
    float accumulator = 0.f;
    for (
        int column_block = 0;
        column_block < scale_cols;
        ++column_block
    ) {
        float scale = lane == 0 ? scale_row[column_block] : 0.f;
        scale = __shfl_sync(0xffffffffu, scale, 0);
        const float rounded_scale = __bfloat162float(
            __float2bfloat16_rn(scale));
        const int begin = column_block * 128;
        const int end = min(begin + 128, cols);
        const int column = begin + lane * 4;
        if (column + 3 < end) {
            const float4 scaled = fp8x4_scale_to_f32(
                __ldg(reinterpret_cast<const uint32_t*>(
                    weight_row + column)),
                scale);
            accumulator = __fmaf_rn(
                scaled.x,
                __bfloat162float(shared_input[column]),
                accumulator);
            accumulator = __fmaf_rn(
                scaled.y,
                __bfloat162float(shared_input[column + 1]),
                accumulator);
            accumulator = __fmaf_rn(
                scaled.z,
                __bfloat162float(shared_input[column + 2]),
                accumulator);
            accumulator = __fmaf_rn(
                scaled.w,
                __bfloat162float(shared_input[column + 3]),
                accumulator);
        } else {
            for (int tail = column; tail < end; ++tail) {
                __nv_fp8_e4m3 fp8_value;
                fp8_value.__x = weight_row[tail];
                const float rounded_value = __bfloat162float(
                    __float2bfloat16_rn(static_cast<float>(fp8_value)));
                const float scaled_value = __bfloat162float(
                    __float2bfloat16_rn(
                        rounded_value * rounded_scale));
                accumulator = __fmaf_rn(
                    scaled_value,
                    __bfloat162float(shared_input[tail]),
                    accumulator);
            }
        }
    }
    accumulator = warp_sum_f32(accumulator);
    if (lane == 0)
        output[output_row] = accumulator;
}

// Decode-sized grouped projections are bandwidth-bound.  Two independent
// output rows per warp reuse the staged activation and halve CTA/input-stage
// traffic while preserving the exact block-FP8 scale layout.  The host only
// selects this path when all groups share one input row, so a projection
// boundary can never invalidate the shared activation.
template <typename input_t, int warps>
__global__ void block_fp8_grouped_gemv_f32_rows2_kernel(
    const input_t* __restrict__ input,
    const int64_t* __restrict__ weight_ptrs,
    const int64_t* __restrict__ scale_ptrs,
    const int32_t* __restrict__ row_offsets,
    float* __restrict__ output,
    const int groups,
    const int total_rows,
    const int cols,
    const int scale_cols)
{
    extern __shared__ unsigned char fp8_grouped_rows2_shared_raw[];
    auto* shared_input = reinterpret_cast<__nv_bfloat16*>(
        fp8_grouped_rows2_shared_raw);
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    for (int column = linear_thread;
         column < cols;
         column += 32 * warps) {
        shared_input[column] = __float2bfloat16_rn(
            vq_scalar_to_float(input + column));
    }
    __syncthreads();

    const int block_row = blockIdx.x * (2 * warps);
    const int output_row0 = block_row + threadIdx.y;
    const int output_row1 = output_row0 + warps;
    if (output_row0 >= total_rows)
        return;

    int group0 = 0;
    while (group0 + 1 < groups &&
           output_row0 >= row_offsets[group0 + 1])
        ++group0;
    int group1 = group0;
    while (group1 + 1 < groups &&
           output_row1 >= row_offsets[group1 + 1])
        ++group1;
    const int row0 = output_row0 - row_offsets[group0];
    const int row1 = output_row1 - row_offsets[group1];
    const auto* weights0 = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(weight_ptrs[group0]));
    const auto* scales0 = reinterpret_cast<const float*>(
        static_cast<uintptr_t>(scale_ptrs[group0]));
    const auto* weight_row0 = weights0 + static_cast<long>(row0) * cols;
    const auto* scale_row0 = scales0 +
        static_cast<long>(row0 / 128) * scale_cols;

    const bool row1_valid = output_row1 < total_rows;
    const auto* weights1 = row1_valid
        ? reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(weight_ptrs[group1]))
        : weights0;
    const auto* scales1 = row1_valid
        ? reinterpret_cast<const float*>(
            static_cast<uintptr_t>(scale_ptrs[group1]))
        : scales0;
    const auto* weight_row1 = weights1 +
        static_cast<long>(row1_valid ? row1 : row0) * cols;
    const auto* scale_row1 = scales1 +
        static_cast<long>((row1_valid ? row1 : row0) / 128) * scale_cols;
    const bool shared_scale_row = scale_row0 == scale_row1;

    float accumulator0 = 0.f;
    float accumulator1 = 0.f;
    for (int column_block = 0;
         column_block < scale_cols;
         ++column_block) {
        float scale0 = lane == 0 ? scale_row0[column_block] : 0.f;
        scale0 = __shfl_sync(0xffffffffu, scale0, 0);
        float scale1 = scale0;
        if (!shared_scale_row) {
            scale1 = lane == 0 ? scale_row1[column_block] : 0.f;
            scale1 = __shfl_sync(0xffffffffu, scale1, 0);
        }
        const int begin = column_block * 128;
        const int end = min(begin + 128, cols);
        const int column = begin + lane * 4;
        if (column + 3 < end) {
            const auto input0 = __bfloat162float(shared_input[column]);
            const auto input1 = __bfloat162float(shared_input[column + 1]);
            const auto input2 = __bfloat162float(shared_input[column + 2]);
            const auto input3 = __bfloat162float(shared_input[column + 3]);
            const float4 scaled0 = fp8x4_scale_to_f32(
                __ldg(reinterpret_cast<const uint32_t*>(
                    weight_row0 + column)),
                scale0);
            const float4 scaled1 = fp8x4_scale_to_f32(
                __ldg(reinterpret_cast<const uint32_t*>(
                    weight_row1 + column)),
                scale1);
            accumulator0 = __fmaf_rn(scaled0.x, input0, accumulator0);
            accumulator0 = __fmaf_rn(scaled0.y, input1, accumulator0);
            accumulator0 = __fmaf_rn(scaled0.z, input2, accumulator0);
            accumulator0 = __fmaf_rn(scaled0.w, input3, accumulator0);
            accumulator1 = __fmaf_rn(scaled1.x, input0, accumulator1);
            accumulator1 = __fmaf_rn(scaled1.y, input1, accumulator1);
            accumulator1 = __fmaf_rn(scaled1.z, input2, accumulator1);
            accumulator1 = __fmaf_rn(scaled1.w, input3, accumulator1);
        }
    }
    accumulator0 = warp_sum_f32(accumulator0);
    accumulator1 = warp_sum_f32(accumulator1);
    if (lane == 0) {
        output[output_row0] = accumulator0;
        if (row1_valid)
            output[output_row1] = accumulator1;
    }
}

// Four G64 groups are consumed per loop. Four 8-lane subgroups load their
// scales and one uint32 of packed codes per lane, while the final full-warp
// reduction still produces one output row. This keeps row-level occupancy
// and cuts loop/shuffle/address overhead for long reduction dimensions.
template <typename input_t, int rows_per_block>
__global__ void int4_gemv_packed_f32_vector4_kernel(
    const input_t* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scales,
    float* __restrict__ output,
    int rows,
    int cols,
    int groups)
{
    extern __shared__ float shared_x[];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    for (
        int col = linear_thread;
        col < cols;
        col += 32 * rows_per_block
    )
        shared_x[col] = vq_scalar_to_float(x + col);
    __syncthreads();

    const int row =
        blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= rows)
        return;
    const int packed_cols = cols / 2;
    const uint8_t* packed_row =
        packed + static_cast<long>(row) * packed_cols;
    const __half* scale_row =
        scales + static_cast<long>(row) * groups;
    const int group_in_iteration = lane >> 3;
    const int group_lane = lane & 7;
    float accumulator = 0.f;
    for (int group_base = 0; group_base < groups; group_base += 4) {
        const int group = group_base + group_in_iteration;
        float scale = group_lane == 0
            ? __half2float(scale_row[group])
            : 0.f;
        scale = __shfl_sync(0xffffffffu, scale, 0, 8);
        const uint32_t codes = __ldg(
            reinterpret_cast<const uint32_t*>(
                packed_row + group * 32 + group_lane * 4));
        const int col_begin =
            group * 64 + group_lane * 8;
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const uint8_t code =
                static_cast<uint8_t>(codes >> (item * 8));
            const int col = col_begin + item * 2;
            accumulator = __fmaf_rn(
                static_cast<float>((code & 15) - 8) * scale,
                shared_x[col],
                accumulator);
            accumulator = __fmaf_rn(
                static_cast<float>((code >> 4) - 8) * scale,
                shared_x[col + 1],
                accumulator);
        }
    }
    accumulator = warp_sum_f32(accumulator);
    if (lane == 0)
        output[row] = accumulator;
}

template <int rows_per_block>
__global__ void int4_glm_qb_split_kernel(
    const float* __restrict__ input,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scales,
    __nv_bfloat16* __restrict__ nope_output,
    float* __restrict__ rope_output,
    const int heads,
    const int nope_width,
    const int rope_width,
    const int cols,
    const int groups)
{
    extern __shared__ float shared_input[];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    for (
        int col = linear_thread;
        col < cols;
        col += 32 * rows_per_block
    )
        shared_input[col] = input[col];
    __syncthreads();

    const int row =
        blockIdx.x * rows_per_block + threadIdx.y;
    const int head_width = nope_width + rope_width;
    const int rows = heads * head_width;
    if (row >= rows)
        return;
    const uint8_t* packed_row =
        packed + static_cast<long>(row) * (cols / 2);
    const __half* scale_row =
        scales + static_cast<long>(row) * groups;
    float accumulator = 0.f;
    for (int group = 0; group < groups; ++group) {
        float scale = lane == 0
            ? __half2float(scale_row[group])
            : 0.f;
        scale = __shfl_sync(0xffffffffu, scale, 0);
        const uint8_t code =
            __ldg(packed_row + group * 32 + lane);
        const int col = group * 64 + lane * 2;
        accumulator = __fmaf_rn(
            static_cast<float>((code & 15) - 8) * scale,
            shared_input[col],
            accumulator);
        accumulator = __fmaf_rn(
            static_cast<float>((code >> 4) - 8) * scale,
            shared_input[col + 1],
            accumulator);
    }
    accumulator = warp_sum_f32(accumulator);
    if (lane == 0) {
        const int head = row / head_width;
        const int feature = row - head * head_width;
        if (feature < nope_width) {
            nope_output[
                static_cast<long>(head) * nope_width + feature
            ] = __float2bfloat16_rn(accumulator);
        } else {
            rope_output[
                static_cast<long>(head) * rope_width
                + feature - nope_width
            ] = accumulator;
        }
    }
}

template <int rows_per_block>
__global__ void int4_glm_qb_split_vector4_kernel(
    const float* __restrict__ input,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scales,
    __nv_bfloat16* __restrict__ nope_output,
    float* __restrict__ rope_output,
    const int heads,
    const int nope_width,
    const int rope_width,
    const int cols,
    const int groups)
{
    extern __shared__ float shared_input[];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    for (
        int col = linear_thread;
        col < cols;
        col += 32 * rows_per_block
    )
        shared_input[col] = input[col];
    __syncthreads();

    const int row =
        blockIdx.x * rows_per_block + threadIdx.y;
    const int head_width = nope_width + rope_width;
    const int rows = heads * head_width;
    if (row >= rows)
        return;
    const uint8_t* packed_row =
        packed + static_cast<long>(row) * (cols / 2);
    const __half* scale_row =
        scales + static_cast<long>(row) * groups;
    const int group_in_iteration = lane >> 3;
    const int group_lane = lane & 7;
    float accumulator = 0.f;
    for (int group_base = 0; group_base < groups; group_base += 4) {
        const int group = group_base + group_in_iteration;
        float scale = group_lane == 0
            ? __half2float(scale_row[group])
            : 0.f;
        scale = __shfl_sync(0xffffffffu, scale, 0, 8);
        const uint32_t codes = __ldg(
            reinterpret_cast<const uint32_t*>(
                packed_row + group * 32 + group_lane * 4));
        const int col_begin =
            group * 64 + group_lane * 8;
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const uint8_t code =
                static_cast<uint8_t>(codes >> (item * 8));
            const int col = col_begin + item * 2;
            accumulator = __fmaf_rn(
                static_cast<float>((code & 15) - 8) * scale,
                shared_input[col],
                accumulator);
            accumulator = __fmaf_rn(
                static_cast<float>((code >> 4) - 8) * scale,
                shared_input[col + 1],
                accumulator);
        }
    }
    accumulator = warp_sum_f32(accumulator);
    if (lane == 0) {
        const int head = row / head_width;
        const int feature = row - head * head_width;
        if (feature < nope_width) {
            nope_output[
                static_cast<long>(head) * nope_width + feature
            ] = __float2bfloat16_rn(accumulator);
        } else {
            rope_output[
                static_cast<long>(head) * rope_width
                + feature - nope_width
            ] = accumulator;
        }
    }
}

std::vector<torch::Tensor> int4_glm_qb_split(
    torch::Tensor input,
    torch::Tensor packed,
    torch::Tensor scales,
    long cols,
    long group_size,
    bool group_vector,
    long heads,
    long nope_width,
    long rope_width,
    c10::optional<torch::Tensor> nope_output_buffer,
    c10::optional<torch::Tensor> rope_output_buffer)
{
    TORCH_CHECK(
        input.is_cuda() && packed.is_cuda() && scales.is_cuda(),
        "GLM Q-B split tensors must be CUDA");
    TORCH_CHECK(
        input.scalar_type() == at::kFloat &&
        packed.scalar_type() == at::kByte &&
        scales.scalar_type() == at::kHalf &&
        input.is_contiguous() &&
        packed.is_contiguous() &&
        scales.is_contiguous() &&
        input.sizes() == torch::IntArrayRef({1, cols}) &&
        packed.dim() == 2 &&
        scales.dim() == 2,
        "GLM Q-B split input layouts do not match");
    TORCH_CHECK(
        group_size == 64 &&
        cols > 0 &&
        cols % group_size == 0 &&
        heads > 0 &&
        nope_width > 0 &&
        rope_width > 0 &&
        packed.size(0) == heads * (nope_width + rope_width) &&
        packed.size(1) * 2 == cols &&
        scales.sizes() == torch::IntArrayRef(
            {packed.size(0), cols / group_size}),
        "GLM Q-B split shapes do not match");
    const int device = input.get_device();
    TORCH_CHECK(
        packed.get_device() == device &&
        scales.get_device() == device,
        "GLM Q-B split tensors must share one device");
    auto nope_output = nope_output_buffer.has_value()
        ? nope_output_buffer.value()
        : torch::empty(
            {heads, 1, nope_width},
            input.options().dtype(at::kBFloat16));
    auto rope_output = rope_output_buffer.has_value()
        ? rope_output_buffer.value()
        : torch::empty(
            {heads, 1, rope_width},
            input.options());
    TORCH_CHECK(
        nope_output.is_cuda() &&
        nope_output.scalar_type() == at::kBFloat16 &&
        nope_output.is_contiguous() &&
        nope_output.sizes() == torch::IntArrayRef(
            {heads, 1, nope_width}) &&
        nope_output.get_device() == device,
        "GLM Q-B no-PE output must be contiguous BF16 [H,1,D]");
    TORCH_CHECK(
        rope_output.is_cuda() &&
        rope_output.scalar_type() == at::kFloat &&
        rope_output.is_contiguous() &&
        rope_output.sizes() == torch::IntArrayRef(
            {heads, 1, rope_width}) &&
        rope_output.get_device() == device,
        "GLM Q-B RoPE output must be contiguous FP32 [H,1,D]");
    constexpr int rows_per_block = 32;
    const int rows = static_cast<int>(packed.size(0));
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto grid = (rows + rows_per_block - 1) / rows_per_block;
    const auto block = dim3(32, rows_per_block);
    const auto shared = static_cast<int>(cols) * sizeof(float);
    if (group_vector) {
        TORCH_CHECK(
            (cols / group_size) % 4 == 0,
            "GLM Q-B vector path requires a multiple of four groups");
        int4_glm_qb_split_vector4_kernel<rows_per_block><<<
            grid,
            block,
            shared,
            stream>>>(
                input.data_ptr<float>(),
                packed.data_ptr<uint8_t>(),
                reinterpret_cast<const __half*>(
                    scales.data_ptr<at::Half>()),
                reinterpret_cast<__nv_bfloat16*>(
                    nope_output.data_ptr<at::BFloat16>()),
                rope_output.data_ptr<float>(),
                static_cast<int>(heads),
                static_cast<int>(nope_width),
                static_cast<int>(rope_width),
                static_cast<int>(cols),
                static_cast<int>(cols / group_size));
    } else {
        int4_glm_qb_split_kernel<rows_per_block><<<
            grid,
            block,
            shared,
            stream>>>(
                input.data_ptr<float>(),
                packed.data_ptr<uint8_t>(),
                reinterpret_cast<const __half*>(
                    scales.data_ptr<at::Half>()),
                reinterpret_cast<__nv_bfloat16*>(
                    nope_output.data_ptr<at::BFloat16>()),
                rope_output.data_ptr<float>(),
                static_cast<int>(heads),
                static_cast<int>(nope_width),
                static_cast<int>(rope_width),
                static_cast<int>(cols),
                static_cast<int>(cols / group_size));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {nope_output, rope_output};
}

// Decode-only GLM input RMSNorm plus the two projections that consume the
// same normalized hidden row (Q-A and KV-A). Every CTA reproduces the exact
// 256-thread RMS reduction into its local activation tile, then evaluates
// distinct output rows. This removes the global normalized row and avoids
// staging it once for each projection.
template <bool ADD_RESIDUAL, int ROWS_PER_CTA>
__global__ void glm_norm_qkv_int4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual_update,
    const float* __restrict__ norm_weight,
    const uint8_t* __restrict__ q_packed,
    const __half* __restrict__ q_scales,
    const uint8_t* __restrict__ kv_packed,
    const __half* __restrict__ kv_scales,
    float* __restrict__ q_output,
    float* __restrict__ kv_output,
    float* __restrict__ residual_output,
    const int q_rows,
    const int kv_rows,
    const int cols,
    const int groups,
    const float eps)
{
    extern __shared__ float shared_x[];
    __shared__ float reduction[32];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;

    float square_sum = 0.f;
    if (linear_thread < 256) {
        for (
            int col = linear_thread;
            col < cols;
            col += 256
        ) {
            const float value = ADD_RESIDUAL
                ? x[col] + residual_update[col]
                : x[col];
            square_sum += value * value;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            square_sum += __shfl_down_sync(
                0xffffffffu,
                square_sum,
                offset);
        if ((linear_thread & 31) == 0)
            reduction[linear_thread >> 5] = square_sum;
    }
    __syncthreads();
    if (linear_thread < 32) {
        float value = linear_thread < 8
            ? reduction[linear_thread]
            : 0.f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(
                0xffffffffu,
                value,
                offset);
        if (linear_thread == 0)
            reduction[0] = value;
    }
    __syncthreads();
    const float norm_scale = rsqrtf(
        reduction[0] / static_cast<float>(cols) + eps);
    for (
        int col = linear_thread;
        col < cols;
        col += 32 * ROWS_PER_CTA
    ) {
        const float value = ADD_RESIDUAL
            ? x[col] + residual_update[col]
            : x[col];
        shared_x[col] =
            norm_weight[col] * (value * norm_scale);
        if (ADD_RESIDUAL && blockIdx.x == 0)
            residual_output[col] = value;
    }
    __syncthreads();

    const int combined_row =
        blockIdx.x * ROWS_PER_CTA + threadIdx.y;
    if (combined_row >= q_rows + kv_rows)
        return;
    const bool is_q = combined_row < q_rows;
    const int row = is_q ? combined_row : combined_row - q_rows;
    const uint8_t* packed = is_q ? q_packed : kv_packed;
    const __half* scales = is_q ? q_scales : kv_scales;
    float* output = is_q ? q_output : kv_output;
    const int packed_cols = cols / 2;
    const uint8_t* packed_row =
        packed + static_cast<long>(row) * packed_cols;
    const __half* scale_row =
        scales + static_cast<long>(row) * groups;
    float accumulator = 0.f;
    for (int group = 0; group < groups; ++group) {
        float scale = lane == 0
            ? __half2float(scale_row[group])
            : 0.f;
        scale = __shfl_sync(0xffffffffu, scale, 0);
        const uint8_t code =
            __ldg(packed_row + group * 32 + lane);
        const int col = group * 64 + lane * 2;
        accumulator = __fmaf_rn(
            static_cast<float>((code & 15) - 8) * scale,
            shared_x[col],
            accumulator);
        accumulator = __fmaf_rn(
            static_cast<float>((code >> 4) - 8) * scale,
            shared_x[col + 1],
            accumulator);
    }
    accumulator = warp_sum_f32(accumulator);
    if (lane == 0)
        output[row] = accumulator;
}

// Decode-only residual add + post-attention RMSNorm + router projection.
// Eight router rows share one normalized hidden tile per CTA.  CTA 0 also
// materializes the residual and normalized rows needed by the expert MLP.
__global__ void glm_residual_norm_router_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ update,
    const float* __restrict__ norm_weight,
    const float* __restrict__ router_weight,
    float* __restrict__ residual_output,
    float* __restrict__ norm_output,
    float* __restrict__ logits_output,
    const int rows,
    const int cols,
    const float eps)
{
    extern __shared__ float shared_norm[];
    __shared__ float reduction[32];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;

    float square_sum = 0.f;
    for (
        int col = linear_thread;
        col < cols;
        col += 256
    ) {
        const float value = residual[col] + update[col];
        square_sum += value * value;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        square_sum += __shfl_down_sync(
            0xffffffffu,
            square_sum,
            offset);
    if ((linear_thread & 31) == 0)
        reduction[linear_thread >> 5] = square_sum;
    __syncthreads();
    if (linear_thread < 32) {
        float value = linear_thread < 8
            ? reduction[linear_thread]
            : 0.f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(
                0xffffffffu,
                value,
                offset);
        if (linear_thread == 0)
            reduction[0] = value;
    }
    __syncthreads();
    const float norm_scale = rsqrtf(
        reduction[0] / static_cast<float>(cols) + eps);
    for (
        int col = linear_thread;
        col < cols;
        col += 256
    ) {
        const float value = residual[col] + update[col];
        const float normalized =
            norm_weight[col] * (value * norm_scale);
        shared_norm[col] = normalized;
        if (blockIdx.x == 0) {
            residual_output[col] = value;
            norm_output[col] = normalized;
        }
    }
    __syncthreads();

    const int row = blockIdx.x * 8 + threadIdx.y;
    if (row >= rows)
        return;
    const float* weight =
        router_weight + static_cast<long>(row) * cols;
    float accumulator = 0.f;
    for (int col = lane; col < cols; col += 32)
        accumulator = __fmaf_rn(
            shared_norm[col],
            weight[col],
            accumulator);
    accumulator = warp_sum_f32(accumulator);
    if (lane == 0)
        logits_output[row] = accumulator;
}

__global__ void glm_moe_residual_add_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ routed,
    const float* __restrict__ shared,
    float* __restrict__ output,
    const int count)
{
    for (
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < count;
        index += blockDim.x * gridDim.x
    ) {
        const float expert_sum = __fadd_rn(
            routed[index],
            shared[index]);
        output[index] = __fadd_rn(
            residual[index],
            expert_sum);
    }
}

__global__ void residual_add3_bf16_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ routed,
    const __nv_bfloat16* __restrict__ shared,
    __nv_bfloat16* __restrict__ output,
    const int count)
{
    for (
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < count;
        index += blockDim.x * gridDim.x
    ) {
        const __nv_bfloat16 expert_sum = __float2bfloat16_rn(
            __bfloat162float(routed[index])
            + __bfloat162float(shared[index]));
        output[index] = __float2bfloat16_rn(
            __bfloat162float(residual[index])
            + __bfloat162float(expert_sum));
    }
}

__global__ void glm_ep_reduce_residual_kernel(
    float* __restrict__ primary_partial,
    const float* __restrict__ partial_1,
    const float* __restrict__ partial_2,
    const float* __restrict__ partial_3,
    const float* __restrict__ partial_4,
    const float* __restrict__ partial_5,
    const float* __restrict__ partial_6,
    const float* __restrict__ partial_7,
    const float* __restrict__ partial_8,
    const float* __restrict__ partial_9,
    const float* __restrict__ partial_10,
    const float* __restrict__ partial_11,
    const float* __restrict__ partial_12,
    const float* __restrict__ partial_13,
    const float* __restrict__ partial_14,
    const float* __restrict__ partial_15,
    const float* __restrict__ residual,
    const int contribution_count,
    const int count)
{
    for (
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < count;
        index += blockDim.x * gridDim.x
    ) {
        float routed = primary_partial[index];
        if (contribution_count > 1)
            routed = __fadd_rn(routed, partial_1[index]);
        if (contribution_count > 2)
            routed = __fadd_rn(routed, partial_2[index]);
        if (contribution_count > 3)
            routed = __fadd_rn(routed, partial_3[index]);
        if (contribution_count > 4)
            routed = __fadd_rn(routed, partial_4[index]);
        if (contribution_count > 5)
            routed = __fadd_rn(routed, partial_5[index]);
        if (contribution_count > 6)
            routed = __fadd_rn(routed, partial_6[index]);
        if (contribution_count > 7)
            routed = __fadd_rn(routed, partial_7[index]);
        if (contribution_count > 8)
            routed = __fadd_rn(routed, partial_8[index]);
        if (contribution_count > 9)
            routed = __fadd_rn(routed, partial_9[index]);
        if (contribution_count > 10)
            routed = __fadd_rn(routed, partial_10[index]);
        if (contribution_count > 11)
            routed = __fadd_rn(routed, partial_11[index]);
        if (contribution_count > 12)
            routed = __fadd_rn(routed, partial_12[index]);
        if (contribution_count > 13)
            routed = __fadd_rn(routed, partial_13[index]);
        if (contribution_count > 14)
            routed = __fadd_rn(routed, partial_14[index]);
        if (contribution_count > 15)
            routed = __fadd_rn(routed, partial_15[index]);
        primary_partial[index] = __fadd_rn(
            residual[index],
            routed);
    }
}

template <typename output_t>
__global__ void tp_all_rank_reduce_kernel(
    output_t* __restrict__ output,
    const float* __restrict__ partial_0,
    const float* __restrict__ partial_1,
    const float* __restrict__ partial_2,
    const float* __restrict__ partial_3,
    const float* __restrict__ partial_4,
    const float* __restrict__ partial_5,
    const float* __restrict__ partial_6,
    const float* __restrict__ partial_7,
    const float* __restrict__ partial_8,
    const float* __restrict__ partial_9,
    const float* __restrict__ partial_10,
    const float* __restrict__ partial_11,
    const float* __restrict__ partial_12,
    const float* __restrict__ partial_13,
    const float* __restrict__ partial_14,
    const float* __restrict__ partial_15,
    const int contribution_count,
    const int count)
{
    for (
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < count;
        index += blockDim.x * gridDim.x
    ) {
        float value = partial_0[index];
        if (contribution_count > 1)
            value = __fadd_rn(value, partial_1[index]);
        if (contribution_count > 2)
            value = __fadd_rn(value, partial_2[index]);
        if (contribution_count > 3)
            value = __fadd_rn(value, partial_3[index]);
        if (contribution_count > 4)
            value = __fadd_rn(value, partial_4[index]);
        if (contribution_count > 5)
            value = __fadd_rn(value, partial_5[index]);
        if (contribution_count > 6)
            value = __fadd_rn(value, partial_6[index]);
        if (contribution_count > 7)
            value = __fadd_rn(value, partial_7[index]);
        if (contribution_count > 8)
            value = __fadd_rn(value, partial_8[index]);
        if (contribution_count > 9)
            value = __fadd_rn(value, partial_9[index]);
        if (contribution_count > 10)
            value = __fadd_rn(value, partial_10[index]);
        if (contribution_count > 11)
            value = __fadd_rn(value, partial_11[index]);
        if (contribution_count > 12)
            value = __fadd_rn(value, partial_12[index]);
        if (contribution_count > 13)
            value = __fadd_rn(value, partial_13[index]);
        if (contribution_count > 14)
            value = __fadd_rn(value, partial_14[index]);
        if (contribution_count > 15)
            value = __fadd_rn(value, partial_15[index]);
        if constexpr (std::is_same_v<output_t, float>)
            output[index] = value;
        else
            output[index] = __float2bfloat16_rn(value);
    }
}

__global__ void tp_moe_finalize_all_rank_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ routed_0,
    const float* __restrict__ routed_1,
    const float* __restrict__ routed_2,
    const float* __restrict__ routed_3,
    const float* __restrict__ routed_4,
    const float* __restrict__ routed_5,
    const float* __restrict__ routed_6,
    const float* __restrict__ routed_7,
    const float* __restrict__ routed_8,
    const float* __restrict__ routed_9,
    const float* __restrict__ routed_10,
    const float* __restrict__ routed_11,
    const float* __restrict__ routed_12,
    const float* __restrict__ routed_13,
    const float* __restrict__ routed_14,
    const float* __restrict__ routed_15,
    const float* __restrict__ shared_0,
    const float* __restrict__ shared_1,
    const float* __restrict__ shared_2,
    const float* __restrict__ shared_3,
    const float* __restrict__ shared_4,
    const float* __restrict__ shared_5,
    const float* __restrict__ shared_6,
    const float* __restrict__ shared_7,
    const float* __restrict__ shared_8,
    const float* __restrict__ shared_9,
    const float* __restrict__ shared_10,
    const float* __restrict__ shared_11,
    const float* __restrict__ shared_12,
    const float* __restrict__ shared_13,
    const float* __restrict__ shared_14,
    const float* __restrict__ shared_15,
    const int contribution_count,
    const int count)
{
    for (
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < count;
        index += blockDim.x * gridDim.x
    ) {
        float routed = routed_0[index];
        float shared = shared_0[index];
#define CCCP_ACCUMULATE_MOE_RANK(rank) \
        if (contribution_count > rank) { \
            routed = __fadd_rn(routed, routed_##rank[index]); \
            shared = __fadd_rn(shared, shared_##rank[index]); \
        }
        CCCP_ACCUMULATE_MOE_RANK(1)
        CCCP_ACCUMULATE_MOE_RANK(2)
        CCCP_ACCUMULATE_MOE_RANK(3)
        CCCP_ACCUMULATE_MOE_RANK(4)
        CCCP_ACCUMULATE_MOE_RANK(5)
        CCCP_ACCUMULATE_MOE_RANK(6)
        CCCP_ACCUMULATE_MOE_RANK(7)
        CCCP_ACCUMULATE_MOE_RANK(8)
        CCCP_ACCUMULATE_MOE_RANK(9)
        CCCP_ACCUMULATE_MOE_RANK(10)
        CCCP_ACCUMULATE_MOE_RANK(11)
        CCCP_ACCUMULATE_MOE_RANK(12)
        CCCP_ACCUMULATE_MOE_RANK(13)
        CCCP_ACCUMULATE_MOE_RANK(14)
        CCCP_ACCUMULATE_MOE_RANK(15)
#undef CCCP_ACCUMULATE_MOE_RANK
        // Preserve the exact three-kernel rounding contract: both reductions
        // first materialize BF16, then routed+shared and residual are rounded
        // independently.  Only the intermediate global-memory traffic and
        // launch boundaries disappear.
        const __nv_bfloat16 routed_bf16 =
            __float2bfloat16_rn(routed);
        const __nv_bfloat16 shared_bf16 =
            __float2bfloat16_rn(shared);
        const __nv_bfloat16 expert_sum = __float2bfloat16_rn(
            __bfloat162float(routed_bf16)
            + __bfloat162float(shared_bf16));
        output[index] = __float2bfloat16_rn(
            __bfloat162float(residual[index])
            + __bfloat162float(expert_sum));
    }
}

// Complete the batch-1 no-owner MoE publication directly into a
// Hyper-Connection state.  This preserves the established rounding order
// (rank FP32 sums -> BF16 routed/shared -> BF16 expert sum -> FP32 HC mix ->
// BF16 output) while removing the intermediate [D] publication and the
// following per-rank HC-post launch.
__global__ void tp_moe_hc_finalize_all_rank_bf16_kernel(
    __nv_bfloat16* __restrict__ output,          // [1,4,D]
    const __nv_bfloat16* __restrict__ residual,  // [1,4,D]
    const __nv_bfloat16* __restrict__ post,      // [1,4]
    const __nv_bfloat16* __restrict__ comb,      // [1,4,4]
    const float* __restrict__ routed_0,
    const float* __restrict__ routed_1,
    const float* __restrict__ routed_2,
    const float* __restrict__ routed_3,
    const float* __restrict__ routed_4,
    const float* __restrict__ routed_5,
    const float* __restrict__ routed_6,
    const float* __restrict__ routed_7,
    const float* __restrict__ routed_8,
    const float* __restrict__ routed_9,
    const float* __restrict__ routed_10,
    const float* __restrict__ routed_11,
    const float* __restrict__ routed_12,
    const float* __restrict__ routed_13,
    const float* __restrict__ routed_14,
    const float* __restrict__ routed_15,
    const float* __restrict__ shared_0,
    const float* __restrict__ shared_1,
    const float* __restrict__ shared_2,
    const float* __restrict__ shared_3,
    const float* __restrict__ shared_4,
    const float* __restrict__ shared_5,
    const float* __restrict__ shared_6,
    const float* __restrict__ shared_7,
    const float* __restrict__ shared_8,
    const float* __restrict__ shared_9,
    const float* __restrict__ shared_10,
    const float* __restrict__ shared_11,
    const float* __restrict__ shared_12,
    const float* __restrict__ shared_13,
    const float* __restrict__ shared_14,
    const float* __restrict__ shared_15,
    const int contribution_count,
    const int D)
{
    const int channel = blockIdx.y;
    __shared__ float coeff[5];
    if (threadIdx.x == 0) {
        coeff[0] = __bfloat162float(post[channel]);
        #pragma unroll
        for (int source = 0; source < 4; ++source)
            coeff[1 + source] = __bfloat162float(
                comb[source * 4 + channel]);
    }
    __syncthreads();
    for (
        int d = blockIdx.x * blockDim.x + threadIdx.x;
        d < D;
        d += blockDim.x * gridDim.x
    ) {
        float routed = routed_0[d];
        float shared = shared_0[d];
#define CCCP_ACCUMULATE_HC_MOE_RANK(rank) \
        if (contribution_count > rank) { \
            routed = __fadd_rn(routed, routed_##rank[d]); \
            shared = __fadd_rn(shared, shared_##rank[d]); \
        }
        CCCP_ACCUMULATE_HC_MOE_RANK(1)
        CCCP_ACCUMULATE_HC_MOE_RANK(2)
        CCCP_ACCUMULATE_HC_MOE_RANK(3)
        CCCP_ACCUMULATE_HC_MOE_RANK(4)
        CCCP_ACCUMULATE_HC_MOE_RANK(5)
        CCCP_ACCUMULATE_HC_MOE_RANK(6)
        CCCP_ACCUMULATE_HC_MOE_RANK(7)
        CCCP_ACCUMULATE_HC_MOE_RANK(8)
        CCCP_ACCUMULATE_HC_MOE_RANK(9)
        CCCP_ACCUMULATE_HC_MOE_RANK(10)
        CCCP_ACCUMULATE_HC_MOE_RANK(11)
        CCCP_ACCUMULATE_HC_MOE_RANK(12)
        CCCP_ACCUMULATE_HC_MOE_RANK(13)
        CCCP_ACCUMULATE_HC_MOE_RANK(14)
        CCCP_ACCUMULATE_HC_MOE_RANK(15)
#undef CCCP_ACCUMULATE_HC_MOE_RANK
        const __nv_bfloat16 routed_bf16 =
            __float2bfloat16_rn(routed);
        const __nv_bfloat16 shared_bf16 =
            __float2bfloat16_rn(shared);
        const __nv_bfloat16 expert_sum = __float2bfloat16_rn(
            __bfloat162float(routed_bf16) +
            __bfloat162float(shared_bf16));
        float value = coeff[0] * __bfloat162float(expert_sum);
        #pragma unroll
        for (int source = 0; source < 4; ++source) {
            value = fmaf(
                coeff[1 + source],
                __bfloat162float(residual[source * D + d]),
                value);
        }
        output[channel * D + d] = __float2bfloat16_rn(value);
    }
}

template <typename input_t>
__global__ void int4_swiglu_packed_f32_kernel(
    const input_t* __restrict__ x,
    const uint8_t* __restrict__ gate_packed,
    const __half* __restrict__ gate_scales,
    const uint8_t* __restrict__ up_packed,
    const __half* __restrict__ up_scales,
    float* __restrict__ output,
    int rows,
    int cols,
    int groups) {
    extern __shared__ float shared_x[];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    for (int col = linear_thread;
         col < cols;
         col += 32 * INT4_ROWS_PER_BLOCK) {
        shared_x[col] = vq_scalar_to_float(x + col);
    }
    __syncthreads();

    const int row =
        blockIdx.x * INT4_ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;
    const int packed_cols = cols / 2;
    const int64_t packed_base =
        static_cast<int64_t>(row) * packed_cols;
    const int64_t scale_base =
        static_cast<int64_t>(row) * groups;
    float gate = 0.0f;
    float up = 0.0f;
    for (int group = 0; group < groups; ++group) {
        float gate_scale = lane == 0
            ? __half2float(gate_scales[scale_base + group])
            : 0.0f;
        float up_scale = lane == 0
            ? __half2float(up_scales[scale_base + group])
            : 0.0f;
        gate_scale = __shfl_sync(
            0xffffffffu, gate_scale, 0);
        up_scale = __shfl_sync(
            0xffffffffu, up_scale, 0);
        const int64_t byte_index =
            packed_base + group * 32 + lane;
        const uint8_t gate_q = __ldg(gate_packed + byte_index);
        const uint8_t up_q = __ldg(up_packed + byte_index);
        const int col = group * 64 + lane * 2;
        const float x0 = shared_x[col];
        const float x1 = shared_x[col + 1];
        gate = __fmaf_rn(
            static_cast<float>((gate_q & 15) - 8) * gate_scale,
            x0,
            gate);
        gate = __fmaf_rn(
            static_cast<float>((gate_q >> 4) - 8) * gate_scale,
            x1,
            gate);
        up = __fmaf_rn(
            static_cast<float>((up_q & 15) - 8) * up_scale,
            x0,
            up);
        up = __fmaf_rn(
            static_cast<float>((up_q >> 4) - 8) * up_scale,
            x1,
            up);
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0) {
        const float silu = gate / (1.0f + expf(-gate));
        output[row] = silu * up;
    }
}

template <typename input_t>
__global__ void int4_swiglu_packed_f32_vector4_kernel(
    const input_t* __restrict__ x,
    const uint8_t* __restrict__ gate_packed,
    const __half* __restrict__ gate_scales,
    const uint8_t* __restrict__ up_packed,
    const __half* __restrict__ up_scales,
    float* __restrict__ output,
    int rows,
    int cols,
    int groups)
{
    extern __shared__ float shared_x[];
    const int lane = threadIdx.x;
    const int linear_thread = threadIdx.y * 32 + lane;
    for (
        int col = linear_thread;
        col < cols;
        col += 32 * INT4_ROWS_PER_BLOCK
    )
        shared_x[col] = vq_scalar_to_float(x + col);
    __syncthreads();

    const int row =
        blockIdx.x * INT4_ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows)
        return;
    const int packed_cols = cols / 2;
    const long packed_base =
        static_cast<long>(row) * packed_cols;
    const long scale_base =
        static_cast<long>(row) * groups;
    const int group_in_iteration = lane >> 3;
    const int group_lane = lane & 7;
    float gate = 0.f;
    float up = 0.f;
    for (int group_base = 0; group_base < groups; group_base += 4) {
        const int group = group_base + group_in_iteration;
        float gate_scale = group_lane == 0
            ? __half2float(gate_scales[scale_base + group])
            : 0.f;
        float up_scale = group_lane == 0
            ? __half2float(up_scales[scale_base + group])
            : 0.f;
        gate_scale = __shfl_sync(
            0xffffffffu,
            gate_scale,
            0,
            8);
        up_scale = __shfl_sync(
            0xffffffffu,
            up_scale,
            0,
            8);
        const long byte_index =
            packed_base + group * 32 + group_lane * 4;
        const uint32_t gate_codes = __ldg(
            reinterpret_cast<const uint32_t*>(
                gate_packed + byte_index));
        const uint32_t up_codes = __ldg(
            reinterpret_cast<const uint32_t*>(
                up_packed + byte_index));
        const int col_begin =
            group * 64 + group_lane * 8;
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const uint8_t gate_code =
                static_cast<uint8_t>(
                    gate_codes >> (item * 8));
            const uint8_t up_code =
                static_cast<uint8_t>(
                    up_codes >> (item * 8));
            const int col = col_begin + item * 2;
            const float x0 = shared_x[col];
            const float x1 = shared_x[col + 1];
            gate = __fmaf_rn(
                static_cast<float>((gate_code & 15) - 8) *
                    gate_scale,
                x0,
                gate);
            gate = __fmaf_rn(
                static_cast<float>((gate_code >> 4) - 8) *
                    gate_scale,
                x1,
                gate);
            up = __fmaf_rn(
                static_cast<float>((up_code & 15) - 8) *
                    up_scale,
                x0,
                up);
            up = __fmaf_rn(
                static_cast<float>((up_code >> 4) - 8) *
                    up_scale,
                x1,
                up);
        }
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0) {
        const float silu = gate / (1.f + expf(-gate));
        output[row] = silu * up;
    }
}

template <typename input_t, int rows_per_block>
void launch_int4_gemv_packed_f32_rows(
    const input_t* x,
    const uint8_t* packed,
    const __half* scales,
    float* output,
    int rows,
    int cols,
    int groups,
    int device,
    cudaStream_t stream) {
    dim3 block(32, rows_per_block);
    const int blocks =
        (rows + rows_per_block - 1) /
        rows_per_block;
    const size_t shared_bytes =
        static_cast<size_t>(cols) * sizeof(float);
    constexpr int tracked_devices = 32;
    static size_t configured_shared_bytes[tracked_devices] = {};
    TORCH_CHECK(
        device >= 0 && device < tracked_devices,
        "INT4 GEMV device index is out of tracked range");
    if (configured_shared_bytes[device] < shared_bytes) {
        int optin_limit = 0;
        const auto query_status = cudaDeviceGetAttribute(
            &optin_limit,
            cudaDevAttrMaxSharedMemoryPerBlockOptin,
            device);
        TORCH_CHECK(
            query_status == cudaSuccess,
            "failed to query opt-in shared memory: ",
            cudaGetErrorString(query_status));
        TORCH_CHECK(
            shared_bytes <= static_cast<size_t>(optin_limit),
            "INT4 GEMV activation row needs ",
            shared_bytes,
            " bytes shared memory, device limit is ",
            optin_limit);
        const auto attr_status = cccp_gpu_func_set_attribute(
            int4_gemv_packed_f32_kernel<
                input_t,
                rows_per_block>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        TORCH_CHECK(
            attr_status == cudaSuccess,
            "failed to configure INT4 GEMV shared memory: ",
            cudaGetErrorString(attr_status));
        configured_shared_bytes[device] = shared_bytes;
    }
    int4_gemv_packed_f32_kernel<input_t, rows_per_block>
        <<<blocks, block, shared_bytes, stream>>>(
            x,
            packed,
            scales,
            output,
            rows,
            cols,
            groups);
}

template <typename input_t, int rows_per_block>
void launch_int4_gemv_packed_f32_vector4_rows(
    const input_t* x,
    const uint8_t* packed,
    const __half* scales,
    float* output,
    int rows,
    int cols,
    int groups,
    int device,
    cudaStream_t stream)
{
    dim3 block(32, rows_per_block);
    const int blocks =
        (rows + rows_per_block - 1) / rows_per_block;
    const size_t shared_bytes =
        static_cast<size_t>(cols) * sizeof(float);
    constexpr int tracked_devices = 32;
    static size_t configured_shared_bytes[tracked_devices] = {};
    TORCH_CHECK(
        device >= 0 && device < tracked_devices,
        "INT4 vector GEMV device index is out of tracked range");
    if (configured_shared_bytes[device] < shared_bytes) {
        int optin_limit = 0;
        const auto query_status = cudaDeviceGetAttribute(
            &optin_limit,
            cudaDevAttrMaxSharedMemoryPerBlockOptin,
            device);
        TORCH_CHECK(
            query_status == cudaSuccess &&
            shared_bytes <= static_cast<size_t>(optin_limit),
            "INT4 vector GEMV shared-memory requirement is unsupported");
        const auto attr_status = cccp_gpu_func_set_attribute(
            int4_gemv_packed_f32_vector4_kernel<
                input_t,
                rows_per_block>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        TORCH_CHECK(
            attr_status == cudaSuccess,
            "failed to configure INT4 vector GEMV shared memory: ",
            cudaGetErrorString(attr_status));
        configured_shared_bytes[device] = shared_bytes;
    }
    int4_gemv_packed_f32_vector4_kernel<
        input_t,
        rows_per_block><<<
            blocks,
            block,
            shared_bytes,
            stream>>>(
                x,
                packed,
                scales,
                output,
                rows,
                cols,
                groups);
}


template <typename input_t, int rows_per_block>
void launch_int4_gemv_packed_f32_vector8_rows(
    const input_t* x,
    const uint8_t* packed,
    const __half* scales,
    float* output,
    int rows,
    int cols,
    int groups,
    int device,
    cudaStream_t stream)
{
    dim3 block(32, rows_per_block);
    const int blocks = (rows + rows_per_block - 1) / rows_per_block;
    const size_t shared_bytes = static_cast<size_t>(cols) * sizeof(float);
    constexpr int tracked_devices = 32;
    static size_t configured[tracked_devices] = {};
    TORCH_CHECK(device >= 0 && device < tracked_devices, "v8 device range");
    if (configured[device] < shared_bytes) {
        int optin = 0;
        cudaDeviceGetAttribute(
            &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        TORCH_CHECK(
            shared_bytes <= static_cast<size_t>(optin),
            "int4 v8 shared memory unsupported");
        cccp_gpu_func_set_attribute(
            int4_gemv_packed_f32_vector8_kernel<input_t, rows_per_block>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        configured[device] = shared_bytes;
    }
    int4_gemv_packed_f32_vector8_kernel<input_t, rows_per_block>
        <<<blocks, block, shared_bytes, stream>>>(
            x, packed, scales, output, rows, cols, groups);
}

template <typename input_t>
void launch_int4_gemv_packed_f32(
    const input_t* x,
    const uint8_t* packed,
    const __half* scales,
    float* output,
    int rows,
    int cols,
    int groups,
    int device,
    cudaStream_t stream,
    bool group_vector)
{
    static const bool use_vector8 = [] {
        const char* flag = std::getenv("CCCP_INT4_GEMV_V8");
        return flag && flag[0] == '1';
    }();
    if (group_vector && groups % 8 == 0 && use_vector8) {
        if (rows <= 2048) {
            launch_int4_gemv_packed_f32_vector8_rows<input_t, 8>(
                x, packed, scales, output, rows, cols, groups,
                device, stream);
        } else {
            launch_int4_gemv_packed_f32_vector8_rows<input_t, 32>(
                x, packed, scales, output, rows, cols, groups,
                device, stream);
        }
        return;
    }
    if (group_vector && groups % 4 == 0) {
        if (rows <= 2048) {
            launch_int4_gemv_packed_f32_vector4_rows<input_t, 8>(
                x, packed, scales, output, rows, cols, groups,
                device, stream);
        } else if (
            (rows == 6144 && cols == 16384) ||
            (rows <= 6144 && cols <= 2048)
        ) {
            launch_int4_gemv_packed_f32_vector4_rows<input_t, 16>(
                x, packed, scales, output, rows, cols, groups,
                device, stream);
        } else {
            launch_int4_gemv_packed_f32_vector4_rows<input_t, 32>(
                x, packed, scales, output, rows, cols, groups,
                device, stream);
        }
        return;
    }
    if (rows <= 2048) {
        launch_int4_gemv_packed_f32_rows<input_t, 8>(
            x, packed, scales, output, rows, cols, groups,
            device, stream);
    } else if (
        (rows == 6144 && cols == 16384) ||
        (rows <= 6144 && cols <= 2048)
    ) {
        launch_int4_gemv_packed_f32_rows<input_t, 16>(
            x, packed, scales, output, rows, cols, groups,
            device, stream);
    } else {
        launch_int4_gemv_packed_f32_rows<input_t, 32>(
            x, packed, scales, output, rows, cols, groups,
            device, stream);
    }
}

template <typename input_t>
void launch_int4_swiglu_packed_f32(
    const input_t* x,
    const uint8_t* gate_packed,
    const __half* gate_scales,
    const uint8_t* up_packed,
    const __half* up_scales,
    float* output,
    int rows,
    int cols,
    int groups,
    int device,
    cudaStream_t stream,
    bool group_vector) {
    dim3 block(32, INT4_ROWS_PER_BLOCK);
    const int blocks =
        (rows + INT4_ROWS_PER_BLOCK - 1) /
        INT4_ROWS_PER_BLOCK;
    const size_t shared_bytes =
        static_cast<size_t>(cols) * sizeof(float);
    constexpr int tracked_devices = 32;
    static size_t configured_shared_bytes[tracked_devices] = {};
    TORCH_CHECK(
        device >= 0 && device < tracked_devices,
        "INT4 SwiGLU device index is out of tracked range");
    if (configured_shared_bytes[device] < shared_bytes) {
        const auto attr_status = cccp_gpu_func_set_attribute(
            int4_swiglu_packed_f32_kernel<input_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        TORCH_CHECK(
            attr_status == cudaSuccess,
            "failed to configure INT4 SwiGLU shared memory: ",
            cudaGetErrorString(attr_status));
        configured_shared_bytes[device] = shared_bytes;
    }
    if (group_vector && groups % 4 == 0) {
        constexpr int vector_tracked_devices = 32;
        static size_t vector_shared_bytes[
            vector_tracked_devices
        ] = {};
        if (vector_shared_bytes[device] < shared_bytes) {
            const auto vector_attr_status = cccp_gpu_func_set_attribute(
                int4_swiglu_packed_f32_vector4_kernel<input_t>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shared_bytes));
            TORCH_CHECK(
                vector_attr_status == cudaSuccess,
                "failed to configure vector INT4 SwiGLU shared memory: ",
                cudaGetErrorString(vector_attr_status));
            vector_shared_bytes[device] = shared_bytes;
        }
        int4_swiglu_packed_f32_vector4_kernel<input_t>
            <<<blocks, block, shared_bytes, stream>>>(
                x,
                gate_packed,
                gate_scales,
                up_packed,
                up_scales,
                output,
                rows,
                cols,
                groups);
    } else {
        int4_swiglu_packed_f32_kernel<input_t>
            <<<blocks, block, shared_bytes, stream>>>(
                x,
                gate_packed,
                gate_scales,
                up_packed,
                up_scales,
                output,
                rows,
                cols,
                groups);
    }
}

torch::Tensor int4_gemv_packed_f32(
    torch::Tensor x,
    torch::Tensor packed,
    torch::Tensor scales,
    long cols,
    long group_size,
    bool group_vector,
    c10::optional<torch::Tensor> output_buffer) {
    TORCH_CHECK(
        x.is_cuda() && packed.is_cuda() && scales.is_cuda(),
        "INT4 GEMV tensors must be CUDA");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat ||
        x.scalar_type() == at::kBFloat16,
        "INT4 GEMV input must be float32 or bfloat16");
    TORCH_CHECK(
        packed.scalar_type() == at::kByte,
        "packed INT4 weights must be uint8");
    TORCH_CHECK(
        scales.scalar_type() == at::kHalf,
        "INT4 scales must be float16");
    TORCH_CHECK(
        x.dim() == 2 && x.size(0) == 1,
        "INT4 GEMV input must be [1,C]");
    TORCH_CHECK(
        packed.dim() == 2 && scales.dim() == 2,
        "INT4 weights and scales must be matrices");
    TORCH_CHECK(
        group_size == 64,
        "direct INT4 GEMV currently requires group size 64");
    TORCH_CHECK(
        cols > 0 && cols % 64 == 0,
        "INT4 columns must be a positive multiple of 64");
    TORCH_CHECK(
        x.size(1) == cols && packed.size(1) * 2 == cols,
        "INT4 GEMV input/weight column mismatch");
    const int rows = static_cast<int>(packed.size(0));
    const int groups = static_cast<int>(cols / group_size);
    TORCH_CHECK(
        scales.size(0) == rows && scales.size(1) == groups,
        "INT4 scale shape mismatch");

    auto xc = x.contiguous();
    auto qc = packed.contiguous();
    auto sc = scales.contiguous();
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty(
            {1, rows},
            x.options().dtype(at::kFloat));
    TORCH_CHECK(
        output.is_cuda() &&
        output.scalar_type() == at::kFloat &&
        output.is_contiguous() &&
        output.sizes() == torch::IntArrayRef({1, rows}) &&
        output.get_device() == packed.get_device(),
        "INT4 GEMV output buffer must be contiguous float32 [1,R] "
        "on the weight device");
    const int device = packed.get_device();
    auto stream = at::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == at::kFloat) {
        launch_int4_gemv_packed_f32<float>(
            xc.data_ptr<float>(),
            qc.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(sc.data_ptr<at::Half>()),
            output.data_ptr<float>(),
            rows,
            static_cast<int>(cols),
            groups,
            device,
            stream,
            group_vector);
    } else {
        launch_int4_gemv_packed_f32<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(
                xc.data_ptr<at::BFloat16>()),
            qc.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(sc.data_ptr<at::Half>()),
            output.data_ptr<float>(),
            rows,
            static_cast<int>(cols),
            groups,
            device,
            stream,
            group_vector);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <typename input_t, int rows_per_block>
void launch_block_fp8_gemv_f32_rows(
    const input_t* input,
    const uint8_t* weights,
    const float* scales,
    float* output,
    const int rows,
    const int cols,
    const int scale_cols,
    cudaStream_t stream)
{
    const size_t shared_bytes =
        static_cast<size_t>(cols) * sizeof(__nv_bfloat16);
    if (shared_bytes > 48 * 1024) {
        const auto status = cccp_gpu_func_set_attribute(
            block_fp8_gemv_f32_kernel<
                input_t,
                rows_per_block>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        TORCH_CHECK(
            status == cudaSuccess,
            "failed to configure block-FP8 GEMV shared memory: ",
            cudaGetErrorString(status));
    }
    block_fp8_gemv_f32_kernel<
        input_t,
        rows_per_block><<<
            (rows + rows_per_block - 1) / rows_per_block,
            dim3(32, rows_per_block),
            shared_bytes,
            stream>>>(
                input,
                weights,
                scales,
                output,
                rows,
                cols,
                scale_cols);
}

template <typename input_t>
void launch_block_fp8_gemv_f32(
    const input_t* input,
    const uint8_t* weights,
    const float* scales,
    float* output,
    const int rows,
    const int cols,
    const int scale_cols,
    cudaStream_t stream)
{
    const char* setting = std::getenv("CCCP_FP8_GEMV_WARPS");
    const int warps = setting == nullptr ? 16 : std::atoi(setting);
    if (warps <= 8) {
        launch_block_fp8_gemv_f32_rows<input_t, 8>(
            input, weights, scales, output,
            rows, cols, scale_cols, stream);
    } else if (warps <= 16) {
        launch_block_fp8_gemv_f32_rows<input_t, 16>(
            input, weights, scales, output,
            rows, cols, scale_cols, stream);
    } else {
        launch_block_fp8_gemv_f32_rows<input_t, 32>(
            input, weights, scales, output,
            rows, cols, scale_cols, stream);
    }
}

torch::Tensor block_fp8_gemv_f32(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor scales,
    long cols,
    long block_size,
    c10::optional<torch::Tensor> output_buffer)
{
    TORCH_CHECK(
        input.is_cuda() && weights.is_cuda() && scales.is_cuda(),
        "block-FP8 GEMV tensors must be CUDA");
    TORCH_CHECK(
        input.scalar_type() == at::kFloat ||
        input.scalar_type() == at::kBFloat16,
        "block-FP8 GEMV input must be float32 or bfloat16");
    TORCH_CHECK(
        weights.scalar_type() == at::kByte &&
        scales.scalar_type() == at::kFloat,
        "block-FP8 weights/scales must be uint8/float32");
    TORCH_CHECK(
        input.dim() == 2 && input.size(0) == 1 &&
        weights.dim() == 2 && scales.dim() == 2,
        "block-FP8 GEMV expects input [1,C] and matrix weights/scales");
    TORCH_CHECK(
        block_size == 128 && cols > 0 &&
        input.size(1) == cols && weights.size(1) == cols,
        "block-FP8 GEMV currently requires 128x128 blocks");
    const int rows = static_cast<int>(weights.size(0));
    const int scale_rows = (rows + 127) / 128;
    const int scale_cols = (static_cast<int>(cols) + 127) / 128;
    TORCH_CHECK(
        scales.size(0) == scale_rows &&
        scales.size(1) == scale_cols,
        "block-FP8 scale matrix shape mismatch");
    TORCH_CHECK(
        input.get_device() == weights.get_device() &&
        input.get_device() == scales.get_device(),
        "block-FP8 GEMV tensors must share one device");
    auto contiguous_input = input.contiguous();
    auto contiguous_weights = weights.contiguous();
    auto contiguous_scales = scales.contiguous();
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty(
            {1, rows},
            input.options().dtype(at::kFloat));
    TORCH_CHECK(
        output.is_cuda() &&
        output.scalar_type() == at::kFloat &&
        output.is_contiguous() &&
        output.sizes() == torch::IntArrayRef({1, rows}) &&
        output.get_device() == input.get_device(),
        "block-FP8 output must be contiguous float32 [1,R]");
    auto stream = at::cuda::getCurrentCUDAStream();
    if (input.scalar_type() == at::kFloat) {
        launch_block_fp8_gemv_f32<float>(
            contiguous_input.data_ptr<float>(),
            contiguous_weights.data_ptr<uint8_t>(),
            contiguous_scales.data_ptr<float>(),
            output.data_ptr<float>(),
            rows,
            static_cast<int>(cols),
            scale_cols,
            stream);
    } else {
        launch_block_fp8_gemv_f32<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(
                contiguous_input.data_ptr<at::BFloat16>()),
            contiguous_weights.data_ptr<uint8_t>(),
            contiguous_scales.data_ptr<float>(),
            output.data_ptr<float>(),
            rows,
            static_cast<int>(cols),
            scale_cols,
            stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <typename input_t, int rows_per_block>
void launch_block_fp8_grouped_gemv_f32_rows(
    const input_t* input,
    const int64_t* weight_ptrs,
    const int64_t* scale_ptrs,
    const int32_t* row_offsets,
    float* output,
    const int groups,
    const int input_rows,
    const int total_rows,
    const int cols,
    const int scale_cols,
    cudaStream_t stream)
{
    const size_t shared_bytes =
        static_cast<size_t>(cols) * sizeof(__nv_bfloat16);
    if (shared_bytes > 48 * 1024) {
        const auto status = cccp_gpu_func_set_attribute(
            block_fp8_grouped_gemv_f32_kernel<
                input_t,
                rows_per_block>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        TORCH_CHECK(
            status == cudaSuccess,
            "failed to configure grouped block-FP8 GEMV shared memory: ",
            cudaGetErrorString(status));
    }
    block_fp8_grouped_gemv_f32_kernel<
        input_t,
        rows_per_block><<<
            (total_rows + rows_per_block - 1) / rows_per_block,
            dim3(32, rows_per_block),
            shared_bytes,
            stream>>>(
                input,
                weight_ptrs,
                scale_ptrs,
                row_offsets,
                output,
                groups,
                input_rows,
                total_rows,
                cols,
                scale_cols);
}

template <typename input_t, int warps>
void launch_block_fp8_grouped_gemv_f32_rows2(
    const input_t* input,
    const int64_t* weight_ptrs,
    const int64_t* scale_ptrs,
    const int32_t* row_offsets,
    float* output,
    const int groups,
    const int total_rows,
    const int cols,
    const int scale_cols,
    cudaStream_t stream)
{
    const size_t shared_bytes =
        static_cast<size_t>(cols) * sizeof(__nv_bfloat16);
    if (shared_bytes > 48 * 1024) {
        const auto status = cccp_gpu_func_set_attribute(
            block_fp8_grouped_gemv_f32_rows2_kernel<input_t, warps>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        TORCH_CHECK(
            status == cudaSuccess,
            "failed to configure two-row grouped block-FP8 GEMV shared "
            "memory: ",
            cudaGetErrorString(status));
    }
    block_fp8_grouped_gemv_f32_rows2_kernel<input_t, warps><<<
        (total_rows + 2 * warps - 1) / (2 * warps),
        dim3(32, warps),
        shared_bytes,
        stream>>>(
            input,
            weight_ptrs,
            scale_ptrs,
            row_offsets,
            output,
            groups,
            total_rows,
            cols,
            scale_cols);
}

template <typename input_t>
void launch_block_fp8_grouped_gemv_f32(
    const input_t* input,
    const int64_t* weight_ptrs,
    const int64_t* scale_ptrs,
    const int32_t* row_offsets,
    float* output,
    const int groups,
    const int input_rows,
    const int total_rows,
    const int cols,
    const int scale_cols,
    cudaStream_t stream)
{
    const char* rows_setting =
        std::getenv("CCCP_FP8_GEMV_ROWS_PER_WARP");
    const int rows_per_warp =
        rows_setting == nullptr ? 2 : std::atoi(rows_setting);
    const char* setting = std::getenv("CCCP_FP8_GEMV_WARPS");
    const int warps = setting == nullptr
        ? (rows_per_warp == 2 ? 8 : 16)
        : std::atoi(setting);
    if (rows_per_warp == 2 && input_rows == 1) {
        if (warps <= 8) {
            launch_block_fp8_grouped_gemv_f32_rows2<input_t, 8>(
                input, weight_ptrs, scale_ptrs, row_offsets, output,
                groups, total_rows, cols, scale_cols, stream);
        } else if (warps <= 16) {
            launch_block_fp8_grouped_gemv_f32_rows2<input_t, 16>(
                input, weight_ptrs, scale_ptrs, row_offsets, output,
                groups, total_rows, cols, scale_cols, stream);
        } else {
            launch_block_fp8_grouped_gemv_f32_rows2<input_t, 32>(
                input, weight_ptrs, scale_ptrs, row_offsets, output,
                groups, total_rows, cols, scale_cols, stream);
        }
        return;
    }
    if (warps <= 8) {
        launch_block_fp8_grouped_gemv_f32_rows<
            input_t, 8>(
            input, weight_ptrs, scale_ptrs, row_offsets, output,
            groups, input_rows, total_rows, cols, scale_cols, stream);
    } else if (warps <= 16) {
        launch_block_fp8_grouped_gemv_f32_rows<
            input_t, 16>(
            input, weight_ptrs, scale_ptrs, row_offsets, output,
            groups, input_rows, total_rows, cols, scale_cols, stream);
    } else {
        launch_block_fp8_grouped_gemv_f32_rows<
            input_t, 32>(
            input, weight_ptrs, scale_ptrs, row_offsets, output,
            groups, input_rows, total_rows, cols, scale_cols, stream);
    }
}

torch::Tensor block_fp8_grouped_gemv_f32(
    torch::Tensor input,
    torch::Tensor weight_ptrs,
    torch::Tensor scale_ptrs,
    torch::Tensor row_offsets,
    long total_rows_value,
    long cols,
    long block_size,
    c10::optional<torch::Tensor> output_buffer)
{
    TORCH_CHECK(
        input.is_cuda() && weight_ptrs.is_cuda() &&
        scale_ptrs.is_cuda() && row_offsets.is_cuda(),
        "grouped block-FP8 GEMV metadata must be CUDA");
    TORCH_CHECK(
        input.scalar_type() == at::kFloat ||
        input.scalar_type() == at::kBFloat16,
        "grouped block-FP8 input must be float32 or bfloat16");
    TORCH_CHECK(
        weight_ptrs.scalar_type() == at::kLong &&
        scale_ptrs.scalar_type() == at::kLong &&
        row_offsets.scalar_type() == at::kInt,
        "grouped block-FP8 pointer/offset metadata dtype mismatch");
    TORCH_CHECK(
        input.dim() == 2 &&
        weight_ptrs.dim() == 1 && scale_ptrs.dim() == 1 &&
        row_offsets.dim() == 1,
        "grouped block-FP8 GEMV metadata must be one-dimensional");
    TORCH_CHECK(
        block_size == 128 && cols > 0 && input.size(1) == cols,
        "grouped block-FP8 GEMV currently requires 128x128 blocks");
    const int groups = static_cast<int>(weight_ptrs.numel());
    TORCH_CHECK(
        groups > 0 && scale_ptrs.numel() == groups &&
        row_offsets.numel() == groups + 1,
        "grouped block-FP8 metadata length mismatch");
    TORCH_CHECK(
        input.size(0) == 1 || input.size(0) == groups,
        "grouped block-FP8 input rows must be one or match groups");
    TORCH_CHECK(
        input.get_device() == weight_ptrs.get_device() &&
        input.get_device() == scale_ptrs.get_device() &&
        input.get_device() == row_offsets.get_device(),
        "grouped block-FP8 tensors must share one device");
    auto contiguous_input = input.contiguous();
    auto contiguous_weight_ptrs = weight_ptrs.contiguous();
    auto contiguous_scale_ptrs = scale_ptrs.contiguous();
    auto contiguous_row_offsets = row_offsets.contiguous();
    const int total_rows = static_cast<int>(total_rows_value);
    TORCH_CHECK(total_rows > 0, "grouped block-FP8 rows must be positive");
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty(
            {1, total_rows},
            input.options().dtype(at::kFloat));
    TORCH_CHECK(
        output.is_cuda() && output.scalar_type() == at::kFloat &&
        output.is_contiguous() &&
        output.sizes() == torch::IntArrayRef({1, total_rows}) &&
        output.get_device() == input.get_device(),
        "grouped block-FP8 output must be contiguous float32 [1,R]");
    const int scale_cols = (static_cast<int>(cols) + 127) / 128;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (input.scalar_type() == at::kFloat) {
        launch_block_fp8_grouped_gemv_f32<float>(
            contiguous_input.data_ptr<float>(),
            contiguous_weight_ptrs.data_ptr<int64_t>(),
            contiguous_scale_ptrs.data_ptr<int64_t>(),
            contiguous_row_offsets.data_ptr<int32_t>(),
            output.data_ptr<float>(),
            groups,
            static_cast<int>(input.size(0)),
            total_rows,
            static_cast<int>(cols),
            scale_cols,
            stream);
    } else {
        launch_block_fp8_grouped_gemv_f32<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(
                contiguous_input.data_ptr<at::BFloat16>()),
            contiguous_weight_ptrs.data_ptr<int64_t>(),
            contiguous_scale_ptrs.data_ptr<int64_t>(),
            contiguous_row_offsets.data_ptr<int32_t>(),
            output.data_ptr<float>(),
            groups,
            static_cast<int>(input.size(0)),
            total_rows,
            static_cast<int>(cols),
            scale_cols,
            stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<torch::Tensor> glm_norm_qkv_int4(
    torch::Tensor x,
    torch::Tensor norm_weight,
    torch::Tensor q_packed,
    torch::Tensor q_scales,
    torch::Tensor kv_packed,
    torch::Tensor kv_scales,
    long cols,
    long group_size,
    double eps,
    c10::optional<torch::Tensor> residual_update,
    c10::optional<torch::Tensor> q_output_buffer,
    c10::optional<torch::Tensor> kv_output_buffer)
{
    TORCH_CHECK(
        x.is_cuda() && norm_weight.is_cuda() &&
        q_packed.is_cuda() && q_scales.is_cuda() &&
        kv_packed.is_cuda() && kv_scales.is_cuda(),
        "GLM fused Q/K inputs must be CUDA");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat &&
        norm_weight.scalar_type() == at::kFloat &&
        q_packed.scalar_type() == at::kByte &&
        kv_packed.scalar_type() == at::kByte &&
        q_scales.scalar_type() == at::kHalf &&
        kv_scales.scalar_type() == at::kHalf,
        "GLM fused Q/K dtypes do not match");
    TORCH_CHECK(
        x.is_contiguous() && norm_weight.is_contiguous() &&
        q_packed.is_contiguous() && q_scales.is_contiguous() &&
        kv_packed.is_contiguous() && kv_scales.is_contiguous(),
        "GLM fused Q/K tensors must be contiguous");
    TORCH_CHECK(
        x.dim() == 2 && x.size(0) == 1 &&
        norm_weight.dim() == 1 &&
        q_packed.dim() == 2 && kv_packed.dim() == 2 &&
        q_scales.dim() == 2 && kv_scales.dim() == 2 &&
        group_size == 64 && cols > 0 && cols % 64 == 0 &&
        x.size(1) == cols && norm_weight.size(0) == cols &&
        q_packed.size(1) * 2 == cols &&
        kv_packed.size(1) * 2 == cols,
        "GLM fused Q/K shapes do not match");
    const int q_rows = static_cast<int>(q_packed.size(0));
    const int kv_rows = static_cast<int>(kv_packed.size(0));
    const int groups = static_cast<int>(cols / group_size);
    TORCH_CHECK(
        q_rows > 0 && kv_rows > 0 &&
        q_rows % INT4_ROWS_PER_BLOCK == 0 &&
        q_scales.sizes() ==
            torch::IntArrayRef({q_rows, groups}) &&
        kv_scales.sizes() ==
            torch::IntArrayRef({kv_rows, groups}),
        "GLM fused Q/K row/scale shapes do not match");
    const int device = x.get_device();
    TORCH_CHECK(
        norm_weight.get_device() == device &&
        q_packed.get_device() == device &&
        q_scales.get_device() == device &&
        kv_packed.get_device() == device &&
        kv_scales.get_device() == device,
        "GLM fused Q/K tensors must share one device");
    if (residual_update.has_value()) {
        const auto update = residual_update.value();
        TORCH_CHECK(
            update.is_cuda() &&
            update.scalar_type() == at::kFloat &&
            update.is_contiguous() &&
            update.sizes() == x.sizes() &&
            update.get_device() == device,
            "GLM fused residual update must match x");
    }

    auto q_output = q_output_buffer.has_value()
        ? q_output_buffer.value()
        : torch::empty(
            {1, q_rows},
            x.options().dtype(at::kFloat));
    auto kv_output = kv_output_buffer.has_value()
        ? kv_output_buffer.value()
        : torch::empty(
            {1, kv_rows},
            x.options().dtype(at::kFloat));
    TORCH_CHECK(
        q_output.is_cuda() &&
        kv_output.is_cuda() &&
        q_output.scalar_type() == at::kFloat &&
        kv_output.scalar_type() == at::kFloat &&
        q_output.is_contiguous() &&
        kv_output.is_contiguous() &&
        q_output.sizes() ==
            torch::IntArrayRef({1, q_rows}) &&
        kv_output.sizes() ==
            torch::IntArrayRef({1, kv_rows}) &&
        q_output.get_device() == device &&
        kv_output.get_device() == device,
        "GLM fused Q/K output buffers must be contiguous float32 "
        "[1,q_rows]/[1,kv_rows] on the input device");
    auto residual_output = residual_update.has_value()
        ? torch::empty_like(x)
        : torch::Tensor();
    const size_t shared_bytes =
        static_cast<size_t>(cols) * sizeof(float);
    constexpr int tracked_devices = 32;
    static size_t configured_shared_bytes[tracked_devices] = {};
    static size_t configured_residual_shared_bytes[
        tracked_devices
    ] = {};
    TORCH_CHECK(
        device >= 0 && device < tracked_devices,
        "GLM fused Q/K device index is out of tracked range");
    if (configured_shared_bytes[device] < shared_bytes) {
        const auto attr_status = cccp_gpu_func_set_attribute(
            glm_norm_qkv_int4_kernel<false, 32>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        TORCH_CHECK(
            attr_status == cudaSuccess,
            "failed to configure GLM fused Q/K shared memory: ",
            cudaGetErrorString(attr_status));
        configured_shared_bytes[device] = shared_bytes;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    const int rows = q_rows + kv_rows;
    const int blocks =
        (rows + 31) / 32;
    if (residual_update.has_value()) {
        const auto update = residual_update.value();
        if (
            configured_residual_shared_bytes[device] <
            shared_bytes
        ) {
            const auto attr_status = cccp_gpu_func_set_attribute(
                glm_norm_qkv_int4_kernel<true, 32>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shared_bytes));
            TORCH_CHECK(
                attr_status == cudaSuccess,
                "failed to configure residual GLM Q/K shared memory: ",
                cudaGetErrorString(attr_status));
            configured_residual_shared_bytes[device] =
                shared_bytes;
        }
        glm_norm_qkv_int4_kernel<true, 32><<<
            blocks,
            dim3(32, 32),
            shared_bytes,
            stream>>>(
                x.data_ptr<float>(),
                update.data_ptr<float>(),
                norm_weight.data_ptr<float>(),
                q_packed.data_ptr<uint8_t>(),
                reinterpret_cast<const __half*>(
                    q_scales.data_ptr<at::Half>()),
                kv_packed.data_ptr<uint8_t>(),
                reinterpret_cast<const __half*>(
                    kv_scales.data_ptr<at::Half>()),
                q_output.data_ptr<float>(),
                kv_output.data_ptr<float>(),
                residual_output.data_ptr<float>(),
                q_rows,
                kv_rows,
                static_cast<int>(cols),
                groups,
                static_cast<float>(eps));
    } else {
        glm_norm_qkv_int4_kernel<false, 32><<<
            blocks,
            dim3(32, 32),
            shared_bytes,
            stream>>>(
                x.data_ptr<float>(),
                nullptr,
                norm_weight.data_ptr<float>(),
                q_packed.data_ptr<uint8_t>(),
                reinterpret_cast<const __half*>(
                    q_scales.data_ptr<at::Half>()),
                kv_packed.data_ptr<uint8_t>(),
                reinterpret_cast<const __half*>(
                    kv_scales.data_ptr<at::Half>()),
                q_output.data_ptr<float>(),
                kv_output.data_ptr<float>(),
                nullptr,
                q_rows,
                kv_rows,
                static_cast<int>(cols),
                groups,
                static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (residual_update.has_value())
        return {q_output, kv_output, residual_output};
    return {q_output, kv_output};
}

std::vector<torch::Tensor> glm_residual_norm_router(
    torch::Tensor residual,
    torch::Tensor update,
    torch::Tensor norm_weight,
    torch::Tensor router_weight,
    double eps)
{
    TORCH_CHECK(
        residual.is_cuda() && update.is_cuda() &&
        norm_weight.is_cuda() && router_weight.is_cuda(),
        "GLM residual/router tensors must be CUDA");
    TORCH_CHECK(
        residual.scalar_type() == at::kFloat &&
        update.scalar_type() == at::kFloat &&
        norm_weight.scalar_type() == at::kFloat &&
        router_weight.scalar_type() == at::kFloat,
        "GLM residual/router tensors must be float32");
    TORCH_CHECK(
        residual.is_contiguous() && update.is_contiguous() &&
        norm_weight.is_contiguous() && router_weight.is_contiguous(),
        "GLM residual/router tensors must be contiguous");
    TORCH_CHECK(
        residual.dim() == 2 && residual.size(0) == 1 &&
        update.sizes() == residual.sizes() &&
        norm_weight.dim() == 1 &&
        norm_weight.size(0) == residual.size(1) &&
        router_weight.dim() == 2 &&
        router_weight.size(1) == residual.size(1),
        "GLM residual/router shapes do not match");
    const int device = residual.get_device();
    TORCH_CHECK(
        update.get_device() == device &&
        norm_weight.get_device() == device &&
        router_weight.get_device() == device,
        "GLM residual/router tensors must share one device");
    int rows = static_cast<int>(router_weight.size(0));
    int cols = static_cast<int>(residual.size(1));
    TORCH_CHECK(
        rows > 0 && cols > 0 && cols % 256 == 0,
        "GLM residual/router dimensions are not supported");

    auto residual_output = torch::empty_like(residual);
    auto norm_output = torch::empty_like(residual);
    auto logits_output = torch::empty(
        {1, rows},
        residual.options());
    dim3 block(32, 8);
    const int blocks = (rows + 7) / 8;
    const size_t shared_bytes =
        static_cast<size_t>(cols) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    glm_residual_norm_router_kernel<<<
        blocks,
        block,
        shared_bytes,
        stream>>>(
            residual.data_ptr<float>(),
            update.data_ptr<float>(),
            norm_weight.data_ptr<float>(),
            router_weight.data_ptr<float>(),
            residual_output.data_ptr<float>(),
            norm_output.data_ptr<float>(),
            logits_output.data_ptr<float>(),
            rows,
            cols,
            static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {residual_output, norm_output, logits_output};
}

std::vector<torch::Tensor> glm_residual_norm_router_norm_out(
    torch::Tensor residual,
    torch::Tensor update,
    torch::Tensor norm_weight,
    torch::Tensor router_weight,
    double eps,
    torch::Tensor norm_output,
    c10::optional<torch::Tensor> residual_output_buffer,
    c10::optional<torch::Tensor> logits_output_buffer)
{
    TORCH_CHECK(
        residual.is_cuda() && update.is_cuda() &&
        norm_weight.is_cuda() && router_weight.is_cuda() &&
        norm_output.is_cuda(),
        "GLM residual/router output-buffer tensors must be CUDA");
    TORCH_CHECK(
        residual.scalar_type() == at::kFloat &&
        update.scalar_type() == at::kFloat &&
        norm_weight.scalar_type() == at::kFloat &&
        router_weight.scalar_type() == at::kFloat &&
        norm_output.scalar_type() == at::kFloat,
        "GLM residual/router output-buffer tensors must be float32");
    TORCH_CHECK(
        residual.is_contiguous() && update.is_contiguous() &&
        norm_weight.is_contiguous() && router_weight.is_contiguous() &&
        norm_output.is_contiguous() &&
        residual.dim() == 2 && residual.size(0) == 1 &&
        update.sizes() == residual.sizes() &&
        norm_output.sizes() == residual.sizes() &&
        norm_weight.dim() == 1 &&
        norm_weight.size(0) == residual.size(1) &&
        router_weight.dim() == 2 &&
        router_weight.size(1) == residual.size(1),
        "GLM residual/router output-buffer shapes do not match");
    const int device = residual.get_device();
    TORCH_CHECK(
        update.get_device() == device &&
        norm_weight.get_device() == device &&
        router_weight.get_device() == device &&
        norm_output.get_device() == device,
        "GLM residual/router output-buffer tensors must share one device");
    const int rows = static_cast<int>(router_weight.size(0));
    const int cols = static_cast<int>(residual.size(1));
    TORCH_CHECK(
        rows > 0 && cols > 0 && cols % 256 == 0,
        "GLM residual/router output-buffer dimensions are unsupported");
    auto residual_output = residual_output_buffer.has_value()
        ? residual_output_buffer.value()
        : torch::empty_like(residual);
    auto logits_output = logits_output_buffer.has_value()
        ? logits_output_buffer.value()
        : torch::empty(
            {1, rows},
            residual.options());
    TORCH_CHECK(
        residual_output.is_cuda() &&
        logits_output.is_cuda() &&
        residual_output.scalar_type() == at::kFloat &&
        logits_output.scalar_type() == at::kFloat &&
        residual_output.is_contiguous() &&
        logits_output.is_contiguous() &&
        residual_output.sizes() == residual.sizes() &&
        logits_output.sizes() ==
            torch::IntArrayRef({1, rows}) &&
        residual_output.get_device() == device &&
        logits_output.get_device() == device,
        "GLM residual/router caller outputs must be contiguous float32 "
        "on the input device");
    dim3 block(32, 8);
    const int blocks = (rows + 7) / 8;
    const size_t shared_bytes =
        static_cast<size_t>(cols) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    glm_residual_norm_router_kernel<<<
        blocks,
        block,
        shared_bytes,
        stream>>>(
            residual.data_ptr<float>(),
            update.data_ptr<float>(),
            norm_weight.data_ptr<float>(),
            router_weight.data_ptr<float>(),
            residual_output.data_ptr<float>(),
            norm_output.data_ptr<float>(),
            logits_output.data_ptr<float>(),
            rows,
            cols,
            static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {residual_output, norm_output, logits_output};
}

torch::Tensor residual_add3(
    torch::Tensor residual,
    torch::Tensor routed,
    torch::Tensor shared)
{
    TORCH_CHECK(
        residual.is_cuda() && routed.is_cuda() && shared.is_cuda(),
        "three-way residual tensors must be CUDA");
    TORCH_CHECK(
        (
            residual.scalar_type() == at::kFloat
            || residual.scalar_type() == at::kBFloat16
        )
        && routed.scalar_type() == residual.scalar_type()
        && shared.scalar_type() == residual.scalar_type(),
        "three-way residual tensors must share float32 or bfloat16 dtype");
    TORCH_CHECK(
        residual.is_contiguous() &&
        routed.is_contiguous() &&
        shared.is_contiguous() &&
        routed.sizes() == residual.sizes() &&
        shared.sizes() == residual.sizes(),
        "three-way residual tensors must be contiguous and shape-equal");
    const int device = residual.get_device();
    TORCH_CHECK(
        routed.get_device() == device &&
        shared.get_device() == device,
        "three-way residual tensors must share one device");
    auto output = torch::empty_like(residual);
    const int count = static_cast<int>(residual.numel());
    const int blocks = std::min(1024, (count + 255) / 256);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (residual.scalar_type() == at::kFloat) {
        glm_moe_residual_add_kernel<<<blocks, 256, 0, stream>>>(
            residual.data_ptr<float>(),
            routed.data_ptr<float>(),
            shared.data_ptr<float>(),
            output.data_ptr<float>(),
            count);
    } else {
        residual_add3_bf16_kernel<<<blocks, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                routed.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                shared.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            count);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor glm_ep_reduce_residual(
    std::vector<torch::Tensor> contributions,
    torch::Tensor residual)
{
    TORCH_CHECK(
        !contributions.empty() && contributions.size() <= 16,
        "GLM TP reduction requires 1 to 16 contributions");
    auto primary_partial = contributions[0];
    TORCH_CHECK(
        primary_partial.is_cuda() &&
        residual.is_cuda(),
        "GLM EP reduction tensors must be CUDA");
    TORCH_CHECK(
        primary_partial.scalar_type() == at::kFloat &&
        residual.scalar_type() == at::kFloat,
        "GLM EP reduction tensors must be float32");
    TORCH_CHECK(
        primary_partial.is_contiguous() &&
        residual.is_contiguous() &&
        primary_partial.numel() == residual.numel(),
        "GLM EP reduction tensors must be contiguous and size-equal");
    const int device = primary_partial.get_device();
    TORCH_CHECK(
        residual.get_device() == device,
        "GLM TP residual must share the primary contribution device");
    for (const auto contribution : contributions) {
        TORCH_CHECK(
            contribution.is_cuda() &&
            contribution.scalar_type() == at::kFloat &&
            contribution.is_contiguous() &&
            contribution.numel() == primary_partial.numel(),
            "GLM TP contributions must be contiguous float32 tensors "
            "with matching size");
        if (contribution.get_device() != device)
            ensure_peer_access(
                device,
                contribution.get_device(),
                "GLM TP contribution reduction");
    }
    const float* contribution_ptrs[16] = {};
    for (size_t index = 1; index < contributions.size(); ++index)
        contribution_ptrs[index] = contributions[index].data_ptr<float>();
    const int count = static_cast<int>(primary_partial.numel());
    const int blocks = std::min(1024, (count + 255) / 256);
    auto stream = at::cuda::getCurrentCUDAStream();
    glm_ep_reduce_residual_kernel<<<blocks, 256, 0, stream>>>(
        primary_partial.data_ptr<float>(),
        contribution_ptrs[1],
        contribution_ptrs[2],
        contribution_ptrs[3],
        contribution_ptrs[4],
        contribution_ptrs[5],
        contribution_ptrs[6],
        contribution_ptrs[7],
        contribution_ptrs[8],
        contribution_ptrs[9],
        contribution_ptrs[10],
        contribution_ptrs[11],
        contribution_ptrs[12],
        contribution_ptrs[13],
        contribution_ptrs[14],
        contribution_ptrs[15],
        residual.data_ptr<float>(),
        static_cast<int>(contributions.size()),
        count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return primary_partial.view(residual.sizes());
}

__global__ void tp1_publish_bf16_kernel(
    const float* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const int count)
{
    for (
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < count;
        index += blockDim.x * gridDim.x
    )
        output[index] = __float2bfloat16_rn(input[index]);
}

void launch_tp_all_rank_reduce_one(
    const std::vector<torch::Tensor>& contributions,
    torch::Tensor output,
    cudaStream_t stream)
{
    TORCH_CHECK(
        !contributions.empty() && contributions.size() <= 16,
        "TP all-rank reduction requires 1 to 16 contributions");
    TORCH_CHECK(
        output.is_cuda() &&
        output.is_contiguous() &&
        (
            output.scalar_type() == at::kFloat ||
            output.scalar_type() == at::kBFloat16
        ),
        "TP all-rank output must be contiguous CUDA float32/BF16");
    const int target = output.get_device();
    const int count = static_cast<int>(output.numel());
    const float* pointers[16] = {};
    for (size_t index = 0; index < contributions.size(); ++index) {
        const auto contribution = contributions[index];
        TORCH_CHECK(
            contribution.is_cuda() &&
            contribution.scalar_type() == at::kFloat &&
            contribution.is_contiguous() &&
            contribution.numel() == count,
            "TP all-rank contributions must be matching contiguous "
            "CUDA float32 tensors");
        if (contribution.get_device() != target)
            ensure_peer_access(
                target,
                contribution.get_device(),
                "TP all-rank contribution");
        pointers[index] = contribution.data_ptr<float>();
    }
    // Width-one TP is a publication, not a collective.  Avoid the generic
    // sixteen-pointer reduction kernel and its all-rank bookkeeping.  A
    // float output can use an asynchronous D2D copy; BF16 needs only a narrow
    // conversion kernel (the specialization is defined below).
    if (contributions.size() == 1 && output.scalar_type() == at::kFloat) {
        if (output.data_ptr<float>() != pointers[0])
            C10_CUDA_CHECK(cudaMemcpyAsync(
                output.data_ptr<float>(),
                pointers[0],
                static_cast<size_t>(count) * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream));
        return;
    }
    if (contributions.size() == 1) {
        const int blocks = std::min(1024, (count + 255) / 256);
        tp1_publish_bf16_kernel<<<blocks, 256, 0, stream>>>(
            pointers[0],
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    const int blocks = std::min(1024, (count + 255) / 256);
#define CCCP_ALL_RANK_ARGUMENTS \
    pointers[0], pointers[1], pointers[2], pointers[3], \
    pointers[4], pointers[5], pointers[6], pointers[7], \
    pointers[8], pointers[9], pointers[10], pointers[11], \
    pointers[12], pointers[13], pointers[14], pointers[15], \
    static_cast<int>(contributions.size()), count
    if (output.scalar_type() == at::kFloat) {
        tp_all_rank_reduce_kernel<float><<<
            blocks, 256, 0, stream>>>(
                output.data_ptr<float>(),
                CCCP_ALL_RANK_ARGUMENTS);
    } else {
        tp_all_rank_reduce_kernel<__nv_bfloat16><<<
            blocks, 256, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(
                    output.data_ptr<at::BFloat16>()),
                CCCP_ALL_RANK_ARGUMENTS);
    }
#undef CCCP_ALL_RANK_ARGUMENTS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_tp_moe_finalize_one(
    const std::vector<torch::Tensor>& routed_contributions,
    const std::vector<torch::Tensor>& shared_contributions,
    torch::Tensor residual,
    torch::Tensor routed_workspace,
    torch::Tensor shared_workspace,
    torch::Tensor output,
    cudaStream_t stream,
    const bool fused)
{
    TORCH_CHECK(
        residual.is_cuda() &&
        routed_workspace.is_cuda() &&
        shared_workspace.is_cuda() &&
        output.is_cuda() &&
        residual.scalar_type() == at::kBFloat16 &&
        routed_workspace.scalar_type() == at::kBFloat16 &&
        shared_workspace.scalar_type() == at::kBFloat16 &&
        output.scalar_type() == at::kBFloat16 &&
        residual.is_contiguous() &&
        routed_workspace.is_contiguous() &&
        shared_workspace.is_contiguous() &&
        output.is_contiguous() &&
        residual.numel() == output.numel() &&
        routed_workspace.numel() == output.numel() &&
        shared_workspace.numel() == output.numel(),
        "TP MoE finalizer buffers must be matching contiguous BF16 CUDA "
        "tensors");
    const int target = output.get_device();
    TORCH_CHECK(
        residual.get_device() == target &&
        routed_workspace.get_device() == target &&
        shared_workspace.get_device() == target,
        "TP MoE finalizer buffers must share one target device");
    if (fused) {
        TORCH_CHECK(
            !routed_contributions.empty() &&
            routed_contributions.size() == shared_contributions.size() &&
            routed_contributions.size() <= 16,
            "fused TP MoE finalizer requires matching 1 to 16 rank "
            "contributions");
        const float* routed_ptrs[16] = {};
        const float* shared_ptrs[16] = {};
        for (size_t rank = 0; rank < routed_contributions.size(); ++rank) {
            const auto routed = routed_contributions[rank];
            const auto shared = shared_contributions[rank];
            TORCH_CHECK(
                routed.is_cuda() &&
                shared.is_cuda() &&
                routed.scalar_type() == at::kFloat &&
                shared.scalar_type() == at::kFloat &&
                routed.is_contiguous() &&
                shared.is_contiguous() &&
                routed.numel() == output.numel() &&
                shared.numel() == output.numel(),
                "fused TP MoE contributions must be matching contiguous "
                "CUDA float32 tensors");
            if (routed.get_device() != target)
                ensure_peer_access(
                    target,
                    routed.get_device(),
                    "fused TP routed contribution");
            if (shared.get_device() != target)
                ensure_peer_access(
                    target,
                    shared.get_device(),
                    "fused TP shared contribution");
            routed_ptrs[rank] = routed.data_ptr<float>();
            shared_ptrs[rank] = shared.data_ptr<float>();
        }
        const int count = static_cast<int>(output.numel());
        const int blocks = std::min(1024, (count + 255) / 256);
#define CCCP_FUSED_MOE_POINTERS(prefix) \
        prefix##_ptrs[0], prefix##_ptrs[1], prefix##_ptrs[2], \
        prefix##_ptrs[3], prefix##_ptrs[4], prefix##_ptrs[5], \
        prefix##_ptrs[6], prefix##_ptrs[7], prefix##_ptrs[8], \
        prefix##_ptrs[9], prefix##_ptrs[10], prefix##_ptrs[11], \
        prefix##_ptrs[12], prefix##_ptrs[13], prefix##_ptrs[14], \
        prefix##_ptrs[15]
        tp_moe_finalize_all_rank_bf16_kernel<<<
            blocks, 256, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(
                    output.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    residual.data_ptr<at::BFloat16>()),
                CCCP_FUSED_MOE_POINTERS(routed),
                CCCP_FUSED_MOE_POINTERS(shared),
                static_cast<int>(routed_contributions.size()),
                count);
#undef CCCP_FUSED_MOE_POINTERS
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    launch_tp_all_rank_reduce_one(
        routed_contributions,
        routed_workspace,
        stream);
    launch_tp_all_rank_reduce_one(
        shared_contributions,
        shared_workspace,
        stream);
    const int count = static_cast<int>(output.numel());
    const int blocks = std::min(1024, (count + 255) / 256);
    residual_add3_bf16_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            routed_workspace.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            shared_workspace.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            output.data_ptr<at::BFloat16>()),
        count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_tp_moe_hc_finalize_one(
    const std::vector<torch::Tensor>& routed_contributions,
    const std::vector<torch::Tensor>& shared_contributions,
    torch::Tensor residual,
    torch::Tensor post,
    torch::Tensor comb,
    torch::Tensor output,
    cudaStream_t stream)
{
    TORCH_CHECK(
        !routed_contributions.empty() &&
        routed_contributions.size() == shared_contributions.size() &&
        routed_contributions.size() <= 16,
        "TP HC MoE finalizer requires matching 1 to 16 rank "
        "contributions");
    TORCH_CHECK(
        residual.is_cuda() && post.is_cuda() && comb.is_cuda() &&
        output.is_cuda() &&
        residual.scalar_type() == at::kBFloat16 &&
        post.scalar_type() == at::kBFloat16 &&
        comb.scalar_type() == at::kBFloat16 &&
        output.scalar_type() == at::kBFloat16 &&
        residual.is_contiguous() && post.is_contiguous() &&
        comb.is_contiguous() && output.is_contiguous() &&
        residual.dim() >= 2 && residual.size(-2) == 4 &&
        output.numel() == residual.numel() &&
        post.numel() == 4 && comb.numel() == 16,
        "TP HC MoE finalizer requires matching contiguous BF16 HC "
        "buffers");
    const int target = output.get_device();
    const int D = static_cast<int>(residual.size(-1));
    TORCH_CHECK(
        residual.get_device() == target && post.get_device() == target &&
        comb.get_device() == target,
        "TP HC MoE finalizer buffers must share one target device");
    const float* routed_ptrs[16] = {};
    const float* shared_ptrs[16] = {};
    for (size_t rank = 0; rank < routed_contributions.size(); ++rank) {
        const auto routed = routed_contributions[rank];
        const auto shared = shared_contributions[rank];
        TORCH_CHECK(
            routed.is_cuda() && shared.is_cuda() &&
            routed.scalar_type() == at::kFloat &&
            shared.scalar_type() == at::kFloat &&
            routed.is_contiguous() && shared.is_contiguous() &&
            routed.numel() == D && shared.numel() == D,
            "TP HC MoE contributions must be contiguous CUDA float32 "
            "rows");
        if (routed.get_device() != target)
            ensure_peer_access(
                target,
                routed.get_device(),
                "TP HC routed contribution");
        if (shared.get_device() != target)
            ensure_peer_access(
                target,
                shared.get_device(),
                "TP HC shared contribution");
        routed_ptrs[rank] = routed.data_ptr<float>();
        shared_ptrs[rank] = shared.data_ptr<float>();
    }
    const int blocks = std::min(32, (D + 255) / 256);
#define CCCP_FUSED_HC_MOE_POINTERS(prefix) \
    prefix##_ptrs[0], prefix##_ptrs[1], prefix##_ptrs[2], \
    prefix##_ptrs[3], prefix##_ptrs[4], prefix##_ptrs[5], \
    prefix##_ptrs[6], prefix##_ptrs[7], prefix##_ptrs[8], \
    prefix##_ptrs[9], prefix##_ptrs[10], prefix##_ptrs[11], \
    prefix##_ptrs[12], prefix##_ptrs[13], prefix##_ptrs[14], \
    prefix##_ptrs[15]
    tp_moe_hc_finalize_all_rank_bf16_kernel<<<
        dim3(blocks, 4), 256, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                post.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                comb.data_ptr<at::BFloat16>()),
            CCCP_FUSED_HC_MOE_POINTERS(routed),
            CCCP_FUSED_HC_MOE_POINTERS(shared),
            static_cast<int>(routed_contributions.size()),
            D);
#undef CCCP_FUSED_HC_MOE_POINTERS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor tp1_moe_finalize(
    torch::Tensor routed_contribution,
    torch::Tensor shared_contribution,
    torch::Tensor residual,
    torch::Tensor routed_workspace,
    torch::Tensor shared_workspace,
    torch::Tensor output)
{
    TORCH_CHECK(
        routed_contribution.get_device() == output.get_device() &&
        shared_contribution.get_device() == output.get_device(),
        "TP1 MoE contributions must share the output device");
    launch_tp_moe_finalize_one(
        {routed_contribution},
        {shared_contribution},
        residual,
        routed_workspace,
        shared_workspace,
        output,
        at::cuda::getCurrentCUDAStream(output.get_device()),
        true);
    return output;
}

std::vector<torch::Tensor> tp_all_rank_reduce(
    std::vector<torch::Tensor> contributions,
    std::vector<torch::Tensor> outputs)
{
    TORCH_CHECK(
        !outputs.empty() && outputs.size() <= 16,
        "TP all-rank reduction requires 1 to 16 outputs");
    int original_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&original_device));
    for (const auto output : outputs) {
        const int target = output.get_device();
        C10_CUDA_CHECK(cudaSetDevice(target));
        launch_tp_all_rank_reduce_one(
            contributions,
            output,
            at::cuda::getCurrentCUDAStream(target));
    }
    C10_CUDA_CHECK(cudaSetDevice(original_device));
    return outputs;
}

__global__ void head_rmsnorm_rope_kernel(
    float* __restrict__ rows,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    const int width,
    const int rope_width,
    const float eps)
{
    extern __shared__ float scratch[];
    const int row = blockIdx.x;
    float sum = 0.0f;
    for (int item = threadIdx.x; item < width; item += blockDim.x) {
        const float value = rows[row * width + item];
        sum = fmaf(value, value, sum);
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(scratch[0] / width + eps);
    for (int item = threadIdx.x; item < width; item += blockDim.x) {
        rows[row * width + item] =
            rows[row * width + item] * scale *
            __bfloat162float(weight[item]);
    }
    __syncthreads();
    const int rope_pairs = rope_width / 2;
    const int rope_start = width - rope_width;
    for (
        int pair = threadIdx.x;
        pair < rope_pairs;
        pair += blockDim.x
    ) {
        const int offset = row * width + rope_start + 2 * pair;
        const float first = rows[offset];
        const float second = rows[offset + 1];
        const float c = cos[pair];
        const float s = sin[pair];
        rows[offset] = __fsub_rn(first * c, second * s);
        rows[offset + 1] = __fadd_rn(first * s, second * c);
    }
}

bool head_rmsnorm_rope(
    torch::Tensor rows,
    torch::Tensor weight,
    torch::Tensor cos,
    torch::Tensor sin,
    int64_t rope_width,
    double eps)
{
    TORCH_CHECK(
        rows.is_cuda() && weight.is_cuda() &&
        cos.is_cuda() && sin.is_cuda() &&
        rows.is_contiguous() && weight.is_contiguous() &&
        cos.is_contiguous() && sin.is_contiguous(),
        "head norm/RoPE tensors must be contiguous CUDA tensors");
    if (
        rows.scalar_type() != at::kFloat ||
        weight.scalar_type() != at::kBFloat16 ||
        cos.scalar_type() != at::kFloat ||
        sin.scalar_type() != at::kFloat ||
        rows.dim() != 2 || weight.dim() != 1)
        return false;
    const int width = static_cast<int>(rows.size(1));
    if (
        rows.size(0) <= 0 || weight.numel() != width ||
        rope_width <= 0 || rope_width > width || rope_width % 2 ||
        cos.numel() != rope_width / 2 ||
        sin.numel() != rope_width / 2 || width > 4096)
        return false;
    int threads = 1;
    while (threads < width && threads < 256)
        threads <<= 1;
    const auto stream = at::cuda::getCurrentCUDAStream();
    head_rmsnorm_rope_kernel<<<
        static_cast<int>(rows.size(0)),
        threads,
        threads * sizeof(float),
        stream>>>(
            rows.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(
                weight.data_ptr<at::BFloat16>()),
            cos.data_ptr<float>(),
            sin.data_ptr<float>(),
            width,
            static_cast<int>(rope_width),
            static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

__global__ void compressed_state_update_kernel(
    const __nv_bfloat16* projected,
    const __nv_bfloat16* ape,
    float* ckv,
    float* cscore,
    int state_width,
    int kv_rows,
    int score_rows,
    int slot,
    int phase);

bool compressed_state_update(
    torch::Tensor projected,
    torch::Tensor ape,
    torch::Tensor ckv,
    torch::Tensor cscore,
    int64_t ratio,
    int64_t position,
    int64_t kv_rows);

__global__ void compressed_state_update_kernel(
    const __nv_bfloat16* __restrict__ projected,
    const __nv_bfloat16* __restrict__ ape,
    float* __restrict__ ckv,
    float* __restrict__ cscore,
    const int state_width,
    const int kv_rows,
    const int score_rows,
    const int slot,
    const int phase)
{
    for (
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < kv_rows;
        index += blockDim.x * gridDim.x
    ) {
        ckv[slot * state_width + index] =
            __bfloat162float(projected[index]);
    }
    for (
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < score_rows;
        index += blockDim.x * gridDim.x
    ) {
        cscore[slot * state_width + index] = __fadd_rn(
            __bfloat162float(projected[kv_rows + index]),
            __bfloat162float(ape[phase * score_rows + index]));
    }
}

bool compressed_state_update(
    torch::Tensor projected,
    torch::Tensor ape,
    torch::Tensor ckv,
    torch::Tensor cscore,
    int64_t ratio,
    int64_t position,
    int64_t kv_rows)
{
    TORCH_CHECK(
        projected.is_cuda() && ape.is_cuda() &&
        ckv.is_cuda() && cscore.is_cuda() &&
        projected.is_contiguous() && ape.is_contiguous() &&
        ckv.is_contiguous() && cscore.is_contiguous(),
        "compressed state tensors must be contiguous CUDA tensors");
    TORCH_CHECK(
        ckv.scalar_type() == at::kFloat &&
        cscore.scalar_type() == at::kFloat &&
        ckv.sizes() == cscore.sizes() && ckv.dim() == 3 &&
        ckv.size(0) == 1,
        "compressed state outputs must be matching [1,slots,width] FP32");
    TORCH_CHECK(
        projected.dim() == 2 && projected.size(0) == 1 &&
        kv_rows > 0 && kv_rows < projected.numel(),
        "compressed projection must be [1,kv+score]");
    if (
        projected.scalar_type() != at::kBFloat16 ||
        ape.scalar_type() != at::kBFloat16)
        return false;
    const int score_rows = static_cast<int>(
        projected.numel() - kv_rows);
    TORCH_CHECK(
        ckv.size(2) == kv_rows &&
        ape.dim() == 2 && ape.size(0) == ratio &&
        ape.size(1) == score_rows && score_rows <= kv_rows &&
        ratio > 0 && position >= 0,
        "compressed state layout/ratio/position mismatch");
    const bool overlap = ckv.size(1) == 2 * ratio;
    TORCH_CHECK(
        ckv.size(1) == ratio || overlap,
        "compressed state slot count must equal ratio or 2*ratio");
    const int phase = static_cast<int>(position % ratio);
    const int slot = phase + (overlap ? static_cast<int>(ratio) : 0);
    const int blocks = std::min(
        32,
        (std::max(static_cast<int>(kv_rows), score_rows) + 255) / 256);
    const auto stream = at::cuda::getCurrentCUDAStream();
    compressed_state_update_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            projected.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            ape.data_ptr<at::BFloat16>()),
        ckv.data_ptr<float>(),
        cscore.data_ptr<float>(),
        static_cast<int>(ckv.size(2)),
        static_cast<int>(kv_rows),
        score_rows,
        slot,
        phase);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

std::vector<torch::Tensor> tp_all_rank_reduce_from_events(
    std::vector<torch::Tensor> contributions,
    std::vector<int64_t> input_events,
    std::vector<torch::Tensor> outputs,
    std::vector<int64_t> output_events)
{
    TORCH_CHECK(
        !contributions.empty() && contributions.size() <= 16 &&
        input_events.size() == contributions.size() &&
        !outputs.empty() && outputs.size() <= 16 &&
        output_events.size() == outputs.size(),
        "TP event reduction inputs/outputs/events must be non-empty and "
        "size-equal");
    int original_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&original_device));
    for (size_t rank = 0; rank < outputs.size(); ++rank) {
        const int target = outputs[rank].get_device();
        C10_CUDA_CHECK(cudaSetDevice(target));
        const auto stream = at::cuda::getCurrentCUDAStream(target);
        for (const auto raw_event : input_events) {
            C10_CUDA_CHECK(
                cudaStreamWaitEvent(
                    stream,
                    reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(raw_event)),
                    0));
        }
        launch_tp_all_rank_reduce_one(
            contributions,
            outputs[rank],
            stream);
        C10_CUDA_CHECK(
            cudaEventRecord(
                reinterpret_cast<cudaEvent_t>(
                    static_cast<uintptr_t>(output_events[rank])),
                stream));
    }
    C10_CUDA_CHECK(cudaSetDevice(original_device));
    return outputs;
}

std::vector<torch::Tensor> tp_moe_finalize_from_events(
    std::vector<torch::Tensor> routed_contributions,
    std::vector<torch::Tensor> shared_contributions,
    std::vector<int64_t> input_events,
    std::vector<torch::Tensor> residuals,
    std::vector<torch::Tensor> outputs,
    std::vector<int64_t> output_events)
{
    TORCH_CHECK(
        !routed_contributions.empty() &&
        routed_contributions.size() <= 16 &&
        shared_contributions.size() == routed_contributions.size() &&
        input_events.size() == routed_contributions.size() &&
        !outputs.empty() && outputs.size() <= 16 &&
        residuals.size() == outputs.size() &&
        output_events.size() == outputs.size(),
        "TP event MoE finalizer inputs/outputs/events are inconsistent");
    int original_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&original_device));
    for (size_t rank = 0; rank < outputs.size(); ++rank) {
        const int target = outputs[rank].get_device();
        C10_CUDA_CHECK(cudaSetDevice(target));
        const auto stream = at::cuda::getCurrentCUDAStream(target);
        for (const auto raw_event : input_events) {
            C10_CUDA_CHECK(cudaStreamWaitEvent(
                stream,
                reinterpret_cast<cudaEvent_t>(
                    static_cast<uintptr_t>(raw_event)),
                0));
        }
        // The fused path writes directly to output; the workspace arguments
        // are shape validators retained by the shared decode/prefill helper.
        launch_tp_moe_finalize_one(
            routed_contributions,
            shared_contributions,
            residuals[rank],
            outputs[rank],
            outputs[rank],
            outputs[rank],
            stream,
            true);
        C10_CUDA_CHECK(cudaEventRecord(
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(output_events[rank])),
            stream));
    }
    C10_CUDA_CHECK(cudaSetDevice(original_device));
    return outputs;
}

namespace {

bool tp_environment_enabled(const char* name)
{
    const char* setting = std::getenv(name);
    return (
        setting != nullptr &&
        setting[0] == '1' &&
        setting[1] == '\0');
}

inline void graph_dispatch_pause()
{
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || \
    defined(_M_X64)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

int graph_dispatch_spin_iterations()
{
    const char* setting = std::getenv("CCCP_TP_GRAPH_SPIN");
    if (setting == nullptr || setting[0] == '\0')
        return 1 << 20;
    char* end = nullptr;
    const long parsed = std::strtol(setting, &end, 10);
    if (
        end == setting ||
        *end != '\0' ||
        parsed < 0 ||
        parsed > (1 << 26))
        return 1 << 20;
    return static_cast<int>(parsed);
}

struct GraphLaunchTask {
    cudaGraphExec_t graph = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t done = nullptr;
    cudaEvent_t ready = nullptr;
};

// vLLM assigns one process to every TP rank, so CUDA graph submission happens
// concurrently. CCCP keeps one process to preserve the RAM-expert fallback and
// previously paid cudaSetDevice + three CUDA API calls serially for every
// rank and every layer. A persistent worker per secondary device provides the
// same concurrent host-side submission without process duplication.
class GraphLaunchWorker {
public:
    explicit GraphLaunchWorker(int device)
        : device_(device), thread_(&GraphLaunchWorker::run, this)
    {
    }

    ~GraphLaunchWorker()
    {
        {
            std::lock_guard<std::mutex> lock(sleep_mutex_);
            stopping_.store(true, std::memory_order_release);
        }
        wake_.notify_one();
        if (thread_.joinable())
            thread_.join();
    }

    GraphLaunchWorker(const GraphLaunchWorker&) = delete;
    GraphLaunchWorker& operator=(const GraphLaunchWorker&) = delete;

    uint64_t submit(const GraphLaunchTask& task)
    {
        const uint64_t previous =
            requested_.load(std::memory_order_relaxed);
        while (
            completed_.load(std::memory_order_acquire) != previous
        )
            graph_dispatch_pause();
        task_ = task;
        const uint64_t sequence = previous + 1;
        {
            // The producer update and consumer wait share this mutex. This
            // closes the condition-variable notification window without
            // serializing any CUDA work; the worker only holds the mutex
            // while it is entering or leaving its idle wait.
            std::lock_guard<std::mutex> lock(sleep_mutex_);
            requested_.store(sequence, std::memory_order_release);
        }
        wake_.notify_one();
        return sequence;
    }

    void wait(uint64_t sequence)
    {
        while (
            completed_.load(std::memory_order_acquire) != sequence
        )
            graph_dispatch_pause();
        TORCH_CHECK(
            set_device_status_ == cudaSuccess,
            "parallel graph worker cudaSetDevice failed: ",
            cudaGetErrorString(set_device_status_));
        TORCH_CHECK(
            wait_status_ == cudaSuccess,
            "parallel graph worker cudaStreamWaitEvent failed: ",
            cudaGetErrorString(wait_status_));
        TORCH_CHECK(
            launch_status_ == cudaSuccess,
            "parallel graph worker cudaGraphLaunch failed: ",
            cudaGetErrorString(launch_status_));
        TORCH_CHECK(
            record_status_ == cudaSuccess,
            "parallel graph worker cudaEventRecord failed: ",
            cudaGetErrorString(record_status_));
    }

private:
    void run()
    {
        set_device_status_ = cudaSetDevice(device_);
        uint64_t observed = 0;
        while (!stopping_.load(std::memory_order_acquire)) {
            const uint64_t requested =
                requested_.load(std::memory_order_acquire);
            if (requested != observed) {
                const GraphLaunchTask task = task_;
                if (set_device_status_ == cudaSuccess) {
                    wait_status_ = cudaStreamWaitEvent(
                        task.stream,
                        task.ready,
                        0);
                    launch_status_ = (
                        wait_status_ == cudaSuccess
                            ? cudaGraphLaunch(task.graph, task.stream)
                            : wait_status_);
                    record_status_ = (
                        launch_status_ == cudaSuccess
                            ? cudaEventRecord(task.done, task.stream)
                            : launch_status_);
                } else {
                    wait_status_ = set_device_status_;
                    launch_status_ = set_device_status_;
                    record_status_ = set_device_status_;
                }
                observed = requested;
                completed_.store(observed, std::memory_order_release);
                continue;
            }

            // Keep workers hot across adjacent Attention/MoE stages, then
            // sleep when the server is idle so the optimization does not
            // permanently consume one CPU core per rank.
            bool changed = false;
            const int spin_limit = graph_dispatch_spin_iterations();
            for (int spin = 0; spin < spin_limit; ++spin) {
                if (
                    stopping_.load(std::memory_order_relaxed) ||
                    requested_.load(std::memory_order_acquire) != observed
                ) {
                    changed = true;
                    break;
                }
                graph_dispatch_pause();
            }
            if (changed)
                continue;
            std::unique_lock<std::mutex> lock(sleep_mutex_);
            wake_.wait(
                lock,
                [&] {
                    return
                        stopping_.load(std::memory_order_acquire) ||
                        requested_.load(std::memory_order_acquire)
                            != observed;
                });
        }
    }

    int device_;
    GraphLaunchTask task_;
    std::atomic<uint64_t> requested_{0};
    std::atomic<uint64_t> completed_{0};
    std::atomic<bool> stopping_{false};
    std::mutex sleep_mutex_;
    std::condition_variable wake_;
    cudaError_t set_device_status_ = cudaSuccess;
    cudaError_t wait_status_ = cudaSuccess;
    cudaError_t launch_status_ = cudaSuccess;
    cudaError_t record_status_ = cudaSuccess;
    std::thread thread_;
};

constexpr int kMaxGraphDispatchDevices = 16;
std::array<std::unique_ptr<GraphLaunchWorker>, kMaxGraphDispatchDevices>
    graph_launch_workers;
std::array<std::once_flag, kMaxGraphDispatchDevices>
    graph_launch_worker_once;

GraphLaunchWorker& graph_launch_worker(int device)
{
    TORCH_CHECK(
        device >= 0 && device < kMaxGraphDispatchDevices,
        "parallel graph dispatch supports CUDA devices [0, ",
        kMaxGraphDispatchDevices,
        ")");
    std::call_once(
        graph_launch_worker_once[device],
        [device] {
            graph_launch_workers[device] =
                std::make_unique<GraphLaunchWorker>(device);
        });
    return *graph_launch_workers[device];
}

void launch_cuda_graphs_sequential(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& graph_execs,
    const std::vector<int64_t>& streams,
    const std::vector<int64_t>& done_events,
    int64_t source_event)
{
    const auto count = devices.size();
    TORCH_CHECK(
        count > 0 &&
        graph_execs.size() == count &&
        streams.size() == count &&
        done_events.size() == count,
        "batched CUDA Graph launch vectors must be non-empty and size-equal");
    int original_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&original_device));
    const auto ready = reinterpret_cast<cudaEvent_t>(
        static_cast<uintptr_t>(source_event));
    for (size_t index = 0; index < count; ++index) {
        C10_CUDA_CHECK(cudaSetDevice(static_cast<int>(devices[index])));
        const auto stream = reinterpret_cast<cudaStream_t>(
            static_cast<uintptr_t>(streams[index]));
        const auto graph_exec = reinterpret_cast<cudaGraphExec_t>(
            static_cast<uintptr_t>(graph_execs[index]));
        const auto done = reinterpret_cast<cudaEvent_t>(
            static_cast<uintptr_t>(done_events[index]));
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, ready, 0));
        C10_CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
        C10_CUDA_CHECK(cudaEventRecord(done, stream));
    }
    C10_CUDA_CHECK(cudaSetDevice(original_device));
}

void launch_cuda_graphs_parallel(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& graph_execs,
    const std::vector<int64_t>& streams,
    const std::vector<int64_t>& done_events,
    int64_t source_event)
{
    const auto count = devices.size();
    TORCH_CHECK(
        count > 1 &&
        count <= kMaxGraphDispatchDevices &&
        graph_execs.size() == count &&
        streams.size() == count &&
        done_events.size() == count,
        "parallel CUDA Graph launch vectors must be size-equal");
    int original_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&original_device));
    const auto ready = reinterpret_cast<cudaEvent_t>(
        static_cast<uintptr_t>(source_event));

    std::array<GraphLaunchWorker*, kMaxGraphDispatchDevices>
        workers{};
    std::array<uint64_t, kMaxGraphDispatchDevices> sequences{};
    for (size_t index = 1; index < count; ++index) {
        const int device = static_cast<int>(devices[index]);
        GraphLaunchWorker& worker = graph_launch_worker(device);
        workers[index] = &worker;
        sequences[index] = worker.submit({
            reinterpret_cast<cudaGraphExec_t>(
                static_cast<uintptr_t>(graph_execs[index])),
            reinterpret_cast<cudaStream_t>(
                static_cast<uintptr_t>(streams[index])),
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(done_events[index])),
            ready,
        });
    }

    const int primary_device = static_cast<int>(devices[0]);
    if (original_device != primary_device)
        C10_CUDA_CHECK(cudaSetDevice(primary_device));
    const auto primary_stream = reinterpret_cast<cudaStream_t>(
        static_cast<uintptr_t>(streams[0]));
    const auto primary_graph = reinterpret_cast<cudaGraphExec_t>(
        static_cast<uintptr_t>(graph_execs[0]));
    const auto primary_done = reinterpret_cast<cudaEvent_t>(
        static_cast<uintptr_t>(done_events[0]));
    C10_CUDA_CHECK(cudaStreamWaitEvent(primary_stream, ready, 0));
    C10_CUDA_CHECK(cudaGraphLaunch(primary_graph, primary_stream));
    C10_CUDA_CHECK(cudaEventRecord(primary_done, primary_stream));

    for (size_t index = 1; index < count; ++index)
        workers[index]->wait(sequences[index]);
    if (original_device != primary_device)
        C10_CUDA_CHECK(cudaSetDevice(original_device));
}

void launch_cuda_graphs_from_events_sequential(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& graph_execs,
    const std::vector<int64_t>& streams,
    const std::vector<int64_t>& done_events,
    const std::vector<int64_t>& ready_events)
{
    const auto count = devices.size();
    TORCH_CHECK(
        count > 0 &&
        graph_execs.size() == count &&
        streams.size() == count &&
        done_events.size() == count &&
        ready_events.size() == count,
        "replicated CUDA Graph vectors must be non-empty and size-equal");
    int original_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&original_device));
    for (size_t index = 0; index < count; ++index) {
        C10_CUDA_CHECK(cudaSetDevice(static_cast<int>(devices[index])));
        const auto stream = reinterpret_cast<cudaStream_t>(
            static_cast<uintptr_t>(streams[index]));
        const auto graph_exec = reinterpret_cast<cudaGraphExec_t>(
            static_cast<uintptr_t>(graph_execs[index]));
        const auto done = reinterpret_cast<cudaEvent_t>(
            static_cast<uintptr_t>(done_events[index]));
        const auto ready = reinterpret_cast<cudaEvent_t>(
            static_cast<uintptr_t>(ready_events[index]));
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, ready, 0));
        C10_CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
        C10_CUDA_CHECK(cudaEventRecord(done, stream));
    }
    C10_CUDA_CHECK(cudaSetDevice(original_device));
}

void launch_cuda_graphs_from_events_parallel(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& graph_execs,
    const std::vector<int64_t>& streams,
    const std::vector<int64_t>& done_events,
    const std::vector<int64_t>& ready_events)
{
    const auto count = devices.size();
    TORCH_CHECK(
        count > 1 &&
        count <= kMaxGraphDispatchDevices &&
        graph_execs.size() == count &&
        streams.size() == count &&
        done_events.size() == count &&
        ready_events.size() == count,
        "parallel replicated CUDA Graph vectors must be size-equal");
    int original_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&original_device));
    std::array<GraphLaunchWorker*, kMaxGraphDispatchDevices> workers{};
    std::array<uint64_t, kMaxGraphDispatchDevices> sequences{};
    for (size_t index = 1; index < count; ++index) {
        const int device = static_cast<int>(devices[index]);
        GraphLaunchWorker& worker = graph_launch_worker(device);
        workers[index] = &worker;
        sequences[index] = worker.submit({
            reinterpret_cast<cudaGraphExec_t>(
                static_cast<uintptr_t>(graph_execs[index])),
            reinterpret_cast<cudaStream_t>(
                static_cast<uintptr_t>(streams[index])),
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(done_events[index])),
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(ready_events[index])),
        });
    }
    const int primary_device = static_cast<int>(devices[0]);
    if (original_device != primary_device)
        C10_CUDA_CHECK(cudaSetDevice(primary_device));
    const auto primary_stream = reinterpret_cast<cudaStream_t>(
        static_cast<uintptr_t>(streams[0]));
    C10_CUDA_CHECK(
        cudaStreamWaitEvent(
            primary_stream,
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(ready_events[0])),
            0));
    C10_CUDA_CHECK(
        cudaGraphLaunch(
            reinterpret_cast<cudaGraphExec_t>(
                static_cast<uintptr_t>(graph_execs[0])),
            primary_stream));
    C10_CUDA_CHECK(
        cudaEventRecord(
            reinterpret_cast<cudaEvent_t>(
                static_cast<uintptr_t>(done_events[0])),
            primary_stream));
    for (size_t index = 1; index < count; ++index)
        workers[index]->wait(sequences[index]);
    if (original_device != primary_device)
        C10_CUDA_CHECK(cudaSetDevice(original_device));
}

}  // namespace

void launch_cuda_graphs(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& graph_execs,
    const std::vector<int64_t>& streams,
    const std::vector<int64_t>& done_events,
    int64_t source_event)
{
    TORCH_CHECK(
        !devices.empty(),
        "CUDA Graph launch requires at least one TP device");
    int current_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));
    TORCH_CHECK(
        current_device == static_cast<int>(devices[0]),
        "CUDA Graph source event must be recorded on the primary device");
    const auto ready = reinterpret_cast<cudaEvent_t>(
        static_cast<uintptr_t>(source_event));
    C10_CUDA_CHECK(
        cudaEventRecord(
            ready,
            at::cuda::getCurrentCUDAStream(current_device)));
    const char* setting = std::getenv("CCCP_TP_PARALLEL_LAUNCH");
    const bool explicitly_enabled = (
        setting != nullptr &&
        setting[0] == '1' &&
        setting[1] == '\0');
    const bool explicitly_disabled = (
        setting != nullptr &&
        setting[0] == '0' &&
        setting[1] == '\0');
    const bool enabled = (
        devices.size() > 1 &&
        (
            explicitly_enabled ||
            (
                !explicitly_disabled &&
                devices.size() >= 8
            )
        ));
    if (enabled) {
        launch_cuda_graphs_parallel(
            devices,
            graph_execs,
            streams,
            done_events,
            source_event);
    } else {
        launch_cuda_graphs_sequential(
            devices,
            graph_execs,
            streams,
            done_events,
            source_event);
    }
}

void launch_cuda_graphs_from_events(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& graph_execs,
    const std::vector<int64_t>& streams,
    const std::vector<int64_t>& done_events,
    const std::vector<int64_t>& ready_events)
{
    const char* setting = std::getenv("CCCP_TP_PARALLEL_LAUNCH");
    const bool explicitly_enabled = (
        setting != nullptr &&
        setting[0] == '1' &&
        setting[1] == '\0');
    const bool explicitly_disabled = (
        setting != nullptr &&
        setting[0] == '0' &&
        setting[1] == '\0');
    const bool enabled = (
        devices.size() > 1 &&
        (
            explicitly_enabled ||
            (
                !explicitly_disabled &&
                devices.size() >= 8
            )
        ));
    if (enabled) {
        launch_cuda_graphs_from_events_parallel(
            devices,
            graph_execs,
            streams,
            done_events,
            ready_events);
    } else {
        launch_cuda_graphs_from_events_sequential(
            devices,
            graph_execs,
            streams,
            done_events,
            ready_events);
    }
}

torch::Tensor launch_cuda_graphs_reduce(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& graph_execs,
    const std::vector<int64_t>& streams,
    const std::vector<int64_t>& done_events,
    int64_t source_event,
    std::vector<torch::Tensor> contributions,
    torch::Tensor residual)
{
    launch_cuda_graphs(
        devices,
        graph_execs,
        streams,
        done_events,
        source_event);
    TORCH_CHECK(
        !devices.empty() &&
        contributions.size() >= devices.size() &&
        contributions.size() <= 16,
        "TP graph reduction needs at least one contribution per rank "
        "and supports at most 16");
    const int primary_device = contributions[0].get_device();
    int current_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));
    TORCH_CHECK(
        current_device == primary_device,
        "TP graph reduction must return to the primary device");
    const auto primary_stream = at::cuda::getCurrentCUDAStream();
    for (const auto raw_event : done_events) {
        const auto event = reinterpret_cast<cudaEvent_t>(
            static_cast<uintptr_t>(raw_event));
        C10_CUDA_CHECK(
            cudaStreamWaitEvent(primary_stream, event, 0));
    }
    return glm_ep_reduce_residual(
        contributions,
        residual);
}

std::vector<torch::Tensor> launch_cuda_graphs_reduce_many(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& graph_execs,
    const std::vector<int64_t>& streams,
    const std::vector<int64_t>& done_events,
    int64_t source_event,
    std::vector<std::vector<torch::Tensor>> contribution_groups,
    std::vector<torch::Tensor> residuals)
{
    launch_cuda_graphs(
        devices,
        graph_execs,
        streams,
        done_events,
        source_event);
    TORCH_CHECK(
        !devices.empty() &&
        !contribution_groups.empty() &&
        contribution_groups.size() == residuals.size(),
        "TP graph multi-reduction groups/residuals must be non-empty "
        "and size-equal");
    const int primary_device = residuals[0].get_device();
    int current_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));
    TORCH_CHECK(
        current_device == primary_device,
        "TP graph multi-reduction must return to the primary device");
    const auto primary_stream = at::cuda::getCurrentCUDAStream();
    for (const auto raw_event : done_events) {
        const auto event = reinterpret_cast<cudaEvent_t>(
            static_cast<uintptr_t>(raw_event));
        C10_CUDA_CHECK(cudaStreamWaitEvent(primary_stream, event, 0));
    }
    std::vector<torch::Tensor> outputs;
    outputs.reserve(residuals.size());
    for (size_t index = 0; index < residuals.size(); ++index) {
        TORCH_CHECK(
            contribution_groups[index].size() >= devices.size() &&
            contribution_groups[index].size() <= 16,
            "each TP graph multi-reduction needs one contribution per rank");
        TORCH_CHECK(
            residuals[index].get_device() == primary_device,
            "TP graph multi-reduction residuals must share primary device");
        outputs.push_back(
            glm_ep_reduce_residual(
                contribution_groups[index],
                residuals[index]));
    }
    return outputs;
}

std::vector<torch::Tensor> launch_cuda_graphs_reduce_norm_router(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& graph_execs,
    const std::vector<int64_t>& streams,
    const std::vector<int64_t>& done_events,
    int64_t source_event,
    std::vector<torch::Tensor> contributions,
    torch::Tensor attention_zero,
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor router_weight,
    double eps,
    torch::Tensor norm_output,
    c10::optional<torch::Tensor> residual_output,
    c10::optional<torch::Tensor> logits_output)
{
    auto attention_update = launch_cuda_graphs_reduce(
        devices,
        graph_execs,
        streams,
        done_events,
        source_event,
        std::move(contributions),
        attention_zero);
    return glm_residual_norm_router_norm_out(
        residual,
        attention_update,
        norm_weight,
        router_weight,
        eps,
        norm_output,
        residual_output,
        logits_output);
}

class TPGraphLaunchBatch {
public:
    TPGraphLaunchBatch(
        std::vector<int64_t> devices,
        std::vector<int64_t> graph_execs,
        std::vector<int64_t> streams,
        std::vector<int64_t> done_events,
        int64_t source_event)
        : devices_(std::move(devices)),
          graph_execs_(std::move(graph_execs)),
          streams_(std::move(streams)),
          done_events_(std::move(done_events)),
          source_event_(source_event),
          collective_event_barrier_enabled_(
              tp_environment_enabled("CCCP_TP_EVENT_BARRIER")),
          fused_moe_finalize_enabled_(
              tp_environment_enabled("CCCP_TP_FUSED_MOE_FINALIZE"))
    {
        validate_handles();
        initialize_collective_events();
    }

    TPGraphLaunchBatch(
        std::vector<int64_t> devices,
        std::vector<std::vector<int64_t>> child_graphs,
        std::vector<int64_t> streams,
        std::vector<int64_t> done_events,
        int64_t source_event)
        : devices_(std::move(devices)),
          streams_(std::move(streams)),
          done_events_(std::move(done_events)),
          source_event_(source_event),
          collective_event_barrier_enabled_(
              tp_environment_enabled("CCCP_TP_EVENT_BARRIER")),
          fused_moe_finalize_enabled_(
              tp_environment_enabled("CCCP_TP_FUSED_MOE_FINALIZE"))
    {
        TORCH_CHECK(
            !devices_.empty() &&
            child_graphs.size() == devices_.size() &&
            streams_.size() == devices_.size() &&
            done_events_.size() == devices_.size() &&
            source_event_ != 0,
            "TP Graph sequence handles must be non-empty and size-equal");
        int original_device = -1;
        C10_CUDA_CHECK(cudaGetDevice(&original_device));
        owned_graphs_.reserve(devices_.size());
        owned_graph_execs_.reserve(devices_.size());
        graph_execs_.reserve(devices_.size());
        for (size_t rank = 0; rank < devices_.size(); ++rank) {
            TORCH_CHECK(
                !child_graphs[rank].empty(),
                "each TP rank sequence needs at least one child graph");
            C10_CUDA_CHECK(
                cudaSetDevice(static_cast<int>(devices_[rank])));
            cudaGraph_t parent = nullptr;
            C10_CUDA_CHECK(cudaGraphCreate(&parent, 0));
            cudaGraphNode_t previous = nullptr;
            for (const auto raw_child : child_graphs[rank]) {
                TORCH_CHECK(
                    raw_child != 0,
                    "TP Graph sequence child handle must be non-zero");
                cudaGraphNode_t child_node = nullptr;
                C10_CUDA_CHECK(
                    cudaGraphAddChildGraphNode(
                        &child_node,
                        parent,
                        previous == nullptr ? nullptr : &previous,
                        previous == nullptr ? 0 : 1,
                        reinterpret_cast<cudaGraph_t>(
                            static_cast<uintptr_t>(raw_child))));
                previous = child_node;
            }
            cudaGraphExec_t executable = nullptr;
            C10_CUDA_CHECK(
                cudaGraphInstantiateWithFlags(
                    &executable,
                    parent,
                    0));
            owned_graphs_.push_back(parent);
            owned_graph_execs_.push_back(executable);
            graph_execs_.push_back(
                static_cast<int64_t>(
                    reinterpret_cast<uintptr_t>(executable)));
        }
        C10_CUDA_CHECK(cudaSetDevice(original_device));
        validate_handles();
        initialize_collective_events();
    }

    TPGraphLaunchBatch(
        std::vector<int64_t> devices,
        std::vector<std::vector<std::vector<int64_t>>> graph_stages,
        std::vector<int64_t> streams,
        std::vector<int64_t> done_events,
        int64_t source_event)
        : devices_(std::move(devices)),
          streams_(std::move(streams)),
          done_events_(std::move(done_events)),
          source_event_(source_event),
          collective_event_barrier_enabled_(
              tp_environment_enabled("CCCP_TP_EVENT_BARRIER")),
          fused_moe_finalize_enabled_(
              tp_environment_enabled("CCCP_TP_FUSED_MOE_FINALIZE"))
    {
        TORCH_CHECK(
            !devices_.empty() &&
            graph_stages.size() == devices_.size() &&
            streams_.size() == devices_.size() &&
            done_events_.size() == devices_.size() &&
            source_event_ != 0,
            "TP Graph DAG handles must be non-empty and size-equal");
        int original_device = -1;
        C10_CUDA_CHECK(cudaGetDevice(&original_device));
        owned_graphs_.reserve(devices_.size());
        owned_graph_execs_.reserve(devices_.size());
        graph_execs_.reserve(devices_.size());
        for (size_t rank = 0; rank < devices_.size(); ++rank) {
            TORCH_CHECK(
                !graph_stages[rank].empty(),
                "each TP rank DAG needs at least one stage");
            C10_CUDA_CHECK(
                cudaSetDevice(static_cast<int>(devices_[rank])));
            cudaGraph_t parent = nullptr;
            C10_CUDA_CHECK(cudaGraphCreate(&parent, 0));
            std::vector<cudaGraphNode_t> previous;
            for (const auto& stage : graph_stages[rank]) {
                TORCH_CHECK(
                    !stage.empty(),
                    "TP Graph DAG stages must be non-empty");
                std::vector<cudaGraphNode_t> current;
                current.reserve(stage.size());
                for (const auto raw_child : stage) {
                    TORCH_CHECK(
                        raw_child != 0,
                        "TP Graph DAG child handle must be non-zero");
                    cudaGraphNode_t child_node = nullptr;
                    C10_CUDA_CHECK(
                        cudaGraphAddChildGraphNode(
                            &child_node,
                            parent,
                            previous.empty()
                                ? nullptr
                                : previous.data(),
                            previous.size(),
                            reinterpret_cast<cudaGraph_t>(
                                static_cast<uintptr_t>(raw_child))));
                    current.push_back(child_node);
                }
                previous = std::move(current);
            }
            cudaGraphExec_t executable = nullptr;
            C10_CUDA_CHECK(
                cudaGraphInstantiateWithFlags(
                    &executable,
                    parent,
                    0));
            owned_graphs_.push_back(parent);
            owned_graph_execs_.push_back(executable);
            graph_execs_.push_back(
                static_cast<int64_t>(
                    reinterpret_cast<uintptr_t>(executable)));
        }
        C10_CUDA_CHECK(cudaSetDevice(original_device));
        validate_handles();
        initialize_collective_events();
    }

    TPGraphLaunchBatch(const TPGraphLaunchBatch&) = delete;
    TPGraphLaunchBatch& operator=(const TPGraphLaunchBatch&) = delete;

    ~TPGraphLaunchBatch()
    {
        int original_device = -1;
        if (cudaGetDevice(&original_device) != cudaSuccess)
            return;
        if (
            collective_ready_event_ != nullptr &&
            cudaSetDevice(static_cast<int>(devices_[0])) == cudaSuccess
        )
            cudaEventDestroy(collective_ready_event_);
        for (size_t index = 0; index < collective_events_.size(); ++index) {
            if (
                cudaSetDevice(static_cast<int>(devices_[index]))
                == cudaSuccess
            )
                cudaEventDestroy(collective_events_[index]);
        }
        for (size_t index = 0; index < owned_graphs_.size(); ++index) {
            if (
                cudaSetDevice(static_cast<int>(devices_[index]))
                != cudaSuccess)
                continue;
            if (owned_graph_execs_[index] != nullptr)
                cudaGraphExecDestroy(owned_graph_execs_[index]);
            if (owned_graphs_[index] != nullptr)
                cudaGraphDestroy(owned_graphs_[index]);
        }
        cudaSetDevice(original_device);
    }

    void launch() const
    {
        launch_cuda_graphs(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            source_event_);
    }

    void launch_tp1() const
    {
        TORCH_CHECK(
            devices_.size() == 1 && graph_execs_.size() == 1,
            "TP1 graph launch requires exactly one rank");
        const int device = static_cast<int>(devices_[0]);
        int current = -1;
        C10_CUDA_CHECK(cudaGetDevice(&current));
        TORCH_CHECK(
            current == device,
            "TP1 graph must launch under its CUDA device");
        // Launch the retained graph directly on the caller's stream.  The
        // preceding fixed-buffer writes and following consumers are thereby
        // ordered without CUDA Event creation, waits, or records.
        C10_CUDA_CHECK(cudaGraphLaunch(
            reinterpret_cast<cudaGraphExec_t>(
                static_cast<uintptr_t>(graph_execs_[0])),
            at::cuda::getCurrentCUDAStream(device)));
    }

    std::vector<int64_t> raw_graphs() const
    {
        TORCH_CHECK(
            owned_graphs_.size() == devices_.size(),
            "raw child graphs are available only for composed batches");
        std::vector<int64_t> result;
        result.reserve(owned_graphs_.size());
        for (const auto graph : owned_graphs_)
            result.push_back(static_cast<int64_t>(
                reinterpret_cast<uintptr_t>(graph)));
        return result;
    }

    void launch_from_events(
        std::vector<int64_t> input_events) const
    {
        TORCH_CHECK(
            input_events.size() == devices_.size(),
            "TP graph input events must match graph ranks");
        launch_cuda_graphs_from_events(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            input_events);
    }

    torch::Tensor launch_reduce(
        std::vector<torch::Tensor> contributions,
        torch::Tensor residual) const
    {
        return launch_cuda_graphs_reduce(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            source_event_,
            std::move(contributions),
            residual);
    }

    std::vector<torch::Tensor> launch_reduce_many(
        std::vector<std::vector<torch::Tensor>> contribution_groups,
        std::vector<torch::Tensor> residuals) const
    {
        return launch_cuda_graphs_reduce_many(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            source_event_,
            std::move(contribution_groups),
            std::move(residuals));
    }

    std::vector<torch::Tensor> launch_all_rank(
        std::vector<torch::Tensor> contributions,
        std::vector<torch::Tensor> outputs) const
    {
        launch_cuda_graphs(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            source_event_);
        TORCH_CHECK(
            outputs.size() == devices_.size() &&
            collective_events_.size() == devices_.size(),
            "TP all-rank outputs must match graph rank count");
        int original_device = -1;
        C10_CUDA_CHECK(cudaGetDevice(&original_device));
        TORCH_CHECK(
            original_device == static_cast<int>(devices_[0]),
            "TP all-rank launch must begin on the primary graph rank");
        const bool event_barrier = collective_event_barrier_enabled();
        if (event_barrier)
            record_collective_ready(done_events_);
        for (size_t rank = 0; rank < devices_.size(); ++rank) {
            const int target = static_cast<int>(devices_[rank]);
            TORCH_CHECK(
                outputs[rank].get_device() == target,
                "TP all-rank output device order must match graph ranks");
            C10_CUDA_CHECK(cudaSetDevice(target));
            const auto stream =
                at::cuda::getCurrentCUDAStream(target);
            if (event_barrier) {
                C10_CUDA_CHECK(
                    cudaStreamWaitEvent(
                        stream,
                        collective_ready_event_,
                        0));
            } else {
                for (const auto raw_event : done_events_) {
                    const auto done = reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(raw_event));
                    C10_CUDA_CHECK(
                        cudaStreamWaitEvent(stream, done, 0));
                }
            }
            launch_tp_all_rank_reduce_one(
                contributions,
                outputs[rank],
                stream);
            C10_CUDA_CHECK(
                cudaEventRecord(
                    collective_events_[rank],
                    stream));
        }
        C10_CUDA_CHECK(
            cudaSetDevice(static_cast<int>(devices_[0])));
        const auto primary_stream =
            at::cuda::getCurrentCUDAStream(
                static_cast<int>(devices_[0]));
        for (const auto event : collective_events_)
            C10_CUDA_CHECK(
                cudaStreamWaitEvent(primary_stream, event, 0));
        return outputs;
    }

    std::vector<torch::Tensor> launch_all_rank_from_events(
        std::vector<int64_t> input_events,
        std::vector<torch::Tensor> contributions,
        std::vector<torch::Tensor> outputs,
        std::vector<int64_t> output_events) const
    {
        TORCH_CHECK(
            input_events.size() == devices_.size() &&
            !outputs.empty() &&
            outputs.size() <= 16 &&
            output_events.size() == outputs.size(),
            "TPHidden inputs must match graph ranks and outputs/events "
            "must form a non-empty rank set");
        launch_cuda_graphs_from_events(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            input_events);
        int original_device = -1;
        C10_CUDA_CHECK(cudaGetDevice(&original_device));
        const bool event_barrier = collective_event_barrier_enabled();
        if (event_barrier)
            record_collective_ready(done_events_);
        for (size_t rank = 0; rank < outputs.size(); ++rank) {
            const int target = outputs[rank].get_device();
            C10_CUDA_CHECK(cudaSetDevice(target));
            const auto stream =
                at::cuda::getCurrentCUDAStream(target);
            if (event_barrier) {
                C10_CUDA_CHECK(
                    cudaStreamWaitEvent(
                        stream,
                        collective_ready_event_,
                        0));
            } else {
                for (const auto raw_event : done_events_) {
                    const auto done = reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(raw_event));
                    C10_CUDA_CHECK(
                        cudaStreamWaitEvent(stream, done, 0));
                }
            }
            launch_tp_all_rank_reduce_one(
                contributions,
                outputs[rank],
                stream);
            C10_CUDA_CHECK(
                cudaEventRecord(
                    reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(
                            output_events[rank])),
                    stream));
        }
        C10_CUDA_CHECK(cudaSetDevice(original_device));
        return outputs;
    }

    std::vector<std::vector<torch::Tensor>>
    launch_all_rank_many_from_events(
        std::vector<int64_t> input_events,
        std::vector<std::vector<torch::Tensor>> contribution_groups,
        std::vector<std::vector<torch::Tensor>> output_groups,
        std::vector<int64_t> output_events) const
    {
        TORCH_CHECK(
            input_events.size() == devices_.size() &&
            !contribution_groups.empty() &&
            contribution_groups.size() == output_groups.size() &&
            output_events.size() == devices_.size(),
            "TPHidden multi-output collective metadata mismatch");
        for (size_t group = 0; group < contribution_groups.size(); ++group) {
            TORCH_CHECK(
                contribution_groups[group].size() == devices_.size() &&
                output_groups[group].size() == devices_.size(),
                "TPHidden multi-output groups must match graph ranks");
        }
        launch_cuda_graphs_from_events(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            input_events);
        int original_device = -1;
        C10_CUDA_CHECK(cudaGetDevice(&original_device));
        const bool event_barrier = collective_event_barrier_enabled();
        if (event_barrier)
            record_collective_ready(done_events_);
        for (size_t rank = 0; rank < devices_.size(); ++rank) {
            const int target = static_cast<int>(devices_[rank]);
            C10_CUDA_CHECK(cudaSetDevice(target));
            const auto stream =
                at::cuda::getCurrentCUDAStream(target);
            if (event_barrier) {
                C10_CUDA_CHECK(
                    cudaStreamWaitEvent(
                        stream,
                        collective_ready_event_,
                        0));
            } else {
                for (const auto raw_event : done_events_) {
                    const auto done = reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(raw_event));
                    C10_CUDA_CHECK(
                        cudaStreamWaitEvent(stream, done, 0));
                }
            }
            for (
                size_t group = 0;
                group < contribution_groups.size();
                ++group
            ) {
                TORCH_CHECK(
                    output_groups[group][rank].get_device() == target,
                    "TPHidden multi-output device order must match ranks");
                launch_tp_all_rank_reduce_one(
                    contribution_groups[group],
                    output_groups[group][rank],
                    stream);
            }
            C10_CUDA_CHECK(
                cudaEventRecord(
                    reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(
                            output_events[rank])),
                    stream));
        }
        C10_CUDA_CHECK(cudaSetDevice(original_device));
        return output_groups;
    }

    std::vector<std::vector<torch::Tensor>>
    reduce_all_rank_many_from_events(
        std::vector<int64_t> input_events,
        std::vector<std::vector<torch::Tensor>> contribution_groups,
        std::vector<std::vector<torch::Tensor>> output_groups,
        std::vector<int64_t> output_events) const
    {
        TORCH_CHECK(
            input_events.size() == devices_.size() &&
            !contribution_groups.empty() &&
            contribution_groups.size() == output_groups.size() &&
            output_events.size() == devices_.size(),
            "TPHidden collective-only metadata mismatch");
        for (size_t group = 0; group < contribution_groups.size(); ++group) {
            TORCH_CHECK(
                contribution_groups[group].size() == devices_.size() &&
                output_groups[group].size() == devices_.size(),
                "TPHidden collective-only groups must match graph ranks");
        }
        int original_device = -1;
        C10_CUDA_CHECK(cudaGetDevice(&original_device));
        const bool event_barrier = collective_event_barrier_enabled();
        if (event_barrier)
            record_collective_ready(input_events);
        for (size_t rank = 0; rank < devices_.size(); ++rank) {
            const int target = static_cast<int>(devices_[rank]);
            C10_CUDA_CHECK(cudaSetDevice(target));
            const auto stream =
                at::cuda::getCurrentCUDAStream(target);
            if (event_barrier) {
                C10_CUDA_CHECK(
                    cudaStreamWaitEvent(
                        stream,
                        collective_ready_event_,
                        0));
            } else {
                for (const auto raw_event : input_events) {
                    C10_CUDA_CHECK(
                        cudaStreamWaitEvent(
                            stream,
                            reinterpret_cast<cudaEvent_t>(
                                static_cast<uintptr_t>(raw_event)),
                            0));
                }
            }
            for (
                size_t group = 0;
                group < contribution_groups.size();
                ++group
            ) {
                TORCH_CHECK(
                    output_groups[group][rank].get_device() == target,
                    "TPHidden collective-only device order must match ranks");
                launch_tp_all_rank_reduce_one(
                    contribution_groups[group],
                    output_groups[group][rank],
                    stream);
            }
            C10_CUDA_CHECK(
                cudaEventRecord(
                    reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(
                            output_events[rank])),
                    stream));
        }
        C10_CUDA_CHECK(cudaSetDevice(original_device));
        return output_groups;
    }

    std::vector<torch::Tensor> launch_moe_all_rank_from_events(
        std::vector<int64_t> input_events,
        std::vector<torch::Tensor> routed_contributions,
        std::vector<torch::Tensor> shared_contributions,
        std::vector<int64_t> shared_events,
        std::vector<torch::Tensor> residuals,
        std::vector<int64_t> residual_events,
        std::vector<torch::Tensor> routed_workspaces,
        std::vector<torch::Tensor> shared_workspaces,
        std::vector<torch::Tensor> outputs,
        std::vector<int64_t> output_events) const
    {
        TORCH_CHECK(
            input_events.size() == devices_.size() &&
            shared_events.size() == devices_.size() &&
            !outputs.empty() &&
            outputs.size() <= 16 &&
            residuals.size() == outputs.size() &&
            residual_events.size() == outputs.size() &&
            routed_workspaces.size() == outputs.size() &&
            shared_workspaces.size() == outputs.size() &&
            output_events.size() == outputs.size(),
            "TP MoE finalizer input/output ranks are inconsistent");
        launch_cuda_graphs_from_events(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            input_events);
        int original_device = -1;
        C10_CUDA_CHECK(cudaGetDevice(&original_device));
        const bool event_barrier = collective_event_barrier_enabled();
        if (event_barrier)
            record_collective_ready(done_events_, &shared_events);
        for (size_t rank = 0; rank < outputs.size(); ++rank) {
            const int target = outputs[rank].get_device();
            C10_CUDA_CHECK(cudaSetDevice(target));
            const auto stream =
                at::cuda::getCurrentCUDAStream(target);
            if (event_barrier) {
                C10_CUDA_CHECK(
                    cudaStreamWaitEvent(
                        stream,
                        collective_ready_event_,
                        0));
            } else {
                for (const auto raw_event : done_events_) {
                    C10_CUDA_CHECK(
                        cudaStreamWaitEvent(
                            stream,
                            reinterpret_cast<cudaEvent_t>(
                                static_cast<uintptr_t>(raw_event)),
                            0));
                }
                for (const auto raw_event : shared_events) {
                    C10_CUDA_CHECK(
                        cudaStreamWaitEvent(
                            stream,
                            reinterpret_cast<cudaEvent_t>(
                                static_cast<uintptr_t>(raw_event)),
                            0));
                }
            }
            C10_CUDA_CHECK(
                cudaStreamWaitEvent(
                    stream,
                    reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(
                            residual_events[rank])),
                    0));
            launch_tp_moe_finalize_one(
                routed_contributions,
                shared_contributions,
                residuals[rank],
                routed_workspaces[rank],
                shared_workspaces[rank],
                outputs[rank],
                stream,
                fused_moe_finalize_enabled_);
            C10_CUDA_CHECK(
                cudaEventRecord(
                    reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(
                            output_events[rank])),
                    stream));
        }
        C10_CUDA_CHECK(cudaSetDevice(original_device));
        return outputs;
    }

    std::vector<torch::Tensor> launch_moe_hc_all_rank_from_events(
        std::vector<int64_t> input_events,
        std::vector<torch::Tensor> routed_contributions,
        std::vector<torch::Tensor> shared_contributions,
        std::vector<int64_t> shared_events,
        std::vector<torch::Tensor> residuals,
        std::vector<torch::Tensor> posts,
        std::vector<torch::Tensor> combs,
        std::vector<torch::Tensor> outputs,
        std::vector<int64_t> output_events) const
    {
        const auto ranks = devices_.size();
        TORCH_CHECK(
            input_events.size() == ranks &&
            shared_events.size() == ranks &&
            residuals.size() == ranks && posts.size() == ranks &&
            combs.size() == ranks && outputs.size() == ranks &&
            output_events.size() == ranks,
            "TP HC MoE finalizer input/output ranks are inconsistent");
        launch_cuda_graphs_from_events(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            input_events);
        int original_device = -1;
        C10_CUDA_CHECK(cudaGetDevice(&original_device));
        const bool event_barrier = collective_event_barrier_enabled();
        if (event_barrier)
            record_collective_ready(done_events_, &shared_events);
        for (size_t rank = 0; rank < ranks; ++rank) {
            const int target = outputs[rank].get_device();
            C10_CUDA_CHECK(cudaSetDevice(target));
            const auto stream =
                at::cuda::getCurrentCUDAStream(target);
            if (event_barrier) {
                C10_CUDA_CHECK(cudaStreamWaitEvent(
                    stream,
                    collective_ready_event_,
                    0));
            } else {
                for (const auto raw_event : done_events_) {
                    C10_CUDA_CHECK(cudaStreamWaitEvent(
                        stream,
                        reinterpret_cast<cudaEvent_t>(
                            static_cast<uintptr_t>(raw_event)),
                        0));
                }
                for (const auto raw_event : shared_events) {
                    C10_CUDA_CHECK(cudaStreamWaitEvent(
                        stream,
                        reinterpret_cast<cudaEvent_t>(
                            static_cast<uintptr_t>(raw_event)),
                        0));
                }
            }
            launch_tp_moe_hc_finalize_one(
                routed_contributions,
                shared_contributions,
                residuals[rank],
                posts[rank],
                combs[rank],
                outputs[rank],
                stream);
            C10_CUDA_CHECK(cudaEventRecord(
                reinterpret_cast<cudaEvent_t>(
                    static_cast<uintptr_t>(output_events[rank])),
                stream));
        }
        C10_CUDA_CHECK(cudaSetDevice(original_device));
        return outputs;
    }

    std::vector<torch::Tensor> launch_reduce_norm_router(
        std::vector<torch::Tensor> contributions,
        torch::Tensor attention_zero,
        torch::Tensor residual,
        torch::Tensor norm_weight,
        torch::Tensor router_weight,
        double eps,
        torch::Tensor norm_output,
        c10::optional<torch::Tensor> residual_output,
        c10::optional<torch::Tensor> logits_output) const
    {
        return launch_cuda_graphs_reduce_norm_router(
            devices_,
            graph_execs_,
            streams_,
            done_events_,
            source_event_,
            std::move(contributions),
            attention_zero,
            residual,
            norm_weight,
            router_weight,
            eps,
            norm_output,
            residual_output,
            logits_output);
    }

    torch::Tensor launch_moe_layer(
        const TPGraphLaunchBatch& expert_batch,
        std::vector<torch::Tensor> attention_contributions,
        torch::Tensor attention_zero,
        torch::Tensor residual,
        torch::Tensor norm_weight,
        torch::Tensor router_weight,
        double eps,
        torch::Tensor norm_output,
        torch::Tensor residual_output,
        torch::Tensor logits_output,
        torch::Tensor route_bias,
        torch::Tensor route_mask,
        long top_k,
        double routed_scaling,
        torch::Tensor route_weights,
        torch::Tensor route_indices,
        std::vector<torch::Tensor> expert_contributions) const
    {
        auto post = launch_reduce_norm_router(
            std::move(attention_contributions),
            attention_zero,
            residual,
            norm_weight,
            router_weight,
            eps,
            norm_output,
            residual_output,
            logits_output);
        sigmoid_route_out(
            post[2],
            route_bias,
            route_mask,
            top_k,
            routed_scaling,
            route_weights,
            route_indices);
        return expert_batch.launch_reduce(
            std::move(expert_contributions),
            post[0]);
    }

private:
    bool collective_event_barrier_enabled() const
    {
        return collective_event_barrier_enabled_;
    }

    void record_collective_ready(
        const std::vector<int64_t>& events,
        const std::vector<int64_t>* more_events = nullptr) const
    {
        // This event is scheduling metadata only.  It coalesces N identical
        // wait lists into one cross-device wait per output rank; every rank
        // still reduces every contribution into its own TPHidden replica.
        const int primary = static_cast<int>(devices_[0]);
        C10_CUDA_CHECK(cudaSetDevice(primary));
        const auto stream =
            at::cuda::getCurrentCUDAStream(primary);
        for (const auto raw_event : events) {
            C10_CUDA_CHECK(
                cudaStreamWaitEvent(
                    stream,
                    reinterpret_cast<cudaEvent_t>(
                        static_cast<uintptr_t>(raw_event)),
                    0));
        }
        if (more_events != nullptr) {
            for (const auto raw_event : *more_events) {
                C10_CUDA_CHECK(
                    cudaStreamWaitEvent(
                        stream,
                        reinterpret_cast<cudaEvent_t>(
                            static_cast<uintptr_t>(raw_event)),
                        0));
            }
        }
        C10_CUDA_CHECK(
            cudaEventRecord(
                collective_ready_event_,
                stream));
    }

    void validate_handles() const
    {
        TORCH_CHECK(
            !devices_.empty() &&
            graph_execs_.size() == devices_.size() &&
            streams_.size() == devices_.size() &&
            done_events_.size() == devices_.size() &&
            source_event_ != 0,
            "TP Graph batch handles must be non-empty and size-equal");
    }

    void initialize_collective_events()
    {
        int original_device = -1;
        C10_CUDA_CHECK(cudaGetDevice(&original_device));
        C10_CUDA_CHECK(
            cudaSetDevice(static_cast<int>(devices_[0])));
        C10_CUDA_CHECK(
            cudaEventCreateWithFlags(
                &collective_ready_event_,
                cudaEventDisableTiming));
        collective_events_.reserve(devices_.size());
        for (const auto raw_device : devices_) {
            C10_CUDA_CHECK(
                cudaSetDevice(static_cast<int>(raw_device)));
            cudaEvent_t event = nullptr;
            C10_CUDA_CHECK(
                cudaEventCreateWithFlags(
                    &event,
                    cudaEventDisableTiming));
            collective_events_.push_back(event);
        }
        C10_CUDA_CHECK(cudaSetDevice(original_device));
    }

    std::vector<int64_t> devices_;
    std::vector<int64_t> graph_execs_;
    std::vector<int64_t> streams_;
    std::vector<int64_t> done_events_;
    int64_t source_event_;
    bool collective_event_barrier_enabled_;
    bool fused_moe_finalize_enabled_;
    cudaEvent_t collective_ready_event_ = nullptr;
    std::vector<cudaEvent_t> collective_events_;
    std::vector<cudaGraph_t> owned_graphs_;
    std::vector<cudaGraphExec_t> owned_graph_execs_;
};

class TPNoOwnerMoELayerPlan {
public:
    TPNoOwnerMoELayerPlan(
        const TPGraphLaunchBatch& shared_batch,
        const TPGraphLaunchBatch& route_batch,
        const TPGraphLaunchBatch& expert_batch,
        const TPGraphLaunchBatch& final_batch,
        std::vector<int64_t> input_events,
        std::vector<std::vector<torch::Tensor>> route_contribution_groups,
        std::vector<std::vector<torch::Tensor>> route_output_groups,
        std::vector<int64_t> route_output_events,
        std::vector<torch::Tensor> expert_contributions,
        std::vector<torch::Tensor> packed_outputs,
        std::vector<int64_t> packed_output_events,
        std::vector<torch::Tensor> routed_contributions,
        std::vector<torch::Tensor> shared_contributions,
        std::vector<int64_t> shared_events,
        std::vector<torch::Tensor> residuals,
        std::vector<int64_t> residual_events,
        std::vector<torch::Tensor> routed_workspaces,
        std::vector<torch::Tensor> shared_workspaces,
        std::vector<torch::Tensor> outputs,
        std::vector<int64_t> output_events)
        : shared_batch_(&shared_batch),
          route_batch_(&route_batch),
          expert_batch_(&expert_batch),
          final_batch_(&final_batch),
          input_events_(std::move(input_events)),
          route_contribution_groups_(
              std::move(route_contribution_groups)),
          route_output_groups_(std::move(route_output_groups)),
          route_output_events_(std::move(route_output_events)),
          expert_contributions_(std::move(expert_contributions)),
          packed_outputs_(std::move(packed_outputs)),
          packed_output_events_(std::move(packed_output_events)),
          routed_contributions_(std::move(routed_contributions)),
          shared_contributions_(std::move(shared_contributions)),
          shared_events_(std::move(shared_events)),
          residuals_(std::move(residuals)),
          residual_events_(std::move(residual_events)),
          routed_workspaces_(std::move(routed_workspaces)),
          shared_workspaces_(std::move(shared_workspaces)),
          outputs_(std::move(outputs)),
          output_events_(std::move(output_events))
    {
        const auto ranks = input_events_.size();
        TORCH_CHECK(
            ranks > 0 &&
            route_contribution_groups_.size() == 2 &&
            route_output_groups_.size() == 2 &&
            route_output_events_.size() == ranks &&
            expert_contributions_.size() == ranks &&
            packed_outputs_.size() == ranks &&
            packed_output_events_.size() == ranks &&
            routed_contributions_.size() == ranks &&
            shared_contributions_.size() == ranks &&
            shared_events_.size() == ranks &&
            residuals_.size() == ranks &&
            residual_events_.size() == ranks &&
            routed_workspaces_.size() == ranks &&
            shared_workspaces_.size() == ranks &&
            outputs_.size() == ranks &&
            output_events_.size() == ranks,
            "no-owner MoE plan ranks and fixed buffers must match");
        for (size_t group = 0; group < 2; ++group) {
            TORCH_CHECK(
                route_contribution_groups_[group].size() == ranks &&
                route_output_groups_[group].size() == ranks,
                "no-owner MoE route groups must match TP ranks");
        }
    }

    void launch() const
    {
        launch_from_events(input_events_);
    }

    void launch_from_events(
        std::vector<int64_t> input_events) const
    {
        TORCH_CHECK(
            input_events.size() == input_events_.size(),
            "profiled no-owner MoE inputs must match TP ranks");
        // One Python→C++ transition schedules the complete fixed-address
        // no-owner MoE chain.  Every phase is still all-rank: the event
        // boundaries only express true TP collective dependencies.
        shared_batch_->launch_from_events(std::move(input_events));
        route_batch_->reduce_all_rank_many_from_events(
            shared_events_,
            route_contribution_groups_,
            route_output_groups_,
            route_output_events_);
        expert_batch_->launch_all_rank_from_events(
            route_output_events_,
            expert_contributions_,
            packed_outputs_,
            packed_output_events_);
        final_batch_->launch_moe_all_rank_from_events(
            packed_output_events_,
            routed_contributions_,
            shared_contributions_,
            shared_events_,
            residuals_,
            residual_events_,
            routed_workspaces_,
            shared_workspaces_,
            outputs_,
            output_events_);
    }

private:
    // The Python wrapper retains the four owning TPGraphLaunchBatch objects.
    // These pointers therefore only describe immutable scheduling metadata.
    const TPGraphLaunchBatch* shared_batch_;
    const TPGraphLaunchBatch* route_batch_;
    const TPGraphLaunchBatch* expert_batch_;
    const TPGraphLaunchBatch* final_batch_;
    std::vector<int64_t> input_events_;
    std::vector<std::vector<torch::Tensor>> route_contribution_groups_;
    std::vector<std::vector<torch::Tensor>> route_output_groups_;
    std::vector<int64_t> route_output_events_;
    std::vector<torch::Tensor> expert_contributions_;
    std::vector<torch::Tensor> packed_outputs_;
    std::vector<int64_t> packed_output_events_;
    std::vector<torch::Tensor> routed_contributions_;
    std::vector<torch::Tensor> shared_contributions_;
    std::vector<int64_t> shared_events_;
    std::vector<torch::Tensor> residuals_;
    std::vector<int64_t> residual_events_;
    std::vector<torch::Tensor> routed_workspaces_;
    std::vector<torch::Tensor> shared_workspaces_;
    std::vector<torch::Tensor> outputs_;
    std::vector<int64_t> output_events_;
};

class TPNoOwnerDecodeLayerPlan {
public:
    TPNoOwnerDecodeLayerPlan(
        const TPGraphLaunchBatch& attention_batch,
        const TPNoOwnerMoELayerPlan& moe_plan,
        std::vector<torch::Tensor> attention_contributions,
        std::vector<torch::Tensor> attention_outputs,
        std::vector<int64_t> attention_output_events)
        : attention_batch_(&attention_batch),
          moe_plan_(&moe_plan),
          attention_contributions_(
              std::move(attention_contributions)),
          attention_outputs_(std::move(attention_outputs)),
          attention_output_events_(
              std::move(attention_output_events))
    {
        TORCH_CHECK(
            !attention_contributions_.empty() &&
            !attention_outputs_.empty() &&
            attention_outputs_.size()
                == attention_output_events_.size(),
            "no-owner decode plan attention metadata is incomplete");
    }

    void launch_from_events(
        std::vector<int64_t> input_events) const
    {
        // One Python→C++ transition now submits the complete routed layer:
        // Attention Column/Head-TP→Row-TP followed by the fixed all-rank
        // MoE plan.  The attention output events are the only dependency
        // between the two all-rank stages; no hidden owner or broadcast is
        // introduced.
        attention_batch_->launch_all_rank_from_events(
            std::move(input_events),
            attention_contributions_,
            attention_outputs_,
            attention_output_events_);
        moe_plan_->launch_from_events(attention_output_events_);
    }

private:
    const TPGraphLaunchBatch* attention_batch_;
    const TPNoOwnerMoELayerPlan* moe_plan_;
    std::vector<torch::Tensor> attention_contributions_;
    std::vector<torch::Tensor> attention_outputs_;
    std::vector<int64_t> attention_output_events_;
};

class TPNoOwnerHCDecodeLayerPlan {
public:
    TPNoOwnerHCDecodeLayerPlan(
        const TPGraphLaunchBatch& attention_batch,
        const TPGraphLaunchBatch& shared_batch,
        const TPGraphLaunchBatch& route_batch,
        const TPGraphLaunchBatch& expert_batch,
        std::vector<torch::Tensor> attention_contributions,
        std::vector<torch::Tensor> attention_outputs,
        std::vector<int64_t> attention_output_events,
        std::vector<torch::Tensor> attention_residuals,
        std::vector<torch::Tensor> attention_posts,
        std::vector<torch::Tensor> attention_combs,
        std::vector<torch::Tensor> prefixes,
        std::vector<torch::Tensor> ffn_functions,
        std::vector<torch::Tensor> ffn_scales,
        std::vector<torch::Tensor> ffn_bases,
        std::vector<torch::Tensor> ffn_norms,
        std::vector<torch::Tensor> ffn_inputs,
        std::vector<torch::Tensor> ffn_posts,
        std::vector<torch::Tensor> ffn_combs,
        std::vector<int64_t> ffn_events,
        std::vector<int64_t> route_output_events,
        std::vector<torch::Tensor> expert_contributions,
        std::vector<torch::Tensor> shared_contributions,
        std::vector<int64_t> shared_events,
        std::vector<torch::Tensor> outputs,
        std::vector<int64_t> output_events,
        long sinkhorn_iters,
        double eps)
        : attention_batch_(&attention_batch),
          shared_batch_(&shared_batch),
          route_batch_(&route_batch),
          expert_batch_(&expert_batch),
          attention_contributions_(std::move(attention_contributions)),
          attention_outputs_(std::move(attention_outputs)),
          attention_output_events_(std::move(attention_output_events)),
          attention_residuals_(std::move(attention_residuals)),
          attention_posts_(std::move(attention_posts)),
          attention_combs_(std::move(attention_combs)),
          prefixes_(std::move(prefixes)),
          ffn_functions_(std::move(ffn_functions)),
          ffn_scales_(std::move(ffn_scales)),
          ffn_bases_(std::move(ffn_bases)),
          ffn_norms_(std::move(ffn_norms)),
          ffn_inputs_(std::move(ffn_inputs)),
          ffn_posts_(std::move(ffn_posts)),
          ffn_combs_(std::move(ffn_combs)),
          ffn_events_(std::move(ffn_events)),
          route_output_events_(std::move(route_output_events)),
          expert_contributions_(std::move(expert_contributions)),
          shared_contributions_(std::move(shared_contributions)),
          shared_events_(std::move(shared_events)),
          outputs_(std::move(outputs)),
          output_events_(std::move(output_events)),
          sinkhorn_iters_(sinkhorn_iters),
          eps_(eps)
    {
        const auto ranks = attention_outputs_.size();
        TORCH_CHECK(
            ranks > 1 &&
            attention_contributions_.size() == ranks &&
            attention_output_events_.size() == ranks &&
            attention_residuals_.size() == ranks &&
            attention_posts_.size() == ranks &&
            attention_combs_.size() == ranks &&
            prefixes_.size() == ranks &&
            ffn_functions_.size() == ranks &&
            ffn_scales_.size() == ranks &&
            ffn_bases_.size() == ranks &&
            ffn_norms_.size() == ranks &&
            ffn_inputs_.size() == ranks &&
            ffn_posts_.size() == ranks &&
            ffn_combs_.size() == ranks &&
            ffn_events_.size() == ranks &&
            route_output_events_.size() == ranks &&
            expert_contributions_.size() == ranks &&
            shared_contributions_.size() == ranks &&
            shared_events_.size() == ranks &&
            outputs_.size() == ranks &&
            output_events_.size() == ranks,
            "no-owner HC decode plan ranks and fixed buffers must match");
    }

    void launch_from_events(std::vector<int64_t> input_events) const
    {
        const auto ranks = attention_outputs_.size();
        TORCH_CHECK(
            input_events.size() == ranks,
            "no-owner HC decode inputs must match TP ranks");
        attention_batch_->launch_all_rank_from_events(
            std::move(input_events),
            attention_contributions_,
            attention_outputs_,
            attention_output_events_);

        int original_device = -1;
        C10_CUDA_CHECK(cudaGetDevice(&original_device));
        for (size_t rank = 0; rank < ranks; ++rank) {
            const int device = ffn_inputs_[rank].get_device();
            C10_CUDA_CHECK(cudaSetDevice(device));
            const auto stream = at::cuda::getCurrentCUDAStream(device);
            C10_CUDA_CHECK(cudaStreamWaitEvent(
                stream,
                reinterpret_cast<cudaEvent_t>(static_cast<uintptr_t>(
                    attention_output_events_[rank])),
                0));
            dsv4_hc_post_into(
                attention_outputs_[rank],
                attention_residuals_[rank],
                attention_posts_[rank],
                attention_combs_[rank],
                prefixes_[rank]);
            dsv4_hc_pre_norm_into(
                prefixes_[rank],
                ffn_functions_[rank],
                ffn_scales_[rank],
                ffn_bases_[rank],
                ffn_norms_[rank],
                ffn_inputs_[rank],
                ffn_posts_[rank],
                ffn_combs_[rank],
                sinkhorn_iters_,
                eps_);
            C10_CUDA_CHECK(cudaEventRecord(
                reinterpret_cast<cudaEvent_t>(static_cast<uintptr_t>(
                    ffn_events_[rank])),
                stream));
        }
        C10_CUDA_CHECK(cudaSetDevice(original_device));

        // Shared MLP and Router are independent after the HC prefix and are
        // submitted to their existing fixed streams before the packed graph.
        shared_batch_->launch_from_events(ffn_events_);
        route_batch_->launch_from_events(ffn_events_);
        expert_batch_->launch_moe_hc_all_rank_from_events(
            route_output_events_,
            expert_contributions_,
            shared_contributions_,
            shared_events_,
            prefixes_,
            ffn_posts_,
            ffn_combs_,
            outputs_,
            output_events_);
    }

    const std::vector<int64_t>& output_events() const
    {
        return output_events_;
    }

private:
    const TPGraphLaunchBatch* attention_batch_;
    const TPGraphLaunchBatch* shared_batch_;
    const TPGraphLaunchBatch* route_batch_;
    const TPGraphLaunchBatch* expert_batch_;
    std::vector<torch::Tensor> attention_contributions_;
    std::vector<torch::Tensor> attention_outputs_;
    std::vector<int64_t> attention_output_events_;
    std::vector<torch::Tensor> attention_residuals_;
    std::vector<torch::Tensor> attention_posts_;
    std::vector<torch::Tensor> attention_combs_;
    std::vector<torch::Tensor> prefixes_;
    std::vector<torch::Tensor> ffn_functions_;
    std::vector<torch::Tensor> ffn_scales_;
    std::vector<torch::Tensor> ffn_bases_;
    std::vector<torch::Tensor> ffn_norms_;
    std::vector<torch::Tensor> ffn_inputs_;
    std::vector<torch::Tensor> ffn_posts_;
    std::vector<torch::Tensor> ffn_combs_;
    std::vector<int64_t> ffn_events_;
    std::vector<int64_t> route_output_events_;
    std::vector<torch::Tensor> expert_contributions_;
    std::vector<torch::Tensor> shared_contributions_;
    std::vector<int64_t> shared_events_;
    std::vector<torch::Tensor> outputs_;
    std::vector<int64_t> output_events_;
    long sinkhorn_iters_;
    double eps_;
};

template <typename output_t, int ROWS_PER_WARP>
__global__ void bf16_gemv_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    output_t* __restrict__ output,
    const int rows,
    const int cols)
{
    const int lane = threadIdx.x;
    const int row_base =
        blockIdx.x * blockDim.y * ROWS_PER_WARP + threadIdx.y;
    if (row_base >= rows)
        return;
    const auto* input2 =
        reinterpret_cast<const __nv_bfloat162*>(input);
    const __nv_bfloat162* weight2[ROWS_PER_WARP];
    #pragma unroll
    for (int item = 0; item < ROWS_PER_WARP; ++item) {
        const int row = row_base + item * blockDim.y;
        weight2[item] = reinterpret_cast<const __nv_bfloat162*>(
            weight + static_cast<long>(row < rows ? row : row_base) * cols);
    }
    const int pairs = cols >> 1;
    float sums[ROWS_PER_WARP] = {};
    for (int pair = lane; pair < pairs; pair += 32) {
        const float2 x = __bfloat1622float2(__ldg(input2 + pair));
        #pragma unroll
        for (int item = 0; item < ROWS_PER_WARP; ++item) {
            if (row_base + item * blockDim.y < rows) {
                const float2 w = __bfloat1622float2(
                    __ldg(weight2[item] + pair));
                sums[item] = __fmaf_rn(x.x, w.x, sums[item]);
                sums[item] = __fmaf_rn(x.y, w.y, sums[item]);
            }
        }
    }
    #pragma unroll
    for (int item = 0; item < ROWS_PER_WARP; ++item)
        sums[item] = warp_sum_f32(sums[item]);
    if (lane == 0) {
        #pragma unroll
        for (int item = 0; item < ROWS_PER_WARP; ++item) {
            const int row = row_base + item * blockDim.y;
            if (row < rows) {
                if constexpr (std::is_same_v<output_t, float>)
                    output[row] = sums[item];
                else
                    output[row] = __float2bfloat16_rn(sums[item]);
            }
        }
    }
}

torch::Tensor bf16_gemv_out(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output)
{
    TORCH_CHECK(
        input.is_cuda() &&
        weight.is_cuda() &&
        output.is_cuda() &&
        input.scalar_type() == at::kBFloat16 &&
        weight.scalar_type() == at::kBFloat16 &&
        (
            output.scalar_type() == at::kBFloat16 ||
            output.scalar_type() == at::kFloat
        ),
        "BF16 GEMV requires CUDA BF16 input/weight and BF16/FP32 output");
    TORCH_CHECK(
        input.dim() == 2 &&
        input.size(0) == 1 &&
        weight.dim() == 2 &&
        weight.size(1) == input.size(1) &&
        input.size(1) > 0 &&
        input.size(1) % 2 == 0 &&
        output.dim() == 2 &&
        output.size(0) == 1 &&
        output.size(1) == weight.size(0) &&
        input.is_contiguous() &&
        weight.is_contiguous() &&
        output.is_contiguous() &&
        input.get_device() == weight.get_device() &&
        input.get_device() == output.get_device(),
        "BF16 GEMV tensor shapes/layout/devices are inconsistent");
    constexpr int warps = 8;
    constexpr int rows_per_warp = 2;
    const int rows = static_cast<int>(weight.size(0));
    const int cols = static_cast<int>(weight.size(1));
    const int blocks =
        (rows + warps * rows_per_warp - 1) /
        (warps * rows_per_warp);
    const dim3 threads(32, warps);
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (output.scalar_type() == at::kFloat) {
        bf16_gemv_kernel<float, rows_per_warp><<<
            blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                weight.data_ptr<at::BFloat16>()),
            output.data_ptr<float>(),
            rows,
            cols);
    } else {
        bf16_gemv_kernel<__nv_bfloat16, rows_per_warp><<<
            blocks, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    input.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(
                    weight.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(
                    output.data_ptr<at::BFloat16>()),
                rows,
                cols);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor int4_swiglu_packed_f32(
    torch::Tensor x,
    torch::Tensor gate_packed,
    torch::Tensor gate_scales,
    torch::Tensor up_packed,
    torch::Tensor up_scales,
    long cols,
    long group_size,
    bool group_vector,
    c10::optional<torch::Tensor> output_buffer) {
    TORCH_CHECK(
        x.is_cuda() && gate_packed.is_cuda() &&
        gate_scales.is_cuda() && up_packed.is_cuda() &&
        up_scales.is_cuda(),
        "INT4 SwiGLU tensors must be CUDA");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat ||
        x.scalar_type() == at::kBFloat16,
        "INT4 SwiGLU input must be float32 or bfloat16");
    TORCH_CHECK(
        gate_packed.scalar_type() == at::kByte &&
        up_packed.scalar_type() == at::kByte,
        "INT4 SwiGLU packed weights must be uint8");
    TORCH_CHECK(
        gate_scales.scalar_type() == at::kHalf &&
        up_scales.scalar_type() == at::kHalf,
        "INT4 SwiGLU scales must be float16");
    TORCH_CHECK(
        x.dim() == 2 && x.size(0) == 1,
        "INT4 SwiGLU input must be [1,C]");
    TORCH_CHECK(
        gate_packed.dim() == 2 && up_packed.dim() == 2 &&
        gate_scales.dim() == 2 && up_scales.dim() == 2,
        "INT4 SwiGLU weights and scales must be matrices");
    TORCH_CHECK(
        group_size == 64 && cols > 0 && cols % 64 == 0,
        "INT4 SwiGLU requires positive g64-aligned columns");
    TORCH_CHECK(
        x.size(1) == cols &&
        gate_packed.sizes() == up_packed.sizes() &&
        gate_scales.sizes() == up_scales.sizes() &&
        gate_packed.size(1) * 2 == cols,
        "INT4 SwiGLU input/weight shape mismatch");
    const int rows = static_cast<int>(gate_packed.size(0));
    const int groups = static_cast<int>(cols / group_size);
    TORCH_CHECK(
        gate_scales.size(0) == rows &&
        gate_scales.size(1) == groups,
        "INT4 SwiGLU scale shape mismatch");
    TORCH_CHECK(
        x.get_device() == gate_packed.get_device() &&
        x.get_device() == gate_scales.get_device() &&
        x.get_device() == up_packed.get_device() &&
        x.get_device() == up_scales.get_device(),
        "INT4 SwiGLU tensors must be on one CUDA device");

    auto xc = x.contiguous();
    auto gate_q = gate_packed.contiguous();
    auto gate_s = gate_scales.contiguous();
    auto up_q = up_packed.contiguous();
    auto up_s = up_scales.contiguous();
    auto output = output_buffer.has_value()
        ? output_buffer.value()
        : torch::empty(
            {1, rows},
            x.options().dtype(at::kFloat));
    TORCH_CHECK(
        output.is_cuda() &&
        output.scalar_type() == at::kFloat &&
        output.is_contiguous() &&
        output.sizes() == torch::IntArrayRef({1, rows}) &&
        output.get_device() == x.get_device(),
        "INT4 SwiGLU output buffer must be contiguous float32 [1,R] "
        "on the input device");
    const int device = x.get_device();
    auto stream = at::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == at::kFloat) {
        launch_int4_swiglu_packed_f32<float>(
            xc.data_ptr<float>(),
            gate_q.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(
                gate_s.data_ptr<at::Half>()),
            up_q.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(
                up_s.data_ptr<at::Half>()),
            output.data_ptr<float>(),
            rows,
            static_cast<int>(cols),
            groups,
            device,
            stream,
            group_vector);
    } else {
        launch_int4_swiglu_packed_f32<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(
                xc.data_ptr<at::BFloat16>()),
            gate_q.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(
                gate_s.data_ptr<at::Half>()),
            up_q.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(
                up_s.data_ptr<at::Half>()),
            output.data_ptr<float>(),
            rows,
            static_cast<int>(cols),
            groups,
            device,
            stream,
            group_vector);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void flashinfer_mla_batch1_plan_kernel(
    uint8_t* __restrict__ workspace,
    int32_t* __restrict__ kv_indptr,
    int32_t* __restrict__ kv_indices,
    int32_t* __restrict__ kv_len_arr,
    int length,
    int page_size,
    int heads,
    int cluster_size,
    int num_clusters,
    int num_sms,
    int64_t q_indptr_offset,
    int64_t kv_indptr_offset,
    int64_t partial_indptr_offset,
    int64_t merge_start_offset,
    int64_t merge_end_offset,
    int64_t merge_partial_start_offset,
    int64_t merge_partial_end_offset,
    int64_t merge_stride_offset,
    int64_t q_len_offset,
    int64_t kv_len_offset,
    int64_t q_start_offset,
    int64_t kv_start_offset,
    int64_t kv_end_offset,
    int64_t work_indptr_offset) {
    int avg_kv = (length + num_clusters - 1) / num_clusters;
    int kv_limit;
    if (avg_kv <= 8) {
        kv_limit = 32;
    } else if (avg_kv <= 16) {
        kv_limit = 64;
    } else if (avg_kv <= 32) {
        kv_limit = 128;
    } else if (avg_kv <= 64) {
        kv_limit = 192;
    } else {
        kv_limit = ((avg_kv + 255) / 256) * 256;
    }
    const bool split = length > kv_limit;
    const int num_works = (length + kv_limit - 1) / kv_limit;
    const int qo_chunks = max(length * cluster_size / kv_limit, 1);
    const int row_chunk = (heads + qo_chunks - 1) / qo_chunks;
    const int merge_count = split
        ? (heads + row_chunk - 1) / row_chunk
        : 0;
    const int blocks = (length + page_size - 1) / page_size;

    int32_t* q_indptr_plan = reinterpret_cast<int32_t*>(
        workspace + q_indptr_offset);
    int32_t* kv_indptr_plan = reinterpret_cast<int32_t*>(
        workspace + kv_indptr_offset);
    int32_t* partial_indptr = reinterpret_cast<int32_t*>(
        workspace + partial_indptr_offset);
    int32_t* merge_start = reinterpret_cast<int32_t*>(
        workspace + merge_start_offset);
    int32_t* merge_end = reinterpret_cast<int32_t*>(
        workspace + merge_end_offset);
    int32_t* merge_partial_start = reinterpret_cast<int32_t*>(
        workspace + merge_partial_start_offset);
    int32_t* merge_partial_end = reinterpret_cast<int32_t*>(
        workspace + merge_partial_end_offset);
    int32_t* merge_stride = reinterpret_cast<int32_t*>(
        workspace + merge_stride_offset);
    int32_t* q_len = reinterpret_cast<int32_t*>(
        workspace + q_len_offset);
    int32_t* kv_len = reinterpret_cast<int32_t*>(
        workspace + kv_len_offset);
    int32_t* q_start = reinterpret_cast<int32_t*>(
        workspace + q_start_offset);
    int32_t* kv_start = reinterpret_cast<int32_t*>(
        workspace + kv_start_offset);
    int32_t* kv_end = reinterpret_cast<int32_t*>(
        workspace + kv_end_offset);
    int32_t* work_indptr = reinterpret_cast<int32_t*>(
        workspace + work_indptr_offset);

    for (int i = threadIdx.x; i < num_sms; i += blockDim.x) {
        if (i < merge_count) {
            const int start = i * row_chunk;
            merge_start[i] = start;
            merge_end[i] = min(start + row_chunk, heads);
            merge_partial_start[i] = start;
            merge_partial_end[i] = num_works * heads;
            merge_stride[i] = heads;
        } else {
            merge_start[i] = 0;
            merge_end[i] = 0;
            merge_partial_start[i] = 0;
            merge_partial_end[i] = 0;
            merge_stride[i] = 0;
        }
    }
    if (threadIdx.x == 0) {
        // FlashInfer's MinHeap deliberately has no index tie-break.  With
        // equal zero costs, std::pop_heap therefore assigns chunks to the
        // cluster order 0,2,6,... rather than 0,1,2,... .  Reproduce that
        // exact heap schedule so the dynamic CUDA plan is byte-identical to
        // the official host planner.  Batch-1 decode always has at most one
        // work item per cluster because kv_limit >= ceil(length/clusters).
        int heap[256];
        int cost[256];
        int cluster_work[256];
        for (int cluster = 0; cluster < num_clusters; ++cluster) {
            heap[cluster] = cluster;
            cost[cluster] = 0;
            cluster_work[cluster] = -1;
        }
        int heap_size = num_clusters;
        for (int work = 0; work < num_works; ++work) {
            const int selected = heap[0];
            const int value = heap[heap_size - 1];
            --heap_size;

            if (heap_size > 0) {
                int hole = 0;
                int right_child = 2;
                while (right_child < heap_size) {
                    // std::adjust_heap chooses the right child on a tie.
                    if (cost[heap[right_child]] >
                        cost[heap[right_child - 1]]) {
                        --right_child;
                    }
                    heap[hole] = heap[right_child];
                    hole = right_child;
                    right_child = 2 * (hole + 1);
                }
                if (right_child == heap_size) {
                    heap[hole] = heap[right_child - 1];
                    hole = right_child - 1;
                }
                while (hole > 0) {
                    const int parent = (hole - 1) / 2;
                    if (!(cost[heap[parent]] > cost[value])) {
                        break;
                    }
                    heap[hole] = heap[parent];
                    hole = parent;
                }
                heap[hole] = value;
            }

            cluster_work[selected] = work;
            cost[selected] = 1;

            int hole = heap_size;
            while (hole > 0) {
                const int parent = (hole - 1) / 2;
                if (!(cost[heap[parent]] > cost[selected])) {
                    break;
                }
                heap[hole] = heap[parent];
                hole = parent;
            }
            heap[hole] = selected;
            ++heap_size;
        }

        int output_work = 0;
        for (int cluster = 0; cluster < num_clusters; ++cluster) {
            work_indptr[cluster] = output_work;
            const int work = cluster_work[cluster];
            if (work < 0) {
                continue;
            }
            const int start = work * kv_limit;
            q_indptr_plan[output_work] = 0;
            kv_indptr_plan[output_work] = 0;
            partial_indptr[output_work] = split ? work * heads : -1;
            q_len[output_work] = 1;
            kv_len[output_work] = length;
            q_start[output_work] = 0;
            kv_start[output_work] = start;
            kv_end[output_work] = min(start + kv_limit, length);
            ++output_work;
        }
        work_indptr[num_clusters] = output_work;
    }
    for (int i = threadIdx.x; i < blocks; i += blockDim.x) {
        kv_indices[i] = i;
    }
    if (threadIdx.x == 0) {
        kv_indptr[0] = 0;
        kv_indptr[1] = blocks;
        kv_len_arr[0] = length;
    }
}

bool flashinfer_mla_batch1_plan(
    torch::Tensor int_workspace,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    torch::Tensor kv_len_arr,
    long length,
    long page_size,
    long heads,
    std::vector<int64_t> plan_info) {
    TORCH_CHECK(
        int_workspace.is_cuda() &&
        kv_indptr.is_cuda() &&
        kv_indices.is_cuda() &&
        kv_len_arr.is_cuda(),
        "FlashInfer MLA plan buffers must be CUDA");
    TORCH_CHECK(
        int_workspace.scalar_type() == at::kByte &&
        kv_indptr.scalar_type() == at::kInt &&
        kv_indices.scalar_type() == at::kInt &&
        kv_len_arr.scalar_type() == at::kInt,
        "FlashInfer MLA plan buffer dtypes are invalid");
    TORCH_CHECK(
        int_workspace.is_contiguous() &&
        kv_indptr.is_contiguous() &&
        kv_indices.is_contiguous() &&
        kv_len_arr.is_contiguous(),
        "FlashInfer MLA plan buffers must be contiguous");
    TORCH_CHECK(
        kv_indptr.numel() == 2 &&
        kv_len_arr.numel() == 1 &&
        plan_info.size() == 18,
        "FlashInfer MLA batch-1 plan shape mismatch");
    TORCH_CHECK(
        length > 0 &&
        page_size > 0 &&
        heads > 0 &&
        (length + page_size - 1) / page_size <= kv_indices.numel(),
        "FlashInfer MLA batch-1 context length is out of range");
    const int cluster_size = static_cast<int>(plan_info[0]);
    const int num_clusters = static_cast<int>(plan_info[1]);
    const int num_sms = cluster_size * num_clusters;
    TORCH_CHECK(
        (cluster_size == 1 || cluster_size == 2) &&
        num_clusters > 0 &&
        num_clusters <= 256 &&
        plan_info[15] +
            static_cast<int64_t>(num_clusters + 1) * sizeof(int32_t) <=
            int_workspace.numel(),
        "Unsupported FlashInfer MLA plan layout");

    auto stream = at::cuda::getCurrentCUDAStream();
    flashinfer_mla_batch1_plan_kernel<<<1, 256, 0, stream>>>(
        int_workspace.data_ptr<uint8_t>(),
        kv_indptr.data_ptr<int32_t>(),
        kv_indices.data_ptr<int32_t>(),
        kv_len_arr.data_ptr<int32_t>(),
        static_cast<int>(length),
        static_cast<int>(page_size),
        static_cast<int>(heads),
        cluster_size,
        num_clusters,
        num_sms,
        plan_info[2],
        plan_info[3],
        plan_info[4],
        plan_info[5],
        plan_info[6],
        plan_info[7],
        plan_info[8],
        plan_info[9],
        plan_info[10],
        plan_info[11],
        plan_info[12],
        plan_info[13],
        plan_info[14],
        plan_info[15]);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

__global__ void packed_route_slots_kernel(
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ directory,
    int64_t* __restrict__ selected,
    bool* __restrict__ hit_mask,
    const int rows,
    const int experts,
    const int top_k) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * top_k;
    if (item >= total)
        return;
    const int row = item / top_k;
    const int rank = item - row * top_k;
    const int64_t expert = route_ids[rank];
    const bool valid_id = expert >= 0 && expert < experts;
    const int64_t value = valid_id
        ? directory[expert * static_cast<int64_t>(rows) + row]
        : 0;
    selected[item] = value;
    if (row == 0)
        hit_mask[rank] = value != 0;
}

bool packed_route_slots_out(
    torch::Tensor route_ids,
    torch::Tensor directory,
    torch::Tensor selected,
    torch::Tensor hit_mask) {
    TORCH_CHECK(
        route_ids.is_cuda() && directory.is_cuda() &&
        selected.is_cuda() && hit_mask.is_cuda(),
        "packed route-slot tensors must be CUDA");
    TORCH_CHECK(
        route_ids.scalar_type() == at::kLong &&
        directory.scalar_type() == at::kLong &&
        selected.scalar_type() == at::kLong &&
        hit_mask.scalar_type() == at::kBool,
        "packed route-slot tensor dtypes are invalid");
    TORCH_CHECK(
        route_ids.is_contiguous() && directory.is_contiguous() &&
        selected.is_contiguous() && hit_mask.is_contiguous(),
        "packed route-slot tensors must be contiguous");
    TORCH_CHECK(
        route_ids.dim() == 1 && directory.dim() == 2 &&
        selected.dim() == 2 && hit_mask.dim() == 1 &&
        selected.size(0) == directory.size(1) &&
        selected.size(1) == route_ids.numel() &&
        hit_mask.numel() == route_ids.numel(),
        "packed route-slot tensor shapes are invalid");
    TORCH_CHECK(
        route_ids.get_device() == directory.get_device() &&
        route_ids.get_device() == selected.get_device() &&
        route_ids.get_device() == hit_mask.get_device(),
        "packed route-slot tensors must share one CUDA device");
    const int experts = static_cast<int>(directory.size(0));
    const int rows = static_cast<int>(directory.size(1));
    const int top_k = static_cast<int>(route_ids.numel());
    TORCH_CHECK(
        rows > 0 && experts > 0 && top_k > 0 && top_k <= 16,
        "packed route-slot dimensions are invalid");
    auto stream = at::cuda::getCurrentCUDAStream();
    const int total = rows * top_k;
    packed_route_slots_kernel<<<(total + 127) / 128, 128, 0, stream>>>(
        route_ids.data_ptr<int64_t>(),
        directory.data_ptr<int64_t>(),
        selected.data_ptr<int64_t>(),
        hit_mask.data_ptr<bool>(),
        rows,
        experts,
        top_k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

// Graph-resident segmented LRU for packed routed experts.  The complete
// ownership table and the fixed-shape output plan stay on the device, so a
// decode graph does not need route D2H copies or Python OrderedDict updates.
// One controller thread is intentional: Top-K is at most 16, while keeping
// the state transition atomic and deterministic is more important than
// parallelising a few dozen integer operations.
__global__ void packed_cache_plan_kernel(
    const int64_t* __restrict__ route_ids,
    const int route_count,
    const int layer,
    const int experts,
    const int32_t* __restrict__ signature_of_id,
    const int32_t* __restrict__ segment_offsets,
    const int signature_count,
    int32_t* __restrict__ slot_for_id,
    int32_t* __restrict__ id_of_slot,
    int64_t* __restrict__ last_used,
    int64_t* __restrict__ step,
    int32_t* __restrict__ route_slots,
    int32_t* __restrict__ source_ids,
    int32_t* __restrict__ destination_slots,
    int32_t* __restrict__ counts,
    const int max_routes) {
    if (blockIdx.x != 0 || threadIdx.x >= 32)
        return;
    const int lane = static_cast<int>(threadIdx.x);
    __shared__ int32_t unique_ids[16];
    __shared__ int32_t unique_slots[16];
    __shared__ int unique_count;
    __shared__ int hit_count;
    __shared__ int fetch_count;
    __shared__ int action;
    __shared__ int current_expert;
    __shared__ int current_logical_id;
    __shared__ int current_slot;
    __shared__ int segment_begin_shared;
    __shared__ int segment_end_shared;
    __shared__ int64_t fast_step_base;
    __shared__ bool failed;

    if (lane < max_routes) {
        route_slots[lane] = -1;
        source_ids[lane] = -1;
        destination_slots[lane] = -1;
    }
    if (lane == 0) {
        counts[0] = route_count;
        counts[1] = 0;
        counts[2] = 0;
        counts[3] = 0;
        unique_count = 0;
        hit_count = 0;
        fetch_count = 0;
        failed = false;
    }
    __syncwarp();

    // packed-cache all-hit warp fast path.  Decode routes hit the segmented
    // slab overwhelmingly often; probing all Top-K entries in parallel avoids
    // the serial controller loop without weakening exact LRU semantics.  A
    // single miss leaves every timestamp untouched and enters the original
    // deterministic miss/eviction path below.
    int32_t fast_logical_id = -1;
    int32_t fast_slot = -1;
    bool fast_hit = false;
    if (lane < route_count) {
        const int64_t expert64 = route_ids[lane];
        if (expert64 >= 0 && expert64 < experts) {
            fast_logical_id = layer * experts + static_cast<int>(expert64);
            const int signature = signature_of_id[fast_logical_id];
            if (signature >= 0 && signature < signature_count) {
                const int segment_begin = segment_offsets[signature];
                const int segment_end = segment_offsets[signature + 1];
                fast_slot = slot_for_id[fast_logical_id];
                fast_hit =
                    segment_begin >= 0 && segment_end > segment_begin &&
                    fast_slot >= segment_begin && fast_slot < segment_end &&
                    id_of_slot[fast_slot] == fast_logical_id;
            }
        }
        unique_ids[lane] = fast_logical_id;
        unique_slots[lane] = fast_slot;
    }
    __syncwarp();
    const unsigned fast_route_mask = (1u << route_count) - 1u;
    const unsigned fast_hit_bits = __ballot_sync(
        0xffffffffu, lane < route_count && fast_hit);
    if ((fast_hit_bits & fast_route_mask) == fast_route_mask) {
        bool first_occurrence = lane < route_count;
        if (first_occurrence) {
            for (int prior = 0; prior < lane; ++prior) {
                if (unique_ids[prior] == fast_logical_id) {
                    first_occurrence = false;
                    break;
                }
            }
            route_slots[lane] = fast_slot;
        }
        const unsigned fast_unique_bits = __ballot_sync(
            0xffffffffu, first_occurrence);
        const int fast_unique_count = __popc(
            fast_unique_bits & fast_route_mask);
        if (lane == 0) {
            fast_step_base = *step;
            *step = fast_step_base + fast_unique_count;
            counts[0] = route_count;
            counts[1] = fast_unique_count;
            counts[2] = fast_unique_count;
            counts[3] = 0;
        }
        __syncwarp();
        if (first_occurrence) {
            const unsigned prior_mask = (1u << lane) - 1u;
            const int recency_rank = __popc(
                fast_unique_bits & prior_mask);
            last_used[fast_slot] = fast_step_base + recency_rank + 1;
        }
        return;
    }

    for (int route_index = 0; route_index < route_count; ++route_index) {
        if (lane == 0) {
            action = -1;
            current_slot = -1;
            const int64_t expert64 = route_ids[route_index];
            if (expert64 < 0 || expert64 >= experts) {
                failed = true;
            } else {
                current_expert = static_cast<int>(expert64);
                current_logical_id = layer * experts + current_expert;
                int duplicate = -1;
                for (int item = 0; item < unique_count; ++item) {
                    if (unique_ids[item] == current_logical_id) {
                        duplicate = item;
                        break;
                    }
                }
                if (duplicate >= 0) {
                    route_slots[route_index] = unique_slots[duplicate];
                    action = 0;
                } else {
                    const int signature =
                        signature_of_id[current_logical_id];
                    if (signature < 0 || signature >= signature_count) {
                        failed = true;
                    } else {
                        segment_begin_shared = segment_offsets[signature];
                        segment_end_shared = segment_offsets[signature + 1];
                        if (segment_begin_shared < 0 ||
                            segment_end_shared <= segment_begin_shared) {
                            failed = true;
                        } else {
                            current_slot = slot_for_id[current_logical_id];
                            const bool hit =
                                current_slot >= segment_begin_shared &&
                                current_slot < segment_end_shared &&
                                id_of_slot[current_slot] == current_logical_id;
                            if (hit) {
                                ++hit_count;
                                action = 1;
                            } else {
                                current_slot = -1;
                                action = 2;
                            }
                        }
                    }
                }
            }
        }
        __syncwarp();
        if (failed)
            break;

        if (action == 2) {
            int local_empty = INT_MAX;
            for (int candidate = segment_begin_shared + lane;
                 candidate < segment_end_shared;
                 candidate += 32) {
                bool reserved = false;
                for (int item = 0; item < unique_count; ++item) {
                    reserved = reserved || unique_slots[item] == candidate;
                }
                if (!reserved && id_of_slot[candidate] < 0)
                    local_empty = min(local_empty, candidate);
            }
            // A free slot always wins. Otherwise all 32 lanes scan the exact
            // LRU timestamps and reduce the oldest (then lowest slot) pair.
            for (int offset = 16; offset > 0; offset >>= 1)
                local_empty = min(
                    local_empty,
                    __shfl_down_sync(0xffffffffu, local_empty, offset));
            if (lane == 0 && local_empty != INT_MAX)
                current_slot = local_empty;
            __syncwarp();

            if (current_slot < 0) {
                long long local_oldest = LLONG_MAX;
                int local_slot = -1;
                for (int candidate = segment_begin_shared + lane;
                     candidate < segment_end_shared;
                     candidate += 32) {
                    bool reserved = false;
                    for (int item = 0; item < unique_count; ++item) {
                        reserved = reserved || unique_slots[item] == candidate;
                    }
                    const long long age = static_cast<long long>(
                        last_used[candidate]);
                    if (!reserved &&
                        (age < local_oldest ||
                         (age == local_oldest &&
                          (local_slot < 0 || candidate < local_slot)))) {
                        local_oldest = age;
                        local_slot = candidate;
                    }
                }
                for (int offset = 16; offset > 0; offset >>= 1) {
                    const long long other_oldest = __shfl_down_sync(
                        0xffffffffu, local_oldest, offset);
                    const int other_slot = __shfl_down_sync(
                        0xffffffffu, local_slot, offset);
                    if (other_oldest < local_oldest ||
                        (other_oldest == local_oldest && other_slot >= 0 &&
                         (local_slot < 0 || other_slot < local_slot))) {
                        local_oldest = other_oldest;
                        local_slot = other_slot;
                    }
                }
                if (lane == 0)
                    current_slot = local_slot;
            }
            __syncwarp();
            if (lane == 0) {
                if (current_slot < 0) {
                    failed = true;
                } else {
                    const int previous = id_of_slot[current_slot];
                    if (previous >= 0)
                        slot_for_id[previous] = -1;
                    id_of_slot[current_slot] = current_logical_id;
                    slot_for_id[current_logical_id] = current_slot;
                    source_ids[fetch_count] = current_expert;
                    destination_slots[fetch_count] = current_slot;
                    ++fetch_count;
                }
            }
            __syncwarp();
            if (failed)
                break;
        }

        if (lane == 0 && action != 0) {
            const int64_t timestamp = ++(*step);
            last_used[current_slot] = timestamp;
            unique_ids[unique_count] = current_logical_id;
            unique_slots[unique_count] = current_slot;
            ++unique_count;
            route_slots[route_index] = current_slot;
        }
        __syncwarp();
    }

    if (lane == 0) {
        counts[1] = unique_count;
        counts[2] = hit_count;
        counts[3] = failed ? -1 : fetch_count;
    }
}

bool packed_cache_plan(
    torch::Tensor route_ids,
    int64_t layer,
    int64_t experts,
    torch::Tensor signature_of_id,
    torch::Tensor segment_offsets,
    torch::Tensor slot_for_id,
    torch::Tensor id_of_slot,
    torch::Tensor last_used,
    torch::Tensor step,
    torch::Tensor route_slots,
    torch::Tensor source_ids,
    torch::Tensor destination_slots,
    torch::Tensor counts) {
    const std::array<torch::Tensor, 11> tensors = {
        route_ids, signature_of_id, segment_offsets, slot_for_id, id_of_slot,
        last_used, step, route_slots, source_ids, destination_slots, counts};
    const int device = route_ids.get_device();
    for (const auto& tensor : tensors) {
        TORCH_CHECK(tensor.is_cuda(), "packed cache-plan tensors must be CUDA");
        TORCH_CHECK(tensor.is_contiguous(),
                    "packed cache-plan tensors must be contiguous");
        TORCH_CHECK(tensor.get_device() == device,
                    "packed cache-plan tensors must share one CUDA device");
    }
    TORCH_CHECK(route_ids.scalar_type() == at::kLong && route_ids.dim() == 1,
                "packed cache-plan route_ids must be one-dimensional int64");
    TORCH_CHECK(route_ids.numel() > 0 && route_ids.numel() <= 16,
                "packed cache-plan route count must be in [1, 16]");
    TORCH_CHECK(
        signature_of_id.scalar_type() == at::kInt &&
        segment_offsets.scalar_type() == at::kInt &&
        slot_for_id.scalar_type() == at::kInt &&
        id_of_slot.scalar_type() == at::kInt &&
        route_slots.scalar_type() == at::kInt &&
        source_ids.scalar_type() == at::kInt &&
        destination_slots.scalar_type() == at::kInt &&
        counts.scalar_type() == at::kInt,
        "packed cache-plan index tensors must be int32");
    TORCH_CHECK(last_used.scalar_type() == at::kLong &&
                step.scalar_type() == at::kLong,
                "packed cache-plan recency tensors must be int64");
    TORCH_CHECK(layer >= 0 && experts > 0 &&
                signature_of_id.numel() % experts == 0 &&
                layer < signature_of_id.numel() / experts,
                "packed cache-plan layer/expert dimensions are invalid");
    TORCH_CHECK(segment_offsets.dim() == 1 && segment_offsets.numel() >= 2,
                "packed cache-plan segment offsets are invalid");
    TORCH_CHECK(slot_for_id.numel() == signature_of_id.numel() &&
                id_of_slot.numel() == last_used.numel(),
                "packed cache-plan ownership shapes are invalid");
    TORCH_CHECK(step.numel() == 1 && counts.numel() >= 4,
                "packed cache-plan scalar/count shapes are invalid");
    const int max_routes = static_cast<int>(route_slots.numel());
    TORCH_CHECK(max_routes >= route_ids.numel() && max_routes <= 16 &&
                source_ids.numel() == max_routes &&
                destination_slots.numel() == max_routes,
                "packed cache-plan output buffers are invalid");

    auto stream = at::cuda::getCurrentCUDAStream(device);
    packed_cache_plan_kernel<<<1, 32, 0, stream>>>(
        route_ids.data_ptr<int64_t>(),
        static_cast<int>(route_ids.numel()),
        static_cast<int>(layer),
        static_cast<int>(experts),
        signature_of_id.data_ptr<int32_t>(),
        segment_offsets.data_ptr<int32_t>(),
        static_cast<int>(segment_offsets.numel() - 1),
        slot_for_id.data_ptr<int32_t>(),
        id_of_slot.data_ptr<int32_t>(),
        last_used.data_ptr<int64_t>(),
        step.data_ptr<int64_t>(),
        route_slots.data_ptr<int32_t>(),
        source_ids.data_ptr<int32_t>(),
        destination_slots.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>(),
        max_routes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

__device__ __forceinline__ uint4 cccp_uva_load16(
    const uint4* __restrict__ source) {
#if defined(__HIP_PLATFORM_AMD__)
    return *source;
#else
    uint32_t x, y, z, w;
    asm volatile(
        "ld.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3},[%4];"
        : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
        : "l"(source));
    return make_uint4(x, y, z, w);
#endif
}

__device__ __forceinline__ void cccp_uva_store16(
    uint4* __restrict__ destination,
    const uint4& value) {
#if defined(__HIP_PLATFORM_AMD__)
    *destination = value;
#else
    asm volatile(
        "st.global.wt.v4.b32 [%0],{%1,%2,%3,%4};"
        :
        : "l"(destination), "r"(value.x), "r"(value.y),
          "r"(value.z), "r"(value.w));
#endif
}

// Fixed-grid mapped-host gather.  ``packed_cache_plan_kernel`` writes the
// source expert ids, destination slots and device-side valid count consumed
// here.  Every launch therefore has stable tensor addresses and is suitable
// for CUDA Graph capture even though the route and miss count change.
__global__ void packed_cache_uva_copy_kernel(
    const int64_t* __restrict__ source_ptr_of_id,
    const int64_t* __restrict__ destination_ptr_of_slot,
    const int32_t* __restrict__ signature_of_id,
    const int64_t* __restrict__ signature_bytes,
    const int32_t* __restrict__ source_ids,
    const int32_t* __restrict__ destination_slots,
    const int32_t* __restrict__ counts,
    const int layer,
    const int experts,
    const int max_routes,
    const int blocks_per_copy) {
    const int copy_index = static_cast<int>(blockIdx.x) / blocks_per_copy;
    if (copy_index >= max_routes)
        return;
    const int fetch_count = counts[3];
    if (fetch_count <= 0 || copy_index >= fetch_count)
        return;
    const int expert = source_ids[copy_index];
    const int destination_slot = destination_slots[copy_index];
    if (expert < 0 || expert >= experts || destination_slot < 0)
        return;
    const int logical_id = layer * experts + expert;
    const int signature = signature_of_id[logical_id];
    if (signature < 0)
        return;
    const int64_t source_value = source_ptr_of_id[logical_id];
    const int64_t destination_value = destination_ptr_of_slot[destination_slot];
    const int64_t bytes = signature_bytes[signature];
    if (source_value == 0 || destination_value == 0 || bytes <= 0)
        return;

    const auto* source = reinterpret_cast<const uint8_t*>(source_value);
    auto* destination = reinterpret_cast<uint8_t*>(destination_value);
    const int block_in_copy = static_cast<int>(blockIdx.x) -
        copy_index * blocks_per_copy;
    const int64_t vector_count = bytes >> 4;
    const int64_t stride =
        static_cast<int64_t>(blocks_per_copy) * blockDim.x;
    for (int64_t vector_index =
             static_cast<int64_t>(block_in_copy) * blockDim.x + threadIdx.x;
         vector_index < vector_count;
         vector_index += stride) {
        const int64_t offset = vector_index << 4;
        const uint4 value = cccp_uva_load16(
            reinterpret_cast<const uint4*>(source + offset));
        cccp_uva_store16(
            reinterpret_cast<uint4*>(destination + offset), value);
    }
    const int64_t tail_begin = vector_count << 4;
    if (block_in_copy == 0 && threadIdx.x == 0) {
        for (int64_t offset = tail_begin; offset < bytes; ++offset)
            destination[offset] = source[offset];
    }
}

bool packed_cache_uva_copy(
    torch::Tensor source_ptr_of_id,
    torch::Tensor destination_ptr_of_slot,
    torch::Tensor signature_of_id,
    torch::Tensor signature_bytes,
    torch::Tensor source_ids,
    torch::Tensor destination_slots,
    torch::Tensor counts,
    int64_t layer,
    int64_t experts,
    int64_t blocks_per_copy) {
    const std::array<torch::Tensor, 7> tensors = {
        source_ptr_of_id, destination_ptr_of_slot, signature_of_id,
        signature_bytes, source_ids, destination_slots, counts};
    const int device = source_ptr_of_id.get_device();
    for (const auto& tensor : tensors) {
        TORCH_CHECK(tensor.is_cuda(), "packed UVA-copy tensors must be CUDA");
        TORCH_CHECK(tensor.is_contiguous(),
                    "packed UVA-copy tensors must be contiguous");
        TORCH_CHECK(tensor.get_device() == device,
                    "packed UVA-copy tensors must share one CUDA device");
    }
    TORCH_CHECK(source_ptr_of_id.scalar_type() == at::kLong &&
                destination_ptr_of_slot.scalar_type() == at::kLong &&
                signature_bytes.scalar_type() == at::kLong,
                "packed UVA-copy pointer/size tensors must be int64");
    TORCH_CHECK(signature_of_id.scalar_type() == at::kInt &&
                source_ids.scalar_type() == at::kInt &&
                destination_slots.scalar_type() == at::kInt &&
                counts.scalar_type() == at::kInt,
                "packed UVA-copy index tensors must be int32");
    TORCH_CHECK(layer >= 0 && experts > 0 &&
                source_ptr_of_id.numel() % experts == 0 &&
                layer < source_ptr_of_id.numel() / experts &&
                signature_of_id.numel() == source_ptr_of_id.numel(),
                "packed UVA-copy layer/expert dimensions are invalid");
    TORCH_CHECK(source_ids.numel() > 0 && source_ids.numel() <= 16 &&
                destination_slots.numel() == source_ids.numel() &&
                counts.numel() >= 4,
                "packed UVA-copy plan shapes are invalid");
    TORCH_CHECK(blocks_per_copy > 0 && blocks_per_copy <= 128,
                "packed UVA-copy blocks_per_copy must be in [1, 128]");

    auto stream = at::cuda::getCurrentCUDAStream(device);
    const int max_routes = static_cast<int>(source_ids.numel());
    packed_cache_uva_copy_kernel<<<
        max_routes * static_cast<int>(blocks_per_copy), 256, 0, stream>>>(
        source_ptr_of_id.data_ptr<int64_t>(),
        destination_ptr_of_slot.data_ptr<int64_t>(),
        signature_of_id.data_ptr<int32_t>(),
        signature_bytes.data_ptr<int64_t>(),
        source_ids.data_ptr<int32_t>(),
        destination_slots.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>(),
        static_cast<int>(layer),
        static_cast<int>(experts),
        max_routes,
        static_cast<int>(blocks_per_copy));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

__global__ void packed_cache_metadata_kernel(
    const int64_t* __restrict__ route_ids,
    const int32_t* __restrict__ route_slots,
    const int32_t* __restrict__ signature_of_id,
    const int64_t* __restrict__ destination_ptr_of_slot,
    const int64_t* __restrict__ projection_offsets,
    const int64_t* __restrict__ metadata_of_id,
    int64_t* __restrict__ output,
    const int layer,
    const int experts,
    const int route_count,
    const int metadata_rows,
    const int projection_count) {
    const int item = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = route_count * metadata_rows;
    if (item >= total)
        return;
    const int row = item / route_count;
    const int route_index = item - row * route_count;
    const int expert = static_cast<int>(route_ids[route_index]);
    if (expert < 0 || expert >= experts) {
        output[item] = 0;
        return;
    }
    const int logical_id = layer * experts + expert;
    int64_t value = metadata_of_id[
        static_cast<int64_t>(logical_id) * metadata_rows + row];
    if (row < projection_count * 5 && row % 5 == 0) {
        const int slot = route_slots[route_index];
        const int signature = signature_of_id[logical_id];
        if (slot < 0 || signature < 0) {
            value = 0;
        } else {
            value = destination_ptr_of_slot[slot] + projection_offsets[
                static_cast<int64_t>(signature) * projection_count + row / 5];
        }
    }
    output[item] = value;
}

bool packed_cache_metadata(
    torch::Tensor route_ids,
    torch::Tensor route_slots,
    torch::Tensor signature_of_id,
    torch::Tensor destination_ptr_of_slot,
    torch::Tensor projection_offsets,
    torch::Tensor metadata_of_id,
    torch::Tensor output,
    int64_t layer,
    int64_t experts) {
    const std::array<torch::Tensor, 7> tensors = {
        route_ids, route_slots, signature_of_id, destination_ptr_of_slot,
        projection_offsets, metadata_of_id, output};
    const int device = route_ids.get_device();
    for (const auto& tensor : tensors) {
        TORCH_CHECK(tensor.is_cuda(), "packed cache-metadata tensors must be CUDA");
        TORCH_CHECK(tensor.is_contiguous(),
                    "packed cache-metadata tensors must be contiguous");
        TORCH_CHECK(tensor.get_device() == device,
                    "packed cache-metadata tensors must share one CUDA device");
    }
    TORCH_CHECK(route_ids.scalar_type() == at::kLong &&
                destination_ptr_of_slot.scalar_type() == at::kLong &&
                projection_offsets.scalar_type() == at::kLong &&
                metadata_of_id.scalar_type() == at::kLong &&
                output.scalar_type() == at::kLong,
                "packed cache-metadata values must be int64");
    TORCH_CHECK(route_slots.scalar_type() == at::kInt &&
                signature_of_id.scalar_type() == at::kInt,
                "packed cache-metadata indices must be int32");
    const int route_count = static_cast<int>(route_ids.numel());
    TORCH_CHECK(route_ids.dim() == 1 && route_count > 0 && route_count <= 16 &&
                route_slots.numel() >= route_count,
                "packed cache-metadata route shapes are invalid");
    TORCH_CHECK(projection_offsets.dim() == 2 &&
                (projection_offsets.size(1) == 2 ||
                 projection_offsets.size(1) == 3),
                "packed cache-metadata projection offsets are invalid");
    TORCH_CHECK(layer >= 0 && experts > 0 &&
                signature_of_id.numel() % experts == 0 &&
                layer < signature_of_id.numel() / experts,
                "packed cache-metadata layer/expert dimensions are invalid");
    TORCH_CHECK(metadata_of_id.dim() == 2 &&
                metadata_of_id.size(0) == signature_of_id.numel() &&
                output.dim() == 2 &&
                output.size(0) == metadata_of_id.size(1) &&
                output.size(1) == route_count,
                "packed cache-metadata table/output shapes are invalid");

    auto stream = at::cuda::getCurrentCUDAStream(device);
    const int metadata_rows = static_cast<int>(output.size(0));
    const int total = route_count * metadata_rows;
    packed_cache_metadata_kernel<<<(total + 127) / 128, 128, 0, stream>>>(
        route_ids.data_ptr<int64_t>(),
        route_slots.data_ptr<int32_t>(),
        signature_of_id.data_ptr<int32_t>(),
        destination_ptr_of_slot.data_ptr<int64_t>(),
        projection_offsets.data_ptr<int64_t>(),
        metadata_of_id.data_ptr<int64_t>(),
        output.data_ptr<int64_t>(),
        static_cast<int>(layer),
        static_cast<int>(experts),
        route_count,
        metadata_rows,
        static_cast<int>(projection_offsets.size(1)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

torch::Tensor packed_moe_topk_grouped_stub(
    torch::Tensor input,
    torch::Tensor token_ids,
    torch::Tensor group_experts,
    torch::Tensor group_offsets,
    torch::Tensor weights,
    torch::Tensor metadata,
    int64_t activation_kind_value,
    double beta,
    double linear_beta,
    double limit,
    torch::Tensor hidden_workspace,
    torch::Tensor result,
    int64_t projection_layout_tag_value,
    int64_t max_group_tiles_value)
{
    return packed_moe_topk_grouped(
        input, token_ids, group_experts, group_offsets, weights, metadata,
        activation_kind_value, beta, linear_beta, limit, hidden_workspace,
        result, projection_layout_tag_value, max_group_tiles_value);
}

bool packed_h2d_batch(
    std::vector<torch::Tensor> sources,
    std::vector<torch::Tensor> destinations) {
    TORCH_CHECK(
        !sources.empty() && sources.size() == destinations.size(),
        "packed H2D batch source/destination counts must match");
    TORCH_CHECK(
        sources.size() <= 128,
        "packed H2D batch supports at most 128 copies");
    const int device = destinations.front().get_device();
    std::vector<void*> src_ptrs;
    std::vector<void*> dst_ptrs;
    std::vector<size_t> sizes;
    src_ptrs.reserve(sources.size());
    dst_ptrs.reserve(destinations.size());
    sizes.reserve(sources.size());
    for (size_t index = 0; index < sources.size(); ++index) {
        const auto& source = sources[index];
        const auto& destination = destinations[index];
        TORCH_CHECK(
            source.device().is_cpu() && destination.is_cuda(),
            "packed H2D batch requires CPU sources and CUDA destinations");
        TORCH_CHECK(
            source.is_pinned(),
            "packed H2D batch requires page-locked CPU sources");
        TORCH_CHECK(
            source.is_contiguous() && destination.is_contiguous(),
            "packed H2D batch tensors must be contiguous");
        TORCH_CHECK(
            destination.get_device() == device,
            "packed H2D batch destinations must share one CUDA device");
        TORCH_CHECK(
            source.nbytes() == destination.nbytes(),
            "packed H2D batch byte counts must match");
        src_ptrs.push_back(source.data_ptr());
        dst_ptrs.push_back(destination.data_ptr());
        sizes.push_back(static_cast<size_t>(source.nbytes()));
    }
#if defined(_WIN32)
    // cudaMemcpyBatchAsync has twice produced a delayed illegal-address fault
    // on CUDA 13/WDDM consumer drivers despite returning cudaSuccess. Keep the
    // layer as one Python->C++ submission, but enqueue its independent pinned
    // H2D copies through the mature cudaMemcpyAsync path. One stream and one
    // tail event preserve ordering and compute overlap without a CPU bounce
    // buffer or per-expert Python dispatch.
    auto stream = at::cuda::getCurrentCUDAStream(device);
    for (size_t index = 0; index < sizes.size(); ++index) {
        const cudaError_t status = cudaMemcpyAsync(
            dst_ptrs[index],
            src_ptrs[index],
            sizes[index],
            cudaMemcpyHostToDevice,
            stream);
        TORCH_CHECK(
            status == cudaSuccess,
            "compiled packed H2D batch failed at copy ",
            index,
            ": ",
            cudaGetErrorString(status));
    }
    return true;
#elif CUDART_VERSION >= 12080
    cudaMemcpyAttributes attributes{};
    attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attributes.flags = cudaMemcpyFlagPreferOverlapWithCompute;
    size_t attributes_index = 0;
    auto stream = at::cuda::getCurrentCUDAStream(device);
#if CUDART_VERSION >= 13000
    // CUDA 13 removed the failIdx output and made source pointers const.
    // Keep one public operator source compatible with both Blackwell's CUDA
    // 13 toolchain and the CUDA 12.8 Hopper deployment.
    const cudaError_t status = cudaMemcpyBatchAsync(
        dst_ptrs.data(),
        reinterpret_cast<const void* const*>(src_ptrs.data()),
        sizes.data(),
        sizes.size(),
        &attributes,
        &attributes_index,
        1,
        stream);
    TORCH_CHECK(
        status == cudaSuccess,
        "cudaMemcpyBatchAsync failed: ",
        cudaGetErrorString(status));
#else
    size_t failed_index = SIZE_MAX;
    const cudaError_t status = cudaMemcpyBatchAsync(
        dst_ptrs.data(),
        src_ptrs.data(),
        sizes.data(),
        sizes.size(),
        &attributes,
        &attributes_index,
        1,
        &failed_index,
        stream);
    TORCH_CHECK(
        status == cudaSuccess,
        "cudaMemcpyBatchAsync failed at copy ",
        failed_index,
        ": ",
        cudaGetErrorString(status));
#endif
    return true;
#else
    return false;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vq_gemv", &vq_gemv, "VQ grouped GEMV (fused codebook lookup + dot)");
    m.def("dense_vq_gemv_packed", &dense_vq_gemv_packed,
          "Dense VQ Linear GEMV directly from p8-p16 indices");
    m.def("dense_vq_gemv_grouped_fp8_codebook",
          &dense_vq_gemv_grouped_fp8_codebook,
          "Grouped GGUF-style BF16 Decode directly from packed VQ");
    m.def("dense_vq_gemv_packed_fp8_codebook",
          &dense_vq_gemv_packed_fp8_codebook,
          "Single-projection GEMV directly from packed VQ and E4M3 codebook");
    m.def("dense_vq_gemv_packed_q8_codebook",
          &dense_vq_gemv_packed_q8_codebook,
          "Single-projection DP4A GEMV directly from packed VQ and Q8 codebook");
    m.def("dense_vq_dequant_fp8_packed", &dense_vq_dequant_fp8_packed,
        "Expand packed VQ to FP8 rows with a single tensor scale.");
    m.def("dense_vq_compile_int4_g64", &dense_vq_compile_int4_g64,
          "Compile Dense VQ into a resident INT4-G64 execution image");
    m.def("dense_vq_expand_fp8_tile_out", &dense_vq_expand_fp8_tile_out,
          "Expand one packed Dense VQ row tile into a fixed E4M3 workspace");
    m.def("dense_vq_mma_packed_m1", &dense_vq_mma_packed_m1,
          "Decode packed Dense VQ directly into one Tensor Core MMA row");
    m.def("dense_vq_quantize_fp8_codebook", &dense_vq_quantize_fp8_codebook,
          "Quantize a compact Dense VQ codebook into tensor-scaled E4M3");
    m.def("dense_vq_dequant_packed", &dense_vq_dequant_packed,
          "Dense VQ BF16 dequantization with optional row selection");
    m.def("dense_vq_expand_native8", &dense_vq_expand_native8,
          "Expand packed VQ through a pre-quantized E4M3/INT8 codebook");
    m.def("dense_fp8_quantize_rows", &dense_fp8_quantize_rows,
          "Row-wise BF16/F32 to E4M3 activation conversion");
    m.def("gated_activation_fp8_quantize_rows",
          &gated_activation_fp8_quantize_rows,
          "Fused gated BF16 activation and row-wise E4M3 conversion");
    m.def("routed_weighted_reduce", &routed_weighted_reduce,
          "Fused routed row reorder, weight and Top-K reduction");
    m.def("kimi_short_conv3", &kimi_short_conv3,
          "Kimi three-way one-token short convolution");
    m.def("qwen35_conv1d_update", &qwen35_conv1d_update,
          "Qwen3.5 one-token cached depthwise convolution");
    m.def("qwen35_delta_recurrent", &qwen35_delta_recurrent,
          "Qwen3.5 one-token gated-delta recurrent update");
    m.def("qwen35_delta_recurrent_batch", &qwen35_delta_recurrent_batch,
          "Qwen3.5 ordered batched gated-delta recurrent update");
    m.def(
          "qwen35_delta_recurrent_batch_checkpoint",
          &qwen35_delta_recurrent_batch_checkpoint,
          "Qwen3.5 batched gated-delta update with token checkpoints");
    m.def("kimi_kda_recurrent", &kimi_kda_recurrent,
          "Kimi KDA one-token recurrent update with V-first FP32 state");
    m.def("kimi_kda_recurrent_batch", &kimi_kda_recurrent_batch,
          "Kimi ordered block-prefill recurrent update with V-first FP32 state");
    m.def("kimi_gated_rmsnorm", &kimi_gated_rmsnorm,
          "Kimi one-token gated RMSNorm");
    m.def(
          "packed_moe_topk",
          &packed_moe_topk,
          "Packed 8/9/10/12/14/16-bit Top-K routed expert MLP");
    m.def(
          "packed_moe_topk_compact_fp8_codebook",
          &packed_moe_topk_compact_fp8_codebook,
          "Top-K MoE directly from compact VQ and E4M3 codebooks");
    m.def(
          "packed_moe_topk_compact_q8_codebook",
          &packed_moe_topk_compact_q8_codebook,
          "Top-K MoE directly from compact VQ and Q8 codebooks");
    m.def(
          "compact_q8_codebook_l2_prefetch",
          &compact_q8_codebook_l2_prefetch,
          "Prefetch selected compact Q8 codebooks into L2");
    m.def(
          "vq_projection_dequant",
          &vq_projection_dequant,
          "Dequantize packed three-projection experts to dense BF16");
    m.def(
          "vq_projection_expand_native8",
          &vq_projection_expand_native8,
          "Expand packed projections to native E4M3/INT8 execution images");
    m.def(
          "packed_moe_topk_grouped",
          &packed_moe_topk_grouped_stub,
          "Grouped token-by-expert three-projection packed MoE prefill");
    m.def(
          "packed_stage_topk_three_projection",
          &packed_stage_topk_three_projection,
          "Stage mapped-host Top-K three-projection experts into fixed VRAM");
    m.def(
          "packed_route_slots_out",
          &packed_route_slots_out,
          "Gather stable packed expert slot metadata on CUDA");
    m.def("packed_cache_plan",
          &packed_cache_plan,
          "Plan signature-segmented packed expert LRU entirely on CUDA");
    m.def("packed_cache_uva_copy",
          &packed_cache_uva_copy,
          "Copy planned packed expert misses from mapped host RAM on CUDA");
    m.def("packed_cache_metadata",
          &packed_cache_metadata,
          "Publish planned packed expert metadata entirely on CUDA");
    m.def(
          "packed_h2d_batch",
          &packed_h2d_batch,
          "Submit independent compact host-to-device copies as one batch");
    m.def(
          "kimi_moe_packed",
          &packed_moe_topk,
          "Compatibility alias for packed_moe_topk");
    m.def("vq_gemv_slots_out", &vq_gemv_slots_out,
          "Stable-slot grouped BF16 VQ GEMV into caller workspace");
    m.def("moe_mlp_slots", &moe_mlp_slots,
          "Stable-slot BF16 VQ MLP (GU + SwiGLU + DN + weighted sum)");
    m.def("moe_mlp_routed_slots", &moe_mlp_routed_slots,
          "Device-routed full-resident VQ MLP partial");
    m.def("moe_mlp_routed_vv", &moe_mlp_routed_vv,
          "Shared-codebook D4/K4096 full-resident VQ MLP partial");
    m.def("moe_mlp_routed_codegemm", &moe_mlp_routed_codegemm,
          "CodeGEMM Psumbook full-resident VQ MLP partial");
    m.def("pack_vq_tensor_shard_codegemm",
          &pack_vq_tensor_shard_codegemm,
          "Pack a tensor-sharded v256/D4 expert for CodeGEMM");
    m.def("unpack_vq_codegemm", &unpack_vq_codegemm,
          "Restore CodeGEMM indices to CCCP row-major layout");
    m.def("expert_dispatch_pack", &expert_dispatch_pack,
          "Pack one full-resident expert dispatch from a peer GPU");
    m.def("tp_peer_copy", &tp_peer_copy,
          "Graph-safe peer copy for TP rank source tensors");
    m.def("tp_attention_peer_dispatch", &tp_attention_peer_dispatch,
          "Graph-safe fused Attention TP peer dispatch");
    m.def("tp_attention_source_pack", &tp_attention_source_pack,
          "Fused Attention TP primary source packing");
    m.def("hc_sinkhorn", &hc_sinkhorn, "HC 4x4 sinkhorn (fused softmax + iterations)");
    m.def("rmsnorm", &rmsnorm, "RMSNorm (fused, f32)");
    m.def(
        "rmsnorm_bf16",
        &rmsnorm_bf16,
        "RMSNorm (fused, BF16 input)");
    m.def(
        "attention_residual_bf16",
        &attention_residual_bf16,
        "Attention residual mixer (fused, BF16)");
    m.def(
        "gated_activation_bf16",
        &gated_activation_bf16,
        "SiLU/SiTU gated activation (fused, BF16)");
    m.def("glm_mla_bmm_decode", &glm_mla_bmm_decode,
          "Direct cuBLAS strided-batched MLA decode GEMM");
    m.def("flashinfer_mla_batch1_plan",
          &flashinfer_mla_batch1_plan,
          "Device-side exact batch-1 FlashInfer MLA scheduler");
    m.def("rope1", &rope1, "RoPE interleaved (decode single-phase, f32)");
    m.def(
          "head_rmsnorm_rope",
          &head_rmsnorm_rope,
          "In-place per-head RMSNorm and interleaved tail RoPE");
    m.def("glm_rope_qk", &glm_rope_qk,
          "GLM MLA Q/K RoPE (fused, HF cat layout, f32)");
    m.def("glm_latent_kv_decode_prepare",
          &glm_latent_kv_decode_prepare,
          "GLM decode RMS/RoPE and BF16 latent KV writes");
    m.def("glm_merge_scores", &glm_merge_scores,
          "GLM latent attention score scale/add (fused, BF16 to f32)");
    m.def(
        "latent_mla_attention_decode",
        &latent_mla_attention_decode,
        "Dynamic-length BF16 latent MLA decode");
    m.def(
        "latent_mla_attention_scores",
        &latent_mla_attention_scores,
        "Dynamic-length BF16 latent MLA score preparation");
    m.def("dsv4_attn_decode", &dsv4_attn_decode, "DSV4 decode attention core (fused, f32)");
    m.def("dsv4_kv_commit_controlled", &dsv4_kv_commit_controlled,
          "Commit one DSV4 KV row from a device decode-control position");
    m.def("dsv4_compressor_step_controlled", &dsv4_compressor_step_controlled,
          "Update and pool paged compressed state from device control");
    m.def("paged_indexer_query_fp8", &paged_indexer_query_fp8,
          "Paged Indexer fused RoPE Hadamard FP8 query transform");
    m.def("paged_indexer_reduce_logits", &paged_indexer_reduce_logits,
          "Paged Indexer ReLU head-weight reduction");
    m.def("sparse_attention_inverse_rope", &sparse_attention_inverse_rope,
          "Sparse Attention BF16 output inverse RoPE");
    m.def("dsv4_attn_decode_controlled", &dsv4_attn_decode_controlled,
          "Fixed-shape DSV4 attention driven by device decode control");
    m.def("dsv4_hc_pre", &dsv4_hc_pre, "DSV4 HC pre (fused RMS/GEMV/sinkhorn/reduce)");
    m.def("dsv4_hc_pre_norm", &dsv4_hc_pre_norm,
          "DSV4 BF16 HC pre + RMSNorm (FP32 reductions)");
    m.def("dsv4_hc_pre_norm_out", &dsv4_hc_pre_norm_out,
          "DSV4 BF16 HC pre + RMSNorm into caller-owned buffers");
    m.def("dsv4_hc_post", &dsv4_hc_post,
          "DSV4 BF16 HC post residual mix (FP32 accumulation)");
    m.def("dsv4_hc_post_out", &dsv4_hc_post_out,
          "DSV4 BF16 HC post into caller-owned hidden buffer");
    m.def("dsv4_hc_post_moe", &dsv4_hc_post_moe,
          "DSV4 BF16 routed/shared merge plus HC post");
    m.def("dsv4_hc_post_moe_out", &dsv4_hc_post_moe_out,
          "DSV4 routed/shared merge plus HC post into caller buffer");
    m.def("dsv4_route_post", &dsv4_route_post,
          "DSV4 learned-route top-k, gather, normalize and scale");
    m.def("sigmoid_route", &sigmoid_route,
          "Sigmoid corrected Top-K with normalized route weights");
    m.def("sigmoid_route_out", &sigmoid_route_out,
          "Sigmoid route into caller-owned decode buffers");
    m.def("sqrtsoftplus_route_out", &sqrtsoftplus_route_out,
          "Sqrt-softplus route into caller-owned decode buffers");
    m.def(
          "linear_sigmoid_route_out",
          &linear_sigmoid_route_out,
          "FP32 linear projection plus sigmoid Top-K routing");
    m.def("glm_route", &sigmoid_route,
          "Compatibility alias for sigmoid_route");
    m.def("glm_route_out", &sigmoid_route_out,
          "Compatibility alias for sigmoid_route_out");
    m.def("paged_gather_bf16", &paged_gather_bf16,
          "Gather batch-1 BF16 entries from stable paged storage");
    m.def("hadamard_bf16", &hadamard_bf16,
          "Normalized BF16 Walsh-Hadamard transform with FP32 butterflies");
    m.def("int4_embedding_lookup", &int4_embedding_lookup,
          "Packed INT4-G64 single-row embedding lookup");
    m.def("int4_embedding_lookup_device_row",
          &int4_embedding_lookup_device_row,
          "Packed INT4-G64 embedding lookup with a CUDA row index");
    m.def("int4_gemv_packed_f32_v4s", &int4_gemv_packed_f32_v4s<float>,
        "Segmented vector4 INT4 GEMV (conflict-free, staged activation).");
    m.def("int4_gemv_packed_f32_v4s_bf16", &int4_gemv_packed_f32_v4s<__nv_bfloat16>,
        "Segmented vector4 INT4 GEMV, bf16 activations.");
    m.def("int4_gemv_v1b", &int4_gemv_v1b,
        "v1b batched GEMV (bit-identical to v1, B<=5).");
    m.def("int4_gemv_v30", &int4_gemv_v30,
        "v30 MMVQ-style Q8xQ4 dp4a batched GEMV, B<=6, native layout.");
    m.def("int4_v30_quant", &int4_v30_quant,
        "v30 per-64-col-group Q8 activation quantizer (single kernel).");
    m.def("int4_repack_v21b", &int4_repack_v21b,
        "v21b repack (same layout).");
    m.def("int4_gemv_v21b", &int4_gemv_v21b,
        "v21b batched GEMV [B,cols]->[B,rows], B<=5.");
    m.def("int4_repack_v21", &int4_repack_v21,
        "v21 superstep-interleave repack.");
    m.def("int4_gemv_v21", &int4_gemv_v21,
        "v21 interleave GEMV (892 GB/s cold-stream, SM75+).");
    m.def("int4_gemv_v17", &int4_gemv_v17,
        "v17: marlin tiles + FMA GEMV (731-1000 GB/s, SM75+).");
    m.def("int4_repack_marlin", &int4_repack_marlin,
        "Repack INT4-G64 into marlin superstep tiles.");
    m.def("int4_gemv_marlin", &int4_gemv_marlin,
        "Marlin-layout INT4 GEMV (coalesced 4B/lane, mma.sync SM75+).");
    m.def("int4_gemv_packed_f32_v2", &int4_gemv_packed_f32_v2<float>,
        "Split-K INT4 batch-1 GEMV (vectorized, all architectures).");
    m.def("int4_gemv_packed_f32_v2_bf16", &int4_gemv_packed_f32_v2<__nv_bfloat16>,
        "Split-K INT4 batch-1 GEMV, bf16 activations.");
    m.def("int4_gemv_packed_f32", &int4_gemv_packed_f32,
          "Shared-input packed INT4-G64 GEMV for float32 decode");
    m.def("block_fp8_gemv_f32", &block_fp8_gemv_f32,
          "Native E4M3 block-scaled GEMV for float32 decode");
    m.def("block_fp8_grouped_gemv_f32", &block_fp8_grouped_gemv_f32,
          "Grouped native E4M3 block-scaled GEMV for decode");
    m.def("int4_glm_qb_split", &int4_glm_qb_split,
          "Packed GLM Q-B GEMV into BF16 no-PE and FP32 RoPE outputs");
    m.def("glm_norm_qkv_int4", &glm_norm_qkv_int4,
          "GLM decode RMSNorm plus packed Q-A/KV-A projections");
    m.def("glm_residual_norm_router",
          &glm_residual_norm_router,
          "GLM decode residual add plus RMSNorm and router projection");
    m.def("glm_residual_norm_router_norm_out",
          &glm_residual_norm_router_norm_out,
          "GLM residual/router with caller-owned normalized output");
    m.def("residual_add3",
          &residual_add3,
          "Three-way residual addition for float32 or bfloat16");
    m.def("glm_moe_residual_add",
          &residual_add3,
          "Compatibility alias for three-way residual addition");
    m.def("glm_ep_reduce_residual",
          &glm_ep_reduce_residual,
          "GLM TP routed/shared contribution reduction plus residual");
    m.def("tp_all_rank_reduce",
          &tp_all_rank_reduce,
          "Reduce FP32 TP partials into fixed outputs on every rank");
    m.def("tp1_moe_finalize",
          &tp1_moe_finalize,
          "Fuse TP1 routed/shared partials with a BF16 residual");
    m.def(
          "compressed_state_update",
          &compressed_state_update,
          "Write projected KV/score plus phase bias into ring state");
    m.def(
          "tp_all_rank_reduce_from_events",
          &tp_all_rank_reduce_from_events,
          "Wait fixed rank events and reduce TP partials into outputs");
    m.def(
          "tp_moe_finalize_from_events",
          &tp_moe_finalize_from_events,
          "Wait rank events and finalize routed/shared TP MoE outputs");
    m.def("tp_hidden_add_batch",
          &tp_hidden_add_batch,
          "Add fixed BF16 TPHidden replicas in one host call");
    m.def("tp_hidden_rmsnorm_batch",
          &tp_hidden_rmsnorm_batch,
          "RMSNorm fixed BF16 TPHidden replicas in one host call");
    m.def("tp_hidden_residual_mix_batch",
          &tp_hidden_residual_mix_batch,
          "Residual-mix fixed BF16 TPHidden replicas in one host call");
    m.def("launch_cuda_graphs",
          &launch_cuda_graphs,
          "Launch one prepared CUDA Graph per TP rank in one host call");
    m.def("launch_cuda_graphs_reduce",
          &launch_cuda_graphs_reduce,
          "Launch TP graphs, wait ranks and reduce in one host call");
    m.def(
          "launch_cuda_graphs_reduce_norm_router",
          &launch_cuda_graphs_reduce_norm_router,
          "Launch TP Attention graphs then reduce, normalize and route");
    pybind11::class_<TPGraphLaunchBatch>(m, "TPGraphLaunchBatch")
        .def(pybind11::init<
             std::vector<int64_t>,
             std::vector<int64_t>,
             std::vector<int64_t>,
             std::vector<int64_t>,
             int64_t>())
        .def(pybind11::init<
             std::vector<int64_t>,
             std::vector<std::vector<int64_t>>,
             std::vector<int64_t>,
             std::vector<int64_t>,
             int64_t>())
        .def(pybind11::init<
             std::vector<int64_t>,
             std::vector<std::vector<std::vector<int64_t>>>,
             std::vector<int64_t>,
             std::vector<int64_t>,
             int64_t>())
        .def("launch", &TPGraphLaunchBatch::launch)
        .def("launch_tp1", &TPGraphLaunchBatch::launch_tp1)
        .def("raw_graphs", &TPGraphLaunchBatch::raw_graphs)
        .def("launch_reduce", &TPGraphLaunchBatch::launch_reduce)
        .def(
            "launch_reduce_many",
            &TPGraphLaunchBatch::launch_reduce_many)
        .def(
            "launch_all_rank",
            &TPGraphLaunchBatch::launch_all_rank)
        .def(
            "launch_all_rank_from_events",
            &TPGraphLaunchBatch::launch_all_rank_from_events)
        .def(
            "launch_all_rank_many_from_events",
            &TPGraphLaunchBatch::launch_all_rank_many_from_events)
        .def(
            "reduce_all_rank_many_from_events",
            &TPGraphLaunchBatch::reduce_all_rank_many_from_events)
        .def(
            "launch_from_events",
            &TPGraphLaunchBatch::launch_from_events)
        .def(
            "launch_moe_all_rank_from_events",
            &TPGraphLaunchBatch::launch_moe_all_rank_from_events)
        .def(
            "launch_moe_hc_all_rank_from_events",
            &TPGraphLaunchBatch::launch_moe_hc_all_rank_from_events)
        .def(
            "launch_reduce_norm_router",
            &TPGraphLaunchBatch::launch_reduce_norm_router)
        .def(
            "launch_moe_layer",
            &TPGraphLaunchBatch::launch_moe_layer);
    pybind11::class_<TPNoOwnerMoELayerPlan>(
        m,
        "TPNoOwnerMoELayerPlan")
        .def(
            pybind11::init<
                const TPGraphLaunchBatch&,
                const TPGraphLaunchBatch&,
                const TPGraphLaunchBatch&,
                const TPGraphLaunchBatch&,
                std::vector<int64_t>,
                std::vector<std::vector<torch::Tensor>>,
                std::vector<std::vector<torch::Tensor>>,
                std::vector<int64_t>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<int64_t>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<int64_t>,
                std::vector<torch::Tensor>,
                std::vector<int64_t>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<int64_t>>(),
            pybind11::keep_alive<1, 2>(),
            pybind11::keep_alive<1, 3>(),
            pybind11::keep_alive<1, 4>(),
            pybind11::keep_alive<1, 5>())
        .def("launch", &TPNoOwnerMoELayerPlan::launch)
        .def(
            "launch_from_events",
            &TPNoOwnerMoELayerPlan::launch_from_events);
    pybind11::class_<TPNoOwnerDecodeLayerPlan>(
        m,
        "TPNoOwnerDecodeLayerPlan")
        .def(
            pybind11::init<
                const TPGraphLaunchBatch&,
                const TPNoOwnerMoELayerPlan&,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<int64_t>>(),
            pybind11::keep_alive<1, 2>(),
            pybind11::keep_alive<1, 3>())
        .def(
            "launch_from_events",
            &TPNoOwnerDecodeLayerPlan::launch_from_events);
    pybind11::class_<TPNoOwnerHCDecodeLayerPlan>(
        m,
        "TPNoOwnerHCDecodeLayerPlan")
        .def(
            pybind11::init<
                const TPGraphLaunchBatch&,
                const TPGraphLaunchBatch&,
                const TPGraphLaunchBatch&,
                const TPGraphLaunchBatch&,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<int64_t>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<int64_t>,
                std::vector<int64_t>,
                std::vector<torch::Tensor>,
                std::vector<torch::Tensor>,
                std::vector<int64_t>,
                std::vector<torch::Tensor>,
                std::vector<int64_t>,
                long,
                double>(),
            pybind11::keep_alive<1, 2>(),
            pybind11::keep_alive<1, 3>(),
            pybind11::keep_alive<1, 4>(),
            pybind11::keep_alive<1, 5>())
        .def(
            "launch_from_events",
            &TPNoOwnerHCDecodeLayerPlan::launch_from_events);
    m.def(
          "bf16_gemv_out",
          &bf16_gemv_out,
          "Fixed-output BF16 GEMV with FP32 accumulation");
    m.def("int4_swiglu_packed_f32", &int4_swiglu_packed_f32,
          "Fused gate/up packed INT4-G64 GEMV plus FP32 SwiGLU");
}
