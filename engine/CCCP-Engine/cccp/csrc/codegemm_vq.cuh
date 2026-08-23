// CodeGEMM-style Psumbook kernels for CCCP's full-resident v256/D4 experts.
//
// The packed layout is [K/16, M] int32: each word contains four consecutive
// uint8 D4 code indices. The byte count is identical to CCCP's row-major
// [M,K/4] representation. RAM-resident experts keep the original layout.

constexpr int CCCP_CODEGEMM_CODES = 256;
constexpr int CCCP_CODEGEMM_VECTOR = 4;
constexpr int CCCP_CODEGEMM_K_TILE = 128;
constexpr int CCCP_CODEGEMM_M_TILE = 2048;
constexpr int CCCP_CODEGEMM_THREADS = 256;

// PyTorch's Windows HIPIFY converts the top-level .cu file but does not
// recursively rewrite this included .cuh file.  Keep the exact same kernels
// and expose a small backend-neutral surface for CUDA and ROCm builds.
#if defined(__HIP_PLATFORM_AMD__)
using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
#define CCCP_GPU_GET_DEVICE hipGetDevice
#define CCCP_GPU_SUCCESS hipSuccess
#define CCCP_GPU_MEMSET_ASYNC hipMemsetAsync
#define CCCP_GPU_CURRENT_STREAM() c10::hip::getCurrentHIPStream()
#define CCCP_GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()
#else
#define CCCP_GPU_GET_DEVICE cudaGetDevice
#define CCCP_GPU_SUCCESS cudaSuccess
#define CCCP_GPU_MEMSET_ASYNC cudaMemsetAsync
#define CCCP_GPU_CURRENT_STREAM() at::cuda::getCurrentCUDAStream()
#define CCCP_GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

__device__ __forceinline__ float cccp_codegemm_dot4(
    const __nv_bfloat16* code,
    const __nv_bfloat16* input)
{
    const auto* c2 = reinterpret_cast<const __nv_bfloat162*>(code);
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(input);
    const float2 c0 = __bfloat1622float2(c2[0]);
    const float2 x0 = __bfloat1622float2(x2[0]);
    const float2 c1 = __bfloat1622float2(c2[1]);
    const float2 x1 = __bfloat1622float2(x2[1]);
    float result = fmaf(c0.x, x0.x, 0.f);
    result = fmaf(c0.y, x0.y, result);
    result = fmaf(c1.x, x1.x, result);
    return fmaf(c1.y, x1.y, result);
}

__global__ void cccp_pack_gu_tensor_shard_kernel(
    const uint8_t* __restrict__ source,
    uint32_t* __restrict__ target,
    const int source_blocks,
    const int global_intermediate,
    const int shard_start,
    const int local_intermediate)
{
    const int target_rows = 2 * local_intermediate;
    const int packed_k = source_blocks / 4;
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= packed_k * target_rows) return;
    const int row = item % target_rows;
    const int word_k = item / target_rows;
    const int source_row =
        row < local_intermediate
        ? shard_start + row
        : global_intermediate + shard_start + row - local_intermediate;
    const uint8_t* bytes =
        source + (long)source_row * source_blocks + word_k * 4;
    target[item] =
        static_cast<uint32_t>(bytes[0]) |
        (static_cast<uint32_t>(bytes[1]) << 8) |
        (static_cast<uint32_t>(bytes[2]) << 16) |
        (static_cast<uint32_t>(bytes[3]) << 24);
}

__global__ void cccp_pack_dn_tensor_shard_kernel(
    const uint8_t* __restrict__ source,
    uint32_t* __restrict__ target,
    const int rows,
    const int source_blocks,
    const int source_block_start,
    const int target_blocks)
{
    const int packed_k = target_blocks / 4;
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= packed_k * rows) return;
    const int row = item % rows;
    const int word_k = item / rows;
    const uint8_t* bytes =
        source + (long)row * source_blocks +
        source_block_start + word_k * 4;
    target[item] =
        static_cast<uint32_t>(bytes[0]) |
        (static_cast<uint32_t>(bytes[1]) << 8) |
        (static_cast<uint32_t>(bytes[2]) << 16) |
        (static_cast<uint32_t>(bytes[3]) << 24);
}

void pack_vq_tensor_shard_codegemm(
    torch::Tensor source_gu,
    torch::Tensor source_dn,
    torch::Tensor target_gu,
    torch::Tensor target_dn,
    long global_intermediate,
    long shard_start,
    long local_intermediate)
{
    TORCH_CHECK(
        source_gu.is_cuda() && source_dn.is_cuda() &&
        target_gu.is_cuda() && target_dn.is_cuda(),
        "CodeGEMM pack tensors must be CUDA");
    TORCH_CHECK(
        source_gu.scalar_type() == at::kByte &&
        source_dn.scalar_type() == at::kByte &&
        target_gu.scalar_type() == at::kByte &&
        target_dn.scalar_type() == at::kByte,
        "CodeGEMM packing requires uint8 v256 indices");
    TORCH_CHECK(
        source_gu.is_contiguous() && source_dn.is_contiguous() &&
        target_gu.is_contiguous() && target_dn.is_contiguous(),
        "CodeGEMM pack tensors must be contiguous");
    TORCH_CHECK(
        source_gu.dim() == 2 && source_dn.dim() == 2 &&
        target_gu.dim() == 2 && target_dn.dim() == 2,
        "CodeGEMM pack tensors must be matrices");
    TORCH_CHECK(
        source_gu.size(0) == 2 * global_intermediate &&
        source_dn.size(1) * CCCP_CODEGEMM_VECTOR == global_intermediate,
        "source expert dimensions do not match");
    TORCH_CHECK(
        target_gu.size(0) == 2 * local_intermediate &&
        target_dn.size(0) == source_dn.size(0) &&
        target_dn.size(1) * CCCP_CODEGEMM_VECTOR == local_intermediate,
        "target expert shard dimensions do not match");
    TORCH_CHECK(
        source_gu.size(1) % 4 == 0 &&
        target_dn.size(1) % 4 == 0,
        "CodeGEMM reduction dimensions must be divisible by 16");

    int current = -1;
    const auto status = CCCP_GPU_GET_DEVICE(&current);
    TORCH_CHECK(
        status == CCCP_GPU_SUCCESS && current == target_gu.get_device() &&
        current == target_dn.get_device(),
        "CodeGEMM pack must run on the target CUDA device");
    ensure_peer_access(
        current,
        source_gu.get_device(),
        "CodeGEMM expert packing");
    TORCH_CHECK(
        source_dn.get_device() == source_gu.get_device(),
        "CodeGEMM pack sources must share one device");

    auto stream = CCCP_GPU_CURRENT_STREAM();
    const int gu_items =
        (int)(source_gu.size(1) / 4 * target_gu.size(0));
    cccp_pack_gu_tensor_shard_kernel<<<
        (gu_items + 255) / 256,
        256,
        0,
        stream>>>(
            source_gu.data_ptr<uint8_t>(),
            reinterpret_cast<uint32_t*>(target_gu.data_ptr<uint8_t>()),
            (int)source_gu.size(1),
            (int)global_intermediate,
            (int)shard_start,
            (int)local_intermediate);
    const int dn_items =
        (int)(target_dn.size(1) / 4 * target_dn.size(0));
    cccp_pack_dn_tensor_shard_kernel<<<
        (dn_items + 255) / 256,
        256,
        0,
        stream>>>(
            source_dn.data_ptr<uint8_t>(),
            reinterpret_cast<uint32_t*>(target_dn.data_ptr<uint8_t>()),
            (int)source_dn.size(0),
            (int)source_dn.size(1),
            (int)(shard_start / CCCP_CODEGEMM_VECTOR),
            (int)target_dn.size(1));
    CCCP_GPU_KERNEL_LAUNCH_CHECK();
}

__global__ void cccp_unpack_codegemm_kernel(
    const uint32_t* __restrict__ packed,
    uint8_t* __restrict__ row_major,
    const int rows,
    const int blocks)
{
    const int packed_k = blocks / 4;
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= packed_k * rows) return;
    const int row = item % rows;
    const int word_k = item / rows;
    const uint32_t word = packed[item];
    uint8_t* output =
        row_major + (long)row * blocks + word_k * 4;
    output[0] = static_cast<uint8_t>(word);
    output[1] = static_cast<uint8_t>(word >> 8);
    output[2] = static_cast<uint8_t>(word >> 16);
    output[3] = static_cast<uint8_t>(word >> 24);
}

torch::Tensor unpack_vq_codegemm(
    torch::Tensor storage,
    long rows,
    long blocks)
{
    TORCH_CHECK(
        storage.is_cuda() && storage.scalar_type() == at::kByte &&
        storage.is_contiguous(),
        "CodeGEMM storage must be contiguous CUDA uint8");
    TORCH_CHECK(
        rows > 0 && blocks > 0 && blocks % 4 == 0 &&
        storage.numel() == rows * blocks,
        "CodeGEMM unpack shape does not match storage");
    auto output = torch::empty(
        {rows, blocks},
        storage.options());
    const int items = (int)(rows * blocks / 4);
    auto stream = CCCP_GPU_CURRENT_STREAM();
    cccp_unpack_codegemm_kernel<<<
        (items + 255) / 256,
        256,
        0,
        stream>>>(
            reinterpret_cast<const uint32_t*>(
                storage.data_ptr<uint8_t>()),
            output.data_ptr<uint8_t>(),
            (int)rows,
            (int)blocks);
    CCCP_GPU_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void cccp_psumbook_routed_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int64_t* __restrict__ route_ids,
    const int64_t* __restrict__ metadata,
    float* __restrict__ output,
    const int top_k,
    const int expert_count,
    const int metadata_base,
    const int output_dim,
    const int input_dim,
    const long input_stride)
{
    const int position = blockIdx.z;
    if (position >= top_k) return;
    const int expert = static_cast<int>(route_ids[position]);
    if (expert < 0 || expert >= expert_count) return;
    const int64_t packed_address =
        metadata[(long)metadata_base * expert_count + expert];
    if (packed_address == 0) return;
    const int64_t codebook_address =
        metadata[(long)(metadata_base + 1) * expert_count + expert];
    const int dtype_tag = static_cast<int>(
        metadata[(long)(metadata_base + 4) * expert_count + expert]);
    // Mixed-codebook layers keep v256 experts in the CodeGEMM Psumbook
    // layout and vv/K4096 experts in their native uint16 row-major layout.
    // Both formats share one metadata table; never reinterpret a uint16
    // address as four packed uint8 Psumbook codes.
    if (dtype_tag != 0) return;
    const auto* packed = reinterpret_cast<const uint32_t*>(
        static_cast<uintptr_t>(packed_address));
    const auto* codebook = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(codebook_address));

    const int input_tile = blockIdx.y * CCCP_CODEGEMM_K_TILE;
    __shared__ float lut[
        CCCP_CODEGEMM_K_TILE / CCCP_CODEGEMM_VECTOR
    ][CCCP_CODEGEMM_CODES];
    const __nv_bfloat16* input_row =
        input + (long)position * input_stride;
    const int code = threadIdx.x;
    #pragma unroll
    for (
        int vector = 0;
        vector < CCCP_CODEGEMM_K_TILE / CCCP_CODEGEMM_VECTOR;
        ++vector
    ) {
        lut[vector][code] = cccp_codegemm_dot4(
            codebook + code * CCCP_CODEGEMM_VECTOR,
            input_row + input_tile + vector * CCCP_CODEGEMM_VECTOR);
    }
    __syncthreads();

    const int packed_tile = input_tile / 16;
    const int output_begin =
        blockIdx.x * CCCP_CODEGEMM_M_TILE + threadIdx.x * 2;
    const int output_end =
        min(
            (int)((blockIdx.x + 1) * CCCP_CODEGEMM_M_TILE),
            output_dim);
    for (
        int row = output_begin;
        row < output_end;
        row += CCCP_CODEGEMM_THREADS * 2
    ) {
        float value0 = 0.f;
        float value1 = 0.f;
        #pragma unroll
        for (
            int vector = 0;
            vector < CCCP_CODEGEMM_K_TILE / CCCP_CODEGEMM_VECTOR;
            ++vector
        ) {
            const int packed_offset = packed_tile + vector / 4;
            const int shift = (vector & 3) * 8;
            const uint32_t word0 =
                packed[(long)packed_offset * output_dim + row];
            const uint32_t word1 =
                packed[(long)packed_offset * output_dim + row + 1];
            value0 += lut[vector][(word0 >> shift) & 255u];
            value1 += lut[vector][(word1 >> shift) & 255u];
        }
        float* destination =
            output + (long)position * output_dim + row;
#if __CUDA_ARCH__ >= 900
        atomicAdd(
            reinterpret_cast<float2*>(destination),
            make_float2(value0, value1));
#else
        atomicAdd(destination, value0);
        atomicAdd(destination + 1, value1);
#endif
    }
}

__global__ void cccp_codegemm_swiglu_kernel(
    const float* __restrict__ gate_up,
    __nv_bfloat16* __restrict__ activation,
    const int top_k,
    const int intermediate)
{
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= top_k * intermediate) return;
    const int position = item / intermediate;
    const int column = item - position * intermediate;
    const long base = (long)position * 2 * intermediate;
    const float gate = __bfloat162float(
        __float2bfloat16_rn(gate_up[base + column]));
    const float up = __bfloat162float(
        __float2bfloat16_rn(
            gate_up[base + intermediate + column]));
    const float silu = gate / (1.f + expf(-gate));
    activation[item] = __float2bfloat16_rn(silu * up);
}

__global__ void cccp_codegemm_weighted_kernel(
    const float* __restrict__ rows,
    const float* __restrict__ weights,
    float* __restrict__ result,
    const int top_k,
    const int hidden)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= hidden) return;
    float value = 0.f;
    #pragma unroll
    for (int position = 0; position < 8; ++position) {
        if (position < top_k) {
            const float row_value = __bfloat162float(
                __float2bfloat16_rn(
                    rows[(long)position * hidden + column]));
            value = fmaf(row_value, weights[position], value);
        }
    }
    result[column] = value;
}

torch::Tensor moe_mlp_routed_codegemm(
    torch::Tensor input,
    torch::Tensor route_ids,
    torch::Tensor weights,
    torch::Tensor metadata,
    torch::Tensor gu_sum,
    torch::Tensor activation,
    torch::Tensor dn_sum,
    torch::Tensor result)
{
    TORCH_CHECK(
        input.is_cuda() && input.scalar_type() == at::kBFloat16 &&
        input.is_contiguous() && input.dim() == 2 && input.size(0) == 1,
        "CodeGEMM input must be contiguous CUDA BF16 [1,D]");
    TORCH_CHECK(
        route_ids.is_cuda() && route_ids.scalar_type() == at::kLong &&
        route_ids.is_contiguous() && route_ids.dim() == 1,
        "CodeGEMM route IDs must be contiguous CUDA int64 [K]");
    const int top_k = (int)route_ids.numel();
    TORCH_CHECK(top_k > 0 && top_k <= 8, "CodeGEMM Top-K must be in [1,8]");
    TORCH_CHECK(
        weights.is_cuda() && weights.scalar_type() == at::kFloat &&
        weights.is_contiguous() && weights.sizes() == route_ids.sizes(),
        "CodeGEMM weights must be contiguous float32 [K]");
    TORCH_CHECK(
        metadata.is_cuda() && metadata.scalar_type() == at::kLong &&
        metadata.is_contiguous() && metadata.dim() == 2 &&
        metadata.size(0) == 10,
        "CodeGEMM metadata must be contiguous int64 [10,E]");
    TORCH_CHECK(
        input.get_device() == route_ids.get_device() &&
        input.get_device() == weights.get_device() &&
        input.get_device() == metadata.get_device() &&
        input.get_device() == gu_sum.get_device() &&
        input.get_device() == activation.get_device() &&
        input.get_device() == dn_sum.get_device(),
        "CodeGEMM compute tensors must share one device");
    TORCH_CHECK(
        result.is_cuda() && result.scalar_type() == at::kFloat,
        "CodeGEMM result must be CUDA float32");

    const int hidden = (int)input.size(1);
    TORCH_CHECK(
        gu_sum.scalar_type() == at::kFloat &&
        gu_sum.is_contiguous() && gu_sum.dim() == 2 &&
        gu_sum.size(0) == top_k && gu_sum.size(1) % 2 == 0,
        "CodeGEMM GU workspace must be float32 [K,2I]");
    const int intermediate = (int)(gu_sum.size(1) / 2);
    TORCH_CHECK(
        activation.scalar_type() == at::kBFloat16 &&
        activation.is_contiguous() &&
        activation.sizes() ==
            torch::IntArrayRef({top_k, intermediate}),
        "CodeGEMM activation workspace must be BF16 [K,I]");
    TORCH_CHECK(
        dn_sum.scalar_type() == at::kFloat &&
        dn_sum.is_contiguous() &&
        dn_sum.sizes() == torch::IntArrayRef({top_k, hidden}),
        "CodeGEMM DN workspace must be float32 [K,D]");
    TORCH_CHECK(
        result.is_contiguous() && result.dim() == 1 &&
        result.numel() == hidden,
        "CodeGEMM result must be contiguous float32 [D]");
    TORCH_CHECK(
        hidden % CCCP_CODEGEMM_K_TILE == 0 &&
        intermediate % CCCP_CODEGEMM_K_TILE == 0,
        "CodeGEMM reduction dimensions must be divisible by 32");

    int current = -1;
    const auto status = CCCP_GPU_GET_DEVICE(&current);
    TORCH_CHECK(
        status == CCCP_GPU_SUCCESS && current == input.get_device(),
        "CodeGEMM kernel must run on its input CUDA device");
    ensure_peer_access(current, result.get_device(), "CodeGEMM direct return");

    const int expert_count = (int)metadata.size(1);
    auto stream = CCCP_GPU_CURRENT_STREAM();
    CCCP_GPU_MEMSET_ASYNC(
        gu_sum.data_ptr<float>(),
        0,
        gu_sum.numel() * sizeof(float),
        stream);
    cccp_psumbook_routed_kernel<<<
        dim3(
            (2 * intermediate + CCCP_CODEGEMM_M_TILE - 1) /
                CCCP_CODEGEMM_M_TILE,
            hidden / CCCP_CODEGEMM_K_TILE,
            top_k),
        CCCP_CODEGEMM_THREADS,
        0,
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                input.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            metadata.data_ptr<int64_t>(),
            gu_sum.data_ptr<float>(),
            top_k,
            expert_count,
            0,
            2 * intermediate,
            hidden,
            0);
    cccp_codegemm_swiglu_kernel<<<
        (top_k * intermediate + 255) / 256,
        256,
        0,
        stream>>>(
            gu_sum.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(
                activation.data_ptr<at::BFloat16>()),
            top_k,
            intermediate);
    CCCP_GPU_MEMSET_ASYNC(
        dn_sum.data_ptr<float>(),
        0,
        dn_sum.numel() * sizeof(float),
        stream);
    cccp_psumbook_routed_kernel<<<
        dim3(
            (hidden + CCCP_CODEGEMM_M_TILE - 1) /
                CCCP_CODEGEMM_M_TILE,
            intermediate / CCCP_CODEGEMM_K_TILE,
            top_k),
        CCCP_CODEGEMM_THREADS,
        0,
        stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                activation.data_ptr<at::BFloat16>()),
            route_ids.data_ptr<int64_t>(),
            metadata.data_ptr<int64_t>(),
            dn_sum.data_ptr<float>(),
            top_k,
            expert_count,
            5,
            hidden,
            intermediate,
            intermediate);
    cccp_codegemm_weighted_kernel<<<
        (hidden + 255) / 256,
        256,
        0,
        stream>>>(
            dn_sum.data_ptr<float>(),
            weights.data_ptr<float>(),
            result.data_ptr<float>(),
            top_k,
            hidden);
    CCCP_GPU_KERNEL_LAUNCH_CHECK();
    return result;
}
