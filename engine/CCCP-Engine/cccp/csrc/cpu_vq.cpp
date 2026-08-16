#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__linux__)
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

double moe_phase_seconds[4] = {0.0, 0.0, 0.0, 0.0};
int64_t moe_phase_calls = 0;
double packed_moe_phase_seconds[6] = {
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
int64_t packed_moe_phase_calls = 0;
double three_projection_phase_seconds[5] = {
    0.0, 0.0, 0.0, 0.0, 0.0};
int64_t three_projection_phase_calls = 0;
double resident_moe_phase_seconds[3] = {0.0, 0.0, 0.0};
int64_t resident_moe_phase_calls = 0;
int64_t resident_moe_selected_experts = 0;
int64_t resident_moe_q4_selected_experts = 0;
double latent_moe_phase_seconds[4] = {0.0, 0.0, 0.0, 0.0};
int64_t latent_moe_phase_calls = 0;
double resident_projection_seconds = 0.0;
int64_t resident_projection_calls = 0;
double block_fp8_gemv_seconds = 0.0;
int64_t block_fp8_gemv_calls = 0;
int64_t block_fp8_gemv_weight_elements = 0;
int64_t block_fp8_block_major_calls = 0;
int64_t block_fp8_block_major_bytes = 0;
int64_t block_fp8_numa_bound_tasks = 0;
int64_t block_fp8_rows8_tasks = 0;
int64_t block_fp8_gemm_calls = 0;
int64_t block_fp8_gemm_tokens = 0;

inline double wall_seconds() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

inline bool packed_single_team_enabled() {
  const char* value = std::getenv("CCCP_CPU_PACKED_SINGLE_TEAM");
  return value == nullptr ||
      (value[0] != '\0' && value[0] != '0');
}

inline bool packed_rows16_enabled() {
  const char* value = std::getenv("CCCP_CPU_PACKED_ROWS16");
  return value == nullptr ||
      (value[0] != '\0' && value[0] != '0');
}

inline bool block_fp8_rows8_enabled() {
  const char* value = std::getenv("CCCP_CPU_BLOCK_FP8_ROWS8");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

inline bool packed_direct_rows8_enabled() {
  const char* value = std::getenv("CCCP_CPU_PACKED_DIRECT_ROWS8");
  return value == nullptr ||
      (value[0] != '\0' && value[0] != '0');
}

inline bool packed_fused_gate_up_enabled() {
  const char* value = std::getenv("CCCP_CPU_PACKED_FUSED_GATE_UP");
  return value == nullptr ||
      (value[0] != '\0' && value[0] != '0');
}

inline bool packed_fused_down_reduce_enabled() {
  const char* value = std::getenv("CCCP_CPU_PACKED_FUSED_DOWN_REDUCE");
  return value == nullptr ||
      (value[0] != '\0' && value[0] != '0');
}

inline int64_t packed_l2_task_tiles() {
  const char* value = std::getenv("CCCP_CPU_L2_TASK_TILES");
  if (value == nullptr || value[0] == '\0') {
    return 4;
  }
  return std::max<int64_t>(1, std::min<int64_t>(32, std::atoll(value)));
}

inline bool q4_numa_local_enabled() {
#if !defined(__linux__) || !defined(_OPENMP)
  return false;
#else
  const char* mode = std::getenv("CCCP_CPU_NUMA");
  if (mode == nullptr) {
    return false;
  }
  const std::string value(mode);
  if (value == "local" || value == "shard" || value == "sharded") {
    return true;
  }
  const char* compile = std::getenv("CCCP_CPU_COMPILE");
  return value == "auto" && compile != nullptr &&
      std::string(compile) == "q4";
#endif
}

inline int current_numa_node() {
#if defined(__linux__) && defined(SYS_getcpu)
  unsigned cpu = 0;
  unsigned node = 0;
  if (syscall(SYS_getcpu, &cpu, &node, nullptr) == 0) {
    return static_cast<int>(node);
  }
#endif
  return 0;
}

inline bool bind_to_numa_node(
    void* address, int64_t bytes, unsigned node, bool move_pages = false) {
#if defined(__linux__) && defined(SYS_mbind)
  if (node >= sizeof(unsigned long) * 8) {
    return false;
  }
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return false;
  }
  const uintptr_t first = reinterpret_cast<uintptr_t>(address);
  const uintptr_t aligned_first =
      (first + static_cast<uintptr_t>(page_size) - 1) &
      ~(static_cast<uintptr_t>(page_size) - 1);
  const uintptr_t last = first + static_cast<uintptr_t>(bytes);
  const uintptr_t aligned_last =
      last & ~(static_cast<uintptr_t>(page_size) - 1);
  if (aligned_last <= aligned_first) {
    return false;
  }
  unsigned long mask = 1UL << node;
  return syscall(
      SYS_mbind,
      reinterpret_cast<void*>(aligned_first),
      aligned_last - aligned_first,
      MPOL_BIND,
      &mask,
      sizeof(mask) * 8,
      move_pages ? MPOL_MF_MOVE : 0) == 0;
#else
  (void)address;
  (void)bytes;
  (void)node;
  (void)move_pages;
  return false;
#endif
}

inline bool bind_to_current_numa_node(void* address, int64_t bytes) {
  return bind_to_numa_node(
      address, bytes, static_cast<unsigned>(current_numa_node()));
}

inline void bind_q4_row_shards(
    torch::Tensor& output, int64_t rows, int64_t row_bytes) {
  if (!q4_numa_local_enabled() || rows < 2 || row_bytes <= 0) {
    return;
  }
  const int64_t split = (rows + 1) / 2;
  auto* base = output.data_ptr<uint8_t>();
  bind_to_numa_node(base, split * row_bytes, 0);
  bind_to_numa_node(
      base + split * row_bytes, (rows - split) * row_bytes, 1);
}

// Return the contiguous output-row interval owned by this OpenMP thread.
// With OMP_PLACES=cores and the documented 96-thread dual-socket launch,
// threads 0..47 live on node0 and 48..95 on node1. Each socket partitions
// only its local half of the rows, keeping first-touch and decode reads local.
inline std::pair<int64_t, int64_t> q4_numa_local_row_range(int64_t rows) {
#if defined(_OPENMP)
  const int team = omp_get_num_threads();
  const int tid = omp_get_thread_num();
  if (!q4_numa_local_enabled() || team < 2) {
    return {rows * tid / team, rows * (tid + 1) / team};
  }
  const int node0_threads = (team + 1) / 2;
  const bool node1 = tid >= node0_threads;
  const int local_threads = node1 ? team - node0_threads : node0_threads;
  const int local_tid = node1 ? tid - node0_threads : tid;
  const int64_t split = (rows + 1) / 2;
  const int64_t node_begin = node1 ? split : 0;
  const int64_t node_rows = node1 ? rows - split : split;
  return {
      node_begin + node_rows * local_tid / local_threads,
      node_begin + node_rows * (local_tid + 1) / local_threads};
#else
  return {0, rows};
#endif
}

inline bool q4_numa_thread_is_node1() {
#if defined(_OPENMP)
  return omp_get_thread_num() >= (omp_get_num_threads() + 1) / 2;
#else
  return false;
#endif
}

// Partition an already node-local logical range across only the threads of
// this socket.  This is used to flatten Q/K/V or Gate/Up groups per socket,
// retaining all 48-way parallelism even when one individual projection has
// fewer than 96 output rows.
inline std::pair<int64_t, int64_t> q4_numa_local_item_range(int64_t items) {
#if defined(_OPENMP)
  const int team = omp_get_num_threads();
  const int tid = omp_get_thread_num();
  const int node0_threads = (team + 1) / 2;
  const bool node1 = tid >= node0_threads;
  const int local_threads = node1 ? team - node0_threads : node0_threads;
  const int local_tid = node1 ? tid - node0_threads : tid;
  return {
      items * local_tid / local_threads,
      items * (local_tid + 1) / local_threads};
#else
  return {0, items};
#endif
}

inline uint64_t load_u40_le(const uint8_t* source) {
  // p10 stores four indices in five bytes.  Two naturally unaligned scalar
  // loads are substantially cheaper than a five-iteration byte loop, while
  // memcpy keeps the access defined and lets x86 emit one 32-bit load.
  uint32_t low = 0;
  std::memcpy(&low, source, sizeof(low));
  return static_cast<uint64_t>(low) |
      (static_cast<uint64_t>(source[4]) << 32);
}

inline uint64_t load_u56_le(const uint8_t* source) {
  // p14 stores four indices in seven bytes.  Never read an eighth byte: the
  // final group can end exactly at the packed payload boundary.
  uint32_t low = 0;
  uint16_t middle = 0;
  std::memcpy(&low, source, sizeof(low));
  std::memcpy(&middle, source + 4, sizeof(middle));
  return static_cast<uint64_t>(low) |
      (static_cast<uint64_t>(middle) << 32) |
      (static_cast<uint64_t>(source[6]) << 48);
}

const float* e4m3fn_table() {
  alignas(64) static float values[256];
  static std::once_flag initialized;
  std::call_once(initialized, []() {
    for (int raw = 0; raw < 256; ++raw) {
      const int magnitude = raw & 0x7f;
      const int exponent = (magnitude >> 3) & 0x0f;
      const int mantissa = magnitude & 0x07;
      float value;
      if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa), -9);
      } else if (exponent < 15) {
        value = std::ldexp(
            1.0f + static_cast<float>(mantissa) / 8.0f,
            exponent - 7);
      } else if (mantissa < 7) {
        value = std::ldexp(
            1.0f + static_cast<float>(mantissa) / 8.0f,
            8);
      } else {
        value = std::numeric_limits<float>::quiet_NaN();
      }
      values[raw] = (raw & 0x80) ? -value : value;
    }
  });
  return values;
}

#if defined(__AVX2__)
// Raptor Lake and many low-end client CPUs expose AVX2/AVX-VNNI but not
// AVX-512.  Keep the common 8-lane conversions explicit: MSVC otherwise
// scalarizes the float accumulation loops used by attention source rows.
// Dense/BFloat16 and E4M3 projection loops deliberately stay on the
// compiler-generated path: their explicit AVX2 variants were slower on the
// supported client CPUs in the isolated release benchmark.
inline float horizontal_sum_f32x8(__m256 value) {
  __m128 sum = _mm_add_ps(
      _mm256_castps256_ps128(value),
      _mm256_extractf128_ps(value, 1));
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}
#endif

#if defined(__AVX512BF16__) && defined(__AVX512BW__)
inline __m512i decode_e4m3fn_bf16x32(const uint8_t* source) {
  // Every finite E4M3FN value is exactly representable as BF16.  Decode 32
  // bytes in registers; only the eight exponent-zero magnitudes need a tiny
  // lane permutation table.  No dequantized weight storage is produced.
  alignas(64) static const uint16_t subnormal_magnitude[32] = {
      0x0000, 0x3b00, 0x3b80, 0x3bc0,
      0x3c00, 0x3c20, 0x3c40, 0x3c60,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
  };
  const __m512i raw = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(source)));
  const __m512i sign = _mm512_slli_epi16(
      _mm512_and_si512(raw, _mm512_set1_epi16(0x80)), 8);
  const __m512i exponent = _mm512_and_si512(
      _mm512_srli_epi16(raw, 3), _mm512_set1_epi16(0x0f));
  const __m512i mantissa = _mm512_and_si512(
      raw, _mm512_set1_epi16(0x07));
  const __m512i normal = _mm512_or_si512(
      sign,
      _mm512_or_si512(
          _mm512_slli_epi16(
              _mm512_add_epi16(exponent, _mm512_set1_epi16(120)),
              7),
          _mm512_slli_epi16(mantissa, 4)));
  const __m512i subnormal = _mm512_or_si512(
      sign,
      _mm512_permutexvar_epi16(
          mantissa,
          _mm512_load_si512(
              reinterpret_cast<const __m512i*>(subnormal_magnitude))));
  const __mmask32 subnormal_mask = _mm512_cmpeq_epi16_mask(
      exponent, _mm512_setzero_si512());
  __m512i decoded = _mm512_mask_blend_epi16(
      subnormal_mask, normal, subnormal);
  const __mmask32 nan_mask =
      _mm512_cmpeq_epi16_mask(
          exponent, _mm512_set1_epi16(15)) &
      _mm512_cmpeq_epi16_mask(
          mantissa, _mm512_set1_epi16(7));
  const __m512i nan = _mm512_or_si512(
      sign, _mm512_set1_epi16(0x7fc0));
  return _mm512_mask_mov_epi16(decoded, nan_mask, nan);
}
#endif

inline float block_fp8_row_dot(
    const float* input_f,
    const at::BFloat16* input_b,
    bool input_bf16,
    const uint8_t* weight_row,
    const float* scale_row,
    const float* lut,
    int64_t cols,
    int64_t block_size) {
  const int64_t col_blocks = (cols + block_size - 1) / block_size;
  float total = 0.0f;
  for (int64_t col_block = 0; col_block < col_blocks; ++col_block) {
    const int64_t start = col_block * block_size;
    const int64_t stop = std::min(start + block_size, cols);
    float partial = 0.0f;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    int64_t col = start;
#if defined(__AVX512BF16__)
    if (input_bf16) {
      __m512 accumulated = _mm512_setzero_ps();
      for (; col + 32 <= stop; col += 32) {
        const __m512i activation = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(input_b + col));
        const __m512i decoded = decode_e4m3fn_bf16x32(
            weight_row + col);
        accumulated = _mm512_dpbf16_ps(
            accumulated,
            (__m512bh)activation,
            (__m512bh)decoded);
      }
      partial = _mm512_reduce_add_ps(accumulated);
    } else
#endif
    {
      __m512 accumulated = _mm512_setzero_ps();
      for (; col + 16 <= stop; col += 16) {
        const __m128i packed = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(weight_row + col));
        const __m512i raw = _mm512_cvtepu8_epi32(packed);
        // E4M3 has only 256 values.  The 1 KiB table remains resident in L1;
        // one indexed load replaces exponent/mantissa reconstruction while
        // preserving the exact scalar fallback, including subnormals/NaN.
        const __m512 decoded = _mm512_i32gather_ps(raw, lut, 4);
        const __m512 activation = _mm512_loadu_ps(input_f + col);
        accumulated = _mm512_fmadd_ps(
            activation, decoded, accumulated);
      }
      partial = _mm512_reduce_add_ps(accumulated);
    }
    for (; col < stop; ++col) {
      const float activation = input_bf16
          ? static_cast<float>(input_b[col])
          : input_f[col];
      partial += activation * lut[weight_row[col]];
    }
#else
    for (int64_t col = start; col < stop; ++col) {
      const float activation = input_bf16
          ? static_cast<float>(input_b[col])
          : input_f[col];
      partial += activation * lut[weight_row[col]];
    }
#endif
    total += partial * scale_row[col_block];
  }
  return total;
}

inline float bf16_dense_row_dot(
    const at::BFloat16* input,
    const at::BFloat16* weight,
    int64_t cols) {
  float total = 0.0f;
  int64_t col = 0;
#if defined(__AVX512BF16__)
  __m512 accumulated = _mm512_setzero_ps();
  for (; col + 32 <= cols; col += 32) {
    const auto activation = (__m512bh)_mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(input + col));
    const auto values = (__m512bh)_mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(weight + col));
    accumulated = _mm512_dpbf16_ps(
        accumulated, activation, values);
  }
  total = _mm512_reduce_add_ps(accumulated);
#endif
  for (; col < cols; ++col) {
    total += static_cast<float>(input[col]) *
        static_cast<float>(weight[col]);
  }
  return total;
}

inline float f32_dense_row_dot(
    const float* input,
    const float* weight,
    int64_t cols) {
  int64_t column = 0;
  float total = 0.0f;
#if defined(__AVX512F__)
  __m512 accumulated = _mm512_setzero_ps();
  for (; column + 16 <= cols; column += 16) {
    accumulated = _mm512_fmadd_ps(
        _mm512_loadu_ps(input + column),
        _mm512_loadu_ps(weight + column),
        accumulated);
  }
  total = _mm512_reduce_add_ps(accumulated);
#endif
  for (; column < cols; ++column) {
    total += input[column] * weight[column];
  }
  return total;
}

// Logical-row view of both public compact block-FP8 CPU layouts.  The
// block-major representation remains one byte per weight; only the address
// calculation changes.  This helper is intentionally shape/format driven so
// fused MoE scheduling can reuse the same primitive for any model family.
inline float block_fp8_logical_row_dot_bf16(
    const at::BFloat16* input,
    const torch::Tensor& weights,
    const torch::Tensor& scales,
    int64_t row,
    int64_t rows,
    int64_t cols,
    int64_t block_size) {
  TORCH_INTERNAL_ASSERT(
      weights.scalar_type() == at::kByte && scales.scalar_type() == at::kFloat);
  const int64_t col_blocks = (cols + block_size - 1) / block_size;
  const float* scale_row =
      scales.data_ptr<float>() + (row / block_size) * col_blocks;
  const float* lut = e4m3fn_table();
  if (weights.dim() == 2) {
    return block_fp8_row_dot(
        nullptr,
        input,
        true,
        weights.data_ptr<uint8_t>() + row * cols,
        scale_row,
        lut,
        cols,
        block_size);
  }
  TORCH_INTERNAL_ASSERT(weights.dim() == 5);
  constexpr int64_t chunk_rows = 32;
  constexpr int64_t row_chunks = 4;
  const int64_t row_block = row / block_size;
  const int64_t local_row = row % block_size;
  const int64_t row_chunk = local_row / chunk_rows;
  const int64_t row_in_chunk = local_row % chunk_rows;
  const uint8_t* base = weights.data_ptr<uint8_t>();
  float total = 0.0f;
  for (int64_t col_block = 0; col_block < col_blocks; ++col_block) {
    const int64_t start = col_block * block_size;
    const int64_t count = std::min(block_size, cols - start);
    const int64_t offset =
        ((((row_block * row_chunks + row_chunk) * col_blocks + col_block) *
           chunk_rows + row_in_chunk) * block_size);
    total += block_fp8_row_dot(
        nullptr,
        input + start,
        true,
        base + offset,
        scale_row + col_block,
        lut,
        count,
        block_size);
  }
  (void)rows;
  return total;
}

torch::Tensor bf16_grouped_gemv_cpu(
    torch::Tensor value,
    torch::Tensor weight_ptrs,
    torch::Tensor row_offsets,
    int64_t total_rows,
    int64_t cols,
    torch::Tensor output) {
  TORCH_CHECK(
      !value.is_cuda() && value.dim() == 2 && value.size(0) == 1 &&
          value.size(1) == cols &&
          (value.scalar_type() == at::kFloat ||
           value.scalar_type() == at::kBFloat16),
      "BF16 grouped GEMV requires CPU FP32/BF16 [1,cols] input");
  TORCH_CHECK(
      !weight_ptrs.is_cuda() && weight_ptrs.scalar_type() == at::kLong &&
          weight_ptrs.dim() == 1 && weight_ptrs.is_contiguous(),
      "BF16 grouped GEMV weight pointers must be CPU int64");
  TORCH_CHECK(
      !row_offsets.is_cuda() && row_offsets.scalar_type() == at::kInt &&
          row_offsets.dim() == 1 && row_offsets.is_contiguous() &&
          row_offsets.numel() == weight_ptrs.numel() + 1,
      "BF16 grouped GEMV row offsets must be CPU int32");
  TORCH_CHECK(
      !output.is_cuda() && output.scalar_type() == at::kBFloat16 &&
          output.is_contiguous() && output.numel() >= total_rows,
      "BF16 grouped GEMV output must be contiguous CPU BF16");
  TORCH_CHECK(total_rows > 0 && cols > 0,
              "BF16 grouped GEMV dimensions must be positive");
  auto input = value.scalar_type() == at::kBFloat16
      ? value.contiguous()
      : value.to(at::kBFloat16).contiguous();
  const at::BFloat16* xp = input.data_ptr<at::BFloat16>();
  const int64_t* pointers = weight_ptrs.data_ptr<int64_t>();
  const int32_t* offsets = row_offsets.data_ptr<int32_t>();
  const int64_t groups = weight_ptrs.numel();
  TORCH_CHECK(offsets[0] == 0 && offsets[groups] == total_rows,
              "BF16 grouped GEMV offsets must cover total_rows");
  at::BFloat16* op = output.data_ptr<at::BFloat16>();
  at::parallel_for(0, total_rows, 1, [&](int64_t begin, int64_t end) {
    int64_t group = std::upper_bound(
        offsets, offsets + groups + 1,
        static_cast<int32_t>(begin)) - offsets - 1;
    group = std::max<int64_t>(0, std::min(group, groups - 1));
    for (int64_t row = begin; row < end; ++row) {
      while (group + 1 < groups && row >= offsets[group + 1]) {
        ++group;
      }
      const auto* weights = reinterpret_cast<const at::BFloat16*>(
          static_cast<uintptr_t>(pointers[group]));
      op[row] = at::BFloat16(bf16_dense_row_dot(
          xp,
          weights + (row - offsets[group]) * cols,
          cols));
    }
  });
  return output.reshape({1, -1}).narrow(1, 0, total_rows);
}

// Process-local linear decode format.  This deliberately matches the small
// Q4_0/Q8_0 blocks used by high-throughput CPU runtimes: weights are compiled
// once while loading, activations are quantized once per projection input, and
// every GEMV becomes a contiguous block dot instead of a random codebook
// gather.  The image is never written back to the model directory.
struct Q4Block32 {
  at::Half d;
  uint8_t qs[16];
};
struct Q8Block32 {
  at::Half d;
  int16_t sum;
  int8_t qs[32];
};
static_assert(sizeof(Q4Block32) == 18, "Q4 block layout changed");
static_assert(sizeof(Q8Block32) == 36, "Q8 block layout changed");

inline void quantize_q4_block32(const float* source, Q4Block32* target) {
  float maximum = 0.0f;
  float signed_maximum = 0.0f;
  for (int64_t column = 0; column < 32; ++column) {
    const float value = source[column];
    if (std::abs(value) > maximum) {
      maximum = std::abs(value);
      signed_maximum = value;
    }
  }
  const float scale = signed_maximum / -8.0f;
  const float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
  target->d = at::Half(scale);
  for (int64_t column = 0; column < 16; ++column) {
    const int first = std::max(
        0, std::min(15, static_cast<int>(source[column] * inverse + 8.5f)));
    const int second = std::max(
        0, std::min(15, static_cast<int>(source[column + 16] * inverse + 8.5f)));
    target->qs[column] = static_cast<uint8_t>(first | (second << 4));
  }
}

inline void quantize_q8_block32(const float* source, Q8Block32* target) {
  float maximum = 0.0f;
#if defined(__AVX2__)
  const __m256 sign = _mm256_set1_ps(-0.0f);
  __m256 max_values = _mm256_setzero_ps();
  for (int64_t column = 0; column < 32; column += 8) {
    max_values = _mm256_max_ps(
        max_values,
        _mm256_andnot_ps(sign, _mm256_loadu_ps(source + column)));
  }
  __m128 lanes = _mm_max_ps(
      _mm256_castps256_ps128(max_values),
      _mm256_extractf128_ps(max_values, 1));
  lanes = _mm_max_ps(lanes, _mm_movehl_ps(lanes, lanes));
  lanes = _mm_max_ss(lanes, _mm_movehdup_ps(lanes));
  maximum = _mm_cvtss_f32(lanes);
#else
  for (int64_t column = 0; column < 32; ++column) {
    maximum = std::max(maximum, std::abs(source[column]));
  }
#endif
  const float scale = maximum / 127.0f;
  const float inverse = maximum == 0.0f ? 0.0f : 127.0f / maximum;
  target->d = at::Half(scale);
  int32_t quantized_sum = 0;
  for (int64_t column = 0; column < 32; ++column) {
    const int value = static_cast<int>(std::nearbyint(source[column] * inverse));
    const int8_t quantized = static_cast<int8_t>(
        std::max(-127, std::min(127, value)));
    target->qs[column] = quantized;
    quantized_sum += quantized;
  }
  target->sum = static_cast<int16_t>(quantized_sum);
}

inline void quantize_q8_row(
    const float* source, int64_t columns, Q8Block32* target) {
  TORCH_INTERNAL_ASSERT(columns % 32 == 0);
  for (int64_t block = 0; block < columns / 32; ++block) {
    quantize_q8_block32(source + block * 32, target + block);
  }
}

#if defined(__AVX2__)
inline int32_t horizontal_sum_i32x8(__m256i value) {
  const __m128i low = _mm256_castsi256_si128(value);
  const __m128i high = _mm256_extracti128_si256(value, 1);
  __m128i sum = _mm_add_epi32(low, high);
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return _mm_cvtsi128_si32(sum);
}

#if defined(_MSC_VER)
inline bool cpu_supports_avx_vnni() {
  static const bool supported = []() {
    int registers[4] = {0, 0, 0, 0};
    __cpuidex(registers, 7, 1);
    return (registers[0] & (1 << 4)) != 0;
  }();
  return supported;
}
#endif
#endif

inline float q4_q8_block_dot(
    const Q4Block32& weight,
    const Q8Block32& activation) {
    int32_t dot = 0;
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    const __m128i packed = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(weight.qs));
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i low = _mm_and_si128(packed, mask);
    const __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
    __m256i unsigned_weights = _mm256_castsi128_si256(low);
    unsigned_weights = _mm256_inserti128_si256(unsigned_weights, high, 1);
    const __m256i values = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(activation.qs));
    const __m256i products = _mm256_dpbusd_epi32(
        _mm256_setzero_si256(), unsigned_weights, values);
    dot = horizontal_sum_i32x8(products) -
        8 * static_cast<int32_t>(activation.sum);
#elif defined(__AVX2__)
    const __m128i packed = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(weight.qs));
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i low = _mm_and_si128(packed, mask);
    const __m128i high = _mm_and_si128(
        _mm_srli_epi16(packed, 4), mask);
    __m256i unsigned_weights = _mm256_castsi128_si256(low);
    unsigned_weights = _mm256_inserti128_si256(
        unsigned_weights, high, 1);
    const __m256i values = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(activation.qs));
#if defined(_MSC_VER)
    if (cpu_supports_avx_vnni()) {
      const __m256i products = _mm256_dpbusd_avx_epi32(
          _mm256_setzero_si256(), unsigned_weights, values);
      dot = horizontal_sum_i32x8(products) -
          8 * static_cast<int32_t>(activation.sum);
    } else
#endif
    {
      const __m256i zero_point = _mm256_set1_epi8(8);
      const __m256i signed_weights = _mm256_sub_epi8(
          unsigned_weights, zero_point);
      const __m256i weight_low = _mm256_cvtepi8_epi16(
          _mm256_castsi256_si128(signed_weights));
      const __m256i weight_high = _mm256_cvtepi8_epi16(
          _mm256_extracti128_si256(signed_weights, 1));
      const __m256i value_low = _mm256_cvtepi8_epi16(
          _mm256_castsi256_si128(values));
      const __m256i value_high = _mm256_cvtepi8_epi16(
          _mm256_extracti128_si256(values, 1));
      dot = horizontal_sum_i32x8(_mm256_add_epi32(
          _mm256_madd_epi16(weight_low, value_low),
          _mm256_madd_epi16(weight_high, value_high)));
    }
#else
    for (int64_t column = 0; column < 16; ++column) {
      dot += (static_cast<int>(weight.qs[column] & 0x0f) - 8) *
          static_cast<int>(activation.qs[column]);
      dot += (static_cast<int>(weight.qs[column] >> 4) - 8) *
          static_cast<int>(activation.qs[column + 16]);
    }
#endif
    return static_cast<float>(dot) *
        static_cast<float>(weight.d) *
        static_cast<float>(activation.d);
}

constexpr int64_t kQ4BlockMajorRows = 8;

inline void q4_q8_block_major_rows8(
    const Q4Block32* weights,
    int64_t first_row,
    int64_t valid_rows,
    const Q8Block32* activation,
    int64_t blocks,
    float* output) {
  TORCH_INTERNAL_ASSERT(
      first_row % kQ4BlockMajorRows == 0 && valid_rows > 0 &&
      valid_rows <= kQ4BlockMajorRows);
  std::fill(output, output + valid_rows, 0.0f);
  const Q4Block32* tile = weights + first_row * blocks;
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
  // Sapphire Rapids and newer Linux servers expose native 256-bit VNNI.
  // One Q8 activation belongs to all eight output rows in this tile; load it
  // once, then evaluate the eight packed Q4 rows before advancing.  The old
  // generic path reloaded and unpacked that same activation eight times.
  if (valid_rows == kQ4BlockMajorRows) {
    __m256 accumulated = _mm256_setzero_ps();
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    alignas(32) int32_t dots[kQ4BlockMajorRows];
    alignas(32) float scales[kQ4BlockMajorRows];
    for (int64_t block = 0; block < blocks; ++block) {
      const Q4Block32* block_rows = tile + block * kQ4BlockMajorRows;
      const Q8Block32& input = activation[block];
      const __m256i values = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(input.qs));
      const int32_t correction = 8 * static_cast<int32_t>(input.sum);
      const float input_scale = static_cast<float>(input.d);
      for (int64_t local = 0; local < kQ4BlockMajorRows; ++local) {
        const Q4Block32& weight = block_rows[local];
        const __m128i packed = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(weight.qs));
        const __m128i low = _mm_and_si128(packed, nibble_mask);
        const __m128i high = _mm_and_si128(
            _mm_srli_epi16(packed, 4), nibble_mask);
        __m256i unsigned_weights = _mm256_castsi128_si256(low);
        unsigned_weights = _mm256_inserti128_si256(
            unsigned_weights, high, 1);
        dots[local] = horizontal_sum_i32x8(_mm256_dpbusd_epi32(
            _mm256_setzero_si256(), unsigned_weights, values)) - correction;
        scales[local] = static_cast<float>(weight.d) * input_scale;
      }
      accumulated = _mm256_fmadd_ps(
          _mm256_cvtepi32_ps(_mm256_load_si256(
              reinterpret_cast<const __m256i*>(dots))),
          _mm256_load_ps(scales),
          accumulated);
    }
    _mm256_storeu_ps(output, accumulated);
    return;
  }
#elif defined(__AVX2__) && defined(_MSC_VER)
  // The release layout always presents complete eight-row tiles for the
  // aligned DSV4 projections. Hoist the common Q8 load/zero-point correction
  // out of the per-row dot and accumulate all eight scaled results together.
  // The generic helper used to reload the same 32-byte activation and branch
  // on AVX-VNNI eight times for every block.
  if (valid_rows == kQ4BlockMajorRows && cpu_supports_avx_vnni()) {
    __m256 accumulated = _mm256_setzero_ps();
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    alignas(32) int32_t dots[kQ4BlockMajorRows];
    alignas(32) float scales[kQ4BlockMajorRows];
    for (int64_t block = 0; block < blocks; ++block) {
      const Q4Block32* block_rows = tile + block * kQ4BlockMajorRows;
      const Q8Block32& input = activation[block];
      const __m256i values = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(input.qs));
      const int32_t correction = 8 * static_cast<int32_t>(input.sum);
      const float input_scale = static_cast<float>(input.d);
      for (int64_t local = 0; local < kQ4BlockMajorRows; ++local) {
        const Q4Block32& weight = block_rows[local];
        const __m128i packed = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(weight.qs));
        const __m128i low = _mm_and_si128(packed, nibble_mask);
        const __m128i high = _mm_and_si128(
            _mm_srli_epi16(packed, 4), nibble_mask);
        __m256i unsigned_weights = _mm256_castsi128_si256(low);
        unsigned_weights = _mm256_inserti128_si256(
            unsigned_weights, high, 1);
        dots[local] = horizontal_sum_i32x8(_mm256_dpbusd_avx_epi32(
            _mm256_setzero_si256(), unsigned_weights, values)) - correction;
        scales[local] = static_cast<float>(weight.d) * input_scale;
      }
      accumulated = _mm256_fmadd_ps(
          _mm256_cvtepi32_ps(_mm256_load_si256(
              reinterpret_cast<const __m256i*>(dots))),
          _mm256_load_ps(scales),
          accumulated);
    }
    _mm256_storeu_ps(output, accumulated);
    return;
  }
#endif
  for (int64_t block = 0; block < blocks; ++block) {
    const Q4Block32* block_rows = tile + block * valid_rows;
    for (int64_t local = 0; local < valid_rows; ++local) {
      output[local] += q4_q8_block_dot(block_rows[local], activation[block]);
    }
  }
}

inline float q4_q8_block_major_row_dot(
    const Q4Block32* weights,
    int64_t row,
    int64_t rows,
    const Q8Block32* activation,
    int64_t blocks) {
  const int64_t first_row = row & ~(kQ4BlockMajorRows - 1);
  const int64_t valid_rows = std::min<int64_t>(
      kQ4BlockMajorRows, rows - first_row);
  const int64_t local = row - first_row;
  const Q4Block32* tile = weights + first_row * blocks;
  float total = 0.0f;
  for (int64_t block = 0; block < blocks; ++block) {
    total += q4_q8_block_dot(
        tile[block * valid_rows + local], activation[block]);
  }
  return total;
}

// One fixed-address decode executor for a logical group containing native
// BF16 and/or compact block-FP8 projections.  The caller owns the model
// tensors; this object retains references and only allocates one BF16 input
// row plus token-sized outputs.  A single OpenMP team covers every logical
// output row, so mixed Attention/Dense input projections do not re-enter the
// worker pool once per weight format.
class CpuResidentProjectionLayer {
 public:
  CpuResidentProjectionLayer(
      std::vector<torch::Tensor> weights,
      std::vector<torch::Tensor> scales,
      std::vector<int64_t> rows,
      std::vector<int64_t> kinds,
      int64_t cols,
      int64_t block_size)
      : weights_(std::move(weights)),
        scales_(std::move(scales)),
        rows_(std::move(rows)),
        kinds_(std::move(kinds)),
        cols_(cols),
        block_size_(block_size) {
    const int64_t groups = weights_.size();
    TORCH_CHECK(groups > 0 && scales_.size() == groups &&
                    rows_.size() == groups && kinds_.size() == groups,
                "resident projection metadata lengths must match");
    TORCH_CHECK(cols_ > 0 && block_size_ > 0,
                "resident projection dimensions must be positive");
    row_offsets_.reserve(groups + 1);
    row_offsets_.push_back(0);
    auto cpu = torch::TensorOptions().device(torch::kCPU);
    for (int64_t group = 0; group < groups; ++group) {
      auto& weight = weights_[group];
      TORCH_CHECK(!weight.is_cuda() && weight.is_contiguous() &&
                      rows_[group] > 0,
                  "resident projection weights must be contiguous CPU tensors");
      if (kinds_[group] == 0) {
        all_q4_ = false;
        TORCH_CHECK(weight.scalar_type() == at::kBFloat16 &&
                        weight.dim() == 2 && weight.size(0) == rows_[group] &&
                        weight.size(1) == cols_,
                    "resident BF16 projection shape mismatch");
      } else if (kinds_[group] == 1) {
        all_q4_ = false;
        TORCH_CHECK(weight.scalar_type() == at::kByte &&
                        (weight.dim() == 2 || weight.dim() == 5),
                    "resident compact projection must be block-FP8");
        TORCH_CHECK(!scales_[group].is_cuda() &&
                        scales_[group].scalar_type() == at::kFloat &&
                        scales_[group].is_contiguous(),
                    "resident block-FP8 scales must be contiguous CPU FP32");
        if (weight.dim() == 2) {
          TORCH_CHECK(weight.size(0) == rows_[group] &&
                          weight.size(1) == cols_,
                      "resident row-major block-FP8 shape mismatch");
        }
      } else {
        TORCH_CHECK(
            kinds_[group] == 2 && weight.scalar_type() == at::kByte &&
                weight.dim() == 1 && cols_ % 32 == 0 &&
                weight.numel() == rows_[group] * (cols_ / 32) *
                    static_cast<int64_t>(sizeof(Q4Block32)),
            "resident Q4 block-dot projection shape mismatch");
        has_q4_ = true;
      }
      row_offsets_.push_back(row_offsets_.back() + rows_[group]);
      outputs_.push_back(torch::empty(
          {1, rows_[group]},
          cpu.dtype(kinds_[group] == 0 ? at::kBFloat16 : at::kFloat)));
    }
    input_bf16_ = torch::empty({1, cols_}, cpu.dtype(at::kBFloat16));
    grouped_input_bf16_ = torch::empty(
        {groups, cols_}, cpu.dtype(at::kBFloat16));
    if (has_q4_) {
      input_float_ = torch::empty({1, cols_}, cpu.dtype(at::kFloat));
      input_q8_ = torch::empty(
          {(cols_ / 32) * static_cast<int64_t>(sizeof(Q8Block32))},
          cpu.dtype(at::kByte));
      grouped_input_float_ = torch::empty(
          {groups, cols_}, cpu.dtype(at::kFloat));
      grouped_input_q8_ = torch::empty(
          {groups * (cols_ / 32) *
               static_cast<int64_t>(sizeof(Q8Block32))},
          cpu.dtype(at::kByte));
    }
    const int64_t total_rows = row_offsets_.back();
    combined_bf16_ = torch::empty(
        {1, total_rows}, cpu.dtype(at::kBFloat16));
    combined_float_ = torch::empty(
        {1, total_rows}, cpu.dtype(at::kFloat));
    row_groups_.resize(total_rows);
    local_rows_.resize(total_rows);
    for (int64_t group = 0; group < groups; ++group) {
      for (int64_t row = 0; row < rows_[group]; ++row) {
        const int64_t logical = row_offsets_[group] + row;
        row_groups_[logical] = group;
        local_rows_[logical] = row;
      }
      if (all_q4_) {
        const int64_t tiles =
            (rows_[group] + kQ4BlockMajorRows - 1) /
            kQ4BlockMajorRows;
        const int64_t split = (tiles + 1) / 2;
        for (int64_t tile = 0; tile < tiles; ++tile) {
          const int64_t first_row = tile * kQ4BlockMajorRows;
          q4_tile_groups_.push_back(group);
          q4_tile_first_rows_.push_back(first_row);
          const int node = tile >= split ? 1 : 0;
          q4_node_tile_groups_[node].push_back(group);
          q4_node_tile_first_rows_[node].push_back(first_row);
        }
      }
    }
  }

  std::vector<torch::Tensor> forward(torch::Tensor value) {
    run(value, torch::Tensor());
    return outputs_;
  }

  torch::Tensor forward_combined(
      torch::Tensor value,
      bool float_output) {
    auto output = float_output ? combined_float_ : combined_bf16_;
    run(value, output);
    return output;
  }

  torch::Tensor forward_grouped(
      torch::Tensor values,
      bool float_output) {
    std::lock_guard<std::mutex> guard(mutex_);
    const double started = wall_seconds();
    const int64_t groups = weights_.size();
    TORCH_CHECK(
        !values.is_cuda() && values.dim() == 2 &&
            values.size(0) == groups && values.size(1) == cols_ &&
            values.is_contiguous() &&
            (values.scalar_type() == at::kFloat ||
             values.scalar_type() == at::kBFloat16),
        "resident grouped projection requires contiguous CPU FP32/BF16 "
        "[groups,cols]");
    auto output = float_output ? combined_float_ : combined_bf16_;
    const bool source_is_float = values.scalar_type() == at::kFloat;
    const float* source_float = source_is_float
        ? values.data_ptr<float>() : nullptr;
    const auto* source_bf16 = source_is_float
        ? nullptr : values.data_ptr<at::BFloat16>();
    auto* input_bf16 = grouped_input_bf16_.data_ptr<at::BFloat16>();
    float* input_float = has_q4_
        ? grouped_input_float_.data_ptr<float>() : nullptr;
    auto* input_q8 = has_q4_
        ? reinterpret_cast<Q8Block32*>(grouped_input_q8_.data_ptr<uint8_t>())
        : nullptr;
    const int64_t q8_blocks = cols_ / 32;
    const int64_t total_rows = row_offsets_.back();
    auto evaluate_row = [&](int64_t logical) {
      const int64_t group = row_groups_[logical];
      const int64_t row = local_rows_[logical];
      const at::BFloat16* group_bf16 = input_bf16 + group * cols_;
      float result;
      if (kinds_[group] == 0) {
        result = bf16_dense_row_dot(
            group_bf16,
            weights_[group].data_ptr<at::BFloat16>() + row * cols_,
            cols_);
      } else if (kinds_[group] == 1) {
        result = block_fp8_logical_row_dot_bf16(
            group_bf16, weights_[group], scales_[group], row,
            rows_[group], cols_, block_size_);
      } else {
        const auto* weight = reinterpret_cast<const Q4Block32*>(
            weights_[group].data_ptr<uint8_t>());
        result = q4_q8_block_major_row_dot(
            weight, row, rows_[group],
            input_q8 + group * q8_blocks, q8_blocks);
      }
      if (float_output) {
        output.data_ptr<float>()[logical] = result;
      } else {
        output.data_ptr<at::BFloat16>()[logical] = at::BFloat16(result);
      }
    };
    auto evaluate_q4_tile = [&](int64_t group, int64_t first_row) {
      const int64_t valid_rows = std::min<int64_t>(
          kQ4BlockMajorRows, rows_[group] - first_row);
      alignas(64) float values[kQ4BlockMajorRows];
      q4_q8_block_major_rows8(
          reinterpret_cast<const Q4Block32*>(
              weights_[group].data_ptr<uint8_t>()),
          first_row, valid_rows, input_q8 + group * q8_blocks,
          q8_blocks, values);
      for (int64_t local = 0; local < valid_rows; ++local) {
        const int64_t logical =
            row_offsets_[group] + first_row + local;
        if (float_output) {
          output.data_ptr<float>()[logical] = values[local];
        } else {
          output.data_ptr<at::BFloat16>()[logical] =
              at::BFloat16(values[local]);
        }
      }
    };
#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (int64_t item = 0; item < groups * cols_; ++item) {
        const float scalar = source_is_float
            ? source_float[item] : static_cast<float>(source_bf16[item]);
        input_bf16[item] = at::BFloat16(scalar);
        if (has_q4_) {
          input_float[item] = scalar;
        }
      }
      if (has_q4_) {
#pragma omp for schedule(static)
        for (int64_t task = 0; task < groups * q8_blocks; ++task) {
          const int64_t group = task / q8_blocks;
          const int64_t block = task - group * q8_blocks;
          quantize_q8_block32(
              input_float + group * cols_ + block * 32,
              input_q8 + group * q8_blocks + block);
        }
      }
      if (all_q4_ && q4_numa_local_enabled()) {
        const int node = q4_numa_thread_is_node1() ? 1 : 0;
        const auto range = q4_numa_local_item_range(
            q4_node_tile_groups_[node].size());
        for (int64_t task = range.first; task < range.second; ++task) {
          evaluate_q4_tile(
              q4_node_tile_groups_[node][task],
              q4_node_tile_first_rows_[node][task]);
        }
      } else if (all_q4_) {
#pragma omp for schedule(static)
        for (int64_t task = 0;
             task < static_cast<int64_t>(q4_tile_groups_.size()); ++task) {
          evaluate_q4_tile(
              q4_tile_groups_[task], q4_tile_first_rows_[task]);
        }
      } else if (q4_numa_local_enabled() && has_q4_) {
        const bool node1 = q4_numa_thread_is_node1();
        int64_t local_total = 0;
        for (int64_t group = 0; group < groups; ++group) {
          const int64_t split = (rows_[group] + 1) / 2;
          local_total += node1 ? rows_[group] - split : split;
        }
        const auto flat = q4_numa_local_item_range(local_total);
        int64_t cursor = 0;
        for (int64_t group = 0; group < groups; ++group) {
          const int64_t split = (rows_[group] + 1) / 2;
          const int64_t row_base = node1 ? split : 0;
          const int64_t local_rows = node1 ? rows_[group] - split : split;
          const int64_t begin = std::max<int64_t>(flat.first, cursor);
          const int64_t end = std::min<int64_t>(flat.second, cursor + local_rows);
          for (int64_t item = begin; item < end; ++item) {
            const int64_t row = row_base + item - cursor;
            evaluate_row(row_offsets_[group] + row);
          }
          cursor += local_rows;
        }
      } else {
#pragma omp for schedule(static)
        for (int64_t logical = 0; logical < total_rows; ++logical) {
          evaluate_row(logical);
        }
      }
    }
    resident_projection_seconds += wall_seconds() - started;
    ++resident_projection_calls;
    return output;
  }

 private:
  void run(torch::Tensor value, torch::Tensor combined) {
    std::lock_guard<std::mutex> guard(mutex_);
    const double started = wall_seconds();
    TORCH_CHECK(!value.is_cuda() && value.dim() == 2 &&
                    value.size(0) == 1 && value.size(1) == cols_ &&
                    value.is_contiguous() &&
                    (value.scalar_type() == at::kFloat ||
                     value.scalar_type() == at::kBFloat16),
                "resident projection requires contiguous CPU FP32/BF16 [1,cols]");
    const bool copy_input = value.scalar_type() == at::kFloat;
    at::BFloat16* input_storage = input_bf16_.data_ptr<at::BFloat16>();
    const at::BFloat16* input = copy_input
        ? input_storage
        : value.data_ptr<at::BFloat16>();
    float* q4_input = has_q4_
        ? input_float_.data_ptr<float>()
        : nullptr;
    const float* source_float = copy_input ? value.data_ptr<float>() : nullptr;
    const auto* source_bf16 = copy_input
        ? nullptr
        : value.data_ptr<at::BFloat16>();
    auto* input_q8 = has_q4_
        ? reinterpret_cast<Q8Block32*>(input_q8_.data_ptr<uint8_t>())
        : nullptr;
    const int64_t total_rows = row_offsets_.back();
    auto evaluate_row = [&](int64_t logical) {
      const int64_t group = row_groups_[logical];
      const int64_t row = local_rows_[logical];
      float result;
      if (kinds_[group] == 0) {
        result = bf16_dense_row_dot(
            input,
            weights_[group].data_ptr<at::BFloat16>() + row * cols_,
            cols_);
      } else if (kinds_[group] == 1) {
        result = block_fp8_logical_row_dot_bf16(
            input, weights_[group], scales_[group], row,
            rows_[group], cols_, block_size_);
      } else {
        const int64_t blocks = cols_ / 32;
        const auto* weight = reinterpret_cast<const Q4Block32*>(
            weights_[group].data_ptr<uint8_t>());
        result = q4_q8_block_major_row_dot(
            weight, row, rows_[group], input_q8, blocks);
      }
      if (combined.defined()) {
        if (combined.scalar_type() == at::kFloat) {
          combined.data_ptr<float>()[logical] = result;
        } else {
          combined.data_ptr<at::BFloat16>()[logical] = at::BFloat16(result);
        }
      } else if (kinds_[group] == 0) {
        outputs_[group].data_ptr<at::BFloat16>()[row] = at::BFloat16(result);
      } else {
        outputs_[group].data_ptr<float>()[row] = result;
      }
    };
    auto evaluate_q4_tile = [&](int64_t group, int64_t first_row) {
      const int64_t blocks = cols_ / 32;
      const int64_t valid_rows = std::min<int64_t>(
          kQ4BlockMajorRows, rows_[group] - first_row);
      alignas(64) float values[kQ4BlockMajorRows];
      q4_q8_block_major_rows8(
          reinterpret_cast<const Q4Block32*>(
              weights_[group].data_ptr<uint8_t>()),
          first_row, valid_rows, input_q8, blocks, values);
      for (int64_t local = 0; local < valid_rows; ++local) {
        const int64_t row = first_row + local;
        if (combined.defined()) {
          const int64_t logical = row_offsets_[group] + row;
          if (combined.scalar_type() == at::kFloat) {
            combined.data_ptr<float>()[logical] = values[local];
          } else {
            combined.data_ptr<at::BFloat16>()[logical] =
                at::BFloat16(values[local]);
          }
        } else {
          outputs_[group].data_ptr<float>()[row] = values[local];
        }
      }
    };
#pragma omp parallel
    {
      if (copy_input || has_q4_) {
#pragma omp for schedule(static)
        for (int64_t column = 0; column < cols_; ++column) {
          const float scalar = copy_input
              ? source_float[column]
              : static_cast<float>(source_bf16[column]);
          if (copy_input) {
            input_storage[column] = at::BFloat16(scalar);
          }
          if (has_q4_) {
            q4_input[column] = scalar;
          }
        }
      }
      if (has_q4_) {
#pragma omp single
        { quantize_q8_row(q4_input, cols_, input_q8); }
      }
      if (all_q4_ && q4_numa_local_enabled()) {
        const int node = q4_numa_thread_is_node1() ? 1 : 0;
        const auto range = q4_numa_local_item_range(
            q4_node_tile_groups_[node].size());
        for (int64_t task = range.first; task < range.second; ++task) {
          evaluate_q4_tile(
              q4_node_tile_groups_[node][task],
              q4_node_tile_first_rows_[node][task]);
        }
      } else if (all_q4_) {
#pragma omp for schedule(static)
        for (int64_t task = 0;
             task < static_cast<int64_t>(q4_tile_groups_.size()); ++task) {
          evaluate_q4_tile(
              q4_tile_groups_[task], q4_tile_first_rows_[task]);
        }
      } else if (q4_numa_local_enabled() && has_q4_) {
        const bool node1 = q4_numa_thread_is_node1();
        int64_t local_total = 0;
        for (int64_t group = 0;
             group < static_cast<int64_t>(weights_.size()); ++group) {
          const int64_t split = (rows_[group] + 1) / 2;
          local_total += node1 ? rows_[group] - split : split;
        }
        const auto flat = q4_numa_local_item_range(local_total);
        int64_t cursor = 0;
        for (int64_t group = 0;
             group < static_cast<int64_t>(weights_.size()); ++group) {
          const int64_t split = (rows_[group] + 1) / 2;
          const int64_t row_base = node1 ? split : 0;
          const int64_t local_rows = node1 ? rows_[group] - split : split;
          const int64_t begin = std::max<int64_t>(flat.first, cursor);
          const int64_t end = std::min<int64_t>(flat.second, cursor + local_rows);
          for (int64_t item = begin; item < end; ++item) {
            const int64_t row = row_base + item - cursor;
            evaluate_row(row_offsets_[group] + row);
          }
          cursor += local_rows;
        }
      } else {
#pragma omp for schedule(static)
        for (int64_t logical = 0; logical < total_rows; ++logical) {
          evaluate_row(logical);
        }
      }
    }
    resident_projection_seconds += wall_seconds() - started;
    ++resident_projection_calls;
  }

  std::vector<torch::Tensor> weights_, scales_, outputs_;
  std::vector<int64_t> rows_, kinds_, row_offsets_;
  std::vector<int64_t> row_groups_, local_rows_;
  std::vector<int64_t> q4_tile_groups_, q4_tile_first_rows_;
  std::vector<int64_t> q4_node_tile_groups_[2];
  std::vector<int64_t> q4_node_tile_first_rows_[2];
  int64_t cols_ = 0;
  int64_t block_size_ = 128;
  torch::Tensor input_bf16_, input_float_, input_q8_;
  torch::Tensor grouped_input_bf16_, grouped_input_float_, grouped_input_q8_;
  torch::Tensor combined_bf16_, combined_float_;
  bool has_q4_ = false;
  bool all_q4_ = true;
  std::mutex mutex_;
};

inline void block_fp8_row_dot_many(
    const float* input_f,
    const at::BFloat16* input_b,
    bool input_bf16,
    int64_t tokens,
    const uint8_t* weight_row,
    int64_t weight_block_stride,
    const float* scale_row,
    const float* lut,
    int64_t cols,
    int64_t block_size,
    float* totals) {
  std::fill(totals, totals + tokens, 0.0f);
  const int64_t col_blocks = (cols + block_size - 1) / block_size;
  for (int64_t col_block = 0; col_block < col_blocks; ++col_block) {
    const int64_t start = col_block * block_size;
    const int64_t stop = std::min(start + block_size, cols);
    const uint8_t* weight_block =
        weight_row + col_block * weight_block_stride;
    alignas(64) float partial[16] = {0.0f};
    int64_t col = start;
#if defined(__AVX512F__) && defined(__AVX512BW__)
#if defined(__AVX512BF16__)
    if (input_bf16) {
      __m512 accumulated[16];
      for (int64_t token = 0; token < tokens; ++token) {
        accumulated[token] = _mm512_setzero_ps();
      }
      for (; col + 32 <= stop; col += 32) {
        const __m512i decoded = decode_e4m3fn_bf16x32(
            weight_block + col - start);
        for (int64_t token = 0; token < tokens; ++token) {
          const __m512i activation = _mm512_loadu_si512(
              reinterpret_cast<const __m512i*>(
                  input_b + token * cols + col));
          accumulated[token] = _mm512_dpbf16_ps(
              accumulated[token],
              (__m512bh)activation,
              (__m512bh)decoded);
        }
      }
      for (int64_t token = 0; token < tokens; ++token) {
        partial[token] = _mm512_reduce_add_ps(accumulated[token]);
      }
    } else
#endif
    {
      __m512 accumulated[16];
      for (int64_t token = 0; token < tokens; ++token) {
        accumulated[token] = _mm512_setzero_ps();
      }
      for (; col + 16 <= stop; col += 16) {
        const __m128i packed = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(
                weight_block + col - start));
        const __m512 decoded = _mm512_i32gather_ps(
            _mm512_cvtepu8_epi32(packed), lut, 4);
        for (int64_t token = 0; token < tokens; ++token) {
          accumulated[token] = _mm512_fmadd_ps(
              _mm512_loadu_ps(input_f + token * cols + col),
              decoded,
              accumulated[token]);
        }
      }
      for (int64_t token = 0; token < tokens; ++token) {
        partial[token] = _mm512_reduce_add_ps(accumulated[token]);
      }
    }
#endif
    for (; col < stop; ++col) {
      const float decoded = lut[weight_block[col - start]];
      for (int64_t token = 0; token < tokens; ++token) {
        const float activation = input_bf16
            ? static_cast<float>(input_b[token * cols + col])
            : input_f[token * cols + col];
        partial[token] += activation * decoded;
      }
    }
    const float scale = scale_row[col_block];
    for (int64_t token = 0; token < tokens; ++token) {
      totals[token] += partial[token] * scale;
    }
  }
}

// Decode layout used by the public CPU block-FP8 backend:
// [row_block, row_chunk32, col_block, row_in_chunk, col_in_block].
// A 32x128 chunk is exactly one 4 KiB page.  The same row-chunk task owns
// that page during packing and GEMV, which gives Linux first-touch/NUMA
// placement a stable unit while keeping the payload at one byte/weight.
torch::Tensor block_fp8_to_block_major_cpu(
    torch::Tensor weights,
    int64_t block_size) {
  TORCH_CHECK(
      !weights.is_cuda() && weights.scalar_type() == at::kByte &&
          weights.dim() == 2 && weights.is_contiguous(),
      "block-major packing requires contiguous CPU uint8 [rows,cols]");
  TORCH_CHECK(block_size == 128,
              "block-major packing currently requires block128");
  const int64_t rows = weights.size(0);
  const int64_t cols = weights.size(1);
  const int64_t row_blocks = (rows + block_size - 1) / block_size;
  const int64_t col_blocks = (cols + block_size - 1) / block_size;
  constexpr int64_t row_chunks = 4;
  constexpr int64_t chunk_rows = 32;
  auto packed = torch::empty(
      {row_blocks, row_chunks, col_blocks, chunk_rows, block_size},
      weights.options());
  const uint8_t* source = weights.data_ptr<uint8_t>();
  uint8_t* destination = packed.data_ptr<uint8_t>();
  const int64_t task_bytes = col_blocks * chunk_rows * block_size;
  const int64_t tasks = row_blocks * row_chunks;
  std::atomic<int64_t> numa_bound{0};
  at::parallel_for(0, tasks, 1, [&](int64_t begin, int64_t end) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t row_block = task / row_chunks;
      const int64_t row_chunk = task % row_chunks;
      const int64_t first_row =
          row_block * block_size + row_chunk * chunk_rows;
      uint8_t* task_output = destination + task * task_bytes;
      if (bind_to_current_numa_node(task_output, task_bytes)) {
        numa_bound.fetch_add(1, std::memory_order_relaxed);
      }
      for (int64_t col_block = 0; col_block < col_blocks; ++col_block) {
        const int64_t first_col = col_block * block_size;
        const int64_t valid_cols = std::min(block_size, cols - first_col);
        uint8_t* tile =
            task_output + col_block * chunk_rows * block_size;
        for (int64_t local_row = 0; local_row < chunk_rows; ++local_row) {
          const int64_t row = first_row + local_row;
          uint8_t* target = tile + local_row * block_size;
          if (row < rows && valid_cols > 0) {
            std::memcpy(
                target,
                source + row * cols + first_col,
                static_cast<size_t>(valid_cols));
          }
        }
      }
    }
  });
  ++block_fp8_block_major_calls;
  block_fp8_block_major_bytes += rows * cols;
  block_fp8_numa_bound_tasks += numa_bound.load(std::memory_order_relaxed);
  return packed;
}

torch::Tensor block_fp8_compile_q4_0_cpu(
    torch::Tensor weights,
    torch::Tensor scales,
    int64_t rows,
    int64_t cols,
    int64_t block_size) {
  TORCH_CHECK(
      !weights.is_cuda() && weights.scalar_type() == at::kByte &&
          weights.dim() == 2 && weights.is_contiguous() &&
          !scales.is_cuda() && scales.scalar_type() == at::kFloat &&
          scales.dim() == 2 && scales.is_contiguous(),
      "Q4 compilation requires row-major CPU block-FP8 tensors");
  TORCH_CHECK(
      rows == weights.size(0) && cols == weights.size(1) &&
          block_size == 128 && cols % 32 == 0,
      "Q4 compilation requires block128 and a column multiple of 32");
  const int64_t q4_blocks = cols / 32;
  auto output = torch::empty(
      {rows * q4_blocks * static_cast<int64_t>(sizeof(Q4Block32))},
      weights.options());
  bind_q4_row_shards(
      output, rows, q4_blocks * static_cast<int64_t>(sizeof(Q4Block32)));
  const uint8_t* source = weights.data_ptr<uint8_t>();
  const float* source_scales = scales.data_ptr<float>();
  const int64_t source_col_blocks = (cols + block_size - 1) / block_size;
  const float* lut = e4m3fn_table();
  auto* destination = reinterpret_cast<Q4Block32*>(
      output.data_ptr<uint8_t>());
#pragma omp parallel
  {
    const int64_t tiles =
        (rows + kQ4BlockMajorRows - 1) / kQ4BlockMajorRows;
    const auto range = q4_numa_local_row_range(tiles);
    alignas(64) float values[32];
    for (int64_t tile_index = range.first;
         tile_index < range.second; ++tile_index) {
      const int64_t first_row = tile_index * kQ4BlockMajorRows;
      const int64_t valid_rows = std::min<int64_t>(
          kQ4BlockMajorRows, rows - first_row);
      for (int64_t qblock = 0; qblock < q4_blocks; ++qblock) {
        const int64_t first_col = qblock * 32;
        for (int64_t local = 0; local < valid_rows; ++local) {
          const int64_t row = first_row + local;
          const float* scale_row = source_scales +
              (row / block_size) * source_col_blocks;
          for (int64_t lane = 0; lane < 32; ++lane) {
            const int64_t column = first_col + lane;
            values[lane] = lut[source[row * cols + column]] *
                scale_row[column / block_size];
          }
          quantize_q4_block32(
              values,
              destination + first_row * q4_blocks +
                  qblock * valid_rows + local);
        }
      }
    }
  }
  return output;
}

torch::Tensor q4_0_gemv_cpu(
    torch::Tensor value,
    torch::Tensor weights,
    int64_t rows,
    int64_t cols,
    torch::Tensor output) {
  TORCH_CHECK(
      !value.is_cuda() && value.dim() == 2 && value.size(0) == 1 &&
          value.size(1) == cols && value.scalar_type() == at::kFloat &&
          value.is_contiguous(),
      "Q4 GEMV requires contiguous CPU FP32 [1,cols] input");
  TORCH_CHECK(
      !weights.is_cuda() && weights.scalar_type() == at::kByte &&
          weights.dim() == 1 && weights.is_contiguous() && cols % 32 == 0 &&
          weights.numel() == rows * (cols / 32) *
              static_cast<int64_t>(sizeof(Q4Block32)),
      "Q4 GEMV weight image mismatch");
  TORCH_CHECK(
      !output.is_cuda() && output.scalar_type() == at::kFloat &&
          output.is_contiguous() && output.numel() >= rows,
      "Q4 GEMV output must be contiguous CPU FP32");
  const int64_t blocks = cols / 32;
  std::vector<Q8Block32> activation(blocks);
  quantize_q8_row(value.data_ptr<float>(), cols, activation.data());
  const auto* wp = reinterpret_cast<const Q4Block32*>(
      weights.data_ptr<uint8_t>());
  float* op = output.data_ptr<float>();
#pragma omp parallel
  {
    const int64_t tiles =
        (rows + kQ4BlockMajorRows - 1) / kQ4BlockMajorRows;
    const auto range = q4_numa_local_row_range(tiles);
    for (int64_t tile = range.first; tile < range.second; ++tile) {
      const int64_t first_row = tile * kQ4BlockMajorRows;
      const int64_t valid_rows = std::min<int64_t>(
          kQ4BlockMajorRows, rows - first_row);
      q4_q8_block_major_rows8(
          wp, first_row, valid_rows, activation.data(), blocks,
          op + first_row);
    }
  }
  return output.reshape({1, -1}).narrow(1, 0, rows);
}

torch::Tensor q4_0_gemm_cpu(
    torch::Tensor value,
    torch::Tensor weights,
    int64_t rows,
    int64_t cols,
    torch::Tensor output) {
  TORCH_CHECK(
      !value.is_cuda() && value.dim() == 2 && value.size(0) >= 2 &&
          value.size(0) <= 64 && value.size(1) == cols &&
          value.scalar_type() == at::kFloat && value.is_contiguous(),
      "Q4 GEMM requires contiguous CPU FP32 [2..64,cols] input");
  TORCH_CHECK(
      !weights.is_cuda() && weights.scalar_type() == at::kByte &&
          weights.dim() == 1 && weights.is_contiguous() && cols % 32 == 0 &&
          weights.numel() == rows * (cols / 32) *
              static_cast<int64_t>(sizeof(Q4Block32)),
      "Q4 GEMM weight image mismatch");
  const int64_t tokens = value.size(0);
  TORCH_CHECK(
      !output.is_cuda() && output.scalar_type() == at::kFloat &&
          output.is_contiguous() && output.sizes() ==
              torch::IntArrayRef({tokens, rows}),
      "Q4 GEMM output must be contiguous CPU FP32 [tokens,rows]");
  const int64_t blocks = cols / 32;
  std::vector<Q8Block32> activations(tokens * blocks);
  const float* source = value.data_ptr<float>();
  const auto* wp = reinterpret_cast<const Q4Block32*>(
      weights.data_ptr<uint8_t>());
  float* op = output.data_ptr<float>();
  const int64_t tiles =
      (rows + kQ4BlockMajorRows - 1) / kQ4BlockMajorRows;
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for (int64_t token = 0; token < tokens; ++token) {
      quantize_q8_row(
          source + token * cols, cols,
          activations.data() + token * blocks);
    }
#pragma omp for schedule(static)
    for (int64_t tile_index = 0; tile_index < tiles; ++tile_index) {
      const int64_t first_row = tile_index * kQ4BlockMajorRows;
      const int64_t valid_rows = std::min<int64_t>(
          kQ4BlockMajorRows, rows - first_row);
      std::vector<float> sums(tokens * valid_rows, 0.0f);
      const Q4Block32* tile = wp + first_row * blocks;
      for (int64_t block = 0; block < blocks; ++block) {
        const Q4Block32* block_rows = tile + block * valid_rows;
        for (int64_t token = 0; token < tokens; ++token) {
          const Q8Block32& activation =
              activations[token * blocks + block];
          float* token_sums = sums.data() + token * valid_rows;
          for (int64_t local = 0; local < valid_rows; ++local) {
            token_sums[local] +=
                q4_q8_block_dot(block_rows[local], activation);
          }
        }
      }
      for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t local = 0; local < valid_rows; ++local) {
          op[token * rows + first_row + local] =
              sums[token * valid_rows + local];
        }
      }
    }
  }
  return output;
}

inline void block_fp8_block_major_task(
    const float* input_f,
    const at::BFloat16* input_b,
    bool input_bf16,
    const uint8_t* task_weights,
    const float* scale_row,
    const float* lut,
    int64_t first_row,
    int64_t rows,
    int64_t cols,
    int64_t col_blocks,
    bool rows8,
    bool output_bf16,
    float* output_f,
    at::BFloat16* output_b) {
  constexpr int64_t block_size = 128;
  constexpr int64_t chunk_rows = 32;
  alignas(64) float sums[chunk_rows] = {0.0f};
  const int64_t valid_rows = std::min(chunk_rows, rows - first_row);
  for (int64_t col_block = 0; col_block < col_blocks; ++col_block) {
    const int64_t first_col = col_block * block_size;
    const int64_t valid_cols = std::min(block_size, cols - first_col);
    const uint8_t* tile =
        task_weights + col_block * chunk_rows * block_size;
    const uint8_t* next_tile =
        col_block + 1 < col_blocks
        ? tile + chunk_rows * block_size
        : nullptr;
    const float scale = scale_row[col_block];
    int64_t local_row = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (rows8) {
      for (; local_row + 8 <= valid_rows; local_row += 8) {
        float partial[8] = {
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f};
        if (next_tile != nullptr) {
          for (int64_t lane = 0; lane < 8; ++lane) {
            const uint8_t* next_row =
                next_tile + (local_row + lane) * block_size;
            _mm_prefetch(
                reinterpret_cast<const char*>(next_row), _MM_HINT_T0);
            _mm_prefetch(
                reinterpret_cast<const char*>(next_row + 64), _MM_HINT_T0);
          }
        }
        int64_t col = 0;
#if defined(__AVX512BF16__)
        if (input_bf16) {
          __m512 accumulated[8];
          for (auto& value : accumulated) {
            value = _mm512_setzero_ps();
          }
          for (; col + 32 <= valid_cols; col += 32) {
            const __m512i activation = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(
                    input_b + first_col + col));
            for (int64_t lane = 0; lane < 8; ++lane) {
              accumulated[lane] = _mm512_dpbf16_ps(
                  accumulated[lane],
                  (__m512bh)activation,
                  (__m512bh)decode_e4m3fn_bf16x32(
                      tile + (local_row + lane) * block_size + col));
            }
          }
          for (int64_t lane = 0; lane < 8; ++lane) {
            partial[lane] = _mm512_reduce_add_ps(accumulated[lane]);
          }
        } else
#endif
        {
          __m512 accumulated[8];
          for (auto& value : accumulated) {
            value = _mm512_setzero_ps();
          }
          for (; col + 16 <= valid_cols; col += 16) {
            const __m512 activation = _mm512_loadu_ps(
                input_f + first_col + col);
            for (int64_t lane = 0; lane < 8; ++lane) {
              const __m128i packed = _mm_loadu_si128(
                  reinterpret_cast<const __m128i*>(
                      tile + (local_row + lane) * block_size + col));
              const __m512 decoded = _mm512_i32gather_ps(
                  _mm512_cvtepu8_epi32(packed), lut, 4);
              accumulated[lane] = _mm512_fmadd_ps(
                  activation, decoded, accumulated[lane]);
            }
          }
          for (int64_t lane = 0; lane < 8; ++lane) {
            partial[lane] = _mm512_reduce_add_ps(accumulated[lane]);
          }
        }
        for (; col < valid_cols; ++col) {
          const float activation = input_bf16
              ? static_cast<float>(input_b[first_col + col])
              : input_f[first_col + col];
          for (int64_t lane = 0; lane < 8; ++lane) {
            partial[lane] += activation *
                lut[tile[(local_row + lane) * block_size + col]];
          }
        }
        for (int64_t lane = 0; lane < 8; ++lane) {
          sums[local_row + lane] += partial[lane] * scale;
        }
      }
    }
    for (; local_row + 4 <= valid_rows; local_row += 4) {
      float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#if defined(__AVX512F__)
      if (next_tile != nullptr) {
        for (int64_t lane = 0; lane < 4; ++lane) {
          const uint8_t* next_row =
              next_tile + (local_row + lane) * block_size;
          _mm_prefetch(
              reinterpret_cast<const char*>(next_row),
              _MM_HINT_T0);
          _mm_prefetch(
              reinterpret_cast<const char*>(next_row + 64),
              _MM_HINT_T0);
        }
      }
#endif
      int64_t col = 0;
#if defined(__AVX512BF16__)
      if (input_bf16) {
        __m512 accumulated0 = _mm512_setzero_ps();
        __m512 accumulated1 = _mm512_setzero_ps();
        __m512 accumulated2 = _mm512_setzero_ps();
        __m512 accumulated3 = _mm512_setzero_ps();
        for (; col + 32 <= valid_cols; col += 32) {
          const __m512i activation = _mm512_loadu_si512(
              reinterpret_cast<const __m512i*>(input_b + first_col + col));
          accumulated0 = _mm512_dpbf16_ps(
              accumulated0,
              (__m512bh)activation,
              (__m512bh)decode_e4m3fn_bf16x32(
                  tile + (local_row + 0) * block_size + col));
          accumulated1 = _mm512_dpbf16_ps(
              accumulated1,
              (__m512bh)activation,
              (__m512bh)decode_e4m3fn_bf16x32(
                  tile + (local_row + 1) * block_size + col));
          accumulated2 = _mm512_dpbf16_ps(
              accumulated2,
              (__m512bh)activation,
              (__m512bh)decode_e4m3fn_bf16x32(
                  tile + (local_row + 2) * block_size + col));
          accumulated3 = _mm512_dpbf16_ps(
              accumulated3,
              (__m512bh)activation,
              (__m512bh)decode_e4m3fn_bf16x32(
                  tile + (local_row + 3) * block_size + col));
        }
        partial[0] = _mm512_reduce_add_ps(accumulated0);
        partial[1] = _mm512_reduce_add_ps(accumulated1);
        partial[2] = _mm512_reduce_add_ps(accumulated2);
        partial[3] = _mm512_reduce_add_ps(accumulated3);
      } else
#endif
      {
        __m512 accumulated0 = _mm512_setzero_ps();
        __m512 accumulated1 = _mm512_setzero_ps();
        __m512 accumulated2 = _mm512_setzero_ps();
        __m512 accumulated3 = _mm512_setzero_ps();
        for (; col + 16 <= valid_cols; col += 16) {
          const __m512 activation = _mm512_loadu_ps(
              input_f + first_col + col);
          const auto decode_row = [&](int64_t row) {
            const __m128i packed = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(
                    tile + row * block_size + col));
            return _mm512_i32gather_ps(
                _mm512_cvtepu8_epi32(packed), lut, 4);
          };
          accumulated0 = _mm512_fmadd_ps(
              activation, decode_row(local_row + 0), accumulated0);
          accumulated1 = _mm512_fmadd_ps(
              activation, decode_row(local_row + 1), accumulated1);
          accumulated2 = _mm512_fmadd_ps(
              activation, decode_row(local_row + 2), accumulated2);
          accumulated3 = _mm512_fmadd_ps(
              activation, decode_row(local_row + 3), accumulated3);
        }
        partial[0] = _mm512_reduce_add_ps(accumulated0);
        partial[1] = _mm512_reduce_add_ps(accumulated1);
        partial[2] = _mm512_reduce_add_ps(accumulated2);
        partial[3] = _mm512_reduce_add_ps(accumulated3);
      }
      for (; col < valid_cols; ++col) {
        const float activation = input_bf16
            ? static_cast<float>(input_b[first_col + col])
            : input_f[first_col + col];
        for (int64_t lane = 0; lane < 4; ++lane) {
          partial[lane] += activation *
              lut[tile[(local_row + lane) * block_size + col]];
        }
      }
      for (int64_t lane = 0; lane < 4; ++lane) {
        sums[local_row + lane] += partial[lane] * scale;
      }
    }
#endif
    for (; local_row < valid_rows; ++local_row) {
      sums[local_row] += block_fp8_row_dot(
          input_f == nullptr ? nullptr : input_f + first_col,
          input_b == nullptr ? nullptr : input_b + first_col,
          input_bf16,
          tile + local_row * block_size,
          &scale,
          lut,
          valid_cols,
          block_size);
    }
  }
  for (int64_t local_row = 0; local_row < valid_rows; ++local_row) {
    const int64_t row = first_row + local_row;
    if (output_bf16) {
      output_b[row] = at::BFloat16(sums[local_row]);
    } else {
      output_f[row] = sums[local_row];
    }
  }
}

torch::Tensor block_fp8_gemv_cpu(
    torch::Tensor value,
    torch::Tensor weights,
    torch::Tensor scales,
    int64_t rows,
    int64_t cols,
    int64_t block_size,
    torch::Tensor output) {
  const double started = wall_seconds();
  TORCH_CHECK(
      !value.is_cuda() && !weights.is_cuda() && !scales.is_cuda() &&
          !output.is_cuda(),
      "block FP8 GEMV operands must be on CPU");
  TORCH_CHECK(
      value.dim() == 2 && value.size(0) == 1,
      "block FP8 GEMV requires one input row");
  TORCH_CHECK(
      value.scalar_type() == at::kFloat ||
          value.scalar_type() == at::kBFloat16,
      "block FP8 GEMV input must be float32 or bfloat16");
  const bool block_major = weights.dim() == 5;
  TORCH_CHECK(
      weights.scalar_type() == at::kByte && weights.is_contiguous() &&
          (weights.dim() == 2 || block_major),
      "block FP8 weights must be row-major [rows,cols] or block-major");
  TORCH_CHECK(
      scales.dim() == 2 && scales.scalar_type() == at::kFloat &&
          scales.is_contiguous(),
      "block FP8 scales must be contiguous float32");
  TORCH_CHECK(
      block_size == 128 && rows > 0 && cols > 0 &&
          value.size(1) == cols,
      "block FP8 GEMV currently requires block128 matching columns");
  const int64_t row_blocks = (rows + block_size - 1) / block_size;
  const int64_t col_blocks = (cols + block_size - 1) / block_size;
  if (block_major) {
    TORCH_CHECK(
        weights.size(0) == row_blocks && weights.size(1) == 4 &&
            weights.size(2) == col_blocks && weights.size(3) == 32 &&
            weights.size(4) == block_size,
        "block-major FP8 tensor shape does not match logical matrix");
  } else {
    TORCH_CHECK(weights.size(0) == rows && weights.size(1) == cols,
                "row-major FP8 tensor shape does not match logical matrix");
  }
  TORCH_CHECK(
      scales.size(0) == row_blocks && scales.size(1) == col_blocks,
      "block FP8 scale grid does not match weight shape");
  TORCH_CHECK(
      (output.scalar_type() == at::kFloat ||
       output.scalar_type() == at::kBFloat16) &&
          output.is_contiguous() &&
          output.numel() >= rows,
      "block FP8 output must be contiguous float32 or bfloat16");

  auto input = value.contiguous();
  const bool input_bf16 = input.scalar_type() == at::kBFloat16;
  const float* xp = input_bf16 ? nullptr : input.data_ptr<float>();
  const at::BFloat16* xb = input_bf16
      ? input.data_ptr<at::BFloat16>()
      : nullptr;
  const uint8_t* wp = weights.data_ptr<uint8_t>();
  const float* sp = scales.data_ptr<float>();
  const float* lut = e4m3fn_table();
  const bool output_bf16 = output.scalar_type() == at::kBFloat16;
  const bool rows8 = block_fp8_rows8_enabled();
  float* op = output_bf16 ? nullptr : output.data_ptr<float>();
  at::BFloat16* opb = output_bf16
      ? output.data_ptr<at::BFloat16>()
      : nullptr;
  if (block_major) {
    constexpr int64_t row_chunks = 4;
    constexpr int64_t chunk_rows = 32;
    const int64_t task_bytes = col_blocks * chunk_rows * block_size;
    at::parallel_for(
        0, row_blocks * row_chunks, 1,
        [&](int64_t begin, int64_t end) {
          for (int64_t task = begin; task < end; ++task) {
            const int64_t row_block = task / row_chunks;
            const int64_t row_chunk = task % row_chunks;
            const int64_t first_row =
                row_block * block_size + row_chunk * chunk_rows;
            if (first_row >= rows) {
              continue;
            }
            block_fp8_block_major_task(
                xp, xb, input_bf16,
                wp + task * task_bytes,
                sp + row_block * col_blocks,
                lut, first_row, rows, cols, col_blocks,
                rows8,
                output_bf16, op, opb);
          }
        });
    if (rows8) {
      block_fp8_rows8_tasks += row_blocks * row_chunks;
    }
  } else {
    at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
      for (int64_t row = begin; row < end; ++row) {
        const uint8_t* weight_row = wp + row * cols;
        const float* scale_row = sp + (row / block_size) * col_blocks;
        const float dot = block_fp8_row_dot(
            xp, xb, input_bf16, weight_row, scale_row, lut, cols,
            block_size);
        if (output_bf16) {
          opb[row] = at::BFloat16(dot);
        } else {
          op[row] = dot;
        }
      }
    });
  }
  block_fp8_gemv_seconds += wall_seconds() - started;
  ++block_fp8_gemv_calls;
  block_fp8_gemv_weight_elements += rows * cols;
  return output.reshape({-1}).narrow(0, 0, rows).reshape({1, rows});
}

torch::Tensor block_fp8_gemm_cpu(
    torch::Tensor value,
    torch::Tensor weights,
    torch::Tensor scales,
    int64_t rows,
    int64_t cols,
    int64_t block_size,
    torch::Tensor output) {
  const double started = wall_seconds();
  TORCH_CHECK(
      !value.is_cuda() && !weights.is_cuda() && !scales.is_cuda() &&
          !output.is_cuda(),
      "block FP8 GEMM operands must be on CPU");
  TORCH_CHECK(
      value.dim() == 2 && value.size(0) >= 2 && value.size(0) <= 16 &&
          value.size(1) == cols,
      "block FP8 GEMM requires 2..16 matching input rows");
  TORCH_CHECK(
      value.scalar_type() == at::kFloat ||
          value.scalar_type() == at::kBFloat16,
      "block FP8 GEMM input must be float32 or bfloat16");
  const bool block_major = weights.dim() == 5;
  TORCH_CHECK(
      weights.scalar_type() == at::kByte && weights.is_contiguous() &&
          (weights.dim() == 2 || block_major),
      "block FP8 GEMM weights must stay compact");
  TORCH_CHECK(
      scales.dim() == 2 && scales.scalar_type() == at::kFloat &&
          scales.is_contiguous(),
      "block FP8 GEMM scales must be contiguous float32");
  TORCH_CHECK(block_size == 128 && rows > 0 && cols > 0,
              "block FP8 GEMM currently requires block128");
  const int64_t tokens = value.size(0);
  const int64_t row_blocks = (rows + block_size - 1) / block_size;
  const int64_t col_blocks = (cols + block_size - 1) / block_size;
  if (block_major) {
    TORCH_CHECK(
        weights.size(0) == row_blocks && weights.size(1) == 4 &&
            weights.size(2) == col_blocks && weights.size(3) == 32 &&
            weights.size(4) == block_size,
        "block-major FP8 GEMM shape mismatch");
  } else {
    TORCH_CHECK(weights.size(0) == rows && weights.size(1) == cols,
                "row-major FP8 GEMM shape mismatch");
  }
  TORCH_CHECK(
      scales.size(0) == row_blocks && scales.size(1) == col_blocks,
      "block FP8 GEMM scale grid mismatch");
  TORCH_CHECK(
      (output.scalar_type() == at::kFloat ||
       output.scalar_type() == at::kBFloat16) &&
          output.is_contiguous() && output.dim() == 2 &&
          output.size(0) == tokens && output.size(1) >= rows,
      "block FP8 GEMM output must be contiguous [tokens,rows]");

  auto input = value.contiguous();
  const bool input_bf16 = input.scalar_type() == at::kBFloat16;
  const float* input_f = input_bf16 ? nullptr : input.data_ptr<float>();
  const at::BFloat16* input_b = input_bf16
      ? input.data_ptr<at::BFloat16>()
      : nullptr;
  const uint8_t* weightp = weights.data_ptr<uint8_t>();
  const float* scalep = scales.data_ptr<float>();
  const float* lut = e4m3fn_table();
  const bool output_bf16 = output.scalar_type() == at::kBFloat16;
  float* output_f = output_bf16 ? nullptr : output.data_ptr<float>();
  at::BFloat16* output_b = output_bf16
      ? output.data_ptr<at::BFloat16>()
      : nullptr;
  at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
    float result[16];
    for (int64_t row = begin; row < end; ++row) {
      const uint8_t* source;
      if (block_major) {
        const int64_t row_block = row / block_size;
        const int64_t local = row % block_size;
        const int64_t row_chunk = local / 32;
        const int64_t row_in_chunk = local % 32;
        const int64_t task = row_block * 4 + row_chunk;
        const int64_t task_bytes = col_blocks * 32 * block_size;
        source = weightp + task * task_bytes + row_in_chunk * block_size;
      } else {
        source = weightp + row * cols;
      }
      block_fp8_row_dot_many(
          input_f,
          input_b,
          input_bf16,
          tokens,
          source,
          block_major ? 32 * block_size : block_size,
          scalep + (row / block_size) * col_blocks,
          lut,
          cols,
          block_size,
          result);
      for (int64_t token = 0; token < tokens; ++token) {
        if (output_bf16) {
          output_b[token * output.size(1) + row] =
              at::BFloat16(result[token]);
        } else {
          output_f[token * output.size(1) + row] = result[token];
        }
      }
    }
  });
  block_fp8_gemv_seconds += wall_seconds() - started;
  ++block_fp8_gemv_calls;
  block_fp8_gemv_weight_elements += rows * cols;
  ++block_fp8_gemm_calls;
  block_fp8_gemm_tokens += tokens;
  return output.narrow(1, 0, rows);
}

torch::Tensor block_fp8_grouped_gemv_cpu(
    torch::Tensor value,
    torch::Tensor weight_ptrs,
    torch::Tensor scale_ptrs,
    torch::Tensor row_offsets,
    int64_t total_rows,
    int64_t cols,
    int64_t block_size,
    bool block_major,
    torch::Tensor output) {
  const double started = wall_seconds();
  TORCH_CHECK(
      !value.is_cuda() && !weight_ptrs.is_cuda() &&
          !scale_ptrs.is_cuda() && !row_offsets.is_cuda() &&
          !output.is_cuda(),
      "grouped block FP8 operands must be on CPU");
  TORCH_CHECK(
      value.dim() == 2 && value.size(0) == 1 && value.size(1) == cols,
      "grouped block FP8 requires one matching input row");
  TORCH_CHECK(
      value.scalar_type() == at::kFloat ||
          value.scalar_type() == at::kBFloat16,
      "grouped block FP8 input must be float32 or bfloat16");
  TORCH_CHECK(
      weight_ptrs.scalar_type() == at::kLong && weight_ptrs.dim() == 1 &&
          weight_ptrs.is_contiguous(),
      "grouped block FP8 weight pointers must be contiguous int64");
  TORCH_CHECK(
      scale_ptrs.scalar_type() == at::kLong && scale_ptrs.dim() == 1 &&
          scale_ptrs.is_contiguous() &&
          scale_ptrs.numel() == weight_ptrs.numel(),
      "grouped block FP8 scale pointers must match weights");
  TORCH_CHECK(
      row_offsets.scalar_type() == at::kInt && row_offsets.dim() == 1 &&
          row_offsets.is_contiguous() &&
          row_offsets.numel() == weight_ptrs.numel() + 1,
      "grouped block FP8 row offsets must delimit every weight");
  TORCH_CHECK(
      block_size == 128 && total_rows > 0 && cols > 0,
      "grouped block FP8 currently requires block128");
  TORCH_CHECK(
      (output.scalar_type() == at::kFloat ||
       output.scalar_type() == at::kBFloat16) &&
          output.is_contiguous() &&
          output.numel() >= total_rows,
      "grouped block FP8 output must be contiguous float32 or bfloat16");

  auto input = value.contiguous();
  const bool input_bf16 = input.scalar_type() == at::kBFloat16;
  const float* xp = input_bf16 ? nullptr : input.data_ptr<float>();
  const at::BFloat16* xb = input_bf16
      ? input.data_ptr<at::BFloat16>()
      : nullptr;
  const int64_t* wp = weight_ptrs.data_ptr<int64_t>();
  const int64_t* sp = scale_ptrs.data_ptr<int64_t>();
  const int32_t* offsets = row_offsets.data_ptr<int32_t>();
  const int64_t groups = weight_ptrs.numel();
  TORCH_CHECK(offsets[0] == 0 && offsets[groups] == total_rows,
              "grouped block FP8 offsets must cover total rows");
  for (int64_t group = 0; group < groups; ++group) {
    TORCH_CHECK(offsets[group] <= offsets[group + 1],
                "grouped block FP8 offsets must be sorted");
  }
  const int64_t col_blocks = (cols + block_size - 1) / block_size;
  const float* lut = e4m3fn_table();
  const bool output_bf16 = output.scalar_type() == at::kBFloat16;
  const bool rows8 = block_fp8_rows8_enabled();
  float* op = output_bf16 ? nullptr : output.data_ptr<float>();
  at::BFloat16* opb = output_bf16
      ? output.data_ptr<at::BFloat16>()
      : nullptr;
  if (block_major) {
    constexpr int64_t row_chunks = 4;
    constexpr int64_t chunk_rows = 32;
    std::vector<int64_t> task_offsets(groups + 1, 0);
    for (int64_t group = 0; group < groups; ++group) {
      const int64_t group_rows = offsets[group + 1] - offsets[group];
      task_offsets[group + 1] = task_offsets[group] +
          ((group_rows + block_size - 1) / block_size) * row_chunks;
    }
    const int64_t total_tasks = task_offsets[groups];
    const int64_t task_bytes = col_blocks * chunk_rows * block_size;
    at::parallel_for(0, total_tasks, 1, [&](int64_t begin, int64_t end) {
      int64_t group = std::upper_bound(
          task_offsets.begin(), task_offsets.end(), begin) -
          task_offsets.begin() - 1;
      group = std::max<int64_t>(0, std::min(group, groups - 1));
      for (int64_t task = begin; task < end; ++task) {
        while (group + 1 < groups && task >= task_offsets[group + 1]) {
          ++group;
        }
        const int64_t local_task = task - task_offsets[group];
        const int64_t row_block = local_task / row_chunks;
        const int64_t row_chunk = local_task % row_chunks;
        const int64_t local_first_row =
            row_block * block_size + row_chunk * chunk_rows;
        const int64_t group_rows = offsets[group + 1] - offsets[group];
        if (local_first_row >= group_rows) {
          continue;
        }
        const auto* weight = reinterpret_cast<const uint8_t*>(wp[group]);
        const auto* scale = reinterpret_cast<const float*>(sp[group]);
        block_fp8_block_major_task(
            xp, xb, input_bf16,
            weight + local_task * task_bytes,
            scale + row_block * col_blocks,
            lut,
            offsets[group] + local_first_row,
            offsets[group] + group_rows,
            cols,
            col_blocks,
            rows8,
            output_bf16,
            op,
            opb);
      }
    });
    if (rows8) {
      block_fp8_rows8_tasks += total_tasks;
    }
  } else {
    at::parallel_for(0, total_rows, 1, [&](int64_t begin, int64_t end) {
      int64_t group = std::upper_bound(
          offsets, offsets + groups + 1, static_cast<int32_t>(begin)) -
          offsets - 1;
      group = std::max<int64_t>(0, std::min(group, groups - 1));
      for (int64_t row = begin; row < end; ++row) {
        while (group + 1 < groups && row >= offsets[group + 1]) {
          ++group;
        }
        const int64_t local_row = row - offsets[group];
        const auto* weight = reinterpret_cast<const uint8_t*>(wp[group]);
        const auto* scale = reinterpret_cast<const float*>(sp[group]);
        const float dot = block_fp8_row_dot(
            xp,
            xb,
            input_bf16,
            weight + local_row * cols,
            scale + (local_row / block_size) * col_blocks,
            lut,
            cols,
            block_size);
        if (output_bf16) {
          opb[row] = at::BFloat16(dot);
        } else {
          op[row] = dot;
        }
      }
    });
  }
  block_fp8_gemv_seconds += wall_seconds() - started;
  ++block_fp8_gemv_calls;
  block_fp8_gemv_weight_elements += total_rows * cols;
  return output.reshape({-1})
      .narrow(0, 0, total_rows)
      .reshape({1, total_rows});
}

torch::Tensor block_fp8_grouped_rows_gemv_cpu(
    torch::Tensor value,
    torch::Tensor weight_ptrs,
    torch::Tensor scale_ptrs,
    torch::Tensor row_offsets,
    int64_t total_rows,
    int64_t cols,
    int64_t block_size,
    bool block_major,
    torch::Tensor output) {
  const double started = wall_seconds();
  TORCH_CHECK(
      !value.is_cuda() && !weight_ptrs.is_cuda() &&
          !scale_ptrs.is_cuda() && !row_offsets.is_cuda() &&
          !output.is_cuda(),
      "grouped-row block FP8 operands must be on CPU");
  const int64_t groups = weight_ptrs.numel();
  TORCH_CHECK(
      value.dim() == 2 && value.size(0) == groups &&
          value.size(1) == cols &&
          (value.scalar_type() == at::kFloat ||
           value.scalar_type() == at::kBFloat16),
      "grouped-row block FP8 requires one input per projection");
  TORCH_CHECK(
      weight_ptrs.scalar_type() == at::kLong &&
          scale_ptrs.scalar_type() == at::kLong &&
          weight_ptrs.dim() == 1 && scale_ptrs.dim() == 1 &&
          weight_ptrs.is_contiguous() && scale_ptrs.is_contiguous() &&
          scale_ptrs.numel() == groups,
      "grouped-row pointer arrays must match");
  TORCH_CHECK(
      row_offsets.scalar_type() == at::kInt &&
          row_offsets.dim() == 1 && row_offsets.is_contiguous() &&
          row_offsets.numel() == groups + 1,
      "grouped-row offsets must delimit every projection");
  TORCH_CHECK(
      block_size == 128 && total_rows > 0 && cols > 0,
      "grouped-row block FP8 currently requires block128");
  TORCH_CHECK(
      output.is_contiguous() && output.numel() >= total_rows &&
          (output.scalar_type() == at::kFloat ||
           output.scalar_type() == at::kBFloat16),
      "grouped-row output must be contiguous float32 or bfloat16");

  auto input = value.contiguous();
  const bool input_bf16 = input.scalar_type() == at::kBFloat16;
  const float* input_f = input_bf16 ? nullptr : input.data_ptr<float>();
  const at::BFloat16* input_b = input_bf16
      ? input.data_ptr<at::BFloat16>() : nullptr;
  const int64_t* weights = weight_ptrs.data_ptr<int64_t>();
  const int64_t* scales = scale_ptrs.data_ptr<int64_t>();
  const int32_t* offsets = row_offsets.data_ptr<int32_t>();
  TORCH_CHECK(
      offsets[0] == 0 && offsets[groups] == total_rows,
      "grouped-row offsets must cover total rows");
  const int64_t col_blocks = (cols + block_size - 1) / block_size;
  const float* lut = e4m3fn_table();
  const bool output_bf16 = output.scalar_type() == at::kBFloat16;
  float* output_f = output_bf16 ? nullptr : output.data_ptr<float>();
  at::BFloat16* output_b = output_bf16
      ? output.data_ptr<at::BFloat16>() : nullptr;
  if (block_major) {
    constexpr int64_t row_chunks = 4;
    constexpr int64_t chunk_rows = 32;
    std::vector<int64_t> task_offsets(groups + 1, 0);
    for (int64_t group = 0; group < groups; ++group) {
      const int64_t group_rows = offsets[group + 1] - offsets[group];
      task_offsets[group + 1] = task_offsets[group] +
          ((group_rows + block_size - 1) / block_size) * row_chunks;
    }
    const int64_t total_tasks = task_offsets[groups];
    const int64_t task_bytes = col_blocks * chunk_rows * block_size;
    at::parallel_for(0, total_tasks, 1, [&](int64_t begin, int64_t end) {
      int64_t group = std::upper_bound(
          task_offsets.begin(), task_offsets.end(), begin) -
          task_offsets.begin() - 1;
      group = std::max<int64_t>(0, std::min(group, groups - 1));
      for (int64_t task = begin; task < end; ++task) {
        while (group + 1 < groups && task >= task_offsets[group + 1]) {
          ++group;
        }
        const int64_t local_task = task - task_offsets[group];
        const int64_t row_block = local_task / row_chunks;
        const int64_t row_chunk = local_task % row_chunks;
        const int64_t local_first =
            row_block * block_size + row_chunk * chunk_rows;
        const int64_t group_rows = offsets[group + 1] - offsets[group];
        if (local_first >= group_rows) {
          continue;
        }
        const auto* weight =
            reinterpret_cast<const uint8_t*>(weights[group]);
        const auto* scale =
            reinterpret_cast<const float*>(scales[group]);
        block_fp8_block_major_task(
            input_f ? input_f + group * cols : nullptr,
            input_b ? input_b + group * cols : nullptr,
            input_bf16,
            weight + local_task * task_bytes,
            scale + row_block * col_blocks,
            lut,
            offsets[group] + local_first,
            offsets[group] + group_rows,
            cols,
            col_blocks,
            block_fp8_rows8_enabled(),
            output_bf16,
            output_f,
            output_b);
      }
    });
  } else {
    at::parallel_for(0, total_rows, 1, [&](int64_t begin, int64_t end) {
      int64_t group = std::upper_bound(
          offsets, offsets + groups + 1, static_cast<int32_t>(begin)) -
          offsets - 1;
      group = std::max<int64_t>(0, std::min(group, groups - 1));
      for (int64_t row = begin; row < end; ++row) {
        while (group + 1 < groups && row >= offsets[group + 1]) {
          ++group;
        }
        const int64_t local_row = row - offsets[group];
        const auto* weight =
            reinterpret_cast<const uint8_t*>(weights[group]);
        const auto* scale =
            reinterpret_cast<const float*>(scales[group]);
        const float dot = block_fp8_row_dot(
            input_f ? input_f + group * cols : nullptr,
            input_b ? input_b + group * cols : nullptr,
            input_bf16,
            weight + local_row * cols,
            scale + (local_row / block_size) * col_blocks,
            lut,
            cols,
            block_size);
        if (output_bf16) {
          output_b[row] = at::BFloat16(dot);
        } else {
          output_f[row] = dot;
        }
      }
    });
  }
  block_fp8_gemv_seconds += wall_seconds() - started;
  ++block_fp8_gemv_calls;
  block_fp8_gemv_weight_elements += total_rows * cols;
  return output.reshape({-1}).narrow(0, 0, total_rows).reshape({1, total_rows});
}

torch::Tensor block_fp8_grouped_gemm_cpu(
    torch::Tensor value,
    torch::Tensor weight_ptrs,
    torch::Tensor scale_ptrs,
    torch::Tensor row_offsets,
    int64_t total_rows,
    int64_t cols,
    int64_t block_size,
    bool block_major,
    torch::Tensor output) {
  const double started = wall_seconds();
  TORCH_CHECK(
      !value.is_cuda() && !weight_ptrs.is_cuda() &&
          !scale_ptrs.is_cuda() && !row_offsets.is_cuda() &&
          !output.is_cuda(),
      "grouped block FP8 GEMM operands must be on CPU");
  TORCH_CHECK(
      value.dim() == 2 && value.size(0) >= 2 && value.size(0) <= 16 &&
          value.size(1) == cols &&
          (value.scalar_type() == at::kFloat ||
           value.scalar_type() == at::kBFloat16),
      "grouped block FP8 GEMM requires 2..16 matching rows");
  TORCH_CHECK(
      weight_ptrs.scalar_type() == at::kLong &&
          scale_ptrs.scalar_type() == at::kLong &&
          weight_ptrs.dim() == 1 && scale_ptrs.dim() == 1 &&
          weight_ptrs.is_contiguous() && scale_ptrs.is_contiguous() &&
          weight_ptrs.numel() == scale_ptrs.numel(),
      "grouped block FP8 GEMM pointer arrays must match");
  TORCH_CHECK(
      row_offsets.scalar_type() == at::kInt && row_offsets.dim() == 1 &&
          row_offsets.is_contiguous() &&
          row_offsets.numel() == weight_ptrs.numel() + 1,
      "grouped block FP8 GEMM row offsets must delimit every weight");
  TORCH_CHECK(block_size == 128 && total_rows > 0 && cols > 0,
              "grouped block FP8 GEMM currently requires block128");
  const int64_t tokens = value.size(0);
  TORCH_CHECK(
      output.dim() == 2 && output.size(0) == tokens &&
          output.size(1) >= total_rows && output.stride(1) == 1 &&
          output.stride(0) >= total_rows &&
          (output.scalar_type() == at::kFloat ||
           output.scalar_type() == at::kBFloat16),
      "grouped block FP8 GEMM output must be a dense-row "
      "[tokens,total_rows] view");

  auto input = value.contiguous();
  const bool input_bf16 = input.scalar_type() == at::kBFloat16;
  const float* input_f = input_bf16 ? nullptr : input.data_ptr<float>();
  const at::BFloat16* input_b = input_bf16
      ? input.data_ptr<at::BFloat16>()
      : nullptr;
  const int64_t* weights = weight_ptrs.data_ptr<int64_t>();
  const int64_t* scales = scale_ptrs.data_ptr<int64_t>();
  const int32_t* offsets = row_offsets.data_ptr<int32_t>();
  const int64_t groups = weight_ptrs.numel();
  TORCH_CHECK(offsets[0] == 0 && offsets[groups] == total_rows,
              "grouped block FP8 GEMM offsets must cover total rows");
  for (int64_t group = 0; group < groups; ++group) {
    TORCH_CHECK(offsets[group] <= offsets[group + 1],
                "grouped block FP8 GEMM offsets must be sorted");
  }
  const int64_t col_blocks = (cols + block_size - 1) / block_size;
  const float* lut = e4m3fn_table();
  const bool output_bf16 = output.scalar_type() == at::kBFloat16;
  const int64_t output_stride = output.stride(0);
  float* output_f = output_bf16 ? nullptr : output.data_ptr<float>();
  at::BFloat16* output_b = output_bf16
      ? output.data_ptr<at::BFloat16>()
      : nullptr;
  at::parallel_for(0, total_rows, 1, [&](int64_t begin, int64_t end) {
    int64_t group = std::upper_bound(
        offsets, offsets + groups + 1, static_cast<int32_t>(begin)) -
        offsets - 1;
    group = std::max<int64_t>(0, std::min(group, groups - 1));
    alignas(64) float results[16];
    for (int64_t row = begin; row < end; ++row) {
      while (group + 1 < groups && row >= offsets[group + 1]) {
        ++group;
      }
      const int64_t local_row = row - offsets[group];
      const auto* weight =
          reinterpret_cast<const uint8_t*>(weights[group]);
      const auto* scale = reinterpret_cast<const float*>(scales[group]);
      const uint8_t* source;
      int64_t weight_stride;
      if (block_major) {
        const int64_t row_block = local_row / block_size;
        const int64_t local = local_row % block_size;
        const int64_t row_chunk = local / 32;
        const int64_t row_in_chunk = local % 32;
        const int64_t task = row_block * 4 + row_chunk;
        const int64_t task_bytes = col_blocks * 32 * block_size;
        source = weight + task * task_bytes + row_in_chunk * block_size;
        weight_stride = 32 * block_size;
      } else {
        source = weight + local_row * cols;
        weight_stride = block_size;
      }
      block_fp8_row_dot_many(
          input_f,
          input_b,
          input_bf16,
          tokens,
          source,
          weight_stride,
          scale + (local_row / block_size) * col_blocks,
          lut,
          cols,
          block_size,
          results);
      for (int64_t token = 0; token < tokens; ++token) {
        if (output_bf16) {
          output_b[token * output_stride + row] =
              at::BFloat16(results[token]);
        } else {
          output_f[token * output_stride + row] = results[token];
        }
      }
    }
  });
  block_fp8_gemv_seconds += wall_seconds() - started;
  ++block_fp8_gemv_calls;
  block_fp8_gemv_weight_elements += total_rows * cols;
  ++block_fp8_gemm_calls;
  block_fp8_gemm_tokens += tokens;
  return output.narrow(1, 0, total_rows);
}

void reset_block_fp8_gemv_profile_cpu() {
  block_fp8_gemv_seconds = 0.0;
  block_fp8_gemv_calls = 0;
  block_fp8_gemv_weight_elements = 0;
  block_fp8_rows8_tasks = 0;
  block_fp8_gemm_calls = 0;
  block_fp8_gemm_tokens = 0;
}

std::vector<double> block_fp8_gemv_profile_cpu() {
  return {
      static_cast<double>(block_fp8_gemv_calls),
      block_fp8_gemv_seconds,
      static_cast<double>(block_fp8_gemv_weight_elements),
      static_cast<double>(block_fp8_block_major_calls),
      static_cast<double>(block_fp8_block_major_bytes),
      static_cast<double>(block_fp8_numa_bound_tasks),
      static_cast<double>(block_fp8_rows8_tasks),
      static_cast<double>(block_fp8_gemm_calls),
      static_cast<double>(block_fp8_gemm_tokens),
  };
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
inline __m512 lookup_16(
    const float* score,
    const uint8_t* indices,
    int64_t block,
    int64_t codes,
    __m512i stride) {
  const __m128i packed = _mm_loadu_si128(
      reinterpret_cast<const __m128i*>(indices + block));
  const __m512i selected = _mm512_cvtepu8_epi32(packed);
  const __m512i base = _mm512_set1_epi32(
      static_cast<int>(block * codes));
  const __m512i offsets = _mm512_add_epi32(
      selected, _mm512_add_epi32(base, stride));
  return _mm512_i32gather_ps(offsets, score, 4);
}

inline __m512 lookup_16_u16(
    const float* score,
    const uint16_t* indices,
    int64_t block,
    int64_t codes,
    __m512i stride) {
  const __m256i packed = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(indices + block));
  const __m512i selected = _mm512_cvtepu16_epi32(packed);
  const __m512i base = _mm512_set1_epi32(
      static_cast<int>(block * codes));
  const __m512i offsets = _mm512_add_epi32(
      selected, _mm512_add_epi32(base, stride));
  return _mm512_i32gather_ps(offsets, score, 4);
}

inline __m512 lookup_rows_16(
    const float* block_score,
    const uint8_t* row_indices) {
  const __m128i packed = _mm_loadu_si128(
      reinterpret_cast<const __m128i*>(row_indices));
  const __m512i selected = _mm512_cvtepu8_epi32(packed);
  return _mm512_i32gather_ps(selected, block_score, 4);
}

#if defined(__AVX512VBMI__)
inline __m512i lookup_i8_rows_64(
    const int8_t* table,
    const uint8_t* row_indices) {
  const __m512i indices = _mm512_loadu_si512(row_indices);
  const __m512i low_indices = _mm512_and_si512(
      indices, _mm512_set1_epi8(0x7f));
  const __m512i table0 = _mm512_loadu_si512(table);
  const __m512i table1 = _mm512_loadu_si512(table + 64);
  const __m512i table2 = _mm512_loadu_si512(table + 128);
  const __m512i table3 = _mm512_loadu_si512(table + 192);
  const __m512i low_values = _mm512_permutex2var_epi8(
      table0, low_indices, table1);
  const __m512i high_values = _mm512_permutex2var_epi8(
      table2, low_indices, table3);
  const __mmask64 high_mask = _mm512_cmp_epu8_mask(
      indices, _mm512_set1_epi8(static_cast<char>(0x80)),
      _MM_CMPINT_GE);
  return _mm512_mask_blend_epi8(
      high_mask, low_values, high_values);
}

inline void add_i8_scores_64(
    int16_t* partial,
    const __m512i scores) {
  const __m512i low = _mm512_cvtepi8_epi16(
      _mm512_castsi512_si256(scores));
  const __m512i high = _mm512_cvtepi8_epi16(
      _mm512_extracti64x4_epi64(scores, 1));
  _mm512_storeu_si512(
      partial,
      _mm512_add_epi16(
          _mm512_loadu_si512(partial), low));
  _mm512_storeu_si512(
      partial + 32,
      _mm512_add_epi16(
          _mm512_loadu_si512(partial + 32), high));
}
#endif
#endif

inline float lookup_sum(
    const float* score,
    const uint8_t* row_idx,
    int64_t blocks,
    int64_t codes) {
  float sum = 0.0f;
  int64_t b = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
  const __m512i lanes = _mm512_setr_epi32(
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  const __m512i stride = _mm512_mullo_epi32(
      lanes, _mm512_set1_epi32(static_cast<int>(codes)));
  __m512 accumulated = _mm512_setzero_ps();
  // Issue four independent gathers before consuming them.  The additions
  // remain in the original block order, preserving the existing FP32 result,
  // while Xeon can overlap the otherwise latency-bound score-table reads.
  for (; b + 64 <= blocks; b += 64) {
    const __m512 first =
        lookup_16(score, row_idx, b, codes, stride);
    const __m512 second =
        lookup_16(score, row_idx, b + 16, codes, stride);
    const __m512 third =
        lookup_16(score, row_idx, b + 32, codes, stride);
    const __m512 fourth =
        lookup_16(score, row_idx, b + 48, codes, stride);
    accumulated = _mm512_add_ps(accumulated, first);
    accumulated = _mm512_add_ps(accumulated, second);
    accumulated = _mm512_add_ps(accumulated, third);
    accumulated = _mm512_add_ps(accumulated, fourth);
  }
  for (; b + 16 <= blocks; b += 16) {
    accumulated = _mm512_add_ps(
        accumulated, lookup_16(score, row_idx, b, codes, stride));
  }
  sum = _mm512_reduce_add_ps(accumulated);
#endif
  for (; b < blocks; ++b) {
    sum += score[b * codes + row_idx[b]];
  }
  return sum;
}

inline float lookup_sum_u16(
    const float* score,
    const uint16_t* row_idx,
    int64_t blocks,
    int64_t codes) {
  float sum = 0.0f;
  int64_t block = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
  const __m512i lanes = _mm512_setr_epi32(
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  const __m512i stride = _mm512_mullo_epi32(
      lanes, _mm512_set1_epi32(static_cast<int>(codes)));
  __m512 accumulated = _mm512_setzero_ps();
  for (; block + 64 <= blocks; block += 64) {
    const __m512 first =
        lookup_16_u16(score, row_idx, block, codes, stride);
    const __m512 second =
        lookup_16_u16(score, row_idx, block + 16, codes, stride);
    const __m512 third =
        lookup_16_u16(score, row_idx, block + 32, codes, stride);
    const __m512 fourth =
        lookup_16_u16(score, row_idx, block + 48, codes, stride);
    accumulated = _mm512_add_ps(accumulated, first);
    accumulated = _mm512_add_ps(accumulated, second);
    accumulated = _mm512_add_ps(accumulated, third);
    accumulated = _mm512_add_ps(accumulated, fourth);
  }
  for (; block + 16 <= blocks; block += 16) {
    accumulated = _mm512_add_ps(
        accumulated,
        lookup_16_u16(score, row_idx, block, codes, stride));
  }
  sum = _mm512_reduce_add_ps(accumulated);
#endif
  for (; block < blocks; ++block) {
    sum += score[block * codes + row_idx[block]];
  }
  return sum;
}

inline void lookup_sum_pair(
    const float* score,
    const uint8_t* first_indices,
    const uint8_t* second_indices,
    int64_t blocks,
    int64_t codes,
    float& first_sum,
    float& second_sum) {
  int64_t block = 0;
  first_sum = 0.0f;
  second_sum = 0.0f;
#if defined(__AVX512F__) && defined(__AVX512BW__)
  const __m512i lanes = _mm512_setr_epi32(
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  const __m512i stride = _mm512_mullo_epi32(
      lanes, _mm512_set1_epi32(static_cast<int>(codes)));
  __m512 first_accumulated = _mm512_setzero_ps();
  __m512 second_accumulated = _mm512_setzero_ps();
  for (; block + 64 <= blocks; block += 64) {
    const __m512 first0 =
        lookup_16(score, first_indices, block, codes, stride);
    const __m512 second0 =
        lookup_16(score, second_indices, block, codes, stride);
    const __m512 first1 =
        lookup_16(score, first_indices, block + 16, codes, stride);
    const __m512 second1 =
        lookup_16(score, second_indices, block + 16, codes, stride);
    const __m512 first2 =
        lookup_16(score, first_indices, block + 32, codes, stride);
    const __m512 second2 =
        lookup_16(score, second_indices, block + 32, codes, stride);
    const __m512 first3 =
        lookup_16(score, first_indices, block + 48, codes, stride);
    const __m512 second3 =
        lookup_16(score, second_indices, block + 48, codes, stride);
    first_accumulated = _mm512_add_ps(first_accumulated, first0);
    second_accumulated = _mm512_add_ps(second_accumulated, second0);
    first_accumulated = _mm512_add_ps(first_accumulated, first1);
    second_accumulated = _mm512_add_ps(second_accumulated, second1);
    first_accumulated = _mm512_add_ps(first_accumulated, first2);
    second_accumulated = _mm512_add_ps(second_accumulated, second2);
    first_accumulated = _mm512_add_ps(first_accumulated, first3);
    second_accumulated = _mm512_add_ps(second_accumulated, second3);
  }
  for (; block + 32 <= blocks; block += 32) {
    const __m512 first0 =
        lookup_16(score, first_indices, block, codes, stride);
    const __m512 second0 =
        lookup_16(score, second_indices, block, codes, stride);
    const __m512 first1 =
        lookup_16(score, first_indices, block + 16, codes, stride);
    const __m512 second1 =
        lookup_16(score, second_indices, block + 16, codes, stride);
    first_accumulated = _mm512_add_ps(first_accumulated, first0);
    second_accumulated = _mm512_add_ps(second_accumulated, second0);
    first_accumulated = _mm512_add_ps(first_accumulated, first1);
    second_accumulated = _mm512_add_ps(second_accumulated, second1);
  }
  for (; block + 16 <= blocks; block += 16) {
    const __m512 first =
        lookup_16(score, first_indices, block, codes, stride);
    const __m512 second =
        lookup_16(score, second_indices, block, codes, stride);
    first_accumulated = _mm512_add_ps(first_accumulated, first);
    second_accumulated = _mm512_add_ps(second_accumulated, second);
  }
  first_sum = _mm512_reduce_add_ps(first_accumulated);
  second_sum = _mm512_reduce_add_ps(second_accumulated);
#endif
  for (; block < blocks; ++block) {
    first_sum += score[block * codes + first_indices[block]];
    second_sum += score[block * codes + second_indices[block]];
  }
}

inline float lookup_weighted_many(
    const std::vector<int64_t>& score_offsets,
    const std::vector<const uint8_t*>& index_ptrs,
    const std::vector<int64_t>& blocks,
    const std::vector<int64_t>& codes,
    const float* scores,
    const float* weights,
    int64_t experts,
    int64_t row) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  if (experts <= 16) {
    __m512 accumulated[16];
    __m512i strides[16];
    const __m512i lanes = _mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15);
    for (int64_t expert = 0; expert < experts; ++expert) {
      accumulated[expert] = _mm512_setzero_ps();
      strides[expert] = _mm512_mullo_epi32(
          lanes,
          _mm512_set1_epi32(static_cast<int>(codes[expert])));
    }
    int64_t maximum_blocks = 0;
    for (int64_t expert = 0; expert < experts; ++expert) {
      maximum_blocks = std::max(maximum_blocks, blocks[expert]);
    }
    for (int64_t block = 0;
         block + 16 <= maximum_blocks;
         block += 16) {
      __m512 gathered[16];
      bool active[16] = {};
      for (int64_t expert = 0; expert < experts; ++expert) {
        if (block + 16 <= blocks[expert]) {
          gathered[expert] = lookup_16(
              scores + score_offsets[expert],
              index_ptrs[expert] + row * blocks[expert],
              block,
              codes[expert],
              strides[expert]);
          active[expert] = true;
        }
      }
      for (int64_t expert = 0; expert < experts; ++expert) {
        if (active[expert]) {
          accumulated[expert] =
              _mm512_add_ps(accumulated[expert], gathered[expert]);
        }
      }
    }
    float result = 0.0f;
    for (int64_t expert = 0; expert < experts; ++expert) {
      float sum = _mm512_reduce_add_ps(accumulated[expert]);
      const int64_t tail = (blocks[expert] / 16) * 16;
      const float* score = scores + score_offsets[expert];
      const uint8_t* indices =
          index_ptrs[expert] + row * blocks[expert];
      for (int64_t block = tail; block < blocks[expert]; ++block) {
        sum += score[block * codes[expert] + indices[block]];
      }
      result += weights[expert] * sum;
    }
    return result;
  }
#endif
  float result = 0.0f;
  for (int64_t expert = 0; expert < experts; ++expert) {
    result += weights[expert] *
              lookup_sum(
                  scores + score_offsets[expert],
                  index_ptrs[expert] + row * blocks[expert],
                  blocks[expert],
                  codes[expert]);
  }
  return result;
}

inline float int4_group_dot(
    const float* x,
    const uint8_t* packed,
    int64_t group_size) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  if (group_size == 64) {
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m512i zero_point = _mm512_set1_epi32(8);
    __m512 sum = _mm512_setzero_ps();
    for (int64_t byte_offset = 0; byte_offset < 32; byte_offset += 16) {
      const __m128i values = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(packed + byte_offset));
      const __m128i low = _mm_and_si128(values, nibble_mask);
      const __m128i high = _mm_and_si128(
          _mm_srli_epi16(values, 4), nibble_mask);
      const __m128i first = _mm_unpacklo_epi8(low, high);
      const __m128i second = _mm_unpackhi_epi8(low, high);
      const __m512 first_weight = _mm512_cvtepi32_ps(
          _mm512_sub_epi32(_mm512_cvtepu8_epi32(first), zero_point));
      const __m512 second_weight = _mm512_cvtepi32_ps(
          _mm512_sub_epi32(_mm512_cvtepu8_epi32(second), zero_point));
      const int64_t x_offset = byte_offset * 2;
      sum = _mm512_fmadd_ps(
          _mm512_loadu_ps(x + x_offset), first_weight, sum);
      sum = _mm512_fmadd_ps(
          _mm512_loadu_ps(x + x_offset + 16), second_weight, sum);
    }
    return _mm512_reduce_add_ps(sum);
  }
#endif
  float sum = 0.0f;
  for (int64_t j = 0; j < group_size / 2; ++j) {
    const uint8_t value = packed[j];
    sum += x[2 * j] *
               static_cast<float>(static_cast<int>(value & 15) - 8) +
           x[2 * j + 1] *
               static_cast<float>(static_cast<int>(value >> 4) - 8);
  }
  return sum;
}

inline float int4_row_dot(
    const float* x,
    const uint8_t* packed,
    const at::Half* scales,
    int64_t cols,
    int64_t group_size) {
  const int64_t groups = cols / group_size;
  const int64_t bytes_per_group = group_size / 2;
  float total = 0.0f;
  for (int64_t group = 0; group < groups; ++group) {
    total +=
        int4_group_dot(
            x + group * group_size,
            packed + group * bytes_per_group,
            group_size) *
        static_cast<float>(scales[group]);
  }
  return total;
}

struct Int8Activation {
  std::vector<int16_t> even;
  std::vector<int16_t> odd;
  std::vector<float> scales;
  int64_t cols = 0;
  int64_t group_size = 0;
};

inline bool cpu_w4a8_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("CCCP_CPU_W4A8");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

Int8Activation quantize_int8_activation(
    const float* input,
    int64_t cols,
    int64_t group_size) {
  TORCH_CHECK(
      group_size == 64 && cols % group_size == 0,
      "W4A8 CPU path currently requires group size 64");
  Int8Activation quantized;
  quantized.cols = cols;
  quantized.group_size = group_size;
  quantized.even.resize(cols / 2);
  quantized.odd.resize(cols / 2);
  quantized.scales.resize(cols / group_size);
  for (int64_t group = 0; group < cols / group_size; ++group) {
    const float* values = input + group * group_size;
    float maximum = 0.0f;
    for (int64_t index = 0; index < group_size; ++index) {
      maximum = std::max(maximum, std::abs(values[index]));
    }
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    const float inverse = 1.0f / scale;
    quantized.scales[group] = scale;
    int16_t* even = quantized.even.data() + group * (group_size / 2);
    int16_t* odd = quantized.odd.data() + group * (group_size / 2);
    for (int64_t index = 0; index < group_size / 2; ++index) {
      const int first = static_cast<int>(
          std::nearbyint(values[2 * index] * inverse));
      const int second = static_cast<int>(
          std::nearbyint(values[2 * index + 1] * inverse));
      even[index] = static_cast<int16_t>(
          std::max(-127, std::min(127, first)));
      odd[index] = static_cast<int16_t>(
          std::max(-127, std::min(127, second)));
    }
  }
  return quantized;
}

inline float int4_row_dot_w4a8(
    const Int8Activation& input,
    const uint8_t* packed,
    const at::Half* scales) {
  const int64_t groups = input.cols / input.group_size;
  const int64_t values_per_parity = input.group_size / 2;
  const int64_t bytes_per_group = input.group_size / 2;
  float total = 0.0f;
  for (int64_t group = 0; group < groups; ++group) {
    int32_t integer_dot = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    const __m256i packed_values = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(
            packed + group * bytes_per_group));
    const __m512i words = _mm512_cvtepu8_epi16(packed_values);
    const __m512i mask = _mm512_set1_epi16(0x0f);
    const __m512i offset = _mm512_set1_epi16(8);
    const __m512i low = _mm512_sub_epi16(
        _mm512_and_si512(words, mask), offset);
    const __m512i high = _mm512_sub_epi16(
        _mm512_and_si512(_mm512_srli_epi16(words, 4), mask),
        offset);
    const __m512i even = _mm512_loadu_si512(
        input.even.data() + group * values_per_parity);
    const __m512i odd = _mm512_loadu_si512(
        input.odd.data() + group * values_per_parity);
    const __m512i products = _mm512_add_epi32(
        _mm512_madd_epi16(low, even),
        _mm512_madd_epi16(high, odd));
    integer_dot = _mm512_reduce_add_epi32(products);
#else
    const int16_t* even =
        input.even.data() + group * values_per_parity;
    const int16_t* odd =
        input.odd.data() + group * values_per_parity;
    const uint8_t* weights = packed + group * bytes_per_group;
    for (int64_t index = 0; index < values_per_parity; ++index) {
      integer_dot +=
          even[index] * (static_cast<int>(weights[index] & 15) - 8) +
          odd[index] * (static_cast<int>(weights[index] >> 4) - 8);
    }
#endif
    total += static_cast<float>(integer_dot) *
             input.scales[group] *
             static_cast<float>(scales[group]);
  }
  return total;
}

struct Bf16Activation {
  std::vector<at::BFloat16> values;
  int64_t cols = 0;
  int64_t group_size = 0;
};

inline bool cpu_w4abf16_enabled() {
  const char* value = std::getenv("CCCP_CPU_W4ABF16");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

Bf16Activation quantize_bf16_activation(
    const float* input,
    int64_t cols,
    int64_t group_size) {
  TORCH_CHECK(
      group_size == 64 && cols % group_size == 0,
      "W4ABF16 CPU path currently requires group size 64");
  Bf16Activation converted;
  converted.cols = cols;
  converted.group_size = group_size;
  converted.values.resize(cols);
  int64_t index = 0;
#if defined(__AVX512BF16__)
  for (; index + 16 <= cols; index += 16) {
    const __m256bh packed =
        _mm512_cvtneps_pbh(_mm512_loadu_ps(input + index));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(converted.values.data() + index),
        (__m256i)packed);
  }
#endif
  for (; index < cols; ++index) {
    converted.values[index] = at::BFloat16(input[index]);
  }
  return converted;
}

inline float int4_row_dot_w4abf16(
    const Bf16Activation& input,
    const uint8_t* packed,
    const at::Half* scales) {
  const int64_t groups = input.cols / input.group_size;
  const int64_t bytes_per_group = input.group_size / 2;
  float total = 0.0f;
  for (int64_t group = 0; group < groups; ++group) {
    float dot = 0.0f;
#if defined(__AVX512BF16__) && defined(__AVX512BW__)
    const __m512i zero_point = _mm512_set1_epi32(8);
    __m512 accumulated = _mm512_setzero_ps();
    for (int64_t byte_offset = 0;
         byte_offset < bytes_per_group;
         byte_offset += 16) {
      const __m128i values = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(
              packed + group * bytes_per_group + byte_offset));
      const __m128i nibble_mask = _mm_set1_epi8(0x0f);
      const __m128i low = _mm_and_si128(values, nibble_mask);
      const __m128i high = _mm_and_si128(
          _mm_srli_epi16(values, 4), nibble_mask);
      const __m128i first = _mm_unpacklo_epi8(low, high);
      const __m128i second = _mm_unpackhi_epi8(low, high);
      const __m512 first_weight = _mm512_cvtepi32_ps(
          _mm512_sub_epi32(
              _mm512_cvtepu8_epi32(first), zero_point));
      const __m512 second_weight = _mm512_cvtepi32_ps(
          _mm512_sub_epi32(
              _mm512_cvtepu8_epi32(second), zero_point));
      const __m512bh weight = _mm512_cvtne2ps_pbh(
          second_weight, first_weight);
      const __m512bh activation = (__m512bh)_mm512_loadu_si512(
          input.values.data() +
          group * input.group_size + byte_offset * 2);
      accumulated =
          _mm512_dpbf16_ps(accumulated, activation, weight);
    }
    dot = _mm512_reduce_add_ps(accumulated);
#else
    const at::BFloat16* values =
        input.values.data() + group * input.group_size;
    const uint8_t* weights =
        packed + group * bytes_per_group;
    for (int64_t index = 0; index < input.group_size / 2; ++index) {
      dot += static_cast<float>(values[2 * index]) *
                 static_cast<float>(
                     static_cast<int>(weights[index] & 15) - 8) +
             static_cast<float>(values[2 * index + 1]) *
                 static_cast<float>(
                     static_cast<int>(weights[index] >> 4) - 8);
    }
#endif
    total += dot * static_cast<float>(scales[group]);
  }
  return total;
}

struct ExpandedBf16Weight {
  torch::Tensor packed_reference;
  torch::Tensor values;
};

std::unordered_map<const void*, ExpandedBf16Weight>
    expanded_bf16_weights;

inline bool cpu_expand_bf16_enabled() {
  const char* value = std::getenv("CCCP_CPU_EXPAND_BF16");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

torch::Tensor expand_int4_bf16(
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    int64_t cols,
    int64_t group_size) {
  const void* key = packed.data_ptr<uint8_t>();
  auto found = expanded_bf16_weights.find(key);
  if (found != expanded_bf16_weights.end()) {
    return found->second.values;
  }
  TORCH_CHECK(
      !packed.is_cuda() && !scales.is_cuda() &&
          packed.scalar_type() == at::kByte &&
          scales.scalar_type() == at::kHalf &&
          packed.dim() == 2 && scales.dim() == 2 &&
          packed.size(1) * 2 == cols &&
          scales.size(0) == packed.size(0) &&
          scales.size(1) * group_size == cols,
      "CPU BF16 expansion shape mismatch");
  auto output = torch::empty(
      {packed.size(0), cols},
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU));
  const uint8_t* qp = packed.data_ptr<uint8_t>();
  const at::Half* sp = scales.data_ptr<at::Half>();
  at::BFloat16* op = output.data_ptr<at::BFloat16>();
  const int64_t rows = packed.size(0);
  const int64_t groups = cols / group_size;
  const int64_t bytes_per_row = cols / 2;
#pragma omp parallel for schedule(static)
  for (int64_t row = 0; row < rows; ++row) {
    const uint8_t* weights = qp + row * bytes_per_row;
    const at::Half* row_scales = sp + row * groups;
    at::BFloat16* destination = op + row * cols;
    for (int64_t group = 0; group < groups; ++group) {
      const float scale = static_cast<float>(row_scales[group]);
      const uint8_t* values =
          weights + group * (group_size / 2);
      for (int64_t index = 0; index < group_size / 2; ++index) {
        destination[group * group_size + 2 * index] =
            at::BFloat16(
                static_cast<float>(
                    static_cast<int>(values[index] & 15) - 8) *
                scale);
        destination[group * group_size + 2 * index + 1] =
            at::BFloat16(
                static_cast<float>(
                    static_cast<int>(values[index] >> 4) - 8) *
                scale);
      }
    }
  }
  expanded_bf16_weights.emplace(
      key, ExpandedBf16Weight{packed, output});
  return output;
}

inline float bf16_row_dot(
    const at::BFloat16* input,
    const at::BFloat16* weight,
    int64_t cols) {
  float result = 0.0f;
  int64_t index = 0;
#if defined(__AVX512BF16__)
  __m512 accumulated = _mm512_setzero_ps();
  for (; index + 32 <= cols; index += 32) {
    const __m512bh x = (__m512bh)_mm512_loadu_si512(input + index);
    const __m512bh w = (__m512bh)_mm512_loadu_si512(weight + index);
    accumulated = _mm512_dpbf16_ps(accumulated, x, w);
  }
  result = _mm512_reduce_add_ps(accumulated);
#endif
  for (; index < cols; ++index) {
    result += static_cast<float>(input[index]) *
              static_cast<float>(weight[index]);
  }
  return result;
}

inline float float_dot(const float* left, const float* right, int64_t size) {
  float sum = 0.0f;
  int64_t index = 0;
#if defined(__AVX512F__)
  __m512 accumulated = _mm512_setzero_ps();
  for (; index + 16 <= size; index += 16) {
    accumulated = _mm512_fmadd_ps(
        _mm512_loadu_ps(left + index),
        _mm512_loadu_ps(right + index),
        accumulated);
  }
  sum = _mm512_reduce_add_ps(accumulated);
#elif defined(__AVX2__)
  __m256 accumulated = _mm256_setzero_ps();
  for (; index + 8 <= size; index += 8) {
    accumulated = _mm256_fmadd_ps(
        _mm256_loadu_ps(left + index),
        _mm256_loadu_ps(right + index),
        accumulated);
  }
  sum = horizontal_sum_f32x8(accumulated);
#endif
  for (; index < size; ++index) {
    sum += left[index] * right[index];
  }
  return sum;
}

inline void codebook_scores(
    const float* input,
    const float* transposed_codebook,
    float* output,
    int64_t codes,
    int64_t dimension) {
  int64_t code = 0;
#if defined(__AVX512F__)
  // Kimi code vectors are only 4/8 floats wide.  Vectorising across codes
  // keeps 16 independent dot products in one register instead of calling a
  // tiny scalar dot routine K times.
  for (; code + 16 <= codes; code += 16) {
    __m512 accumulated = _mm512_setzero_ps();
    for (int64_t index = 0; index < dimension; ++index) {
      accumulated = _mm512_fmadd_ps(
          _mm512_set1_ps(input[index]),
          _mm512_loadu_ps(
              transposed_codebook + index * codes + code),
          accumulated);
    }
    _mm512_storeu_ps(output + code, accumulated);
  }
#endif
  for (; code < codes; ++code) {
    float sum = 0.0f;
    for (int64_t index = 0; index < dimension; ++index) {
      sum += input[index] *
             transposed_codebook[index * codes + code];
    }
    output[code] = sum;
  }
}

inline void codebook_scores_range(
    const float* input,
    const float* transposed_codebook,
    float* output,
    int64_t codes,
    int64_t dimension,
    int64_t begin,
    int64_t end) {
  int64_t code = begin;
#if defined(__AVX512F__)
  for (; code + 16 <= end; code += 16) {
    __m512 accumulated = _mm512_setzero_ps();
    for (int64_t index = 0; index < dimension; ++index) {
      accumulated = _mm512_fmadd_ps(
          _mm512_set1_ps(input[index]),
          _mm512_loadu_ps(
              transposed_codebook + index * codes + code),
          accumulated);
    }
    _mm512_storeu_ps(output + code, accumulated);
  }
#endif
  for (; code < end; ++code) {
    float sum = 0.0f;
    for (int64_t index = 0; index < dimension; ++index) {
      sum += input[index] *
          transposed_codebook[index * codes + code];
    }
    output[code] = sum;
  }
}

struct TransposedCodebook {
  torch::Tensor source;
  torch::Tensor values;
};

std::unordered_map<const void*, TransposedCodebook>
    transposed_codebooks;
std::mutex transposed_codebooks_mutex;

torch::Tensor cached_transposed_codebook(torch::Tensor codebook) {
  const void* key = codebook.data_ptr<float>();
  std::lock_guard<std::mutex> guard(transposed_codebooks_mutex);
  auto found = transposed_codebooks.find(key);
  if (found != transposed_codebooks.end()) {
    return found->second.values;
  }
  const int64_t codes = codebook.size(0);
  const int64_t dimension = codebook.size(1);
  auto output = torch::empty(
      {dimension, codes},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  const float* source = codebook.data_ptr<float>();
  float* destination = output.data_ptr<float>();
  // These codebooks are only a few hundred KiB.  A serial transpose avoids
  // launching the large host thread pool during every layer's first access.
  for (int64_t code = 0; code < codes; ++code) {
    for (int64_t index = 0; index < dimension; ++index) {
      destination[index * codes + code] =
          source[code * dimension + index];
    }
  }
  transposed_codebooks.emplace(
      key, TransposedCodebook{codebook, output});
  return output;
}

struct PairedBf16Codebook {
  torch::Tensor source;
  torch::Tensor values;
};

std::unordered_map<const void*, PairedBf16Codebook> paired_bf16_codebooks;
std::mutex paired_bf16_codebooks_mutex;

torch::Tensor cached_paired_bf16_codebook(torch::Tensor codebook) {
  const void* key = codebook.data_ptr<float>();
  std::lock_guard<std::mutex> guard(paired_bf16_codebooks_mutex);
  auto found = paired_bf16_codebooks.find(key);
  if (found != paired_bf16_codebooks.end()) {
    return found->second.values;
  }
  const int64_t codes = codebook.size(0);
  const int64_t dimension = codebook.size(1);
  TORCH_CHECK(dimension % 2 == 0,
              "paired BF16 codebook requires an even code dimension");
  auto output = torch::empty(
      {dimension / 2, codes, 2},
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU));
  const float* source = codebook.data_ptr<float>();
  at::BFloat16* destination = output.data_ptr<at::BFloat16>();
  for (int64_t pair = 0; pair < dimension / 2; ++pair) {
    for (int64_t code = 0; code < codes; ++code) {
      destination[(pair * codes + code) * 2] =
          at::BFloat16(source[code * dimension + pair * 2]);
      destination[(pair * codes + code) * 2 + 1] =
          at::BFloat16(source[code * dimension + pair * 2 + 1]);
    }
  }
  paired_bf16_codebooks.emplace(
      key, PairedBf16Codebook{codebook, output});
  return output;
}

inline void codebook_scores_bf16_range(
    const at::BFloat16* input,
    const at::BFloat16* paired_codebook,
    float* output,
    int64_t codes,
    int64_t dimension,
    int64_t begin,
    int64_t end) {
  int64_t code = begin;
#if defined(__AVX512BF16__)
  for (; code + 16 <= end; code += 16) {
    __m512 accumulated = _mm512_setzero_ps();
    for (int64_t pair = 0; pair < dimension / 2; ++pair) {
      uint32_t activation_pair = 0;
      std::memcpy(
          &activation_pair,
          input + pair * 2,
          sizeof(activation_pair));
      const auto activation = (__m512bh)_mm512_set1_epi32(
          static_cast<int32_t>(activation_pair));
      const auto values = (__m512bh)_mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(
              paired_codebook + (pair * codes + code) * 2));
      accumulated = _mm512_dpbf16_ps(
          accumulated, activation, values);
    }
    _mm512_storeu_ps(output + code, accumulated);
  }
#endif
  for (; code < end; ++code) {
    float sum = 0.0f;
    for (int64_t lane = 0; lane < dimension; ++lane) {
      const int64_t pair = lane / 2;
      const int64_t within = lane % 2;
      sum += static_cast<float>(input[lane]) *
          static_cast<float>(
              paired_codebook[(pair * codes + code) * 2 + within]);
    }
    output[code] = sum;
  }
}

inline void float_axpy(
    float* output,
    const float* value,
    float weight,
    int64_t size) {
  int64_t index = 0;
#if defined(__AVX512F__)
  const __m512 scale = _mm512_set1_ps(weight);
  for (; index + 16 <= size; index += 16) {
    _mm512_storeu_ps(
        output + index,
        _mm512_fmadd_ps(
            _mm512_loadu_ps(value + index),
            scale,
          _mm512_loadu_ps(output + index)));
  }
#elif defined(__AVX2__)
  const __m256 scale = _mm256_set1_ps(weight);
  for (; index + 8 <= size; index += 8) {
    _mm256_storeu_ps(
        output + index,
        _mm256_fmadd_ps(
            _mm256_loadu_ps(value + index),
            scale,
            _mm256_loadu_ps(output + index)));
  }
#endif
  for (; index < size; ++index) {
    output[index] += value[index] * weight;
  }
}

torch::Tensor vq_gemv_cpu(
    torch::Tensor x_rows,
    torch::Tensor indices,
    torch::Tensor codebooks) {
  TORCH_CHECK(!x_rows.is_cuda(), "x_rows must be on CPU");
  TORCH_CHECK(!indices.is_cuda(), "indices must be on CPU");
  TORCH_CHECK(!codebooks.is_cuda(), "codebooks must be on CPU");
  TORCH_CHECK(x_rows.dim() == 2, "x_rows must have shape [N|1,C]");
  TORCH_CHECK(indices.dim() == 3, "indices must have shape [N|1,R,B]");
  TORCH_CHECK(codebooks.dim() == 3, "codebooks must have shape [N|1,K,D]");
  const bool indices_u8 = indices.scalar_type() == at::kByte;
  const bool indices_u16 = indices.scalar_type() == at::kUInt16;
  TORCH_CHECK(
      indices_u8 || indices_u16,
      "CPU VQ GEMV supports uint8 or uint16 indices");

  // Converting these two small operands once is much cheaper than expanding
  // every uint8 expert index to int64 and materialising [N,B,R] with gather.
  auto x = x_rows.to(torch::kFloat32).contiguous();
  auto cb = codebooks.to(torch::kFloat32).contiguous();
  auto idx = indices.contiguous();

  const int64_t xn = x.size(0);
  const int64_t in = idx.size(0);
  const int64_t cn = cb.size(0);
  const int64_t rows = idx.size(1);
  const int64_t blocks = idx.size(2);
  const int64_t codes = cb.size(1);
  const int64_t dim = cb.size(2);
  const int64_t n = std::max({xn, in, cn});

  TORCH_CHECK(x.size(1) == blocks * dim,
              "x width must equal index blocks * codebook dimension");
  TORCH_CHECK(
      (indices_u8 && codes <= 256) ||
          (indices_u16 && codes <= 65536),
      "index dtype cannot represent every codebook entry");
  TORCH_CHECK(xn == 1 || xn == n, "x batch is not broadcastable");
  TORCH_CHECK(in == 1 || in == n, "index batch is not broadcastable");
  TORCH_CHECK(cn == 1 || cn == n, "codebook batch is not broadcastable");

  // Lookup scores are shared by all output rows of one expert:
  // score[n,b,k] = dot(x[n,b,:], codebook[n,k,:]).
  // Decode normally has x/codebook batch 1, so this is only B*K floats.
  const int64_t score_n = std::max(xn, cn);
  auto scores = torch::empty(
      {score_n, blocks, codes},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  const float* xp = x.data_ptr<float>();
  const float* cp = cb.data_ptr<float>();
  float* sp = scores.data_ptr<float>();
  at::parallel_for(0, score_n * blocks, 8, [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t sn = item / blocks;
      const int64_t b = item - sn * blocks;
      const int64_t xbatch = xn == 1 ? 0 : sn;
      const int64_t cbatch = cn == 1 ? 0 : sn;
      const float* xv = xp + (xbatch * blocks + b) * dim;
      const float* codebook = cp + cbatch * codes * dim;
      float* score = sp + (sn * blocks + b) * codes;
      for (int64_t k = 0; k < codes; ++k) {
        const float* code = codebook + k * dim;
        float sum = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
          sum += xv[d] * code[d];
        }
        score[k] = sum;
      }
    }
  });

  auto out = torch::empty(
      {n, rows},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  const uint8_t* ip8 =
      indices_u8 ? idx.data_ptr<uint8_t>() : nullptr;
  const uint16_t* ip16 =
      indices_u16 ? idx.data_ptr<uint16_t>() : nullptr;
  float* op = out.data_ptr<float>();
  at::parallel_for(0, n * rows, 1, [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t batch = item / rows;
      const int64_t row = item - batch * rows;
      const int64_t ibatch = in == 1 ? 0 : batch;
      const int64_t sbatch = score_n == 1 ? 0 : batch;
      const float* score = sp + sbatch * blocks * codes;
      const int64_t offset = (ibatch * rows + row) * blocks;
      op[item] = indices_u8
          ? lookup_sum(score, ip8 + offset, blocks, codes)
          : lookup_sum_u16(score, ip16 + offset, blocks, codes);
    }
  });
  return out;
}

torch::Tensor vq_gemv_list_cpu(
    torch::Tensor x_rows,
    std::vector<torch::Tensor> index_list,
    torch::Tensor codebook) {
  TORCH_CHECK(!x_rows.is_cuda() && !codebook.is_cuda(),
              "VQ list operands must be on CPU");
  TORCH_CHECK(x_rows.dim() == 2, "x_rows must be [N|1,C]");
  TORCH_CHECK(codebook.dim() == 2, "codebook must be [K,D]");
  TORCH_CHECK(!index_list.empty(), "VQ index list cannot be empty");
  const int64_t n = static_cast<int64_t>(index_list.size());
  const int64_t rows = index_list[0].size(0);
  const int64_t blocks = index_list[0].size(1);
  const auto index_type = index_list[0].scalar_type();
  const bool indices_u8 = index_type == at::kByte;
  const bool indices_u16 = index_type == at::kUInt16;
  TORCH_CHECK(
      indices_u8 || indices_u16,
      "VQ list indices must be CPU uint8 or uint16 tensors");
  for (const auto& index : index_list) {
    TORCH_CHECK(
        !index.is_cuda() && index.scalar_type() == index_type,
        "VQ list indices must be homogeneous CPU tensors");
    TORCH_CHECK(index.dim() == 2 && index.size(0) == rows &&
                    index.size(1) == blocks,
                "VQ list index shapes must match");
  }
  TORCH_CHECK(x_rows.size(0) == 1 || x_rows.size(0) == n,
              "VQ list input batch must be 1 or expert count");
  const int64_t codes = codebook.size(0);
  const int64_t dim = codebook.size(1);
  TORCH_CHECK(
      (indices_u8 && codes <= 256) ||
          (indices_u16 && codes <= 65536),
      "index dtype cannot represent every codebook entry");
  TORCH_CHECK(x_rows.size(1) == blocks * dim,
              "VQ list input width mismatch");

  auto x = x_rows.to(torch::kFloat32).contiguous();
  auto cb = codebook.to(torch::kFloat32).contiguous();
  std::vector<torch::Tensor> indices;
  std::vector<const uint8_t*> index_ptrs_u8;
  std::vector<const uint16_t*> index_ptrs_u16;
  indices.reserve(n);
  index_ptrs_u8.reserve(indices_u8 ? n : 0);
  index_ptrs_u16.reserve(indices_u16 ? n : 0);
  for (auto& index : index_list) {
    indices.push_back(index.contiguous());
    if (indices_u8) {
      index_ptrs_u8.push_back(indices.back().data_ptr<uint8_t>());
    } else {
      index_ptrs_u16.push_back(indices.back().data_ptr<uint16_t>());
    }
  }
  const int64_t score_n = x.size(0);
  auto scores = torch::empty(
      {score_n, blocks, codes},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto cb_transposed = cached_transposed_codebook(cb);
  const float* xp = x.data_ptr<float>();
  const float* cp = cb_transposed.data_ptr<float>();
  float* scorep = scores.data_ptr<float>();
  at::parallel_for(0, score_n * blocks, 8, [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t batch = item / blocks;
      const int64_t block = item - batch * blocks;
      const float* xv = xp + (batch * blocks + block) * dim;
      float* score = scorep + (batch * blocks + block) * codes;
      codebook_scores(xv, cp, score, codes, dim);
    }
  });

  auto out = torch::empty(
      {n, rows},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  float* op = out.data_ptr<float>();
  at::parallel_for(0, n * rows, 1, [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t batch = item / rows;
      const int64_t row = item - batch * rows;
      const int64_t score_batch = score_n == 1 ? 0 : batch;
      const float* score =
          scorep + score_batch * blocks * codes;
      op[item] = indices_u8
          ? lookup_sum(
                score,
                index_ptrs_u8[batch] + row * blocks,
                blocks,
                codes)
          : lookup_sum_u16(
                score,
                index_ptrs_u16[batch] + row * blocks,
                blocks,
                codes);
    }
  });
  return out;
}

inline uint16_t read_odd_packed_index(
    const uint8_t* packed,
    int64_t index_offset,
    int64_t bits) {
  const int64_t bit_offset = index_offset * bits;
  const int64_t byte_offset = bit_offset >> 3;
  const int shift = static_cast<int>(bit_offset & 7);
  uint32_t word =
      static_cast<uint32_t>(packed[byte_offset]) |
      (static_cast<uint32_t>(packed[byte_offset + 1]) << 8);
  if (shift + bits > 16) {
    word |= static_cast<uint32_t>(packed[byte_offset + 2]) << 16;
  }
  return static_cast<uint16_t>(
      (word >> shift) & ((uint32_t{1} << bits) - 1));
}

inline float lookup_sum_packed(
    const float* score,
    const uint8_t* packed,
    int64_t start_index,
    int64_t blocks,
    int64_t codes,
    int64_t bits) {
  float sum = 0.0f;
  if (bits == 8) {
    for (int64_t block = 0; block < blocks; ++block) {
      sum += score[block * codes + packed[start_index + block]];
    }
    return sum;
  }
  if (bits == 16) {
    for (int64_t block = 0; block < blocks; ++block) {
      const int64_t index_offset = start_index + block;
      const int64_t offset = index_offset * 2;
      const uint16_t index = static_cast<uint16_t>(packed[offset]) |
          (static_cast<uint16_t>(packed[offset + 1]) << 8);
      sum += score[block * codes + index];
    }
    return sum;
  }
  if (bits == 9) {
    for (int64_t block = 0; block < blocks; ++block) {
      const int64_t index_offset = start_index + block;
      const int64_t bit_offset = index_offset * 9;
      const int64_t byte_offset = bit_offset >> 3;
      const int shift = static_cast<int>(bit_offset & 7);
      const uint16_t word =
          static_cast<uint16_t>(packed[byte_offset]) |
          (static_cast<uint16_t>(packed[byte_offset + 1]) << 8);
      const uint16_t index = static_cast<uint16_t>(
          (word >> shift) & 0x1ff);
      sum += score[block * codes + index];
    }
    return sum;
  }
  if (bits == 12) {
    if (start_index % 2 == 0 && blocks % 2 == 0) {
      const int64_t start_byte = (start_index / 2) * 3;
      for (int64_t block = 0; block < blocks; block += 2) {
        const int64_t offset = start_byte + (block / 2) * 3;
        const uint16_t first =
            static_cast<uint16_t>(packed[offset]) |
            ((static_cast<uint16_t>(packed[offset + 1]) & 0x0f) << 8);
        const uint16_t second =
            (static_cast<uint16_t>(packed[offset + 1]) >> 4) |
            (static_cast<uint16_t>(packed[offset + 2]) << 4);
        sum += score[block * codes + first];
        sum += score[(block + 1) * codes + second];
      }
      return sum;
    }
    for (int64_t block = 0; block < blocks; ++block) {
      const int64_t index_offset = start_index + block;
      const int64_t offset = (index_offset / 2) * 3;
      const uint16_t index = index_offset % 2 == 0
          ? static_cast<uint16_t>(packed[offset]) |
              ((static_cast<uint16_t>(packed[offset + 1]) & 0x0f) << 8)
          : (static_cast<uint16_t>(packed[offset + 1]) >> 4) |
              (static_cast<uint16_t>(packed[offset + 2]) << 4);
      sum += score[block * codes + index];
    }
    return sum;
  }
  if (bits == 10) {
    if (start_index % 4 == 0 && blocks % 4 == 0) {
      const int64_t start_byte = (start_index / 4) * 5;
      for (int64_t block = 0; block < blocks; block += 4) {
        const int64_t offset = start_byte + (block / 4) * 5;
        const uint64_t word = load_u40_le(packed + offset);
        sum += score[block * codes + (word & 0x3ff)];
        sum += score[(block + 1) * codes + ((word >> 10) & 0x3ff)];
        sum += score[(block + 2) * codes + ((word >> 20) & 0x3ff)];
        sum += score[(block + 3) * codes + ((word >> 30) & 0x3ff)];
      }
      return sum;
    }
    for (int64_t block = 0; block < blocks; ++block) {
      const int64_t index_offset = start_index + block;
      const int64_t offset = (index_offset / 4) * 5;
      const uint64_t word = load_u40_le(packed + offset);
      const int64_t shift = (index_offset % 4) * 10;
      sum += score[block * codes + ((word >> shift) & 0x3ff)];
    }
    return sum;
  }
  if (bits == 14 && start_index % 4 == 0 && blocks % 4 == 0) {
    const int64_t start_byte = (start_index / 4) * 7;
    for (int64_t block = 0; block < blocks; block += 4) {
      const int64_t offset = start_byte + (block / 4) * 7;
      const uint64_t word = load_u56_le(packed + offset);
      sum += score[block * codes + (word & 0x3fff)];
      sum += score[(block + 1) * codes + ((word >> 14) & 0x3fff)];
      sum += score[(block + 2) * codes + ((word >> 28) & 0x3fff)];
      sum += score[(block + 3) * codes + ((word >> 42) & 0x3fff)];
    }
    return sum;
  }
  if (bits == 14) {
    for (int64_t block = 0; block < blocks; ++block) {
      const int64_t index_offset = start_index + block;
      const int64_t offset = (index_offset / 4) * 7;
      const uint64_t word = load_u56_le(packed + offset);
      const int64_t shift = (index_offset % 4) * 14;
      sum += score[block * codes + ((word >> shift) & 0x3fff)];
    }
    return sum;
  }
  for (int64_t block = 0; block < blocks; ++block) {
    const uint16_t index = read_odd_packed_index(
        packed, start_index + block, bits);
    sum += score[block * codes + index];
  }
  return sum;
}

inline bool lookup_sum_packed_rows16(
    const float* score,
    const uint8_t* packed,
    int64_t first_row,
    int64_t rows,
    int64_t blocks,
    int64_t codes,
    int64_t bits,
    float* output) {
#if defined(__AVX512F__)
  if (bits != 8 && bits != 10 && bits != 12) {
    return false;
  }
  const int64_t valid = std::min<int64_t>(16, rows - first_row);
  const int64_t row_bits = blocks * bits;
  if (valid <= 0 || row_bits % 8 != 0) {
    return false;
  }
  const int64_t row_bytes = row_bits / 8;
  alignas(64) int32_t offsets_array[16];
  for (int lane = 0; lane < 16; ++lane) {
    offsets_array[lane] = static_cast<int32_t>(
        (first_row + lane) * row_bytes);
  }
  const __m512i row_offsets = _mm512_load_si512(offsets_array);
  const __mmask16 valid_mask = valid == 16
      ? static_cast<__mmask16>(0xffff)
      : static_cast<__mmask16>((uint32_t{1} << valid) - 1);
  __m512 sums = _mm512_setzero_ps();
  bool overwrite_last = false;

  if (bits == 8) {
    const int64_t vector_blocks = std::max<int64_t>(0, blocks - 3);
    int64_t block = 0;
    for (; block < vector_blocks; ++block) {
      const __m512i packed_words = _mm512_mask_i32gather_epi32(
          _mm512_setzero_si512(),
          valid_mask,
          _mm512_add_epi32(
              row_offsets, _mm512_set1_epi32(static_cast<int>(block))),
          packed,
          1);
      const __m512i indices = _mm512_and_si512(
          packed_words, _mm512_set1_epi32(0xff));
      sums = _mm512_add_ps(
          sums,
          _mm512_mask_i32gather_ps(
              _mm512_setzero_ps(), valid_mask, indices,
              score + block * codes, 4));
    }
    alignas(64) float partial[16];
    _mm512_store_ps(partial, sums);
    for (; block < blocks; ++block) {
      const float* block_score = score + block * codes;
      for (int64_t lane = 0; lane < valid; ++lane) {
        partial[lane] += block_score[
            packed[(first_row + lane) * blocks + block]];
      }
    }
    std::memcpy(output + first_row, partial, valid * sizeof(float));
    return true;
  }

  const int64_t group = bits == 10 ? 4 : 2;
  const int64_t group_bytes = bits == 10 ? 5 : 3;
  TORCH_CHECK(blocks % group == 0, "packed row group is misaligned");
  for (int64_t block = 0; block < blocks; block += group) {
    const int64_t byte_offset = (block / group) * group_bytes;
    __mmask16 load_mask = valid_mask;
    const bool final_group = block + group == blocks;
    if (final_group && first_row + valid == rows) {
      load_mask &= static_cast<__mmask16>(~(
          uint32_t{1} << (valid - 1)));
      overwrite_last = true;
    }
    const __m512i addresses = _mm512_add_epi32(
        row_offsets,
        _mm512_set1_epi32(static_cast<int>(byte_offset)));
    const __m512i low = _mm512_mask_i32gather_epi32(
        _mm512_setzero_si512(), load_mask, addresses, packed, 1);
    if (bits == 10) {
      const __m512i high_words = _mm512_mask_i32gather_epi32(
          _mm512_setzero_si512(), load_mask,
          _mm512_add_epi32(addresses, _mm512_set1_epi32(4)),
          packed, 1);
      const __m512i mask = _mm512_set1_epi32(0x3ff);
      const __m512i indices0 = _mm512_and_si512(low, mask);
      const __m512i indices1 = _mm512_and_si512(
          _mm512_srli_epi32(low, 10), mask);
      const __m512i indices2 = _mm512_and_si512(
          _mm512_srli_epi32(low, 20), mask);
      const __m512i indices3 = _mm512_or_si512(
          _mm512_srli_epi32(low, 30),
          _mm512_slli_epi32(
              _mm512_and_si512(
                  high_words, _mm512_set1_epi32(0xff)),
              2));
      const __m512i indices[4] = {
          indices0, indices1, indices2, indices3};
      for (int lane = 0; lane < 4; ++lane) {
        sums = _mm512_add_ps(
            sums,
            _mm512_mask_i32gather_ps(
                _mm512_setzero_ps(), load_mask, indices[lane],
                score + (block + lane) * codes, 4));
      }
    } else {
      const __m512i mask = _mm512_set1_epi32(0xfff);
      const __m512i indices[2] = {
          _mm512_and_si512(low, mask),
          _mm512_and_si512(_mm512_srli_epi32(low, 12), mask)};
      for (int lane = 0; lane < 2; ++lane) {
        sums = _mm512_add_ps(
            sums,
            _mm512_mask_i32gather_ps(
                _mm512_setzero_ps(), load_mask, indices[lane],
                score + (block + lane) * codes, 4));
      }
    }
  }
  _mm512_mask_storeu_ps(output + first_row, valid_mask, sums);
  if (overwrite_last) {
    const int64_t row = first_row + valid - 1;
    output[row] = lookup_sum_packed(
        score, packed, row * blocks, blocks, codes, bits);
  }
  return true;
#else
  (void)score;
  (void)packed;
  (void)first_row;
  (void)rows;
  (void)blocks;
  (void)codes;
  (void)bits;
  (void)output;
  return false;
#endif
}

inline uint16_t read_packed_index(
    const uint8_t* packed,
    int64_t index_offset,
    int64_t bits) {
  if (bits == 8) {
    return packed[index_offset];
  }
  if (bits == 16) {
    const int64_t offset = index_offset * 2;
    return static_cast<uint16_t>(packed[offset]) |
        (static_cast<uint16_t>(packed[offset + 1]) << 8);
  }
  if (bits == 9 || bits == 11 || bits == 13 || bits == 15) {
    return read_odd_packed_index(packed, index_offset, bits);
  }
  if (bits == 12) {
    const int64_t offset = (index_offset / 2) * 3;
    return index_offset % 2 == 0
        ? static_cast<uint16_t>(packed[offset]) |
              ((static_cast<uint16_t>(packed[offset + 1]) & 0x0f) << 8)
        : (static_cast<uint16_t>(packed[offset + 1]) >> 4) |
              (static_cast<uint16_t>(packed[offset + 2]) << 4);
  }
  if (bits == 10) {
    const int64_t offset = (index_offset / 4) * 5;
    const uint64_t word = load_u40_le(packed + offset);
    return static_cast<uint16_t>(
        (word >> ((index_offset % 4) * 10)) & 0x3ff);
  }
  const int64_t offset = (index_offset / 4) * 7;
  const uint64_t word = load_u56_le(packed + offset);
  return static_cast<uint16_t>(
      (word >> ((index_offset % 4) * 14)) & 0x3fff);
}

torch::Tensor vq_dequant_packed_cpu(
    torch::Tensor payload,
    torch::Tensor codebook,
    int64_t rows,
    int64_t blocks,
    int64_t bits,
    int64_t layout) {
  TORCH_CHECK(
      !payload.is_cuda() && payload.scalar_type() == at::kByte &&
          payload.is_contiguous() && payload.dim() == 1,
      "packed VQ dequant payload must be contiguous CPU uint8");
  TORCH_CHECK(
      !codebook.is_cuda() && codebook.scalar_type() == at::kFloat &&
          codebook.is_contiguous() && codebook.dim() == 2,
      "packed VQ dequant codebook must be contiguous CPU float32");
  TORCH_CHECK(rows > 0 && blocks > 0 && bits >= 8 && bits <= 16 &&
                  layout >= 0 && layout <= 2,
              "packed VQ dequant metadata is invalid");
  const int64_t count = rows * blocks;
  TORCH_CHECK(
      count * bits % 8 == 0 && payload.numel() == count * bits / 8,
      "packed VQ dequant payload length mismatch");
  const int64_t codes = codebook.size(0);
  const int64_t dim = codebook.size(1);
  TORCH_CHECK(
      codes <= (int64_t{1} << bits),
      "packed VQ width cannot represent the codebook");

  auto output = torch::empty(
      {rows, blocks * dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  const uint8_t* packed = payload.data_ptr<uint8_t>();
  const float* cb = codebook.data_ptr<float>();
  float* out = output.data_ptr<float>();
  at::parallel_for(0, count, 256, [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t row = item / blocks;
      const int64_t block = item - row * blocks;
      int64_t physical = item;
      if (layout == 1) {
        physical = block * rows + row;
      } else if (layout == 2) {
        const int64_t first_row = (row / 8) * 8;
        const int64_t valid_rows = std::min<int64_t>(8, rows - first_row);
        physical = first_row * blocks + block * valid_rows +
            (row - first_row);
      }
      const uint16_t index = read_packed_index(packed, physical, bits);
      TORCH_CHECK(index < codes, "packed VQ code index exceeds codebook");
      std::memcpy(
          out + item * dim,
          cb + static_cast<int64_t>(index) * dim,
          static_cast<size_t>(dim) * sizeof(float));
    }
  });
  return output;
}

torch::Tensor q4_0_dequant_cpu(
    torch::Tensor payload,
    int64_t rows,
    int64_t cols) {
  TORCH_CHECK(
      !payload.is_cuda() && payload.scalar_type() == at::kByte &&
          payload.is_contiguous() && payload.dim() == 1 &&
          rows > 0 && cols > 0 && cols % 32 == 0,
      "Q4 dequant requires one contiguous CPU image and aligned columns");
  const int64_t blocks = cols / 32;
  TORCH_CHECK(
      payload.numel() == rows * blocks *
          static_cast<int64_t>(sizeof(Q4Block32)),
      "Q4 dequant payload length mismatch");
  auto output = torch::empty(
      {rows, cols},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  const auto* weights = reinterpret_cast<const Q4Block32*>(
      payload.data_ptr<uint8_t>());
  float* out = output.data_ptr<float>();
  const int64_t tiles = (rows + kQ4BlockMajorRows - 1) /
      kQ4BlockMajorRows;
  at::parallel_for(0, tiles, 1, [&](int64_t begin, int64_t end) {
    for (int64_t tile = begin; tile < end; ++tile) {
      const int64_t first_row = tile * kQ4BlockMajorRows;
      const int64_t valid_rows = std::min<int64_t>(
          kQ4BlockMajorRows, rows - first_row);
      for (int64_t block = 0; block < blocks; ++block) {
        for (int64_t local = 0; local < valid_rows; ++local) {
          const int64_t row = first_row + local;
          const Q4Block32& source = weights[
              first_row * blocks + block * valid_rows + local];
          const float scale = static_cast<float>(source.d);
          float* target = out + row * cols + block * 32;
          for (int64_t lane = 0; lane < 16; ++lane) {
            target[lane] =
                (static_cast<int>(source.qs[lane] & 0x0f) - 8) * scale;
            target[lane + 16] =
                (static_cast<int>(source.qs[lane] >> 4) - 8) * scale;
          }
        }
      }
    }
  });
  return output;
}

inline void write_packed_index(
    uint8_t* packed,
    int64_t index_offset,
    int64_t bits,
    uint16_t value) {
  const int64_t bit_offset = index_offset * bits;
  const int64_t byte_offset = bit_offset >> 3;
  const int shift = static_cast<int>(bit_offset & 7);
  const uint32_t encoded = static_cast<uint32_t>(value) << shift;
  const int bytes = (shift + bits + 7) >> 3;
  for (int byte = 0; byte < bytes; ++byte) {
    packed[byte_offset + byte] |= static_cast<uint8_t>(
        (encoded >> (byte * 8)) & 0xff);
  }
}

torch::Tensor vq_repack_block_major_cpu(
    torch::Tensor payload,
    int64_t rows,
    int64_t blocks,
    int64_t bits) {
  TORCH_CHECK(
      !payload.is_cuda() && payload.scalar_type() == at::kByte &&
          payload.is_contiguous() && payload.dim() == 1,
      "packed VQ relayout requires contiguous CPU uint8 payload");
  TORCH_CHECK(
      rows > 0 && blocks > 0 && bits >= 8 && bits <= 16,
      "packed VQ relayout metadata is invalid");
  const int64_t count = rows * blocks;
  TORCH_CHECK(
      count * bits % 8 == 0 && payload.numel() == count * bits / 8,
      "packed VQ relayout payload length mismatch");
  TORCH_CHECK(
      rows * bits % 8 == 0,
      "block-major VQ layout requires byte-aligned block rows");
  auto output = torch::zeros_like(payload);
  const uint8_t* source = payload.data_ptr<uint8_t>();
  uint8_t* destination = output.data_ptr<uint8_t>();
  const int64_t block_bytes = rows * bits / 8;
  at::parallel_for(0, blocks, 1, [&](int64_t begin, int64_t end) {
    for (int64_t block = begin; block < end; ++block) {
      uint8_t* target = destination + block * block_bytes;
      for (int64_t row = 0; row < rows; ++row) {
        const uint16_t index = read_packed_index(
            source, row * blocks + block, bits);
        write_packed_index(target, row, bits, index);
      }
    }
  });
  return output;
}

torch::Tensor vq_repack_row_tile_cpu(
    torch::Tensor payload,
    int64_t rows,
    int64_t blocks,
    int64_t bits,
    int64_t tile_rows) {
  TORCH_CHECK(
      !payload.is_cuda() && payload.scalar_type() == at::kByte &&
          payload.is_contiguous() && payload.dim() == 1,
      "packed VQ row-tile relayout requires contiguous CPU uint8 payload");
  TORCH_CHECK(
      rows > 0 && blocks > 0 && bits >= 8 && bits <= 16 &&
          tile_rows > 0 && tile_rows % 8 == 0,
      "packed VQ row-tile metadata is invalid");
  const int64_t count = rows * blocks;
  TORCH_CHECK(
      count * bits % 8 == 0 && payload.numel() == count * bits / 8,
      "packed VQ row-tile payload length mismatch");
  auto output = torch::zeros_like(payload);
  const uint8_t* source = payload.data_ptr<uint8_t>();
  uint8_t* destination = output.data_ptr<uint8_t>();
  const int64_t tiles = (rows + tile_rows - 1) / tile_rows;
  at::parallel_for(0, tiles, 1, [&](int64_t begin, int64_t end) {
    for (int64_t tile = begin; tile < end; ++tile) {
      const int64_t first_row = tile * tile_rows;
      const int64_t valid_rows = std::min(tile_rows, rows - first_row);
      const int64_t target_start = first_row * blocks;
      for (int64_t block = 0; block < blocks; ++block) {
        for (int64_t local_row = 0; local_row < valid_rows; ++local_row) {
          const uint16_t index = read_packed_index(
              source, (first_row + local_row) * blocks + block, bits);
          write_packed_index(
              destination,
              target_start + block * valid_rows + local_row,
              bits,
              index);
        }
      }
    }
  });
  return output;
}

// Compile a byte-packed VQ stream into the CPU's native uint16 row-tile
// traversal.  The result is an in-memory execution image, not a new model
// file: callers can release the source payload immediately after this call.
// Each logical index is decoded exactly once during model load, which removes
// p9--p15 bit extraction from every subsequent decode token.
torch::Tensor vq_compile_u16_row_tile_cpu(
    torch::Tensor payload,
    int64_t rows,
    int64_t blocks,
    int64_t bits,
    int64_t tile_rows) {
  TORCH_CHECK(
      !payload.is_cuda() && payload.scalar_type() == at::kByte &&
          payload.is_contiguous() && payload.dim() == 1,
      "packed VQ compilation requires contiguous CPU uint8 payload");
  TORCH_CHECK(
      rows > 0 && blocks > 0 && bits >= 8 && bits <= 16 &&
          tile_rows == 8,
      "packed VQ compilation metadata is invalid");
  const int64_t count = rows * blocks;
  TORCH_CHECK(
      count * bits % 8 == 0 && payload.numel() == count * bits / 8,
      "packed VQ compilation payload length mismatch");
  auto output = torch::empty(
      {count},
      torch::TensorOptions().dtype(torch::kUInt16).device(torch::kCPU));
  const uint8_t* source = payload.data_ptr<uint8_t>();
  uint16_t* destination = output.data_ptr<uint16_t>();
  const int64_t tiles = (rows + tile_rows - 1) / tile_rows;
  at::parallel_for(0, tiles, 1, [&](int64_t begin, int64_t end) {
    for (int64_t tile = begin; tile < end; ++tile) {
      const int64_t first_row = tile * tile_rows;
      const int64_t valid_rows = std::min(tile_rows, rows - first_row);
      const int64_t target_start = first_row * blocks;
      for (int64_t block = 0; block < blocks; ++block) {
        for (int64_t local_row = 0; local_row < valid_rows; ++local_row) {
          destination[target_start + block * valid_rows + local_row] =
              read_packed_index(
                  source,
                  (first_row + local_row) * blocks + block,
                  bits);
        }
      }
    }
  });
  return output;
}

torch::Tensor vq_compile_q4_0_cpu(
    torch::Tensor payload,
    torch::Tensor codebook,
    int64_t rows,
    int64_t blocks,
    int64_t bits) {
  TORCH_CHECK(
      !payload.is_cuda() && payload.scalar_type() == at::kByte &&
          payload.dim() == 1 && payload.is_contiguous() &&
          !codebook.is_cuda() && codebook.scalar_type() == at::kFloat &&
          codebook.dim() == 2 && codebook.is_contiguous(),
      "VQ Q4 compilation requires compact CPU indices and FP32 codebook");
  const int64_t dim = codebook.size(1);
  const int64_t cols = blocks * dim;
  TORCH_CHECK(
      rows > 0 && blocks > 0 && bits >= 8 && bits <= 16 &&
          cols % 32 == 0 && payload.numel() == rows * blocks * bits / 8,
      "VQ Q4 compilation metadata mismatch");
  const int64_t q4_blocks = cols / 32;
  auto output = torch::empty(
      {rows * q4_blocks * static_cast<int64_t>(sizeof(Q4Block32))},
      payload.options());
  bind_q4_row_shards(
      output, rows, q4_blocks * static_cast<int64_t>(sizeof(Q4Block32)));
  const uint8_t* indices = payload.data_ptr<uint8_t>();
  const float* codes = codebook.data_ptr<float>();
  auto* destination = reinterpret_cast<Q4Block32*>(
      output.data_ptr<uint8_t>());
#pragma omp parallel
  {
    const int64_t tiles =
        (rows + kQ4BlockMajorRows - 1) / kQ4BlockMajorRows;
    const auto range = q4_numa_local_row_range(tiles);
    alignas(64) float values[32];
    for (int64_t tile_index = range.first;
         tile_index < range.second; ++tile_index) {
      const int64_t first_row = tile_index * kQ4BlockMajorRows;
      const int64_t valid_rows = std::min<int64_t>(
          kQ4BlockMajorRows, rows - first_row);
      for (int64_t qblock = 0; qblock < q4_blocks; ++qblock) {
        const int64_t first_col = qblock * 32;
        for (int64_t local = 0; local < valid_rows; ++local) {
          const int64_t row = first_row + local;
          for (int64_t lane = 0; lane < 32; ++lane) {
            const int64_t column = first_col + lane;
            const int64_t vq_block = column / dim;
            const int64_t component = column - vq_block * dim;
            const uint16_t index = read_packed_index(
                indices, row * blocks + vq_block, bits);
            TORCH_INTERNAL_ASSERT(index < codebook.size(0));
            values[lane] =
                codes[static_cast<int64_t>(index) * dim + component];
          }
          quantize_q4_block32(
              values,
              destination + first_row * q4_blocks +
                  qblock * valid_rows + local);
        }
      }
    }
  }
  return output;
}

inline float direct_dot_packed(
    const float* input,
    const float* codebook,
    const uint8_t* packed,
    int64_t start_index,
    int64_t blocks,
    int64_t bits,
    int64_t dim) {
  float sum = 0.0f;
  const auto add_code = [&](int64_t block, uint16_t index) {
    const float* code = codebook + static_cast<int64_t>(index) * dim;
    const float* value = input + block * dim;
    for (int64_t lane = 0; lane < dim; ++lane) {
      sum += value[lane] * code[lane];
    }
  };
  if (bits == 8) {
    const uint8_t* row = packed + start_index;
    for (int64_t block = 0; block < blocks; ++block) {
      add_code(block, row[block]);
    }
    return sum;
  }
  if (bits == 12 && start_index % 2 == 0 && blocks % 2 == 0) {
    const uint8_t* row = packed + (start_index / 2) * 3;
    for (int64_t block = 0; block < blocks; block += 2) {
      const int64_t offset = (block / 2) * 3;
      const uint16_t first =
          static_cast<uint16_t>(row[offset]) |
          ((static_cast<uint16_t>(row[offset + 1]) & 0x0f) << 8);
      const uint16_t second =
          (static_cast<uint16_t>(row[offset + 1]) >> 4) |
          (static_cast<uint16_t>(row[offset + 2]) << 4);
      add_code(block, first);
      add_code(block + 1, second);
    }
    return sum;
  }
  if (bits == 10 && start_index % 4 == 0 && blocks % 4 == 0) {
    const uint8_t* row = packed + (start_index / 4) * 5;
    for (int64_t block = 0; block < blocks; block += 4) {
      const int64_t offset = (block / 4) * 5;
      const uint64_t word = load_u40_le(row + offset);
      add_code(block, static_cast<uint16_t>(word & 0x3ff));
      add_code(
          block + 1,
          static_cast<uint16_t>((word >> 10) & 0x3ff));
      add_code(
          block + 2,
          static_cast<uint16_t>((word >> 20) & 0x3ff));
      add_code(
          block + 3,
          static_cast<uint16_t>((word >> 30) & 0x3ff));
    }
    return sum;
  }
  if (bits == 14 && start_index % 4 == 0 && blocks % 4 == 0) {
    const uint8_t* row = packed + (start_index / 4) * 7;
    for (int64_t block = 0; block < blocks; block += 4) {
      const int64_t offset = (block / 4) * 7;
      const uint64_t word = load_u56_le(row + offset);
      add_code(block, static_cast<uint16_t>(word & 0x3fff));
      add_code(
          block + 1,
          static_cast<uint16_t>((word >> 14) & 0x3fff));
      add_code(
          block + 2,
          static_cast<uint16_t>((word >> 28) & 0x3fff));
      add_code(
          block + 3,
          static_cast<uint16_t>((word >> 42) & 0x3fff));
    }
    return sum;
  }
  for (int64_t block = 0; block < blocks; ++block) {
    add_code(
        block,
        read_packed_index(packed, start_index + block, bits));
  }
  return sum;
}

inline float direct_dot_packed_block_major(
    const float* input,
    const float* codebook,
    const uint8_t* packed,
    int64_t row,
    int64_t rows,
    int64_t blocks,
    int64_t bits,
    int64_t dim) {
  float sum = 0.0f;
  for (int64_t block = 0; block < blocks; ++block) {
    const uint16_t index = read_packed_index(
        packed, block * rows + row, bits);
    const float* code = codebook + static_cast<int64_t>(index) * dim;
    const float* value = input + block * dim;
    for (int64_t lane = 0; lane < dim; ++lane) {
      sum += value[lane] * code[lane];
    }
  }
  return sum;
}

inline float direct_dot_packed_row_tile8(
    const float* input,
    const float* codebook,
    const uint8_t* packed,
    int64_t row,
    int64_t rows,
    int64_t blocks,
    int64_t bits,
    int64_t dim) {
  const int64_t first_row = (row / 8) * 8;
  const int64_t local_row = row - first_row;
  const int64_t valid_rows = std::min<int64_t>(8, rows - first_row);
  const int64_t tile_start = first_row * blocks;
  float sum = 0.0f;
  for (int64_t block = 0; block < blocks; ++block) {
    const uint16_t index = read_packed_index(
        packed, tile_start + block * valid_rows + local_row, bits);
    const float* code = codebook + static_cast<int64_t>(index) * dim;
    const float* value = input + block * dim;
    for (int64_t lane = 0; lane < dim; ++lane) {
      sum += value[lane] * code[lane];
    }
  }
  return sum;
}

// Visit one row-major packed block for up to eight output rows at a time.
// Keeping the packed word outside the projection loop is important for p10,
// p12 and p14: one 5/3/7-byte load yields 4/2/4 indices instead of reloading
// the same word once per block.  The callback still observes blocks in their
// original order, so this only changes index decoding, not VQ arithmetic.
template <typename Callback>
inline void for_each_packed_rows8_block(
    const uint8_t* packed,
    int64_t first_row,
    int64_t valid_rows,
    int64_t blocks,
    int64_t bits,
    Callback&& callback) {
  uint16_t indices[4][8];
  if (bits == 8) {
    for (int64_t block = 0; block < blocks; ++block) {
      for (int64_t local_row = 0; local_row < valid_rows; ++local_row) {
        indices[0][local_row] = packed[
            (first_row + local_row) * blocks + block];
      }
      callback(block, indices[0]);
    }
    return;
  }
  if (bits == 16) {
    for (int64_t block = 0; block < blocks; ++block) {
      for (int64_t local_row = 0; local_row < valid_rows; ++local_row) {
        const int64_t offset =
            ((first_row + local_row) * blocks + block) * 2;
        indices[0][local_row] =
            static_cast<uint16_t>(packed[offset]) |
            (static_cast<uint16_t>(packed[offset + 1]) << 8);
      }
      callback(block, indices[0]);
    }
    return;
  }
  if (bits == 12 && blocks % 2 == 0) {
    for (int64_t block = 0; block < blocks; block += 2) {
      for (int64_t local_row = 0; local_row < valid_rows; ++local_row) {
        const int64_t start_index =
            (first_row + local_row) * blocks + block;
        const uint8_t* word = packed + (start_index / 2) * 3;
        indices[0][local_row] =
            static_cast<uint16_t>(word[0]) |
            ((static_cast<uint16_t>(word[1]) & 0x0f) << 8);
        indices[1][local_row] =
            (static_cast<uint16_t>(word[1]) >> 4) |
            (static_cast<uint16_t>(word[2]) << 4);
      }
      callback(block, indices[0]);
      callback(block + 1, indices[1]);
    }
    return;
  }
  if ((bits == 10 || bits == 14) && blocks % 4 == 0) {
    const int64_t bytes = bits == 10 ? 5 : 7;
    const uint64_t mask = bits == 10 ? 0x3ff : 0x3fff;
    for (int64_t block = 0; block < blocks; block += 4) {
      for (int64_t local_row = 0; local_row < valid_rows; ++local_row) {
        const int64_t start_index =
            (first_row + local_row) * blocks + block;
        const uint8_t* address =
            packed + (start_index / 4) * bytes;
        const uint64_t word = bits == 10
            ? load_u40_le(address)
            : load_u56_le(address);
        indices[0][local_row] = static_cast<uint16_t>(word & mask);
        indices[1][local_row] =
            static_cast<uint16_t>((word >> bits) & mask);
        indices[2][local_row] =
            static_cast<uint16_t>((word >> (bits * 2)) & mask);
        indices[3][local_row] =
            static_cast<uint16_t>((word >> (bits * 3)) & mask);
      }
      callback(block, indices[0]);
      callback(block + 1, indices[1]);
      callback(block + 2, indices[2]);
      callback(block + 3, indices[3]);
    }
    return;
  }
  for (int64_t block = 0; block < blocks; ++block) {
    for (int64_t local_row = 0; local_row < valid_rows; ++local_row) {
      indices[0][local_row] = read_packed_index(
          packed, (first_row + local_row) * blocks + block, bits);
    }
    callback(block, indices[0]);
  }
}

inline void decode_contiguous_indices(
    const uint8_t* packed,
    int64_t start_index,
    int64_t count,
    int64_t bits,
    uint16_t* indices) {
  if (bits == 8) {
    for (int64_t index = 0; index < count; ++index) {
      indices[index] = packed[start_index + index];
    }
    return;
  }
  if (bits == 16) {
    const uint8_t* source = packed + start_index * 2;
    for (int64_t index = 0; index < count; ++index) {
      indices[index] = static_cast<uint16_t>(source[index * 2]) |
          (static_cast<uint16_t>(source[index * 2 + 1]) << 8);
    }
    return;
  }
  if (bits == 12 && start_index % 2 == 0 && count % 2 == 0) {
    const uint8_t* source = packed + (start_index / 2) * 3;
    for (int64_t index = 0; index < count; index += 2) {
      indices[index] = static_cast<uint16_t>(source[0]) |
          ((static_cast<uint16_t>(source[1]) & 0x0f) << 8);
      indices[index + 1] = (static_cast<uint16_t>(source[1]) >> 4) |
          (static_cast<uint16_t>(source[2]) << 4);
      source += 3;
    }
    return;
  }
  if ((bits == 10 || bits == 14) &&
      start_index % 4 == 0 && count % 4 == 0) {
    const int64_t bytes = bits == 10 ? 5 : 7;
    const uint64_t mask = bits == 10 ? 0x3ff : 0x3fff;
    const uint8_t* source = packed + (start_index / 4) * bytes;
    for (int64_t index = 0; index < count; index += 4) {
      const uint64_t word = bits == 10
          ? load_u40_le(source) : load_u56_le(source);
      indices[index] = static_cast<uint16_t>(word & mask);
      indices[index + 1] =
          static_cast<uint16_t>((word >> bits) & mask);
      indices[index + 2] =
          static_cast<uint16_t>((word >> (bits * 2)) & mask);
      indices[index + 3] =
          static_cast<uint16_t>((word >> (bits * 3)) & mask);
      source += bytes;
    }
    return;
  }
  for (int64_t index = 0; index < count; ++index) {
    indices[index] = read_packed_index(
        packed, start_index + index, bits);
  }
}

template <typename Callback>
inline void for_each_packed_row_tile8_block(
    const uint8_t* packed,
    int64_t first_row,
    int64_t valid_rows,
    int64_t blocks,
    int64_t bits,
    Callback&& callback) {
  uint16_t indices[8];
  const int64_t tile_start = first_row * blocks;
  for (int64_t block = 0; block < blocks; ++block) {
    decode_contiguous_indices(
        packed,
        tile_start + block * valid_rows,
        valid_rows,
        bits,
        indices);
    callback(block, indices);
  }
}

inline void direct_dot_packed_rows8(
    const float* input,
    const float* codebook,
    const uint8_t* packed,
    int64_t first_row,
    int64_t valid_rows,
    int64_t blocks,
    int64_t bits,
    int64_t dim,
    int64_t layout,
    float* output) {
  if (layout == 3) {
    const int64_t columns = blocks * dim;
    TORCH_INTERNAL_ASSERT(columns % 32 == 0);
    const int64_t q4_blocks = columns / 32;
    std::vector<Q8Block32> activation(q4_blocks);
    quantize_q8_row(input, columns, activation.data());
    const auto* weights = reinterpret_cast<const Q4Block32*>(packed);
    if (first_row % kQ4BlockMajorRows == 0) {
      q4_q8_block_major_rows8(
          weights, first_row, valid_rows, activation.data(), q4_blocks,
          output);
    } else {
      // Scalar/fallback projection paths request one logical row at a time.
      // Q4 images are stored in eight-row tiles, so evaluate the containing
      // tile and copy the requested lanes instead of asserting alignment.
      int64_t completed = 0;
      while (completed < valid_rows) {
        const int64_t logical_row = first_row + completed;
        const int64_t tile_row =
            logical_row & ~(kQ4BlockMajorRows - 1);
        const int64_t lane = logical_row - tile_row;
        const int64_t take = std::min<int64_t>(
            valid_rows - completed, kQ4BlockMajorRows - lane);
        alignas(64) float tile_values[kQ4BlockMajorRows];
        q4_q8_block_major_rows8(
            weights, tile_row, kQ4BlockMajorRows, activation.data(),
            q4_blocks, tile_values);
        std::copy(
            tile_values + lane, tile_values + lane + take,
            output + completed);
        completed += take;
      }
    }
    return;
  }
  const auto visit = [&](auto&& callback) {
    if (layout == 2) {
      for_each_packed_row_tile8_block(
          packed, first_row, valid_rows, blocks, bits,
          std::forward<decltype(callback)>(callback));
    } else {
      for_each_packed_rows8_block(
          packed, first_row, valid_rows, blocks, bits,
          std::forward<decltype(callback)>(callback));
    }
  };
#if defined(__AVX512F__)
  if (dim == 16) {
    __m512 accumulated[8];
    for (int64_t row = 0; row < valid_rows; ++row) {
      accumulated[row] = _mm512_setzero_ps();
    }
    visit([&](int64_t block, const uint16_t* indices) {
      const __m512 activation = _mm512_loadu_ps(input + block * dim);
      for (int64_t local_row = 0; local_row < valid_rows; ++local_row) {
        accumulated[local_row] = _mm512_fmadd_ps(
            activation,
            _mm512_loadu_ps(
                codebook + static_cast<int64_t>(indices[local_row]) * dim),
            accumulated[local_row]);
      }
    });
    for (int64_t row = 0; row < valid_rows; ++row) {
      output[row] = _mm512_reduce_add_ps(accumulated[row]);
    }
    return;
  }
#endif
#if defined(__AVX2__)
  if (dim == 8) {
    __m256 accumulated[8];
    for (int64_t row = 0; row < valid_rows; ++row) {
      accumulated[row] = _mm256_setzero_ps();
    }
    visit([&](int64_t block, const uint16_t* indices) {
      const __m256 activation = _mm256_loadu_ps(input + block * dim);
      for (int64_t local_row = 0; local_row < valid_rows; ++local_row) {
        accumulated[local_row] = _mm256_fmadd_ps(
            activation,
            _mm256_loadu_ps(
                codebook + static_cast<int64_t>(indices[local_row]) * dim),
            accumulated[local_row]);
      }
    });
    for (int64_t row = 0; row < valid_rows; ++row) {
      alignas(32) float lanes[8];
      _mm256_store_ps(lanes, accumulated[row]);
      float sum = 0.0f;
      for (float lane : lanes) {
        sum += lane;
      }
      output[row] = sum;
    }
    return;
  }
  if (dim == 4) {
    __m128 accumulated[8];
    for (int64_t row = 0; row < valid_rows; ++row) {
      accumulated[row] = _mm_setzero_ps();
    }
    visit([&](int64_t block, const uint16_t* indices) {
      const __m128 activation = _mm_loadu_ps(input + block * dim);
      for (int64_t local_row = 0; local_row < valid_rows; ++local_row) {
        accumulated[local_row] = _mm_fmadd_ps(
            activation,
            _mm_loadu_ps(
                codebook + static_cast<int64_t>(indices[local_row]) * dim),
            accumulated[local_row]);
      }
    });
    for (int64_t row = 0; row < valid_rows; ++row) {
      alignas(16) float lanes[4];
      _mm_store_ps(lanes, accumulated[row]);
      output[row] = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    }
    return;
  }
#endif
  for (int64_t local_row = 0; local_row < valid_rows; ++local_row) {
    output[local_row] = direct_dot_packed(
        input,
        codebook,
        packed,
        (first_row + local_row) * blocks,
        blocks,
        bits,
        dim);
  }
}

// llama.cpp keeps the activation block live while several matrix rows consume
// it.  Apply the same execution rule to CCCP's two independent input
// projections after the exact byte-preserving row-tile repack: decode Gate
// and Up indices together, load the activation vector once, and accumulate
// both outputs before moving to the next block.  The logical accumulation
// order of each projection is unchanged.
inline bool direct_dot_packed_gate_up_row_tile8(
    const float* input,
    const float* gate_codebook,
    const uint8_t* gate_packed,
    int64_t gate_blocks,
    int64_t gate_bits,
    int64_t gate_dim,
    int64_t gate_layout,
    const float* up_codebook,
    const uint8_t* up_packed,
    int64_t up_blocks,
    int64_t up_bits,
    int64_t up_dim,
    int64_t up_layout,
    int64_t first_row,
    int64_t valid_rows,
    float* gate_output,
    float* up_output) {
  if (gate_layout != 2 || up_layout != 2 ||
      gate_blocks != up_blocks || gate_dim != up_dim) {
    return false;
  }
  const int64_t blocks = gate_blocks;
  const int64_t dim = gate_dim;
  const int64_t tile_start = first_row * blocks;
  alignas(64) uint16_t gate_indices[8];
  alignas(64) uint16_t up_indices[8];
#if defined(__AVX512F__)
  if (dim == 16) {
    __m512 gate_accumulated[8];
    __m512 up_accumulated[8];
    for (int64_t row = 0; row < valid_rows; ++row) {
      gate_accumulated[row] = _mm512_setzero_ps();
      up_accumulated[row] = _mm512_setzero_ps();
    }
    for (int64_t block = 0; block < blocks; ++block) {
      decode_contiguous_indices(
          gate_packed, tile_start + block * valid_rows,
          valid_rows, gate_bits, gate_indices);
      decode_contiguous_indices(
          up_packed, tile_start + block * valid_rows,
          valid_rows, up_bits, up_indices);
      const __m512 activation = _mm512_loadu_ps(input + block * dim);
      for (int64_t row = 0; row < valid_rows; ++row) {
        gate_accumulated[row] = _mm512_fmadd_ps(
            activation,
            _mm512_loadu_ps(
                gate_codebook +
                static_cast<int64_t>(gate_indices[row]) * dim),
            gate_accumulated[row]);
        up_accumulated[row] = _mm512_fmadd_ps(
            activation,
            _mm512_loadu_ps(
                up_codebook +
                static_cast<int64_t>(up_indices[row]) * dim),
            up_accumulated[row]);
      }
    }
    for (int64_t row = 0; row < valid_rows; ++row) {
      gate_output[row] = _mm512_reduce_add_ps(gate_accumulated[row]);
      up_output[row] = _mm512_reduce_add_ps(up_accumulated[row]);
    }
    return true;
  }
#endif
#if defined(__AVX2__)
  if (dim == 8) {
    __m256 gate_accumulated[8];
    __m256 up_accumulated[8];
    for (int64_t row = 0; row < valid_rows; ++row) {
      gate_accumulated[row] = _mm256_setzero_ps();
      up_accumulated[row] = _mm256_setzero_ps();
    }
    for (int64_t block = 0; block < blocks; ++block) {
      decode_contiguous_indices(
          gate_packed, tile_start + block * valid_rows,
          valid_rows, gate_bits, gate_indices);
      decode_contiguous_indices(
          up_packed, tile_start + block * valid_rows,
          valid_rows, up_bits, up_indices);
      const __m256 activation = _mm256_loadu_ps(input + block * dim);
      for (int64_t row = 0; row < valid_rows; ++row) {
        gate_accumulated[row] = _mm256_fmadd_ps(
            activation,
            _mm256_loadu_ps(
                gate_codebook +
                static_cast<int64_t>(gate_indices[row]) * dim),
            gate_accumulated[row]);
        up_accumulated[row] = _mm256_fmadd_ps(
            activation,
            _mm256_loadu_ps(
                up_codebook +
                static_cast<int64_t>(up_indices[row]) * dim),
            up_accumulated[row]);
      }
    }
    for (int64_t row = 0; row < valid_rows; ++row) {
      alignas(32) float gate_lanes[8];
      alignas(32) float up_lanes[8];
      _mm256_store_ps(gate_lanes, gate_accumulated[row]);
      _mm256_store_ps(up_lanes, up_accumulated[row]);
      float gate_sum = 0.0f;
      float up_sum = 0.0f;
      for (int64_t lane = 0; lane < 8; ++lane) {
        gate_sum += gate_lanes[lane];
        up_sum += up_lanes[lane];
      }
      gate_output[row] = gate_sum;
      up_output[row] = up_sum;
    }
    return true;
  }
  if (dim == 4) {
    __m128 gate_accumulated[8];
    __m128 up_accumulated[8];
    for (int64_t row = 0; row < valid_rows; ++row) {
      gate_accumulated[row] = _mm_setzero_ps();
      up_accumulated[row] = _mm_setzero_ps();
    }
    for (int64_t block = 0; block < blocks; ++block) {
      decode_contiguous_indices(
          gate_packed, tile_start + block * valid_rows,
          valid_rows, gate_bits, gate_indices);
      decode_contiguous_indices(
          up_packed, tile_start + block * valid_rows,
          valid_rows, up_bits, up_indices);
      const __m128 activation = _mm_loadu_ps(input + block * dim);
      for (int64_t row = 0; row < valid_rows; ++row) {
        gate_accumulated[row] = _mm_fmadd_ps(
            activation,
            _mm_loadu_ps(
                gate_codebook +
                static_cast<int64_t>(gate_indices[row]) * dim),
            gate_accumulated[row]);
        up_accumulated[row] = _mm_fmadd_ps(
            activation,
            _mm_loadu_ps(
                up_codebook +
                static_cast<int64_t>(up_indices[row]) * dim),
            up_accumulated[row]);
      }
    }
    for (int64_t row = 0; row < valid_rows; ++row) {
      alignas(16) float gate_lanes[4];
      alignas(16) float up_lanes[4];
      _mm_store_ps(gate_lanes, gate_accumulated[row]);
      _mm_store_ps(up_lanes, up_accumulated[row]);
      gate_output[row] = gate_lanes[0] + gate_lanes[1] +
          gate_lanes[2] + gate_lanes[3];
      up_output[row] = up_lanes[0] + up_lanes[1] +
          up_lanes[2] + up_lanes[3];
    }
    return true;
  }
#endif
  return false;
}

torch::Tensor vq_gemv_packed_list_cpu(
    torch::Tensor x_rows,
    std::vector<torch::Tensor> packed_list,
    torch::Tensor codebook,
    int64_t rows,
    int64_t blocks,
    int64_t bits,
    bool allow_direct) {
  TORCH_CHECK(!x_rows.is_cuda() && !codebook.is_cuda(),
              "packed VQ list operands must be on CPU");
  TORCH_CHECK(x_rows.dim() == 2, "x_rows must be [N|1,C]");
  TORCH_CHECK(codebook.dim() == 2, "codebook must be [K,D]");
  TORCH_CHECK(!packed_list.empty(), "packed VQ list cannot be empty");
  TORCH_CHECK(
      bits >= 8 && bits <= 16,
      "packed VQ width must be in [8,16]");
  TORCH_CHECK(rows > 0 && blocks > 0,
              "packed VQ rows and blocks must be positive");
  const int64_t n = static_cast<int64_t>(packed_list.size());
  const int64_t expected_bits = rows * blocks * bits;
  TORCH_CHECK(expected_bits % 8 == 0,
              "packed VQ payload must be byte aligned");
  const int64_t expected_bytes = expected_bits / 8;
  std::vector<torch::Tensor> payloads;
  std::vector<const uint8_t*> payload_ptrs;
  payloads.reserve(n);
  payload_ptrs.reserve(n);
  for (auto& packed : packed_list) {
    TORCH_CHECK(!packed.is_cuda() && packed.scalar_type() == at::kByte,
                "packed VQ payloads must be CPU uint8 tensors");
    TORCH_CHECK(packed.numel() == expected_bytes,
                "packed VQ payload length mismatch");
    payloads.push_back(packed.contiguous().reshape({-1}));
    payload_ptrs.push_back(payloads.back().data_ptr<uint8_t>());
  }
  TORCH_CHECK(x_rows.size(0) == 1 || x_rows.size(0) == n,
              "packed VQ input batch must be 1 or expert count");
  const int64_t codes = codebook.size(0);
  const int64_t dim = codebook.size(1);
  TORCH_CHECK(codes <= (int64_t{1} << bits),
              "packed width cannot represent every codebook entry");
  TORCH_CHECK(x_rows.size(1) == blocks * dim,
              "packed VQ input width mismatch");

  auto x = x_rows.to(torch::kFloat32).contiguous();
  auto cb = codebook.to(torch::kFloat32).contiguous();
  const int64_t score_n = x.size(0);
  const bool use_direct =
      allow_direct &&
      (score_n == 1 || score_n == n) &&
      n * rows * dim < score_n * codes * dim + n * rows;
#if defined(__AVX512F__)
  // Sparse Top-K decode only needs the selected experts' row indices.
  // Vectorise the direct path for every packed width p8--p16; heterogeneous
  // archives frequently choose p11/p12/p13 for a single selected expert, and
  // leaving those widths on the scalar direct-dot fallback dominated DSV4
  // CPU decode.  Index extraction remains byte-packed and row-local.
  // Vectorise over 16 output rows and
  // gather the referenced transposed code vectors directly.  Keeping one
  // block-local dot before accumulating preserves the score-then-lookup
  // floating-point order while avoiding a blocks*K score table.
  if (use_direct) {
    auto out = torch::empty(
        {n, rows},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto cb_transposed = cached_transposed_codebook(cb);
    const float* xp = x.data_ptr<float>();
    const float* cp = cb_transposed.data_ptr<float>();
    float* op = out.data_ptr<float>();
    const int64_t row_groups = (rows + 15) / 16;
    at::parallel_for(0, n * row_groups, 1, [&](int64_t begin, int64_t end) {
      alignas(64) int32_t gathered_indices[4][16];
      for (int64_t item = begin; item < end; ++item) {
        const int64_t batch = item / row_groups;
        const int64_t row = (item - batch * row_groups) * 16;
        const int64_t valid = std::min<int64_t>(16, rows - row);
        const uint8_t* payload = payload_ptrs[batch];
        const int64_t input_batch = score_n == 1 ? 0 : batch;
        const float* input = xp + input_batch * blocks * dim;
        __m512 accumulated = _mm512_setzero_ps();
        const auto accumulate_block = [&](const int32_t* block_indices,
                                          int64_t block) {
          const __m512i indices = _mm512_load_si512(block_indices);
          __m512 block_score = _mm512_setzero_ps();
          for (int64_t lane = 0; lane < dim; ++lane) {
            const __m512 code_values = _mm512_i32gather_ps(
                indices,
                cp + lane * codes,
                sizeof(float));
            block_score = _mm512_fmadd_ps(
                _mm512_set1_ps(input[block * dim + lane]),
                code_values,
                block_score);
          }
          accumulated = _mm512_add_ps(accumulated, block_score);
        };
        if (bits == 14 && blocks % 4 == 0) {
          for (int64_t block = 0; block < blocks; block += 4) {
            for (int64_t lane = 0; lane < valid; ++lane) {
              const int64_t index_offset =
                  (row + lane) * blocks + block;
              const int64_t offset = (index_offset / 4) * 7;
              const uint64_t word = load_u56_le(payload + offset);
              gathered_indices[0][lane] =
                  static_cast<int32_t>(word & 0x3fff);
              gathered_indices[1][lane] =
                  static_cast<int32_t>((word >> 14) & 0x3fff);
              gathered_indices[2][lane] =
                  static_cast<int32_t>((word >> 28) & 0x3fff);
              gathered_indices[3][lane] =
                  static_cast<int32_t>((word >> 42) & 0x3fff);
            }
            for (int subblock = 0; subblock < 4; ++subblock) {
              for (int64_t lane = valid; lane < 16; ++lane) {
                gathered_indices[subblock][lane] = 0;
              }
              accumulate_block(
                  gathered_indices[subblock], block + subblock);
            }
          }
        } else {
          for (int64_t block = 0; block < blocks; ++block) {
            for (int64_t lane = 0; lane < valid; ++lane) {
              const int64_t index_offset =
                  (row + lane) * blocks + block;
              if (bits == 16) {
                const int64_t offset = index_offset * 2;
                gathered_indices[0][lane] =
                    static_cast<int32_t>(payload[offset]) |
                    (static_cast<int32_t>(payload[offset + 1]) << 8);
              } else {
                gathered_indices[0][lane] = static_cast<int32_t>(
                    read_packed_index(payload, index_offset, bits));
              }
            }
            for (int64_t lane = valid; lane < 16; ++lane) {
              gathered_indices[0][lane] = 0;
            }
            accumulate_block(gathered_indices[0], block);
          }
        }
        if (valid == 16) {
          _mm512_storeu_ps(op + batch * rows + row, accumulated);
        } else {
          const __mmask16 mask =
              static_cast<__mmask16>((uint32_t{1} << valid) - 1);
          _mm512_mask_storeu_ps(
              op + batch * rows + row, mask, accumulated);
        }
      }
    });
    return out;
  }
#endif
  if (use_direct) {
    auto out = torch::empty(
        {n, rows},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    const float* xp = x.data_ptr<float>();
    const float* cp = cb.data_ptr<float>();
    float* op = out.data_ptr<float>();
    at::parallel_for(0, n * rows, 1, [&](int64_t begin, int64_t end) {
      for (int64_t item = begin; item < end; ++item) {
        const int64_t batch = item / rows;
        const int64_t row = item - batch * rows;
        const uint8_t* payload = payload_ptrs[batch];
        const int64_t input_batch = score_n == 1 ? 0 : batch;
        const float* input = xp + input_batch * blocks * dim;
        op[item] = direct_dot_packed(
            input,
            cp,
            payload,
            row * blocks,
            blocks,
            bits,
            dim);
      }
    });
    return out;
  }
  auto scores = torch::empty(
      {score_n, blocks, codes},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto cb_transposed = cached_transposed_codebook(cb);
  const float* xp = x.data_ptr<float>();
  const float* cp = cb_transposed.data_ptr<float>();
  float* scorep = scores.data_ptr<float>();
  at::parallel_for(0, score_n * blocks, 8, [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t batch = item / blocks;
      const int64_t block = item - batch * blocks;
      const float* xv = xp + (batch * blocks + block) * dim;
      float* score = scorep + (batch * blocks + block) * codes;
      codebook_scores(xv, cp, score, codes, dim);
    }
  });

  auto out = torch::empty(
      {n, rows},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  float* op = out.data_ptr<float>();
  at::parallel_for(0, n * rows, 1, [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t batch = item / rows;
      const int64_t row = item - batch * rows;
      const int64_t score_batch = score_n == 1 ? 0 : batch;
      op[item] = lookup_sum_packed(
          scorep + score_batch * blocks * codes,
          payload_ptrs[batch],
          row * blocks,
          blocks,
          codes,
          bits);
    }
  });
  return out;
}

torch::Tensor moe_packed_three_projection_cpu(
    torch::Tensor x_row,
    std::vector<torch::Tensor> gate_payloads,
    torch::Tensor gate_codebook,
    int64_t gate_rows,
    int64_t gate_blocks,
    int64_t gate_bits,
    std::vector<torch::Tensor> up_payloads,
    torch::Tensor up_codebook,
    int64_t up_rows,
    int64_t up_blocks,
    int64_t up_bits,
    std::vector<torch::Tensor> down_payloads,
    torch::Tensor down_codebook,
    int64_t down_rows,
    int64_t down_blocks,
    int64_t down_bits,
    torch::Tensor route_weights,
    double limit,
    std::string activation,
    double beta,
    double linear_beta,
    torch::Tensor workspace,
    torch::Tensor result) {
  const int64_t experts =
      static_cast<int64_t>(gate_payloads.size());
  TORCH_CHECK(
      experts > 0 && experts <= 16 &&
          static_cast<int64_t>(up_payloads.size()) == experts &&
          static_cast<int64_t>(down_payloads.size()) == experts,
      "three-projection packed MoE operand counts must match");
  TORCH_CHECK(
      !x_row.is_cuda() && x_row.dim() == 2 &&
          x_row.size(0) == 1,
      "three-projection packed MoE requires one CPU row");
  TORCH_CHECK(
      !route_weights.is_cuda() &&
          route_weights.numel() == experts,
      "three-projection route weights must match Top-K");
  TORCH_CHECK(
      activation == "situ" || activation == "silu" ||
          activation == "swiglu",
      "three-projection activation must be situ, silu, or swiglu");
  TORCH_CHECK(
      gate_rows == up_rows &&
          down_rows == x_row.size(1) &&
          gate_blocks * gate_codebook.size(1) == x_row.size(1) &&
          up_blocks * up_codebook.size(1) == x_row.size(1) &&
          down_blocks * down_codebook.size(1) == gate_rows,
      "three-projection logical matrix shapes do not match");

  // Small/medium codebooks use score-then-lookup.  Keep every stage in one
  // OpenMP team so the public Top-K call does not enter and leave the host
  // worker pool seven times per layer.  The packed p8--p13 payload remains
  // byte-exact and row-major; no expanded index or weight matrix is created.
  // p14/p16 retain the sparse direct-gather path below.
  const bool score_pipeline =
      packed_single_team_enabled() &&
      gate_bits <= 13 && up_bits <= 13 && down_bits <= 13 &&
      x_row.scalar_type() == at::kFloat && x_row.is_contiguous() &&
      gate_codebook.scalar_type() == at::kFloat &&
      up_codebook.scalar_type() == at::kFloat &&
      down_codebook.scalar_type() == at::kFloat &&
      gate_codebook.is_contiguous() && up_codebook.is_contiguous() &&
      down_codebook.is_contiguous();
  if (score_pipeline) {
    const int64_t gate_codes = gate_codebook.size(0);
    const int64_t up_codes = up_codebook.size(0);
    const int64_t down_codes = down_codebook.size(0);
    TORCH_CHECK(
        gate_codes <= (int64_t{1} << gate_bits) &&
            up_codes <= (int64_t{1} << up_bits) &&
            down_codes <= (int64_t{1} << down_bits),
        "packed width cannot represent projection codebook");
    std::vector<const uint8_t*> gate_ptrs(experts);
    std::vector<const uint8_t*> up_ptrs(experts);
    std::vector<const uint8_t*> down_ptrs(experts);
    const auto collect_payloads = [&](
        const std::vector<torch::Tensor>& payloads,
        int64_t rows,
        int64_t blocks,
        int64_t bits,
        std::vector<const uint8_t*>& pointers) {
      const int64_t expected_bits = rows * blocks * bits;
      TORCH_CHECK(
          expected_bits % 8 == 0,
          "packed projection payload must be byte aligned");
      const int64_t expected_bytes = expected_bits / 8;
      for (int64_t expert = 0; expert < experts; ++expert) {
        const auto& payload = payloads[expert];
        TORCH_CHECK(
            !payload.is_cuda() && payload.scalar_type() == at::kByte &&
                payload.is_contiguous() &&
                payload.numel() == expected_bytes,
            "packed projection payload mismatch");
        pointers[expert] = payload.data_ptr<uint8_t>();
      }
    };
    collect_payloads(
        gate_payloads,
        gate_rows,
        gate_blocks,
        gate_bits,
        gate_ptrs);
    collect_payloads(
        up_payloads,
        up_rows,
        up_blocks,
        up_bits,
        up_ptrs);
    collect_payloads(
        down_payloads,
        down_rows,
        down_blocks,
        down_bits,
        down_ptrs);
    TORCH_CHECK(
        !result.is_cuda() && result.scalar_type() == at::kFloat &&
            result.is_contiguous() && result.numel() >= down_rows,
        "three-projection result workspace is invalid");

    auto gate_cb = cached_transposed_codebook(gate_codebook);
    auto up_cb = cached_transposed_codebook(up_codebook);
    auto down_cb = cached_transposed_codebook(down_codebook);
    auto route = route_weights.to(torch::kFloat32).contiguous();
    const int64_t score_count = std::max(
        gate_blocks * gate_codes,
        up_blocks * up_codes);
    const int64_t gate_offset = score_count;
    const int64_t up_offset = gate_offset + experts * gate_rows;
    const int64_t down_score_offset =
        up_offset + experts * up_rows;
    const int64_t down_value_offset =
        down_score_offset + experts * down_blocks * down_codes;
    TORCH_CHECK(
        !workspace.is_cuda() &&
            workspace.scalar_type() == at::kFloat &&
            workspace.is_contiguous() &&
            workspace.numel() >= down_value_offset + experts * down_rows,
        "three-projection persistent workspace is too small");
    float* workspacep = workspace.data_ptr<float>();
    float* scorep = workspacep;
    float* gatep = workspacep + gate_offset;
    float* upp = workspacep + up_offset;
    float* down_scorep = workspacep + down_score_offset;
    float* downp = workspacep + down_value_offset;
    float* resultp = result.data_ptr<float>();
    const float* xp = x_row.data_ptr<float>();
    const float* gate_cbp = gate_cb.data_ptr<float>();
    const float* up_cbp = up_cb.data_ptr<float>();
    const float* down_cbp = down_cb.data_ptr<float>();
    const float* routep = route.data_ptr<float>();
    const float activation_limit = static_cast<float>(limit);
    const float situ_beta = static_cast<float>(beta);
    const float situ_linear_beta = static_cast<float>(linear_beta);
    const bool rows16 = packed_rows16_enabled();
    double phase_markers[6] = {wall_seconds(), 0.0, 0.0, 0.0, 0.0, 0.0};

#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (int64_t block = 0; block < gate_blocks; ++block) {
        codebook_scores(
            xp + block * gate_codebook.size(1),
            gate_cbp,
            scorep + block * gate_codes,
            gate_codes,
            gate_codebook.size(1));
      }
      if (rows16 &&
          (gate_bits == 8 || gate_bits == 10 || gate_bits == 12)) {
#pragma omp for schedule(static)
        for (int64_t item = 0;
             item < experts * ((gate_rows + 15) / 16);
             ++item) {
          const int64_t groups = (gate_rows + 15) / 16;
          const int64_t expert = item / groups;
          const int64_t first_row = (item - expert * groups) * 16;
          lookup_sum_packed_rows16(
              scorep, gate_ptrs[expert], first_row, gate_rows,
              gate_blocks, gate_codes, gate_bits,
              gatep + expert * gate_rows);
        }
      } else {
#pragma omp for schedule(static)
        for (int64_t item = 0; item < experts * gate_rows; ++item) {
          const int64_t expert = item / gate_rows;
          const int64_t row = item - expert * gate_rows;
          gatep[item] = lookup_sum_packed(
              scorep, gate_ptrs[expert], row * gate_blocks, gate_blocks,
              gate_codes, gate_bits);
        }
      }
#pragma omp single
      { phase_markers[1] = wall_seconds(); }

#pragma omp for schedule(static)
      for (int64_t block = 0; block < up_blocks; ++block) {
        codebook_scores(
            xp + block * up_codebook.size(1),
            up_cbp,
            scorep + block * up_codes,
            up_codes,
            up_codebook.size(1));
      }
      if (rows16 &&
          (up_bits == 8 || up_bits == 10 || up_bits == 12)) {
#pragma omp for schedule(static)
        for (int64_t item = 0;
             item < experts * ((up_rows + 15) / 16);
             ++item) {
          const int64_t groups = (up_rows + 15) / 16;
          const int64_t expert = item / groups;
          const int64_t first_row = (item - expert * groups) * 16;
          lookup_sum_packed_rows16(
              scorep, up_ptrs[expert], first_row, up_rows,
              up_blocks, up_codes, up_bits,
              upp + expert * up_rows);
        }
      } else {
#pragma omp for schedule(static)
        for (int64_t item = 0; item < experts * up_rows; ++item) {
          const int64_t expert = item / up_rows;
          const int64_t row = item - expert * up_rows;
          upp[item] = lookup_sum_packed(
              scorep, up_ptrs[expert], row * up_blocks, up_blocks,
              up_codes, up_bits);
        }
      }
#pragma omp single
      { phase_markers[2] = wall_seconds(); }

#pragma omp for schedule(static)
      for (int64_t item = 0; item < experts * gate_rows; ++item) {
        float gate = gatep[item];
        float up = upp[item];
        if (activation_limit != 0.0f) {
          gate = std::min(gate, activation_limit);
          up = std::max(
              -activation_limit, std::min(up, activation_limit));
        }
        if (activation == "situ") {
          float linear = up;
          if (situ_linear_beta > 0.0f) {
            linear = situ_linear_beta *
                std::tanh(up / situ_linear_beta);
          }
          gatep[item] =
              situ_beta * std::tanh(gate / situ_beta) *
              (1.0f / (1.0f + std::exp(-gate))) * linear;
        } else {
          gatep[item] =
              gate * (1.0f / (1.0f + std::exp(-gate))) * up;
        }
      }
#pragma omp single
      { phase_markers[3] = wall_seconds(); }

#pragma omp for schedule(static)
      for (int64_t item = 0; item < experts * down_blocks; ++item) {
        const int64_t expert = item / down_blocks;
        const int64_t block = item - expert * down_blocks;
        codebook_scores(
            gatep + expert * gate_rows +
                block * down_codebook.size(1),
            down_cbp,
            down_scorep + item * down_codes,
            down_codes,
            down_codebook.size(1));
      }
      if (rows16 &&
          (down_bits == 8 || down_bits == 10 || down_bits == 12)) {
#pragma omp for schedule(static)
        for (int64_t item = 0;
             item < experts * ((down_rows + 15) / 16);
             ++item) {
          const int64_t groups = (down_rows + 15) / 16;
          const int64_t expert = item / groups;
          const int64_t first_row = (item - expert * groups) * 16;
          lookup_sum_packed_rows16(
              down_scorep + expert * down_blocks * down_codes,
              down_ptrs[expert], first_row, down_rows,
              down_blocks, down_codes, down_bits,
              downp + expert * down_rows);
        }
      } else {
#pragma omp for schedule(static)
        for (int64_t item = 0; item < experts * down_rows; ++item) {
          const int64_t expert = item / down_rows;
          const int64_t row = item - expert * down_rows;
          downp[item] = lookup_sum_packed(
              down_scorep + expert * down_blocks * down_codes,
              down_ptrs[expert], row * down_blocks, down_blocks,
              down_codes, down_bits);
        }
      }
#pragma omp single
      { phase_markers[4] = wall_seconds(); }

#pragma omp for schedule(static)
      for (int64_t row = 0; row < down_rows; ++row) {
        float value = 0.0f;
        for (int64_t expert = 0; expert < experts; ++expert) {
          value += downp[expert * down_rows + row] * routep[expert];
        }
        resultp[row] = value;
      }
#pragma omp single
      { phase_markers[5] = wall_seconds(); }
    }
    for (int64_t phase = 0; phase < 5; ++phase) {
      three_projection_phase_seconds[phase] +=
          phase_markers[phase + 1] - phase_markers[phase];
    }
    ++three_projection_phase_calls;
    return result.narrow(0, 0, down_rows);
  }

  const double gate_started = wall_seconds();
  // Gate and Up deliberately keep separate score pages. Combining their
  // unrelated random-gather streams regresses dual-socket LLC locality.
  auto gate_values = vq_gemv_packed_list_cpu(
      x_row,
      std::move(gate_payloads),
      gate_codebook,
      gate_rows,
      gate_blocks,
      gate_bits,
      true);
  const double up_started = wall_seconds();
  auto up_values = vq_gemv_packed_list_cpu(
      x_row,
      std::move(up_payloads),
      up_codebook,
      up_rows,
      up_blocks,
      up_bits,
      true);
  const double activation_started = wall_seconds();
  TORCH_CHECK(
      gate_values.scalar_type() == at::kFloat &&
          up_values.scalar_type() == at::kFloat &&
          gate_values.is_contiguous() && up_values.is_contiguous() &&
          gate_values.sizes() == up_values.sizes(),
      "three-projection activation workspace is invalid");
  // Reuse the gate output as the activation workspace.  This keeps the whole
  // Top-K activation in the public native operator and avoids clone/sigmoid/
  // multiply tensor launches while retaining only O(TopK*intermediate)
  // temporary storage.
  float* gatep = gate_values.data_ptr<float>();
  float* upp = up_values.data_ptr<float>();
  const int64_t activation_count = gate_values.numel();
  const float activation_limit = static_cast<float>(limit);
  const float situ_beta = static_cast<float>(beta);
  const float situ_linear_beta = static_cast<float>(linear_beta);
  at::parallel_for(
      0, activation_count, 256, [&](int64_t begin, int64_t end) {
        for (int64_t index = begin; index < end; ++index) {
          float gate = gatep[index];
          float up = upp[index];
          if (activation_limit != 0.0f) {
            gate = std::min(gate, activation_limit);
            up = std::max(
                -activation_limit, std::min(up, activation_limit));
          }
          if (activation == "situ") {
            float linear = up;
            if (situ_linear_beta > 0.0f) {
              linear = situ_linear_beta *
                  std::tanh(up / situ_linear_beta);
            }
            gatep[index] =
                situ_beta * std::tanh(gate / situ_beta) *
                (1.0f / (1.0f + std::exp(-gate))) * linear;
          } else {
            gatep[index] =
                gate * (1.0f / (1.0f + std::exp(-gate))) * up;
          }
        }
      });
  auto activated = gate_values;
  const double down_started = wall_seconds();
  auto down_values = vq_gemv_packed_list_cpu(
      activated,
      std::move(down_payloads),
      down_codebook,
      down_rows,
      down_blocks,
      down_bits,
      true);
  const double reduce_started = wall_seconds();
  TORCH_CHECK(
      !result.is_cuda() && result.scalar_type() == at::kFloat &&
          result.numel() >= down_rows,
      "three-projection result workspace is invalid");
  auto route = route_weights.to(torch::kFloat32).contiguous();
  TORCH_CHECK(
      down_values.scalar_type() == at::kFloat &&
          down_values.is_contiguous() &&
          down_values.size(0) == experts &&
          down_values.size(1) == down_rows,
      "three-projection down workspace is invalid");
  const float* downp = down_values.data_ptr<float>();
  const float* routep = route.data_ptr<float>();
  float* resultp = result.data_ptr<float>();
  at::parallel_for(0, down_rows, 256, [&](int64_t begin, int64_t end) {
    for (int64_t column = begin; column < end; ++column) {
      float value = 0.0f;
      for (int64_t expert = 0; expert < experts; ++expert) {
        value += downp[expert * down_rows + column] * routep[expert];
      }
      resultp[column] = value;
    }
  });
  const double finished = wall_seconds();
  three_projection_phase_seconds[0] += up_started - gate_started;
  three_projection_phase_seconds[1] +=
      activation_started - up_started;
  three_projection_phase_seconds[2] +=
      down_started - activation_started;
  three_projection_phase_seconds[3] +=
      reduce_started - down_started;
  three_projection_phase_seconds[4] += finished - reduce_started;
  ++three_projection_phase_calls;
  return result.narrow(0, 0, down_rows);
}

class CpuPackedThreeLayer {
 public:
  CpuPackedThreeLayer(
      std::vector<torch::Tensor> gate_payloads,
      std::vector<torch::Tensor> gate_codebooks,
      int64_t gate_rows,
      int64_t gate_blocks,
      int64_t gate_bits,
      std::vector<torch::Tensor> up_payloads,
      std::vector<torch::Tensor> up_codebooks,
      int64_t up_rows,
      int64_t up_blocks,
      int64_t up_bits,
      std::vector<torch::Tensor> down_payloads,
      std::vector<torch::Tensor> down_codebooks,
      int64_t down_rows,
      int64_t down_blocks,
      int64_t down_bits)
      : gate_payloads_(std::move(gate_payloads)),
        gate_codebooks_(std::move(gate_codebooks)),
        up_payloads_(std::move(up_payloads)),
        up_codebooks_(std::move(up_codebooks)),
        down_payloads_(std::move(down_payloads)),
        down_codebooks_(std::move(down_codebooks)),
        gate_rows_(gate_rows),
        gate_blocks_(gate_blocks),
        gate_bits_(gate_bits),
        up_rows_(up_rows),
        up_blocks_(up_blocks),
        up_bits_(up_bits),
        down_rows_(down_rows),
        down_blocks_(down_blocks),
        down_bits_(down_bits) {
    const int64_t experts = gate_payloads_.size();
    TORCH_CHECK(
        experts > 0 && experts == (int64_t)gate_codebooks_.size() &&
            experts == (int64_t)up_payloads_.size() &&
            experts == (int64_t)up_codebooks_.size() &&
            experts == (int64_t)down_payloads_.size() &&
            experts == (int64_t)down_codebooks_.size(),
        "resident packed layer expert counts must match");
    const int64_t top_k = std::min<int64_t>(16, experts);
    const int64_t gate_codes = gate_codebooks_[0].size(0);
    const int64_t up_codes = up_codebooks_[0].size(0);
    const int64_t down_codes = down_codebooks_[0].size(0);
    const int64_t score_count = std::max(
        gate_blocks_ * gate_codes, up_blocks_ * up_codes);
    const int64_t required =
        score_count + top_k * gate_rows_ + top_k * up_rows_ +
        top_k * down_blocks_ * down_codes + top_k * down_rows_;
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32).device(torch::kCPU);
    workspace_ = torch::empty({required}, options);
    result_ = torch::empty({down_rows_}, options);
    empty_ = torch::empty({0}, options);
  }

  torch::Tensor forward(
      torch::Tensor x_row,
      torch::Tensor expert_ids,
      torch::Tensor route_weights,
      double limit,
      std::string activation,
      double beta,
      double linear_beta) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto ids = expert_ids.to(torch::kLong).contiguous();
    TORCH_CHECK(
        !ids.is_cuda() && ids.dim() == 1 && ids.numel() > 0 &&
            ids.numel() <= 16 && route_weights.numel() == ids.numel(),
        "resident packed layer route shape mismatch");
    const int64_t count = ids.numel();
    const int64_t* idp = ids.data_ptr<int64_t>();
    std::vector<torch::Tensor> gate;
    std::vector<torch::Tensor> up;
    std::vector<torch::Tensor> down;
    gate.reserve(count);
    up.reserve(count);
    down.reserve(count);
    torch::Tensor gate_codebook;
    torch::Tensor up_codebook;
    torch::Tensor down_codebook;
    for (int64_t slot = 0; slot < count; ++slot) {
      const int64_t expert = idp[slot];
      TORCH_CHECK(
          expert >= 0 && expert < (int64_t)gate_payloads_.size(),
          "resident packed layer selected an invalid expert");
      if (slot == 0) {
        gate_codebook = gate_codebooks_[expert];
        up_codebook = up_codebooks_[expert];
        down_codebook = down_codebooks_[expert];
      } else if (
          gate_codebooks_[expert].data_ptr() != gate_codebook.data_ptr() ||
          up_codebooks_[expert].data_ptr() != up_codebook.data_ptr() ||
          down_codebooks_[expert].data_ptr() != down_codebook.data_ptr()) {
        return empty_;
      }
      gate.push_back(gate_payloads_[expert]);
      up.push_back(up_payloads_[expert]);
      down.push_back(down_payloads_[expert]);
    }
    return moe_packed_three_projection_cpu(
        x_row,
        std::move(gate),
        gate_codebook,
        gate_rows_,
        gate_blocks_,
        gate_bits_,
        std::move(up),
        up_codebook,
        up_rows_,
        up_blocks_,
        up_bits_,
        std::move(down),
        down_codebook,
        down_rows_,
        down_blocks_,
        down_bits_,
        route_weights,
        limit,
        std::move(activation),
        beta,
        linear_beta,
        workspace_,
        result_);
  }

 private:
  std::vector<torch::Tensor> gate_payloads_;
  std::vector<torch::Tensor> gate_codebooks_;
  std::vector<torch::Tensor> up_payloads_;
  std::vector<torch::Tensor> up_codebooks_;
  std::vector<torch::Tensor> down_payloads_;
  std::vector<torch::Tensor> down_codebooks_;
  int64_t gate_rows_;
  int64_t gate_blocks_;
  int64_t gate_bits_;
  int64_t up_rows_;
  int64_t up_blocks_;
  int64_t up_bits_;
  int64_t down_rows_;
  int64_t down_blocks_;
  int64_t down_bits_;
  torch::Tensor workspace_;
  torch::Tensor result_;
  torch::Tensor empty_;
  std::mutex mutex_;
};

// Resident directory for archives whose projection layout varies per expert.
// It owns only compact payload/codebook tensors and a bounded Top-K workspace;
// no index stream or logical weight matrix is expanded.  All selected Gate,
// Up, activation, Down and route reduction stages stay inside one OpenMP team.
class CpuPackedThreeMixedLayer {
 public:
  CpuPackedThreeMixedLayer(
      std::vector<torch::Tensor> gate_payloads,
      std::vector<torch::Tensor> gate_codebooks,
      std::vector<int64_t> gate_rows,
      std::vector<int64_t> gate_blocks,
      std::vector<int64_t> gate_bits,
      std::vector<int64_t> gate_layouts,
      std::vector<torch::Tensor> up_payloads,
      std::vector<torch::Tensor> up_codebooks,
      std::vector<int64_t> up_rows,
      std::vector<int64_t> up_blocks,
      std::vector<int64_t> up_bits,
      std::vector<int64_t> up_layouts,
      std::vector<torch::Tensor> down_payloads,
      std::vector<torch::Tensor> down_codebooks,
      std::vector<int64_t> down_rows,
      std::vector<int64_t> down_blocks,
      std::vector<int64_t> down_bits,
      std::vector<int64_t> down_layouts)
      : gate_payloads_(std::move(gate_payloads)),
        gate_codebooks_(normalize_codebooks(std::move(gate_codebooks))),
        gate_rows_(std::move(gate_rows)),
        gate_blocks_(std::move(gate_blocks)),
        gate_bits_(std::move(gate_bits)),
        gate_layouts_(std::move(gate_layouts)),
        up_payloads_(std::move(up_payloads)),
        up_codebooks_(normalize_codebooks(std::move(up_codebooks))),
        up_rows_(std::move(up_rows)),
        up_blocks_(std::move(up_blocks)),
        up_bits_(std::move(up_bits)),
        up_layouts_(std::move(up_layouts)),
        down_payloads_(std::move(down_payloads)),
        down_codebooks_(normalize_codebooks(std::move(down_codebooks))),
        down_rows_(std::move(down_rows)),
        down_blocks_(std::move(down_blocks)),
        down_bits_(std::move(down_bits)),
        down_layouts_(std::move(down_layouts)) {
    const int64_t experts = gate_payloads_.size();
    TORCH_CHECK(experts > 0, "mixed resident layer cannot be empty");
    validate_projection(
        "gate", experts, gate_payloads_, gate_codebooks_,
        gate_rows_, gate_blocks_, gate_bits_, gate_layouts_);
    validate_projection(
        "up", experts, up_payloads_, up_codebooks_,
        up_rows_, up_blocks_, up_bits_, up_layouts_);
    validate_projection(
        "down", experts, down_payloads_, down_codebooks_,
        down_rows_, down_blocks_, down_bits_, down_layouts_);
    hidden_ = down_rows_[0];
    intermediate_ = gate_rows_[0];
    int64_t max_scores = 1;
    for (int64_t expert = 0; expert < experts; ++expert) {
      TORCH_CHECK(
          gate_rows_[expert] == intermediate_ &&
              up_rows_[expert] == intermediate_ &&
              down_rows_[expert] == hidden_ &&
              gate_blocks_[expert] * gate_codebooks_[expert].size(1) == hidden_ &&
              up_blocks_[expert] * up_codebooks_[expert].size(1) == hidden_ &&
              down_blocks_[expert] * down_codebooks_[expert].size(1) == intermediate_,
          "mixed resident projection shapes do not match");
      max_scores = std::max(
          max_scores,
          gate_blocks_[expert] * gate_codebooks_[expert].size(0));
      max_scores = std::max(
          max_scores,
          up_blocks_[expert] * up_codebooks_[expert].size(0));
      max_scores = std::max(
          max_scores,
          down_blocks_[expert] * down_codebooks_[expert].size(0));
    }
    gate_transposed_ = transpose_codebooks(gate_codebooks_);
    up_transposed_ = transpose_codebooks(up_codebooks_);
    down_transposed_ = transpose_codebooks(down_codebooks_);
    gate_paired_bf16_ = paired_codebooks(
        gate_codebooks_, gate_layouts_);
    up_paired_bf16_ = paired_codebooks(
        up_codebooks_, up_layouts_);
    has_score_layout_ =
        std::find(gate_layouts_.begin(), gate_layouts_.end(), 1) !=
            gate_layouts_.end() ||
         std::find(up_layouts_.begin(), up_layouts_.end(), 1) !=
             up_layouts_.end();
    q4_experts_ = std::all_of(
        gate_layouts_.begin(), gate_layouts_.end(),
        [](int64_t layout) { return layout == 3; }) &&
        std::all_of(
            up_layouts_.begin(), up_layouts_.end(),
            [](int64_t layout) { return layout == 3; }) &&
        std::all_of(
            down_layouts_.begin(), down_layouts_.end(),
            [](int64_t layout) { return layout == 3; });
    any_q4_experts_ =
        std::find(gate_layouts_.begin(), gate_layouts_.end(), 3) !=
            gate_layouts_.end() ||
        std::find(up_layouts_.begin(), up_layouts_.end(), 3) !=
            up_layouts_.end() ||
        std::find(down_layouts_.begin(), down_layouts_.end(), 3) !=
            down_layouts_.end();
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32).device(torch::kCPU);
    score_ = torch::empty({max_scores}, options);
    gate_values_ = torch::empty({16, intermediate_}, options);
    up_values_ = torch::empty({16, intermediate_}, options);
    down_values_ = torch::empty({16, hidden_}, options);
    result_ = torch::empty({hidden_}, options);
    if (has_score_layout_) {
      input_bf16_ = torch::empty(
          {hidden_}, options.dtype(torch::kBFloat16));
    }
    if (any_q4_experts_) {
      TORCH_CHECK(hidden_ % 32 == 0 && intermediate_ % 32 == 0,
                  "Q4 resident MoE dimensions must be multiples of 32");
      input_q8_ = torch::empty(
          {(hidden_ / 32) * static_cast<int64_t>(sizeof(Q8Block32))},
          options.dtype(torch::kByte));
      activation_q8_ = torch::empty(
          {16 * (intermediate_ / 32) *
               static_cast<int64_t>(sizeof(Q8Block32))},
          options.dtype(torch::kByte));
    }
  }

  void configure_fused_moe(
      torch::Tensor router_weight,
      torch::Tensor router_bias,
      torch::Tensor router_mask,
      std::vector<torch::Tensor> shared_weights,
      std::vector<torch::Tensor> shared_scales,
      std::vector<int64_t> shared_rows,
      std::vector<int64_t> shared_cols,
      int64_t shared_block,
      int64_t top_k,
      bool normalize_route,
      double routed_scaling) {
    std::lock_guard<std::mutex> guard(mutex_);
    TORCH_CHECK(
        !router_weight.is_cuda() && router_weight.dim() == 2 &&
            router_weight.is_contiguous() &&
            (router_weight.scalar_type() == at::kFloat ||
             router_weight.scalar_type() == at::kBFloat16) &&
            router_weight.size(1) == hidden_,
        "resident fused MoE router must be contiguous CPU FP32/BF16");
    const int64_t experts = router_weight.size(0);
    TORCH_CHECK(
        !router_bias.is_cuda() && router_bias.numel() == experts &&
            !router_mask.is_cuda() &&
            router_mask.scalar_type() == at::kBool &&
            router_mask.numel() == experts &&
            top_k > 0 && top_k <= 16 && top_k <= experts,
        "resident fused MoE route metadata mismatch");
    TORCH_CHECK(
        shared_weights.size() == 3 && shared_scales.size() == 3 &&
            shared_rows.size() == 3 && shared_cols.size() == 3 &&
            shared_block == 128,
        "resident fused MoE requires three public compact operands");
    TORCH_CHECK(
        shared_rows[0] == intermediate_ &&
            shared_rows[1] == intermediate_ &&
            shared_rows[2] == hidden_ &&
            shared_cols[0] == hidden_ && shared_cols[1] == hidden_ &&
            shared_cols[2] == intermediate_,
        "resident fused MoE shared projection shapes mismatch");
    for (int64_t projection = 0; projection < 3; ++projection) {
      const auto& weight = shared_weights[projection];
      const auto& scale = shared_scales[projection];
      const int64_t rows = shared_rows[projection];
      const int64_t cols = shared_cols[projection];
      TORCH_CHECK(!weight.is_cuda() && weight.scalar_type() == at::kByte &&
                      weight.is_contiguous(),
                  "resident fused MoE shared payload must be CPU uint8");
      const bool q4 = weight.dim() == 1 && scale.numel() == 0;
      if (q4) {
        TORCH_CHECK(
            cols % 32 == 0 &&
                weight.numel() == rows * (cols / 32) *
                    static_cast<int64_t>(sizeof(Q4Block32)),
            "resident fused MoE shared Q4 image mismatch");
      } else if (weight.dim() == 2) {
        TORCH_CHECK(!scale.is_cuda() && scale.scalar_type() == at::kFloat &&
                        scale.dim() == 2 && scale.is_contiguous(),
                    "resident shared block-FP8 scales mismatch");
        TORCH_CHECK(
            weight.size(0) == rows && weight.size(1) == cols,
            "resident fused MoE row-major block-FP8 shape mismatch");
      } else {
        TORCH_CHECK(!scale.is_cuda() && scale.scalar_type() == at::kFloat &&
                        scale.dim() == 2 && scale.is_contiguous(),
                    "resident shared block-FP8 scales mismatch");
        TORCH_CHECK(
            weight.size(0) == (rows + shared_block - 1) / shared_block &&
                weight.size(1) == 4 &&
                weight.size(2) ==
                    (cols + shared_block - 1) / shared_block &&
                weight.size(3) == 32 && weight.size(4) == shared_block,
            "resident fused MoE block-major block-FP8 shape mismatch");
      }
      shared_q4_.push_back(q4);
    }
    router_weight_ = std::move(router_weight);
    router_bias_ = router_bias.to(torch::kFloat32).contiguous();
    router_mask_ = router_mask.to(torch::kBool).contiguous();
    shared_weights_ = std::move(shared_weights);
    shared_scales_ = std::move(shared_scales);
    shared_rows_ = std::move(shared_rows);
    shared_cols_ = std::move(shared_cols);
    shared_block_ = shared_block;
    route_top_k_ = top_k;
    normalize_route_ = normalize_route;
    routed_scaling_ = static_cast<float>(routed_scaling);
    auto float_options = torch::TensorOptions()
        .dtype(torch::kFloat32).device(torch::kCPU);
    route_scores_ = torch::empty({experts}, float_options);
    route_weights_ = torch::empty({top_k}, float_options);
    shared_activation_ = torch::empty({intermediate_}, float_options);
    shared_activation_bf16_ = torch::empty(
        {intermediate_}, float_options.dtype(torch::kBFloat16));
    input_bf16_ = torch::empty(
        {hidden_}, float_options.dtype(torch::kBFloat16));
    if (std::find(shared_q4_.begin(), shared_q4_.end(), true) !=
        shared_q4_.end()) {
      if (!input_q8_.defined()) {
        input_q8_ = torch::empty(
            {(hidden_ / 32) * static_cast<int64_t>(sizeof(Q8Block32))},
            float_options.dtype(torch::kByte));
      }
      shared_activation_q8_ = torch::empty(
          {(intermediate_ / 32) *
               static_cast<int64_t>(sizeof(Q8Block32))},
          float_options.dtype(torch::kByte));
    }
    selected_.assign(top_k, -1);
    route_choices_.assign(
        top_k, -std::numeric_limits<float>::infinity());
    fused_moe_ready_ = true;
  }

  torch::Tensor forward_fused_moe(
      torch::Tensor x_row,
      double limit,
      std::string activation,
      double beta,
      double linear_beta) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!fused_moe_ready_ || !packed_direct_rows8_enabled() ||
        !packed_fused_gate_up_enabled() ||
        !packed_fused_down_reduce_enabled()) {
      return torch::empty(
          {0}, torch::TensorOptions().dtype(torch::kFloat32));
    }
    auto x = x_row.to(torch::kFloat32).contiguous();
    TORCH_CHECK(
        !x.is_cuda() && x.sizes() == torch::IntArrayRef({1, hidden_}),
        "resident fused MoE requires one CPU input row");
    TORCH_CHECK(
        activation == "situ" || activation == "silu" ||
            activation == "swiglu",
        "resident fused MoE activation must be situ, silu, or swiglu");
    const int64_t experts = router_weight_.size(0);
    for (int64_t expert = 0;
         expert < static_cast<int64_t>(gate_layouts_.size()); ++expert) {
      if (gate_layouts_[expert] == 1 || up_layouts_[expert] == 1 ||
          down_layouts_[expert] == 1) {
        return torch::empty(
            {0}, torch::TensorOptions().dtype(torch::kFloat32));
      }
    }
    const bool* maskp = router_mask_.data_ptr<bool>();
    const float* biasp = router_bias_.data_ptr<float>();
    float* scorep = route_scores_.data_ptr<float>();
    float* routep = route_weights_.data_ptr<float>();
    const float* xp = x.data_ptr<float>();
    at::BFloat16* xb = input_bf16_.data_ptr<at::BFloat16>();
    float* sharedp = shared_activation_.data_ptr<float>();
    at::BFloat16* sharedb =
        shared_activation_bf16_.data_ptr<at::BFloat16>();
    float* routed_activation = gate_values_.data_ptr<float>();
    float* resultp = result_.data_ptr<float>();
    auto* input_q8 = input_q8_.defined()
        ? reinterpret_cast<Q8Block32*>(input_q8_.data_ptr<uint8_t>())
        : nullptr;
    auto* activation_q8 = activation_q8_.defined()
        ? reinterpret_cast<Q8Block32*>(activation_q8_.data_ptr<uint8_t>())
        : nullptr;
    auto* shared_activation_q8 = shared_activation_q8_.defined()
        ? reinterpret_cast<Q8Block32*>(
              shared_activation_q8_.data_ptr<uint8_t>())
        : nullptr;
    const float activation_limit = static_cast<float>(limit);
    const float situ_beta = static_cast<float>(beta);
    const float situ_linear_beta = static_cast<float>(linear_beta);
    const double started = wall_seconds();
    double markers[3] = {0.0, 0.0, 0.0};
    if (input_q8 != nullptr) {
      quantize_q8_row(xp, hidden_, input_q8);
    }

#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (int64_t column = 0; column < hidden_; ++column) {
        xb[column] = at::BFloat16(xp[column]);
      }
      // Router and shared Gate+Up remain in the same persistent team. Router
      // is tiny and interleaved; shared Q4 rows follow the same node-local
      // partition as their compiled pages. `nowait` preserves concurrency.
#pragma omp for schedule(static) nowait
      for (int64_t expert = 0; expert < experts; ++expert) {
        float raw;
        if (router_weight_.scalar_type() == at::kFloat) {
          raw = f32_dense_row_dot(
              xp,
              router_weight_.data_ptr<float>() + expert * hidden_,
              hidden_);
        } else {
          raw = bf16_dense_row_dot(
              xb,
              router_weight_.data_ptr<at::BFloat16>() + expert * hidden_,
              hidden_);
        }
        const float softplus =
            raw > 20.0f ? raw : std::log1p(std::exp(raw));
        scorep[expert] = std::sqrt(softplus);
      }
      auto evaluate_shared_row = [&](int64_t row) {
          const int64_t input_blocks = hidden_ / 32;
          const float gate = shared_q4_[0]
              ? q4_q8_block_major_row_dot(
                    reinterpret_cast<const Q4Block32*>(
                        shared_weights_[0].data_ptr<uint8_t>()),
                    row, intermediate_, input_q8, input_blocks)
              : block_fp8_logical_row_dot_bf16(
                    xb, shared_weights_[0], shared_scales_[0], row,
                    shared_rows_[0], shared_cols_[0], shared_block_);
          const float up = shared_q4_[1]
              ? q4_q8_block_major_row_dot(
                    reinterpret_cast<const Q4Block32*>(
                        shared_weights_[1].data_ptr<uint8_t>()),
                    row, intermediate_, input_q8, input_blocks)
              : block_fp8_logical_row_dot_bf16(
                    xb, shared_weights_[1], shared_scales_[1], row,
                    shared_rows_[1], shared_cols_[1], shared_block_);
          apply_gated_activation(
              gate, up, activation_limit, activation,
              situ_beta, situ_linear_beta, sharedp[row]);
          sharedb[row] = at::BFloat16(sharedp[row]);
      };
      auto evaluate_shared_tile = [&](int64_t group) {
          const int64_t first_row = group * kQ4BlockMajorRows;
          const int64_t valid_rows = std::min<int64_t>(
              kQ4BlockMajorRows, intermediate_ - first_row);
          const int64_t input_blocks = hidden_ / 32;
          alignas(64) float gate_values[kQ4BlockMajorRows];
          alignas(64) float up_values[kQ4BlockMajorRows];
          q4_q8_block_major_rows8(
              reinterpret_cast<const Q4Block32*>(
                  shared_weights_[0].data_ptr<uint8_t>()),
              first_row, valid_rows, input_q8, input_blocks, gate_values);
          q4_q8_block_major_rows8(
              reinterpret_cast<const Q4Block32*>(
                  shared_weights_[1].data_ptr<uint8_t>()),
              first_row, valid_rows, input_q8, input_blocks, up_values);
          for (int64_t local = 0; local < valid_rows; ++local) {
            const int64_t row = first_row + local;
            apply_gated_activation(
                gate_values[local], up_values[local], activation_limit,
                activation, situ_beta, situ_linear_beta, sharedp[row]);
            sharedb[row] = at::BFloat16(sharedp[row]);
          }
      };
      if (shared_q4_[0] && shared_q4_[1]) {
        const int64_t row_groups =
            (intermediate_ + kQ4BlockMajorRows - 1) /
            kQ4BlockMajorRows;
        if (q4_numa_local_enabled()) {
          const auto range = q4_numa_local_row_range(row_groups);
          for (int64_t group = range.first; group < range.second; ++group) {
            evaluate_shared_tile(group);
          }
        } else {
#pragma omp for schedule(static) nowait
          for (int64_t group = 0; group < row_groups; ++group) {
            evaluate_shared_tile(group);
          }
        }
      } else if (q4_numa_local_enabled() &&
                 (shared_q4_[0] || shared_q4_[1])) {
        const auto range = q4_numa_local_row_range(intermediate_);
        for (int64_t row = range.first; row < range.second; ++row) {
          evaluate_shared_row(row);
        }
      } else {
#pragma omp for schedule(static) nowait
        for (int64_t row = 0; row < intermediate_; ++row) {
          evaluate_shared_row(row);
        }
      }
#pragma omp barrier

      if (shared_q4_[2]) {
#pragma omp for schedule(static)
        for (int64_t block = 0; block < intermediate_ / 32; ++block) {
          quantize_q8_block32(
              sharedp + block * 32, shared_activation_q8 + block);
        }
      }

#pragma omp single
      {
        std::fill(selected_.begin(), selected_.end(), int64_t{-1});
        std::fill(
            route_choices_.begin(), route_choices_.end(),
            -std::numeric_limits<float>::infinity());
        for (int64_t expert = 0; expert < experts; ++expert) {
          if (!maskp[expert]) {
            continue;
          }
          const float choice = scorep[expert] + biasp[expert];
          for (int64_t rank = 0; rank < route_top_k_; ++rank) {
            if (choice > route_choices_[rank]) {
              for (int64_t move = route_top_k_ - 1; move > rank; --move) {
                route_choices_[move] = route_choices_[move - 1];
                selected_[move] = selected_[move - 1];
              }
              route_choices_[rank] = choice;
              selected_[rank] = expert;
              break;
            }
          }
        }
        float denominator = normalize_route_ ? 1.0e-20f : 1.0f;
        for (int64_t rank = 0; rank < route_top_k_; ++rank) {
          TORCH_CHECK(
              selected_[rank] >= 0,
              "resident fused MoE has insufficient available experts");
          routep[rank] = scorep[selected_[rank]];
          if (normalize_route_) {
            denominator += routep[rank];
          }
        }
        const float multiplier = routed_scaling_ / denominator;
        for (int64_t rank = 0; rank < route_top_k_; ++rank) {
          routep[rank] *= multiplier;
          const int64_t expert = selected_[rank];
          ++resident_moe_selected_experts;
          if (gate_layouts_[expert] == 3 && up_layouts_[expert] == 3 &&
              down_layouts_[expert] == 3) {
            ++resident_moe_q4_selected_experts;
          }
        }
        markers[0] = wall_seconds();
      }

      if (q4_experts_) {
        evaluate_selected_q4_gate_up_activation(
            input_q8, selected_, gate_payloads_, up_payloads_, hidden_,
            intermediate_, activation_limit, activation,
            situ_beta, situ_linear_beta, routed_activation);
#pragma omp for schedule(static)
        for (int64_t task = 0;
             task < route_top_k_ * (intermediate_ / 32); ++task) {
          const int64_t slot = task / (intermediate_ / 32);
          const int64_t block = task - slot * (intermediate_ / 32);
          quantize_q8_block32(
              routed_activation + slot * intermediate_ + block * 32,
              activation_q8 + slot * (intermediate_ / 32) + block);
        }
      } else {
        evaluate_selected_gate_up_activation(
            xp, input_q8, hidden_, selected_,
            gate_payloads_, gate_codebooks_, gate_blocks_, gate_bits_,
            gate_layouts_,
            up_payloads_, up_codebooks_, up_blocks_, up_bits_, up_layouts_,
            intermediate_, activation_limit, activation,
            situ_beta, situ_linear_beta, routed_activation);
        if (any_q4_experts_) {
#pragma omp for schedule(static)
          for (int64_t task = 0;
               task < route_top_k_ * (intermediate_ / 32); ++task) {
            const int64_t slot = task / (intermediate_ / 32);
            const int64_t block = task - slot * (intermediate_ / 32);
            const int64_t expert = selected_[slot];
            if (down_layouts_[expert] == 3) {
              quantize_q8_block32(
                  routed_activation + slot * intermediate_ + block * 32,
                  activation_q8 + slot * (intermediate_ / 32) + block);
            }
          }
        }
      }
#pragma omp single
      { markers[1] = wall_seconds(); }

      if (q4_experts_) {
        evaluate_selected_q4_down_shared_reduced(
            activation_q8, intermediate_, selected_, down_payloads_, hidden_,
            routep, sharedb, shared_activation_q8, shared_weights_[2],
            shared_scales_[2], shared_q4_[2], shared_block_, resultp);
      } else {
        evaluate_selected_down_shared_reduced(
            routed_activation,
            activation_q8,
            intermediate_,
            selected_,
            down_payloads_,
            down_codebooks_,
            down_blocks_,
            down_bits_,
            down_layouts_,
            hidden_,
            routep,
            sharedb,
            shared_activation_q8,
            shared_weights_[2],
            shared_scales_[2],
            shared_rows_[2],
            shared_cols_[2],
            shared_block_,
            shared_q4_[2],
            resultp);
      }
#pragma omp single
      { markers[2] = wall_seconds(); }
    }
    resident_moe_phase_seconds[0] += markers[0] - started;
    resident_moe_phase_seconds[1] += markers[1] - markers[0];
    resident_moe_phase_seconds[2] += markers[2] - markers[1];
    ++resident_moe_phase_calls;
    return result_;
  }

  // Configure a generic latent-MoE decode layer.  Four projections consume
  // the full hidden row (shared Gate, shared Up, routed latent and Router),
  // packed experts transform the latent row, and two projections return the
  // shared/routed branches to the full hidden width.  The interface is shape
  // and storage-format driven; no model name appears in this executor.
  void configure_latent_moe(
      std::vector<torch::Tensor> input_weights,
      std::vector<torch::Tensor> input_scales,
      std::vector<int64_t> input_rows,
      std::vector<int64_t> input_kinds,
      int64_t input_cols,
      std::vector<torch::Tensor> output_weights,
      std::vector<torch::Tensor> output_scales,
      std::vector<int64_t> output_rows,
      std::vector<int64_t> output_cols,
      std::vector<int64_t> output_kinds,
      torch::Tensor route_correction,
      torch::Tensor route_mask,
      torch::Tensor routed_norm,
      int64_t block_size,
      int64_t top_k,
      bool normalize_route,
      double routed_scaling,
      double rms_eps,
      double limit,
      std::string scoring,
      std::string activation,
      double beta,
      double linear_beta) {
    std::lock_guard<std::mutex> guard(mutex_);
    TORCH_CHECK(
        input_weights.size() == 4 && input_scales.size() == 4 &&
            input_rows.size() == 4 && input_kinds.size() == 4 &&
            output_weights.size() == 2 && output_scales.size() == 2 &&
            output_rows.size() == 2 && output_cols.size() == 2 &&
            output_kinds.size() == 2,
        "latent resident MoE projection metadata mismatch");
    const int64_t experts = gate_payloads_.size();
    TORCH_CHECK(
        input_cols > 0 && input_rows[0] == input_rows[1] &&
            input_rows[2] == hidden_ && input_rows[3] == experts &&
            output_rows[0] == input_cols && output_rows[1] == input_cols &&
            output_cols[0] == input_rows[0] && output_cols[1] == hidden_ &&
            routed_norm.numel() == hidden_ && top_k > 0 && top_k <= 16 &&
            top_k <= experts && route_correction.numel() == experts &&
            route_mask.numel() == experts && block_size == 128,
        "latent resident MoE logical shapes mismatch");
    TORCH_CHECK(
        scoring == "sigmoid" || scoring == "softplus_sqrt",
        "latent resident MoE scoring must be sigmoid or softplus_sqrt");
    TORCH_CHECK(
        activation == "situ" || activation == "silu" ||
            activation == "swiglu",
        "latent resident MoE activation is unsupported");

    const auto validate = [&](
        const torch::Tensor& weight,
        const torch::Tensor& scale,
        int64_t rows,
        int64_t cols,
        int64_t kind) {
      TORCH_CHECK(!weight.is_cuda() && weight.is_contiguous(),
                  "latent resident projection must be contiguous CPU data");
      if (kind == 0 || kind == 2) {
        TORCH_CHECK(
            weight.dim() == 2 && weight.size(0) == rows &&
                weight.size(1) == cols &&
                weight.scalar_type() ==
                    (kind == 0 ? at::kBFloat16 : at::kFloat),
            "latent resident dense projection shape/dtype mismatch");
      } else {
        TORCH_CHECK(
            kind == 1 && weight.scalar_type() == at::kByte &&
                (weight.dim() == 2 || weight.dim() == 5) &&
                !scale.is_cuda() && scale.scalar_type() == at::kFloat &&
                scale.is_contiguous() &&
                scale.size(0) == (rows + block_size - 1) / block_size &&
                scale.size(1) == (cols + block_size - 1) / block_size,
            "latent resident block-FP8 projection mismatch");
      }
    };
    for (int64_t index = 0; index < 4; ++index) {
      validate(
          input_weights[index], input_scales[index], input_rows[index],
          input_cols, input_kinds[index]);
    }
    for (int64_t index = 0; index < 2; ++index) {
      validate(
          output_weights[index], output_scales[index], output_rows[index],
          output_cols[index], output_kinds[index]);
    }
    for (int64_t expert = 0; expert < experts; ++expert) {
      TORCH_CHECK(
          gate_layouts_[expert] != 1 && up_layouts_[expert] != 1 &&
              down_layouts_[expert] != 1,
          "latent resident MoE requires row or row-tile expert indices");
    }

    latent_input_weights_ = std::move(input_weights);
    latent_input_scales_ = std::move(input_scales);
    latent_input_rows_ = std::move(input_rows);
    latent_input_kinds_ = std::move(input_kinds);
    latent_input_cols_ = input_cols;
    latent_output_weights_ = std::move(output_weights);
    latent_output_scales_ = std::move(output_scales);
    latent_output_rows_ = std::move(output_rows);
    latent_output_cols_ = std::move(output_cols);
    latent_output_kinds_ = std::move(output_kinds);
    latent_route_correction_ = route_correction.to(torch::kFloat32).contiguous();
    latent_route_mask_ = route_mask.to(torch::kBool).contiguous();
    latent_routed_norm_ = routed_norm.to(torch::kBFloat16).contiguous();
    latent_block_size_ = block_size;
    route_top_k_ = top_k;
    normalize_route_ = normalize_route;
    routed_scaling_ = static_cast<float>(routed_scaling);
    latent_rms_eps_ = static_cast<float>(rms_eps);
    latent_limit_ = static_cast<float>(limit);
    latent_scoring_ = std::move(scoring);
    latent_activation_ = std::move(activation);
    latent_beta_ = static_cast<float>(beta);
    latent_linear_beta_ = static_cast<float>(linear_beta);

    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto bf16 = f32.dtype(torch::kBFloat16);
    latent_input_float_ = torch::empty({input_cols}, f32);
    latent_input_bf16_ = torch::empty({input_cols}, bf16);
    latent_shared_gate_ = torch::empty({latent_input_rows_[0]}, f32);
    latent_shared_up_ = torch::empty({latent_input_rows_[1]}, f32);
    latent_shared_activation_ = torch::empty({latent_input_rows_[0]}, f32);
    latent_shared_activation_bf16_ = torch::empty({latent_input_rows_[0]}, bf16);
    latent_projected_ = torch::empty({hidden_}, f32);
    latent_routed_normalized_ = torch::empty({hidden_}, f32);
    latent_routed_normalized_bf16_ = torch::empty({hidden_}, bf16);
    route_scores_ = torch::empty({experts}, f32);
    route_weights_ = torch::empty({top_k}, f32);
    latent_full_output_ = torch::empty({1, input_cols}, bf16);
    selected_.assign(top_k, -1);
    route_choices_.assign(
        top_k, -std::numeric_limits<float>::infinity());
    latent_moe_ready_ = true;
  }

  torch::Tensor forward_latent_moe(
      torch::Tensor x_row,
      torch::Tensor residual) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!latent_moe_ready_ || !packed_direct_rows8_enabled() ||
        !packed_fused_gate_up_enabled() ||
        !packed_fused_down_reduce_enabled()) {
      return torch::empty(
          {0}, torch::TensorOptions().dtype(torch::kBFloat16));
    }
    TORCH_CHECK(
        !x_row.is_cuda() && x_row.dim() == 2 && x_row.size(0) == 1 &&
            x_row.size(1) == latent_input_cols_ && x_row.is_contiguous() &&
            (x_row.scalar_type() == at::kFloat ||
             x_row.scalar_type() == at::kBFloat16),
        "latent resident MoE input mismatch");
    TORCH_CHECK(
        !residual.is_cuda() && residual.sizes() == x_row.sizes() &&
            residual.is_contiguous() &&
            (residual.scalar_type() == at::kFloat ||
             residual.scalar_type() == at::kBFloat16),
        "latent resident MoE residual mismatch");
    const int64_t experts = gate_payloads_.size();
    const int64_t shared_intermediate = latent_input_rows_[0];
    const int64_t prelude_rows =
        shared_intermediate * 2 + hidden_ + experts;
    const bool* maskp = latent_route_mask_.data_ptr<bool>();
    const float* correctionp = latent_route_correction_.data_ptr<float>();
    float* inputp = latent_input_float_.data_ptr<float>();
    at::BFloat16* inputb = latent_input_bf16_.data_ptr<at::BFloat16>();
    float* shared_gatep = latent_shared_gate_.data_ptr<float>();
    float* shared_upp = latent_shared_up_.data_ptr<float>();
    float* sharedp = latent_shared_activation_.data_ptr<float>();
    at::BFloat16* sharedb =
        latent_shared_activation_bf16_.data_ptr<at::BFloat16>();
    float* projectedp = latent_projected_.data_ptr<float>();
    float* scorep = route_scores_.data_ptr<float>();
    float* routep = route_weights_.data_ptr<float>();
    float* packed_activationp = gate_values_.data_ptr<float>();
    float* routedp = result_.data_ptr<float>();
    float* normp = latent_routed_normalized_.data_ptr<float>();
    at::BFloat16* normb =
        latent_routed_normalized_bf16_.data_ptr<at::BFloat16>();
    at::BFloat16* outputp = latent_full_output_.data_ptr<at::BFloat16>();
    auto* packed_input_q8 = any_q4_experts_
        ? reinterpret_cast<Q8Block32*>(input_q8_.data_ptr<uint8_t>())
        : nullptr;
    auto* packed_activation_q8 = any_q4_experts_
        ? reinterpret_cast<Q8Block32*>(activation_q8_.data_ptr<uint8_t>())
        : nullptr;
    const at::BFloat16* norm_weightp =
        latent_routed_norm_.data_ptr<at::BFloat16>();
    float inverse_rms = 1.0f;
    const double started = wall_seconds();
    double markers[3] = {started, started, started};

    const auto projection_dot = [&](
        const torch::Tensor& weight,
        const torch::Tensor& scale,
        int64_t rows,
        int64_t cols,
        int64_t kind,
        int64_t row,
        const float* sourcef,
        const at::BFloat16* sourceb) -> float {
      if (kind == 0) {
        return bf16_dense_row_dot(
            sourceb,
            weight.data_ptr<at::BFloat16>() + row * cols,
            cols);
      }
      if (kind == 2) {
        return f32_dense_row_dot(
            sourcef, weight.data_ptr<float>() + row * cols, cols);
      }
      return block_fp8_logical_row_dot_bf16(
          sourceb, weight, scale, row, rows, cols, latent_block_size_);
    };

#pragma omp parallel
    {
#pragma omp for schedule(static)
      for (int64_t column = 0; column < latent_input_cols_; ++column) {
        const float value = x_row.scalar_type() == at::kFloat
            ? x_row.data_ptr<float>()[column]
            : static_cast<float>(x_row.data_ptr<at::BFloat16>()[column]);
        inputp[column] = value;
        inputb[column] = at::BFloat16(value);
      }

#pragma omp for schedule(dynamic, 4)
      for (int64_t logical = 0; logical < prelude_rows; ++logical) {
        int64_t projection;
        int64_t row;
        if (logical < shared_intermediate) {
          projection = 0;
          row = logical;
        } else if (logical < shared_intermediate * 2) {
          projection = 1;
          row = logical - shared_intermediate;
        } else if (logical < shared_intermediate * 2 + hidden_) {
          projection = 2;
          row = logical - shared_intermediate * 2;
        } else {
          projection = 3;
          row = logical - shared_intermediate * 2 - hidden_;
        }
        const float value = projection_dot(
            latent_input_weights_[projection],
            latent_input_scales_[projection],
            latent_input_rows_[projection], latent_input_cols_,
            latent_input_kinds_[projection], row, inputp, inputb);
        if (projection == 0) {
          shared_gatep[row] = value;
        } else if (projection == 1) {
          shared_upp[row] = value;
        } else if (projection == 2) {
          projectedp[row] = value;
        } else {
          scorep[row] = latent_scoring_ == "sigmoid"
              ? 1.0f / (1.0f + std::exp(-value))
              : std::sqrt(value > 20.0f
                    ? value : std::log1p(std::exp(value)));
        }
      }

#pragma omp for schedule(static)
      for (int64_t row = 0; row < shared_intermediate; ++row) {
        apply_gated_activation(
            shared_gatep[row], shared_upp[row], latent_limit_,
            latent_activation_, latent_beta_, latent_linear_beta_,
            sharedp[row]);
        sharedb[row] = at::BFloat16(sharedp[row]);
      }

      if (any_q4_experts_) {
#pragma omp for schedule(static)
        for (int64_t block = 0; block < hidden_ / 32; ++block) {
          quantize_q8_block32(
              projectedp + block * 32, packed_input_q8 + block);
        }
      }

#pragma omp single
      {
        std::fill(selected_.begin(), selected_.end(), int64_t{-1});
        std::fill(
            route_choices_.begin(), route_choices_.end(),
            -std::numeric_limits<float>::infinity());
        for (int64_t expert = 0; expert < experts; ++expert) {
          if (!maskp[expert]) {
            continue;
          }
          const float choice = scorep[expert] + correctionp[expert];
          for (int64_t rank = 0; rank < route_top_k_; ++rank) {
            if (choice > route_choices_[rank]) {
              for (int64_t move = route_top_k_ - 1; move > rank; --move) {
                route_choices_[move] = route_choices_[move - 1];
                selected_[move] = selected_[move - 1];
              }
              route_choices_[rank] = choice;
              selected_[rank] = expert;
              break;
            }
          }
        }
        float denominator = normalize_route_ ? 1.0e-20f : 1.0f;
        for (int64_t rank = 0; rank < route_top_k_; ++rank) {
          TORCH_CHECK(selected_[rank] >= 0,
                      "latent resident route has too few experts");
          routep[rank] = scorep[selected_[rank]];
          if (normalize_route_) {
            denominator += routep[rank];
          }
        }
        const float multiplier = routed_scaling_ / denominator;
        for (int64_t rank = 0; rank < route_top_k_; ++rank) {
          routep[rank] *= multiplier;
        }
        markers[0] = wall_seconds();
      }

      evaluate_selected_gate_up_activation(
          projectedp, packed_input_q8, hidden_, selected_,
          gate_payloads_, gate_codebooks_, gate_blocks_, gate_bits_,
          gate_layouts_, up_payloads_, up_codebooks_, up_blocks_, up_bits_,
          up_layouts_, intermediate_, latent_limit_, latent_activation_,
          latent_beta_, latent_linear_beta_, packed_activationp);
      if (any_q4_experts_) {
#pragma omp for schedule(static)
        for (int64_t task = 0;
             task < route_top_k_ * (intermediate_ / 32); ++task) {
          const int64_t slot = task / (intermediate_ / 32);
          const int64_t block = task - slot * (intermediate_ / 32);
          const int64_t expert = selected_[slot];
          if (down_layouts_[expert] == 3) {
            quantize_q8_block32(
                packed_activationp + slot * intermediate_ + block * 32,
                packed_activation_q8 + slot * (intermediate_ / 32) + block);
          }
        }
      }
      evaluate_selected_direct_rows_reduced(
          packed_activationp, packed_activation_q8, intermediate_, selected_,
          down_payloads_,
          down_codebooks_, down_blocks_, down_bits_, down_layouts_, hidden_,
          routep, routedp);
#pragma omp single
      {
        float sum = 0.0f;
        for (int64_t column = 0; column < hidden_; ++column) {
          sum += routedp[column] * routedp[column];
        }
        inverse_rms = 1.0f / std::sqrt(
            sum / static_cast<float>(hidden_) + latent_rms_eps_);
        markers[1] = wall_seconds();
      }

#pragma omp for schedule(static)
      for (int64_t column = 0; column < hidden_; ++column) {
        const float value = routedp[column] * inverse_rms *
            static_cast<float>(norm_weightp[column]);
        normp[column] = value;
        normb[column] = at::BFloat16(value);
      }

#pragma omp for schedule(static)
      for (int64_t row = 0; row < latent_input_cols_; ++row) {
        const float shared_value = projection_dot(
            latent_output_weights_[0], latent_output_scales_[0],
            latent_output_rows_[0], latent_output_cols_[0],
            latent_output_kinds_[0], row, sharedp, sharedb);
        const float routed_value = projection_dot(
            latent_output_weights_[1], latent_output_scales_[1],
            latent_output_rows_[1], latent_output_cols_[1],
            latent_output_kinds_[1], row, normp, normb);
        const float residual_value = residual.scalar_type() == at::kFloat
            ? residual.data_ptr<float>()[row]
            : static_cast<float>(residual.data_ptr<at::BFloat16>()[row]);
        outputp[row] = at::BFloat16(
            residual_value + shared_value + routed_value);
      }
#pragma omp single
      { markers[2] = wall_seconds(); }
    }
    latent_moe_phase_seconds[0] += markers[0] - started;
    latent_moe_phase_seconds[1] += markers[1] - markers[0];
    latent_moe_phase_seconds[2] += markers[2] - markers[1];
    latent_moe_phase_seconds[3] += markers[2] - started;
    ++latent_moe_phase_calls;
    return latent_full_output_;
  }

  torch::Tensor forward(
      torch::Tensor x_row,
      torch::Tensor expert_ids,
      torch::Tensor route_weights,
      double limit,
      std::string activation,
      double beta,
      double linear_beta) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto x = x_row.to(torch::kFloat32).contiguous();
    auto ids = expert_ids.to(torch::kLong).contiguous();
    auto route = route_weights.to(torch::kFloat32).contiguous();
    TORCH_CHECK(
        !x.is_cuda() && x.sizes() == torch::IntArrayRef({1, hidden_}),
        "mixed resident layer requires one CPU input row");
    TORCH_CHECK(
        !ids.is_cuda() && ids.dim() == 1 && ids.numel() > 0 &&
            ids.numel() <= 16 && route.numel() == ids.numel(),
        "mixed resident layer route shape mismatch");
    TORCH_CHECK(
        activation == "situ" || activation == "silu" ||
            activation == "swiglu",
        "mixed resident activation must be situ, silu, or swiglu");
    const int64_t top_k = ids.numel();
    const int64_t* idp = ids.data_ptr<int64_t>();
    std::vector<int64_t> selected(idp, idp + top_k);
    for (const int64_t expert : selected) {
      TORCH_CHECK(
          expert >= 0 && expert < (int64_t)gate_payloads_.size(),
          "mixed resident layer selected an invalid expert");
    }
    const float* xp = x.data_ptr<float>();
    const float activation_limit = static_cast<float>(limit);
    const float situ_beta = static_cast<float>(beta);
    const float situ_linear_beta = static_cast<float>(linear_beta);
    float* scorep = score_.data_ptr<float>();
    float* gatep = gate_values_.data_ptr<float>();
    float* upp = up_values_.data_ptr<float>();
    float* downp = down_values_.data_ptr<float>();
    float* resultp = result_.data_ptr<float>();
    const float* routep = route.data_ptr<float>();
    auto* q4_input = any_q4_experts_
        ? reinterpret_cast<Q8Block32*>(input_q8_.data_ptr<uint8_t>())
        : nullptr;
    auto* q4_activation = any_q4_experts_
        ? reinterpret_cast<Q8Block32*>(activation_q8_.data_ptr<uint8_t>())
        : nullptr;
    if (any_q4_experts_) {
      quantize_q8_row(xp, hidden_, q4_input);
    }
    at::BFloat16* input_bf16p = has_score_layout_
        ? input_bf16_.data_ptr<at::BFloat16>() : nullptr;
    const auto gate_groups = projection_groups(
        selected, gate_codebooks_, gate_blocks_, gate_bits_, gate_layouts_);
    const auto up_groups = projection_groups(
        selected, up_codebooks_, up_blocks_, up_bits_, up_layouts_);
    const auto gate_score_modes = projection_score_modes(
        selected, gate_groups.first, gate_codebooks_, gate_rows_,
        gate_blocks_, gate_layouts_);
    const auto up_score_modes = projection_score_modes(
        selected, up_groups.first, up_codebooks_, up_rows_,
        up_blocks_, up_layouts_);
    const bool down_direct = selected_direct(
        selected, down_codebooks_, down_rows_, down_layouts_);
    const bool fused_gate_up =
        packed_direct_rows8_enabled() && packed_fused_gate_up_enabled() &&
        std::find(gate_score_modes.begin(), gate_score_modes.end(), uint8_t{1}) ==
            gate_score_modes.end() &&
        std::find(up_score_modes.begin(), up_score_modes.end(), uint8_t{1}) ==
            up_score_modes.end() &&
        selected_row_major(selected, gate_layouts_) &&
        selected_row_major(selected, up_layouts_);
    const bool fused_down_reduce =
        packed_direct_rows8_enabled() &&
        packed_fused_down_reduce_enabled() && down_direct &&
        selected_row_major(selected, down_layouts_);
    double phase_markers[6] = {wall_seconds(), 0.0, 0.0, 0.0, 0.0, 0.0};

#pragma omp parallel
    {
      const auto& gate_payloads = gate_payloads_;
      const auto& up_payloads = up_payloads_;
      const auto& down_payloads = down_payloads_;
      const auto& gate_codebooks = gate_codebooks_;
      const auto& up_codebooks = up_codebooks_;
      const auto& down_codebooks = down_codebooks_;
      if (has_score_layout_) {
#pragma omp for schedule(static)
        for (int64_t column = 0; column < hidden_; ++column) {
          input_bf16p[column] = at::BFloat16(xp[column]);
        }
      }
      if (q4_experts_) {
        evaluate_selected_q4_gate_up_activation(
            q4_input, selected, gate_payloads, up_payloads, hidden_,
            intermediate_, activation_limit, activation,
            situ_beta, situ_linear_beta, gatep);
#pragma omp for schedule(static)
        for (int64_t task = 0;
             task < top_k * (intermediate_ / 32); ++task) {
          const int64_t slot = task / (intermediate_ / 32);
          const int64_t block = task - slot * (intermediate_ / 32);
          quantize_q8_block32(
              gatep + slot * intermediate_ + block * 32,
              q4_activation + slot * (intermediate_ / 32) + block);
        }
#pragma omp single
        {
          phase_markers[1] = wall_seconds();
          phase_markers[2] = phase_markers[1];
          phase_markers[3] = phase_markers[1];
        }
      } else if (fused_gate_up) {
        evaluate_selected_gate_up_activation(
            xp, q4_input, hidden_, selected,
            gate_payloads, gate_codebooks, gate_blocks_, gate_bits_,
            gate_layouts_,
            up_payloads, up_codebooks, up_blocks_, up_bits_, up_layouts_,
            intermediate_, activation_limit, activation,
            situ_beta, situ_linear_beta, gatep);
#pragma omp single
        {
          phase_markers[1] = wall_seconds();
          phase_markers[2] = phase_markers[1];
          phase_markers[3] = phase_markers[1];
        }
      } else {
        evaluate_selected_projection(
            xp, selected, gate_payloads, gate_codebooks,
            gate_transposed_, gate_paired_bf16_, input_bf16p,
            gate_rows_, gate_blocks_, gate_bits_,
            gate_layouts_, gate_groups.first, gate_groups.second,
            gate_score_modes, intermediate_, scorep, gatep);
#pragma omp single
        { phase_markers[1] = wall_seconds(); }

        evaluate_selected_projection(
            xp, selected, up_payloads, up_codebooks,
            up_transposed_, up_paired_bf16_, input_bf16p,
            up_rows_, up_blocks_, up_bits_,
            up_layouts_, up_groups.first, up_groups.second,
            up_score_modes, intermediate_, scorep, upp);
#pragma omp single
        { phase_markers[2] = wall_seconds(); }

#pragma omp for schedule(static)
        for (int64_t item = 0; item < top_k * intermediate_; ++item) {
          float gate = gatep[item];
          float up = upp[item];
          apply_gated_activation(
              gate, up, activation_limit, activation,
              situ_beta, situ_linear_beta, gatep[item]);
        }
#pragma omp single
        { phase_markers[3] = wall_seconds(); }
      }

      if (!q4_experts_ && any_q4_experts_) {
#pragma omp for schedule(static)
        for (int64_t task = 0;
             task < top_k * (intermediate_ / 32); ++task) {
          const int64_t slot = task / (intermediate_ / 32);
          const int64_t block = task - slot * (intermediate_ / 32);
          const int64_t expert = selected[slot];
          if (down_layouts_[expert] == 3) {
            quantize_q8_block32(
                gatep + slot * intermediate_ + block * 32,
                q4_activation + slot * (intermediate_ / 32) + block);
          }
        }
      }

      if (q4_experts_) {
        evaluate_selected_q4_down_reduced(
            q4_activation, intermediate_, selected, down_payloads,
            hidden_, routep, resultp);
      } else if (fused_down_reduce) {
        evaluate_selected_direct_rows_reduced(
            gatep, q4_activation, intermediate_, selected,
            down_payloads, down_codebooks, down_blocks_, down_bits_,
            down_layouts_,
            hidden_, routep, resultp);
      } else if (down_direct) {
        evaluate_selected_direct_rows(
            gatep, intermediate_, selected,
            down_payloads, down_codebooks, down_rows_,
            down_blocks_, down_bits_, down_layouts_, hidden_, downp);
      } else {
        for (int64_t slot = 0; slot < top_k; ++slot) {
          const int64_t expert = selected[slot];
          evaluate_projection(
              gatep + slot * intermediate_, down_payloads[expert],
              down_codebooks[expert], down_transposed_[expert],
              down_rows_[expert], down_blocks_[expert],
              down_bits_[expert], down_layouts_[expert], scorep,
              downp + slot * hidden_);
        }
      }
#pragma omp single
      { phase_markers[4] = wall_seconds(); }

      if (!fused_down_reduce) {
#pragma omp for schedule(static)
        for (int64_t row = 0; row < hidden_; ++row) {
          float value = 0.0f;
          for (int64_t slot = 0; slot < top_k; ++slot) {
            value += downp[slot * hidden_ + row] * routep[slot];
          }
          resultp[row] = value;
        }
      }
#pragma omp single
      { phase_markers[5] = wall_seconds(); }
    }
    for (int64_t phase = 0; phase < 5; ++phase) {
      three_projection_phase_seconds[phase] +=
          phase_markers[phase + 1] - phase_markers[phase];
    }
    ++three_projection_phase_calls;
    return result_;
  }

 private:
  static std::vector<torch::Tensor> normalize_codebooks(
      std::vector<torch::Tensor> codebooks) {
    for (auto& codebook : codebooks) {
      codebook = codebook.to(torch::kFloat32).contiguous();
    }
    return codebooks;
  }

  static std::vector<torch::Tensor> transpose_codebooks(
      const std::vector<torch::Tensor>& codebooks) {
    std::vector<torch::Tensor> output;
    output.reserve(codebooks.size());
    for (const auto& codebook : codebooks) {
      output.push_back(cached_transposed_codebook(codebook));
    }
    return output;
  }

  static std::vector<torch::Tensor> paired_codebooks(
      const std::vector<torch::Tensor>& codebooks,
      const std::vector<int64_t>& layouts) {
    std::vector<torch::Tensor> output(codebooks.size());
    for (int64_t expert = 0;
         expert < static_cast<int64_t>(codebooks.size()); ++expert) {
      if (layouts[expert] == 1) {
        output[expert] = cached_paired_bf16_codebook(
            codebooks[expert]);
      }
    }
    return output;
  }

  static void validate_projection(
      const char* name,
      int64_t experts,
      const std::vector<torch::Tensor>& payloads,
      const std::vector<torch::Tensor>& codebooks,
      const std::vector<int64_t>& rows,
      const std::vector<int64_t>& blocks,
      const std::vector<int64_t>& bits,
      const std::vector<int64_t>& layouts) {
    TORCH_CHECK(
        (int64_t)payloads.size() == experts &&
            (int64_t)codebooks.size() == experts &&
            (int64_t)rows.size() == experts &&
            (int64_t)blocks.size() == experts &&
            (int64_t)bits.size() == experts &&
            (int64_t)layouts.size() == experts,
        name, " mixed projection metadata counts must match");
    for (int64_t expert = 0; expert < experts; ++expert) {
      TORCH_CHECK(
          !payloads[expert].is_cuda() &&
              payloads[expert].scalar_type() == at::kByte &&
              payloads[expert].is_contiguous(),
          name, " payload must be contiguous CPU uint8");
      TORCH_CHECK(
          codebooks[expert].dim() == 2 &&
              codebooks[expert].size(0) <= (int64_t{1} << bits[expert]) &&
              bits[expert] >= 8 && bits[expert] <= 16 &&
              (layouts[expert] >= 0 && layouts[expert] <= 3),
          name, " codebook/packed width mismatch");
      if (layouts[expert] == 3) {
        const int64_t cols = blocks[expert] * codebooks[expert].size(1);
        TORCH_CHECK(
            cols % 32 == 0 &&
                payloads[expert].numel() ==
                    rows[expert] * (cols / 32) *
                        static_cast<int64_t>(sizeof(Q4Block32)),
            name, " Q4 execution image length mismatch");
      } else {
        const int64_t expected_bits =
            rows[expert] * blocks[expert] * bits[expert];
        TORCH_CHECK(
            expected_bits % 8 == 0 &&
                payloads[expert].numel() == expected_bits / 8,
            name, " packed payload length mismatch");
      }
    }
  }

  static void evaluate_projection(
      const float* input,
      const torch::Tensor& payload,
      const torch::Tensor& codebook,
      const torch::Tensor& transposed,
      int64_t rows,
      int64_t blocks,
      int64_t bits,
      int64_t layout,
      float* scores,
      float* output) {
    const int64_t codes = codebook.size(0);
    const int64_t dim = codebook.size(1);
    const bool use_direct =
        layout == 2 || layout == 3 || rows * dim < codes * dim + rows;
    const uint8_t* packed = payload.data_ptr<uint8_t>();
    if (use_direct) {
      const float* cb = codebook.data_ptr<float>();
#pragma omp for schedule(static)
      for (int64_t row = 0; row < rows; ++row) {
        if (layout == 3) {
          direct_dot_packed_rows8(
              input, cb, packed, row, 1, blocks, bits, dim, layout,
              output + row);
        } else {
          output[row] = layout == 0
              ? direct_dot_packed(
                    input, cb, packed, row * blocks, blocks, bits, dim)
              : layout == 1
              ? direct_dot_packed_block_major(
                    input, cb, packed, row, rows, blocks, bits, dim)
              : direct_dot_packed_row_tile8(
                    input, cb, packed, row, rows, blocks, bits, dim);
        }
      }
      return;
    }
    TORCH_INTERNAL_ASSERT(layout == 0);
    const float* cb = transposed.data_ptr<float>();
#pragma omp for schedule(static)
    for (int64_t block = 0; block < blocks; ++block) {
      codebook_scores(
          input + block * dim,
          cb,
          scores + block * codes,
          codes,
          dim);
    }
#pragma omp for schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
      output[row] = lookup_sum_packed(
          scores, packed, row * blocks, blocks, codes, bits);
    }
  }

  static bool selected_direct(
      const std::vector<int64_t>& selected,
      const std::vector<torch::Tensor>& codebooks,
      const std::vector<int64_t>& rows,
      const std::vector<int64_t>& layouts) {
    for (const int64_t expert : selected) {
      if (layouts[expert] == 3) {
        continue;
      }
      const int64_t codes = codebooks[expert].size(0);
      const int64_t dim = codebooks[expert].size(1);
      if (rows[expert] * dim >= codes * dim + rows[expert]) {
        return false;
      }
    }
    return true;
  }

  static bool selected_row_major(
      const std::vector<int64_t>& selected,
      const std::vector<int64_t>& layouts) {
    for (const int64_t expert : selected) {
      if (layouts[expert] == 1) {
        return false;
      }
    }
    return true;
  }

  using SlotGroups = std::vector<std::vector<int64_t>>;

  static std::pair<SlotGroups, std::vector<int64_t>> projection_groups(
      const std::vector<int64_t>& selected,
      const std::vector<torch::Tensor>& codebooks,
      const std::vector<int64_t>& blocks,
      const std::vector<int64_t>& bits,
      const std::vector<int64_t>& layouts) {
    SlotGroups groups;
    std::vector<int64_t> group_for_slot(selected.size(), -1);
    for (int64_t slot = 0;
         slot < static_cast<int64_t>(selected.size()); ++slot) {
      const int64_t expert = selected[slot];
      int64_t match = -1;
      for (int64_t group = 0;
           group < static_cast<int64_t>(groups.size()); ++group) {
        const int64_t other = selected[groups[group][0]];
        if (codebooks[expert].data_ptr() == codebooks[other].data_ptr() &&
            blocks[expert] == blocks[other] && bits[expert] == bits[other] &&
            layouts[expert] == layouts[other]) {
          match = group;
          break;
        }
      }
      if (match < 0) {
        match = groups.size();
        groups.push_back({});
      }
      groups[match].push_back(slot);
      group_for_slot[slot] = match;
    }
    return {std::move(groups), std::move(group_for_slot)};
  }

  static std::vector<uint8_t> projection_score_modes(
      const std::vector<int64_t>& selected,
      const SlotGroups& groups,
      const std::vector<torch::Tensor>& codebooks,
      const std::vector<int64_t>& rows,
      const std::vector<int64_t>& blocks,
      const std::vector<int64_t>& layouts) {
    std::vector<uint8_t> modes(groups.size(), 0);
    for (int64_t group = 0;
         group < static_cast<int64_t>(groups.size()); ++group) {
      const int64_t expert = selected[groups[group][0]];
      if (layouts[expert] != 1) {
        continue;
      }
      const int64_t members = groups[group].size();
      const int64_t codes = codebooks[expert].size(0);
      const int64_t dim = codebooks[expert].size(1);
      const int64_t direct_work =
          members * rows[expert] * blocks[expert] * dim;
      const int64_t score_work =
          blocks[expert] * codes * dim +
          members * rows[expert] * blocks[expert];
      modes[group] = score_work < direct_work ? 1 : 0;
    }
    return modes;
  }

  static inline void apply_gated_activation(
      float gate,
      float up,
      float activation_limit,
      const std::string& activation,
      float situ_beta,
      float situ_linear_beta,
      float& output) {
    if (activation_limit != 0.0f) {
      gate = std::min(gate, activation_limit);
      up = std::max(-activation_limit, std::min(up, activation_limit));
    }
    if (activation == "situ") {
      float linear = up;
      if (situ_linear_beta > 0.0f) {
        linear = situ_linear_beta * std::tanh(up / situ_linear_beta);
      }
      output = situ_beta * std::tanh(gate / situ_beta) *
          (1.0f / (1.0f + std::exp(-gate))) * linear;
    } else {
      output = gate * (1.0f / (1.0f + std::exp(-gate))) * up;
    }
  }

  static void evaluate_selected_q4_gate_up_activation(
      const Q8Block32* input,
      const std::vector<int64_t>& selected,
      const std::vector<torch::Tensor>& gate_payloads,
      const std::vector<torch::Tensor>& up_payloads,
      int64_t input_columns,
      int64_t common_rows,
      float activation_limit,
      const std::string& activation,
      float situ_beta,
      float situ_linear_beta,
      float* output) {
    const int64_t input_blocks = input_columns / 32;
    const int64_t row_groups = (common_rows + 7) / 8;
    const int64_t task_tiles = packed_l2_task_tiles();
    auto evaluate = [&](int64_t group, int64_t slot) {
      const int64_t first_row = group * 8;
      const int64_t valid_rows = std::min<int64_t>(8, common_rows - first_row);
      const int64_t expert = selected[slot];
      const auto* gate = reinterpret_cast<const Q4Block32*>(
          gate_payloads[expert].data_ptr<uint8_t>());
      const auto* up = reinterpret_cast<const Q4Block32*>(
          up_payloads[expert].data_ptr<uint8_t>());
      float* destination = output + slot * common_rows + first_row;
      alignas(64) float gate_values[kQ4BlockMajorRows];
      alignas(64) float up_values[kQ4BlockMajorRows];
      q4_q8_block_major_rows8(
          gate, first_row, valid_rows, input, input_blocks, gate_values);
      q4_q8_block_major_rows8(
          up, first_row, valid_rows, input, input_blocks, up_values);
      for (int64_t local = 0; local < valid_rows; ++local) {
        apply_gated_activation(
            gate_values[local], up_values[local], activation_limit, activation,
            situ_beta, situ_linear_beta, destination[local]);
      }
    };
    if (q4_numa_local_enabled()) {
      const auto range = q4_numa_local_row_range(row_groups);
      for (int64_t group = range.first; group < range.second; ++group) {
        for (int64_t slot = 0;
             slot < static_cast<int64_t>(selected.size()); ++slot) {
          evaluate(group, slot);
        }
      }
    } else {
#pragma omp for schedule(dynamic, task_tiles)
      for (int64_t task = 0;
           task < static_cast<int64_t>(selected.size()) * row_groups; ++task) {
        const int64_t slot = task / row_groups;
        evaluate(task - slot * row_groups, slot);
      }
    }
  }

  static void evaluate_selected_q4_down_shared_reduced(
      const Q8Block32* routed_inputs,
      int64_t input_columns,
      const std::vector<int64_t>& selected,
      const std::vector<torch::Tensor>& down_payloads,
      int64_t common_rows,
      const float* route,
      const at::BFloat16* shared_input_bf16,
      const Q8Block32* shared_input_q8,
      const torch::Tensor& shared_weight,
      const torch::Tensor& shared_scales,
      bool shared_q4,
      int64_t shared_block,
      float* output) {
    const int64_t input_blocks = input_columns / 32;
    const int64_t row_groups = (common_rows + 7) / 8;
    auto evaluate = [&](int64_t group) {
      const int64_t first_row = group * 8;
      const int64_t valid_rows = std::min<int64_t>(8, common_rows - first_row);
      alignas(64) float values[kQ4BlockMajorRows] = {0.0f};
      if (shared_q4) {
        const auto* shared = reinterpret_cast<const Q4Block32*>(
            shared_weight.data_ptr<uint8_t>());
        q4_q8_block_major_rows8(
            shared, first_row, valid_rows, shared_input_q8, input_blocks,
            values);
      } else {
        for (int64_t local = 0; local < valid_rows; ++local) {
          const int64_t row = first_row + local;
          values[local] = block_fp8_logical_row_dot_bf16(
              shared_input_bf16, shared_weight, shared_scales, row,
              common_rows, input_columns, shared_block);
        }
      }
      alignas(64) float routed[kQ4BlockMajorRows];
      for (int64_t slot = 0;
           slot < static_cast<int64_t>(selected.size()); ++slot) {
        const int64_t expert = selected[slot];
        const auto* down = reinterpret_cast<const Q4Block32*>(
            down_payloads[expert].data_ptr<uint8_t>());
        q4_q8_block_major_rows8(
            down, first_row, valid_rows,
            routed_inputs + slot * input_blocks, input_blocks, routed);
        for (int64_t local = 0; local < valid_rows; ++local) {
          values[local] += route[slot] * routed[local];
        }
      }
      for (int64_t local = 0; local < valid_rows; ++local) {
        output[first_row + local] = values[local];
      }
    };
    if (q4_numa_local_enabled()) {
      const auto range = q4_numa_local_row_range(row_groups);
      for (int64_t group = range.first; group < range.second; ++group) {
        evaluate(group);
      }
    } else {
#pragma omp for schedule(static)
      for (int64_t group = 0; group < row_groups; ++group) {
        evaluate(group);
      }
    }
  }

  static void evaluate_selected_q4_down_reduced(
      const Q8Block32* routed_inputs,
      int64_t input_columns,
      const std::vector<int64_t>& selected,
      const std::vector<torch::Tensor>& down_payloads,
      int64_t common_rows,
      const float* route,
      float* output) {
    const int64_t input_blocks = input_columns / 32;
    const int64_t row_groups = (common_rows + 7) / 8;
    auto evaluate = [&](int64_t group) {
      const int64_t first_row = group * 8;
      const int64_t valid_rows = std::min<int64_t>(8, common_rows - first_row);
      alignas(64) float values[kQ4BlockMajorRows] = {0.0f};
      alignas(64) float routed[kQ4BlockMajorRows];
      for (int64_t slot = 0;
           slot < static_cast<int64_t>(selected.size()); ++slot) {
        const int64_t expert = selected[slot];
        const auto* down = reinterpret_cast<const Q4Block32*>(
            down_payloads[expert].data_ptr<uint8_t>());
        q4_q8_block_major_rows8(
            down, first_row, valid_rows,
            routed_inputs + slot * input_blocks, input_blocks, routed);
        for (int64_t local = 0; local < valid_rows; ++local) {
          values[local] += route[slot] * routed[local];
        }
      }
      for (int64_t local = 0; local < valid_rows; ++local) {
        output[first_row + local] = values[local];
      }
    };
    if (q4_numa_local_enabled()) {
      const auto range = q4_numa_local_row_range(row_groups);
      for (int64_t group = range.first; group < range.second; ++group) {
        evaluate(group);
      }
    } else {
#pragma omp for schedule(static)
      for (int64_t group = 0; group < row_groups; ++group) {
        evaluate(group);
      }
    }
  }

  // Gate and Up share the input and output row partition.  Decode both
  // compact projections and apply the gated activation while the eight rows
  // are still hot, eliminating two full intermediate sweeps and two team
  // barriers without materializing either expert matrix.
  static void evaluate_selected_gate_up_activation(
      const float* input,
      const Q8Block32* input_q8,
      int64_t input_columns,
      const std::vector<int64_t>& selected,
      const std::vector<torch::Tensor>& gate_payloads,
      const std::vector<torch::Tensor>& gate_codebooks,
      const std::vector<int64_t>& gate_blocks,
      const std::vector<int64_t>& gate_bits,
      const std::vector<int64_t>& gate_layouts,
      const std::vector<torch::Tensor>& up_payloads,
      const std::vector<torch::Tensor>& up_codebooks,
      const std::vector<int64_t>& up_blocks,
      const std::vector<int64_t>& up_bits,
      const std::vector<int64_t>& up_layouts,
      int64_t common_rows,
      float activation_limit,
      const std::string& activation,
      float situ_beta,
      float situ_linear_beta,
      float* output) {
    const int64_t top_k = selected.size();
    const int64_t row_groups = (common_rows + 7) / 8;
    const int64_t task_tiles = packed_l2_task_tiles();
#pragma omp for schedule(dynamic, task_tiles)
    for (int64_t task = 0; task < top_k * row_groups; ++task) {
      const int64_t slot = task / row_groups;
      const int64_t first_row = (task - slot * row_groups) * 8;
      const int64_t valid_rows = std::min<int64_t>(
          8, common_rows - first_row);
      const int64_t expert = selected[slot];
      alignas(64) float gate_values[8];
      alignas(64) float up_values[8];
      const bool gate_q4 = gate_layouts[expert] == 3;
      const bool up_q4 = up_layouts[expert] == 3;
      bool paired = false;
      if (!gate_q4 && !up_q4) {
        paired = direct_dot_packed_gate_up_row_tile8(
            input,
            gate_codebooks[expert].data_ptr<float>(),
            gate_payloads[expert].data_ptr<uint8_t>(),
            gate_blocks[expert],
            gate_bits[expert],
            gate_codebooks[expert].size(1),
            gate_layouts[expert],
            up_codebooks[expert].data_ptr<float>(),
            up_payloads[expert].data_ptr<uint8_t>(),
            up_blocks[expert],
            up_bits[expert],
            up_codebooks[expert].size(1),
            up_layouts[expert],
            first_row,
            valid_rows,
            gate_values,
            up_values);
      }
      if (!paired && gate_q4) {
        TORCH_INTERNAL_ASSERT(input_q8 != nullptr && input_columns % 32 == 0);
        q4_q8_block_major_rows8(
            reinterpret_cast<const Q4Block32*>(
                gate_payloads[expert].data_ptr<uint8_t>()),
            first_row, valid_rows, input_q8, input_columns / 32,
            gate_values);
      } else if (!paired) {
        direct_dot_packed_rows8(
            input,
            gate_codebooks[expert].data_ptr<float>(),
            gate_payloads[expert].data_ptr<uint8_t>(),
            first_row,
            valid_rows,
            gate_blocks[expert],
            gate_bits[expert],
            gate_codebooks[expert].size(1),
            gate_layouts[expert],
            gate_values);
      }
      if (!paired && up_q4) {
        TORCH_INTERNAL_ASSERT(input_q8 != nullptr && input_columns % 32 == 0);
        q4_q8_block_major_rows8(
            reinterpret_cast<const Q4Block32*>(
                up_payloads[expert].data_ptr<uint8_t>()),
            first_row, valid_rows, input_q8, input_columns / 32,
            up_values);
      } else if (!paired) {
        direct_dot_packed_rows8(
            input,
            up_codebooks[expert].data_ptr<float>(),
            up_payloads[expert].data_ptr<uint8_t>(),
            first_row,
            valid_rows,
            up_blocks[expert],
            up_bits[expert],
            up_codebooks[expert].size(1),
            up_layouts[expert],
            up_values);
      }
      float* destination = output + slot * common_rows + first_row;
      for (int64_t row = 0; row < valid_rows; ++row) {
        apply_gated_activation(
            gate_values[row], up_values[row], activation_limit,
            activation, situ_beta, situ_linear_beta, destination[row]);
      }
    }
  }

  static void evaluate_selected_projection(
      const float* input,
      const std::vector<int64_t>& selected,
      const std::vector<torch::Tensor>& payloads,
      const std::vector<torch::Tensor>& codebooks,
      const std::vector<torch::Tensor>& transposed,
      const std::vector<torch::Tensor>& paired_bf16,
      const at::BFloat16* input_bf16,
      const std::vector<int64_t>& rows,
      const std::vector<int64_t>& blocks,
      const std::vector<int64_t>& bits,
      const std::vector<int64_t>& layouts,
      const SlotGroups& groups,
      const std::vector<int64_t>& group_for_slot,
      const std::vector<uint8_t>& score_modes,
      int64_t common_rows,
      float* scores,
      float* output) {
    const int64_t top_k = selected.size();
    const bool any_score = std::find(
        score_modes.begin(), score_modes.end(), uint8_t{1}) !=
        score_modes.end();
    bool all_direct_layout = true;
    for (const int64_t expert : selected) {
      all_direct_layout = all_direct_layout && layouts[expert] != 1;
    }
    if (packed_direct_rows8_enabled() && !any_score && all_direct_layout) {
      const int64_t row_groups = (common_rows + 7) / 8;
#pragma omp for schedule(static)
      for (int64_t task = 0; task < top_k * row_groups; ++task) {
        const int64_t slot = task / row_groups;
        const int64_t first_row = (task - slot * row_groups) * 8;
        const int64_t valid_rows = std::min<int64_t>(
            8, common_rows - first_row);
        const int64_t expert = selected[slot];
        direct_dot_packed_rows8(
            input,
            codebooks[expert].data_ptr<float>(),
            payloads[expert].data_ptr<uint8_t>(),
            first_row,
            valid_rows,
            blocks[expert],
            bits[expert],
            codebooks[expert].size(1),
            layouts[expert],
            output + slot * common_rows + first_row);
      }
      return;
    }
#pragma omp for schedule(static)
    for (int64_t item = 0; item < top_k * common_rows; ++item) {
      const int64_t slot = item / common_rows;
      const int64_t row = item - slot * common_rows;
      const int64_t expert = selected[slot];
      TORCH_INTERNAL_ASSERT(rows[expert] == common_rows);
      if (score_modes[group_for_slot[slot]]) {
        output[item] = 0.0f;
        continue;
      }
      const float* codebook = codebooks[expert].data_ptr<float>();
      const uint8_t* payload = payloads[expert].data_ptr<uint8_t>();
      const int64_t dim = codebooks[expert].size(1);
      output[item] = layouts[expert] == 0
          ? direct_dot_packed(
                input, codebook, payload, row * blocks[expert],
                blocks[expert], bits[expert], dim)
          : layouts[expert] == 1
          ? direct_dot_packed_block_major(
                input, codebook, payload, row, rows[expert],
                blocks[expert], bits[expert], dim)
          : direct_dot_packed_row_tile8(
                input, codebook, payload, row, rows[expert],
                blocks[expert], bits[expert], dim);
    }

    constexpr int64_t block_chunk = 32;
    constexpr int64_t code_chunk = 256;
    for (int64_t group = 0;
         group < static_cast<int64_t>(groups.size()); ++group) {
      if (!score_modes[group]) {
        continue;
      }
      const int64_t representative = selected[groups[group][0]];
      const int64_t group_blocks = blocks[representative];
      const int64_t codes = codebooks[representative].size(0);
      const int64_t dim = codebooks[representative].size(1);
      const int64_t code_chunks = (codes + code_chunk - 1) / code_chunk;
      const float* codebook = transposed[representative].data_ptr<float>();
      const at::BFloat16* bf16_codebook =
          paired_bf16[representative].data_ptr<at::BFloat16>();
      for (int64_t first_block = 0;
           first_block < group_blocks;
           first_block += block_chunk) {
        const int64_t chunk_blocks = std::min(
            block_chunk, group_blocks - first_block);
#pragma omp for schedule(static)
        for (int64_t task = 0;
             task < chunk_blocks * code_chunks; ++task) {
          const int64_t local_block = task / code_chunks;
          const int64_t code_part = task - local_block * code_chunks;
          const int64_t code_begin = code_part * code_chunk;
          const int64_t code_end = std::min(code_begin + code_chunk, codes);
          const int64_t block = first_block + local_block;
          if (input_bf16 != nullptr) {
            codebook_scores_bf16_range(
                input_bf16 + block * dim,
                bf16_codebook,
                scores + local_block * codes,
                codes,
                dim,
                code_begin,
                code_end);
          } else {
            codebook_scores_range(
                input + block * dim,
                codebook,
                scores + local_block * codes,
                codes,
                dim,
                code_begin,
                code_end);
          }
        }
        const int64_t group_items =
            static_cast<int64_t>(groups[group].size()) * common_rows;
#pragma omp for schedule(static)
        for (int64_t item = 0; item < group_items; ++item) {
          const int64_t member = item / common_rows;
          const int64_t row = item - member * common_rows;
          const int64_t slot = groups[group][member];
          const int64_t expert = selected[slot];
          const uint8_t* payload = payloads[expert].data_ptr<uint8_t>();
          float value = output[slot * common_rows + row];
          for (int64_t local_block = 0;
               local_block < chunk_blocks; ++local_block) {
            const int64_t block = first_block + local_block;
            const uint16_t index = read_packed_index(
                payload, block * common_rows + row, bits[expert]);
            value += scores[local_block * codes + index];
          }
          output[slot * common_rows + row] = value;
        }
      }
    }
  }

  // Decode all routed experts in one work-sharing loop.  The old mixed path
  // entered one omp-for per expert (16 barriers per projection); Top-K is a
  // scheduling dimension too, so one [top_k, rows] range keeps all cores busy
  // and preserves each output row's exact accumulation order.
  static void evaluate_selected_direct(
      const float* input,
      const std::vector<int64_t>& selected,
      const std::vector<torch::Tensor>& payloads,
      const std::vector<torch::Tensor>& codebooks,
      const std::vector<int64_t>& rows,
      const std::vector<int64_t>& blocks,
      const std::vector<int64_t>& bits,
      const std::vector<int64_t>& layouts,
      int64_t common_rows,
      float* output) {
    const int64_t top_k = selected.size();
    bool all_direct_layout = true;
    for (const int64_t expert : selected) {
      all_direct_layout = all_direct_layout && layouts[expert] != 1;
    }
    if (packed_direct_rows8_enabled() && all_direct_layout) {
      const int64_t row_groups = (common_rows + 7) / 8;
#pragma omp for schedule(static)
      for (int64_t task = 0; task < top_k * row_groups; ++task) {
        const int64_t slot = task / row_groups;
        const int64_t first_row = (task - slot * row_groups) * 8;
        const int64_t valid_rows = std::min<int64_t>(
            8, common_rows - first_row);
        const int64_t expert = selected[slot];
        direct_dot_packed_rows8(
            input,
            codebooks[expert].data_ptr<float>(),
            payloads[expert].data_ptr<uint8_t>(),
            first_row,
            valid_rows,
            blocks[expert],
            bits[expert],
            codebooks[expert].size(1),
            layouts[expert],
            output + slot * common_rows + first_row);
      }
      return;
    }
#pragma omp for schedule(static)
    for (int64_t item = 0; item < top_k * common_rows; ++item) {
      const int64_t slot = item / common_rows;
      const int64_t row = item - slot * common_rows;
      const int64_t expert = selected[slot];
      TORCH_INTERNAL_ASSERT(rows[expert] == common_rows);
      output[item] = direct_dot_packed(
          input,
          codebooks[expert].data_ptr<float>(),
          payloads[expert].data_ptr<uint8_t>(),
          row * blocks[expert],
          blocks[expert],
          bits[expert],
          codebooks[expert].size(1));
    }
  }

  static void evaluate_selected_direct_rows(
      const float* inputs,
      int64_t input_stride,
      const std::vector<int64_t>& selected,
      const std::vector<torch::Tensor>& payloads,
      const std::vector<torch::Tensor>& codebooks,
      const std::vector<int64_t>& rows,
      const std::vector<int64_t>& blocks,
      const std::vector<int64_t>& bits,
      const std::vector<int64_t>& layouts,
      int64_t common_rows,
      float* output) {
    const int64_t top_k = selected.size();
    bool all_direct_layout = true;
    for (const int64_t expert : selected) {
      all_direct_layout = all_direct_layout && layouts[expert] != 1;
    }
    if (packed_direct_rows8_enabled() && all_direct_layout) {
      const int64_t row_groups = (common_rows + 7) / 8;
#pragma omp for schedule(static)
      for (int64_t task = 0; task < top_k * row_groups; ++task) {
        const int64_t slot = task / row_groups;
        const int64_t first_row = (task - slot * row_groups) * 8;
        const int64_t valid_rows = std::min<int64_t>(
            8, common_rows - first_row);
        const int64_t expert = selected[slot];
        direct_dot_packed_rows8(
            inputs + slot * input_stride,
            codebooks[expert].data_ptr<float>(),
            payloads[expert].data_ptr<uint8_t>(),
            first_row,
            valid_rows,
            blocks[expert],
            bits[expert],
            codebooks[expert].size(1),
            layouts[expert],
            output + slot * common_rows + first_row);
      }
      return;
    }
#pragma omp for schedule(static)
    for (int64_t item = 0; item < top_k * common_rows; ++item) {
      const int64_t slot = item / common_rows;
      const int64_t row = item - slot * common_rows;
      const int64_t expert = selected[slot];
      TORCH_INTERNAL_ASSERT(rows[expert] == common_rows);
      const float* input = inputs + slot * input_stride;
      const float* codebook = codebooks[expert].data_ptr<float>();
      const uint8_t* payload = payloads[expert].data_ptr<uint8_t>();
      output[item] = layouts[expert] == 0
          ? direct_dot_packed(
                input, codebook, payload, row * blocks[expert],
                blocks[expert], bits[expert], codebooks[expert].size(1))
          : layouts[expert] == 1
          ? direct_dot_packed_block_major(
                input, codebook, payload, row, rows[expert],
                blocks[expert], bits[expert], codebooks[expert].size(1))
          : direct_dot_packed_row_tile8(
                input, codebook, payload, row, rows[expert],
                blocks[expert], bits[expert], codebooks[expert].size(1));
    }
  }

  // Compute every selected Down projection for one output tile and fold the
  // route weights immediately.  Result ownership is by output row, so no
  // atomics, per-expert Down tensor or second reduction pass is required.
  static void evaluate_selected_direct_rows_reduced(
      const float* inputs,
      const Q8Block32* q4_inputs,
      int64_t input_stride,
      const std::vector<int64_t>& selected,
      const std::vector<torch::Tensor>& payloads,
      const std::vector<torch::Tensor>& codebooks,
      const std::vector<int64_t>& blocks,
      const std::vector<int64_t>& bits,
      const std::vector<int64_t>& layouts,
      int64_t common_rows,
      const float* route,
      float* output) {
    const int64_t top_k = selected.size();
    const int64_t row_groups = (common_rows + 7) / 8;
#pragma omp for schedule(static)
    for (int64_t group = 0; group < row_groups; ++group) {
      const int64_t first_row = group * 8;
      const int64_t valid_rows = std::min<int64_t>(
          8, common_rows - first_row);
      alignas(64) float reduced[8] = {};
      alignas(64) float projected[8];
      for (int64_t slot = 0; slot < top_k; ++slot) {
        const int64_t expert = selected[slot];
        if (layouts[expert] == 3) {
          TORCH_INTERNAL_ASSERT(
              q4_inputs != nullptr && input_stride % 32 == 0);
          q4_q8_block_major_rows8(
              reinterpret_cast<const Q4Block32*>(
                  payloads[expert].data_ptr<uint8_t>()),
              first_row, valid_rows,
              q4_inputs + slot * (input_stride / 32), input_stride / 32,
              projected);
        } else {
          direct_dot_packed_rows8(
              inputs + slot * input_stride,
              codebooks[expert].data_ptr<float>(),
              payloads[expert].data_ptr<uint8_t>(),
              first_row,
              valid_rows,
              blocks[expert],
              bits[expert],
              codebooks[expert].size(1),
              layouts[expert],
              projected);
        }
        for (int64_t row = 0; row < valid_rows; ++row) {
          reduced[row] += projected[row] * route[slot];
        }
      }
      for (int64_t row = 0; row < valid_rows; ++row) {
        output[first_row + row] = reduced[row];
      }
    }
  }

  // The shared expert is block-FP8 while routed experts retain their native
  // p8--p16 VQ payloads.  Output-row ownership lets one loop calculate both
  // formats and perform the final route/shared merge without materializing a
  // shared output tensor or launching another reduction operator.
  static void evaluate_selected_down_shared_reduced(
      const float* inputs,
      const Q8Block32* q4_inputs,
      int64_t input_stride,
      const std::vector<int64_t>& selected,
      const std::vector<torch::Tensor>& payloads,
      const std::vector<torch::Tensor>& codebooks,
      const std::vector<int64_t>& blocks,
      const std::vector<int64_t>& bits,
      const std::vector<int64_t>& layouts,
      int64_t common_rows,
      const float* route,
      const at::BFloat16* shared_input,
      const Q8Block32* shared_input_q8,
      const torch::Tensor& shared_weight,
      const torch::Tensor& shared_scales,
      int64_t shared_rows,
      int64_t shared_cols,
      int64_t shared_block,
      bool shared_q4,
      float* output) {
    const int64_t top_k = selected.size();
    const int64_t row_groups = (common_rows + 7) / 8;
#pragma omp for schedule(static)
    for (int64_t group = 0; group < row_groups; ++group) {
      const int64_t first_row = group * 8;
      const int64_t valid_rows = std::min<int64_t>(
          8, common_rows - first_row);
      alignas(64) float reduced[8];
      alignas(64) float projected[8];
      if (shared_q4) {
        TORCH_INTERNAL_ASSERT(
            shared_input_q8 != nullptr && shared_cols % 32 == 0);
        q4_q8_block_major_rows8(
            reinterpret_cast<const Q4Block32*>(
                shared_weight.data_ptr<uint8_t>()),
            first_row,
            valid_rows,
            shared_input_q8,
            shared_cols / 32,
            reduced);
      } else {
        for (int64_t row = 0; row < valid_rows; ++row) {
          reduced[row] = block_fp8_logical_row_dot_bf16(
              shared_input,
              shared_weight,
              shared_scales,
              first_row + row,
              shared_rows,
              shared_cols,
              shared_block);
        }
      }
      for (int64_t slot = 0; slot < top_k; ++slot) {
        const int64_t expert = selected[slot];
        if (layouts[expert] == 3) {
          TORCH_INTERNAL_ASSERT(
              q4_inputs != nullptr && input_stride % 32 == 0);
          q4_q8_block_major_rows8(
              reinterpret_cast<const Q4Block32*>(
                  payloads[expert].data_ptr<uint8_t>()),
              first_row, valid_rows,
              q4_inputs + slot * (input_stride / 32), input_stride / 32,
              projected);
        } else {
          direct_dot_packed_rows8(
              inputs + slot * input_stride,
              codebooks[expert].data_ptr<float>(),
              payloads[expert].data_ptr<uint8_t>(),
              first_row,
              valid_rows,
              blocks[expert],
              bits[expert],
              codebooks[expert].size(1),
              layouts[expert],
              projected);
        }
        for (int64_t row = 0; row < valid_rows; ++row) {
          reduced[row] += projected[row] * route[slot];
        }
      }
      for (int64_t row = 0; row < valid_rows; ++row) {
        output[first_row + row] = reduced[row];
      }
    }
  }

  std::vector<torch::Tensor> gate_payloads_;
  std::vector<torch::Tensor> gate_codebooks_;
  std::vector<torch::Tensor> gate_transposed_;
  std::vector<torch::Tensor> gate_paired_bf16_;
  std::vector<int64_t> gate_rows_, gate_blocks_, gate_bits_, gate_layouts_;
  std::vector<torch::Tensor> up_payloads_;
  std::vector<torch::Tensor> up_codebooks_;
  std::vector<torch::Tensor> up_transposed_;
  std::vector<torch::Tensor> up_paired_bf16_;
  std::vector<int64_t> up_rows_, up_blocks_, up_bits_, up_layouts_;
  std::vector<torch::Tensor> down_payloads_;
  std::vector<torch::Tensor> down_codebooks_;
  std::vector<torch::Tensor> down_transposed_;
  std::vector<int64_t> down_rows_, down_blocks_, down_bits_, down_layouts_;
  torch::Tensor router_weight_, router_bias_, router_mask_;
  std::vector<torch::Tensor> shared_weights_, shared_scales_;
  std::vector<int64_t> shared_rows_, shared_cols_;
  std::vector<bool> shared_q4_;
  torch::Tensor route_scores_, route_weights_;
  torch::Tensor shared_activation_, shared_activation_bf16_;
  std::vector<int64_t> selected_;
  std::vector<float> route_choices_;
  int64_t shared_block_ = 128;
  int64_t route_top_k_ = 0;
  bool normalize_route_ = true;
  float routed_scaling_ = 1.0f;
  bool fused_moe_ready_ = false;
  bool q4_experts_ = false;
  bool any_q4_experts_ = false;
  std::vector<torch::Tensor> latent_input_weights_, latent_input_scales_;
  std::vector<int64_t> latent_input_rows_, latent_input_kinds_;
  std::vector<torch::Tensor> latent_output_weights_, latent_output_scales_;
  std::vector<int64_t> latent_output_rows_, latent_output_cols_;
  std::vector<int64_t> latent_output_kinds_;
  torch::Tensor latent_route_correction_, latent_route_mask_;
  torch::Tensor latent_routed_norm_;
  torch::Tensor latent_input_float_, latent_input_bf16_;
  torch::Tensor latent_shared_gate_, latent_shared_up_;
  torch::Tensor latent_shared_activation_, latent_shared_activation_bf16_;
  torch::Tensor latent_projected_, latent_routed_normalized_;
  torch::Tensor latent_routed_normalized_bf16_, latent_full_output_;
  int64_t latent_input_cols_ = 0;
  int64_t latent_block_size_ = 128;
  float latent_rms_eps_ = 1.0e-6f;
  float latent_limit_ = 0.0f;
  float latent_beta_ = 1.0f;
  float latent_linear_beta_ = -1.0f;
  std::string latent_scoring_ = "sigmoid";
  std::string latent_activation_ = "situ";
  bool latent_moe_ready_ = false;
  int64_t hidden_ = 0;
  int64_t intermediate_ = 0;
  bool has_score_layout_ = false;
  torch::Tensor score_, gate_values_, up_values_, down_values_, result_;
  torch::Tensor input_bf16_, input_q8_, activation_q8_, shared_activation_q8_;
  std::mutex mutex_;
};

void reset_three_projection_phase_profile_cpu() {
  std::fill(
      std::begin(three_projection_phase_seconds),
      std::end(three_projection_phase_seconds),
      0.0);
  three_projection_phase_calls = 0;
}

std::vector<double> three_projection_phase_profile_cpu() {
  return {
      static_cast<double>(three_projection_phase_calls),
      three_projection_phase_seconds[0],
      three_projection_phase_seconds[1],
      three_projection_phase_seconds[2],
      three_projection_phase_seconds[3],
      three_projection_phase_seconds[4],
  };
}

void reset_resident_moe_phase_profile_cpu() {
  std::fill(
      std::begin(resident_moe_phase_seconds),
      std::end(resident_moe_phase_seconds),
      0.0);
  resident_moe_phase_calls = 0;
  resident_moe_selected_experts = 0;
  resident_moe_q4_selected_experts = 0;
}

void reset_latent_moe_phase_profile_cpu() {
  std::fill(
      std::begin(latent_moe_phase_seconds),
      std::end(latent_moe_phase_seconds),
      0.0);
  latent_moe_phase_calls = 0;
}

void reset_resident_projection_profile_cpu() {
  resident_projection_seconds = 0.0;
  resident_projection_calls = 0;
}

std::vector<double> resident_projection_profile_cpu() {
  return {
      static_cast<double>(resident_projection_calls),
      resident_projection_seconds,
  };
}

std::vector<double> resident_moe_phase_profile_cpu() {
  return {
      static_cast<double>(resident_moe_phase_calls),
      resident_moe_phase_seconds[0],
      resident_moe_phase_seconds[1],
      resident_moe_phase_seconds[2],
      static_cast<double>(resident_moe_selected_experts),
      static_cast<double>(resident_moe_q4_selected_experts),
  };
}

std::vector<double> latent_moe_phase_profile_cpu() {
  return {
      static_cast<double>(latent_moe_phase_calls),
      latent_moe_phase_seconds[0],
      latent_moe_phase_seconds[1],
      latent_moe_phase_seconds[2],
      latent_moe_phase_seconds[3],
  };
}

std::vector<torch::Tensor> route_topk_sigmoid_cpu(
    torch::Tensor logits,
    torch::Tensor bias,
    torch::Tensor mask,
    int64_t top_k,
    bool normalize,
    double scaling) {
  TORCH_CHECK(
      !logits.is_cuda() && logits.dim() == 2,
      "CPU sigmoid Top-K requires [batch, experts] logits");
  TORCH_CHECK(
      !bias.is_cuda() && bias.numel() == logits.size(1),
      "CPU sigmoid Top-K bias width mismatch");
  TORCH_CHECK(
      !mask.is_cuda() && mask.scalar_type() == at::kBool &&
          mask.numel() == logits.size(1),
      "CPU sigmoid Top-K mask width mismatch");
  TORCH_CHECK(
      top_k > 0 && top_k <= logits.size(1),
      "CPU sigmoid Top-K count is invalid");
  auto logits_f = logits.to(torch::kFloat32).contiguous();
  auto bias_f = bias.to(torch::kFloat32).contiguous();
  auto mask_c = mask.contiguous();
  auto weights = torch::empty(
      {logits.size(0), top_k},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto indices = torch::empty(
      {logits.size(0), top_k},
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
  const float* logitp = logits_f.data_ptr<float>();
  const float* biasp = bias_f.data_ptr<float>();
  const bool* maskp = mask_c.data_ptr<bool>();
  float* weightp = weights.data_ptr<float>();
  int64_t* indexp = indices.data_ptr<int64_t>();
  const int64_t experts = logits.size(1);
  struct Candidate {
    float choice;
    float score;
    int64_t index;
  };
  at::parallel_for(0, logits.size(0), 1, [&](int64_t begin, int64_t end) {
    std::vector<Candidate> candidates(experts);
    for (int64_t batch = begin; batch < end; ++batch) {
      for (int64_t expert = 0; expert < experts; ++expert) {
        const float value = logitp[batch * experts + expert];
        const float score = 1.0f / (1.0f + std::exp(-value));
        candidates[expert] = Candidate{
            maskp[expert]
                ? score + biasp[expert]
                : -std::numeric_limits<float>::infinity(),
            score,
            expert};
      }
      std::partial_sort(
          candidates.begin(),
          candidates.begin() + top_k,
          candidates.end(),
          [](const Candidate& left, const Candidate& right) {
            return left.choice > right.choice ||
                (left.choice == right.choice && left.index < right.index);
          });
      float denominator = 0.0f;
      if (normalize && top_k > 1) {
        for (int64_t rank = 0; rank < top_k; ++rank) {
          denominator += candidates[rank].score;
        }
        denominator += 1.0e-20f;
      } else {
        denominator = 1.0f;
      }
      for (int64_t rank = 0; rank < top_k; ++rank) {
        weightp[batch * top_k + rank] =
            candidates[rank].score / denominator *
            static_cast<float>(scaling);
        indexp[batch * top_k + rank] = candidates[rank].index;
      }
    }
  });
  return {weights, indices};
}

torch::Tensor moe_packed_topk_cpu(
    torch::Tensor x_row,
    std::vector<torch::Tensor> gu_payload_list,
    std::vector<torch::Tensor> gu_codebook_list,
    std::vector<int64_t> gu_rows,
    std::vector<int64_t> gu_blocks,
    std::vector<int64_t> gu_bits,
    std::vector<torch::Tensor> dn_payload_list,
    std::vector<torch::Tensor> dn_codebook_list,
    std::vector<int64_t> dn_rows,
    std::vector<int64_t> dn_blocks,
    std::vector<int64_t> dn_bits,
    torch::Tensor route_weights,
    double limit,
    std::string activation,
    double beta,
    double linear_beta,
    torch::Tensor workspace,
    torch::Tensor result) {
  const int64_t experts =
      static_cast<int64_t>(gu_payload_list.size());
  TORCH_CHECK(
      experts > 0 && experts <= 16 &&
          static_cast<int64_t>(gu_codebook_list.size()) == experts &&
          static_cast<int64_t>(gu_rows.size()) == experts &&
          static_cast<int64_t>(gu_blocks.size()) == experts &&
          static_cast<int64_t>(gu_bits.size()) == experts &&
          static_cast<int64_t>(dn_payload_list.size()) == experts &&
          static_cast<int64_t>(dn_codebook_list.size()) == experts &&
          static_cast<int64_t>(dn_rows.size()) == experts &&
          static_cast<int64_t>(dn_blocks.size()) == experts &&
          static_cast<int64_t>(dn_bits.size()) == experts,
      "packed Top-K MoE operand counts must match");
  TORCH_CHECK(
      !x_row.is_cuda() && x_row.scalar_type() == at::kFloat &&
          x_row.dim() == 2 && x_row.size(0) == 1 &&
          x_row.is_contiguous(),
      "packed Top-K MoE requires one contiguous CPU float32 row");
  TORCH_CHECK(
      !route_weights.is_cuda() &&
          route_weights.scalar_type() == at::kFloat &&
          route_weights.numel() == experts &&
          route_weights.is_contiguous(),
      "packed Top-K MoE route weights must be contiguous CPU float32");
  TORCH_CHECK(
      !workspace.is_cuda() && workspace.scalar_type() == at::kFloat &&
          workspace.dim() == 1 && workspace.is_contiguous(),
      "packed Top-K MoE workspace must be contiguous CPU float32");
  TORCH_CHECK(
      !result.is_cuda() && result.scalar_type() == at::kFloat &&
          result.dim() == 1 && result.is_contiguous(),
      "packed Top-K MoE result must be contiguous CPU float32");
  TORCH_CHECK(
      activation == "silu" || activation == "swiglu" ||
          activation == "situ",
      "packed Top-K MoE activation must be silu, swiglu, or situ");

  const int64_t hidden = x_row.size(1);
  int64_t intermediate = -1;
  std::vector<const uint8_t*> gu_payload_ptrs(experts);
  std::vector<const uint8_t*> dn_payload_ptrs(experts);
  std::vector<const float*> gu_codebook_ptrs(experts);
  std::vector<const float*> dn_codebook_ptrs(experts);
  std::vector<int64_t> gu_codes(experts);
  std::vector<int64_t> gu_dims(experts);
  std::vector<int64_t> dn_codes(experts);
  std::vector<int64_t> dn_dims(experts);
  std::vector<torch::Tensor> gu_transposed(experts);
  std::vector<torch::Tensor> dn_transposed(experts);
  std::vector<bool> dn_direct(experts);

  for (int64_t expert = 0; expert < experts; ++expert) {
    const auto& gu_payload = gu_payload_list[expert];
    const auto& gu_codebook = gu_codebook_list[expert];
    const auto& dn_payload = dn_payload_list[expert];
    const auto& dn_codebook = dn_codebook_list[expert];
    TORCH_CHECK(
        !gu_payload.is_cuda() &&
            gu_payload.scalar_type() == at::kByte &&
            gu_payload.is_contiguous() &&
            !dn_payload.is_cuda() &&
            dn_payload.scalar_type() == at::kByte &&
            dn_payload.is_contiguous(),
        "packed Top-K MoE payloads must be contiguous CPU uint8");
    TORCH_CHECK(
        !gu_codebook.is_cuda() &&
            gu_codebook.scalar_type() == at::kFloat &&
            gu_codebook.dim() == 2 && gu_codebook.is_contiguous() &&
            !dn_codebook.is_cuda() &&
            dn_codebook.scalar_type() == at::kFloat &&
            dn_codebook.dim() == 2 && dn_codebook.is_contiguous(),
        "packed Top-K MoE codebooks must be contiguous CPU float32");
    TORCH_CHECK(
        gu_bits[expert] >= 8 && gu_bits[expert] <= 16,
        "unsupported packed GU width");
    TORCH_CHECK(
        dn_bits[expert] >= 8 && dn_bits[expert] <= 16,
        "unsupported packed Down width");
    const int64_t gu_dim = gu_codebook.size(1);
    const int64_t dn_dim = dn_codebook.size(1);
    const int64_t current_intermediate =
        dn_blocks[expert] * dn_dim;
    if (intermediate < 0) {
      intermediate = current_intermediate;
    }
    TORCH_CHECK(
        current_intermediate == intermediate &&
            gu_blocks[expert] * gu_dim == hidden &&
            gu_rows[expert] == 2 * intermediate &&
            dn_rows[expert] == hidden,
        "packed Top-K MoE logical matrix shapes do not match");
    TORCH_CHECK(
        gu_payload.numel() * 8 ==
                gu_rows[expert] * gu_blocks[expert] *
                    gu_bits[expert] &&
            dn_payload.numel() * 8 ==
                dn_rows[expert] * dn_blocks[expert] *
                    dn_bits[expert],
        "packed Top-K MoE payload length mismatch");
    TORCH_CHECK(
        gu_codebook.size(0) <=
                (int64_t{1} << gu_bits[expert]) &&
            dn_codebook.size(0) <=
                (int64_t{1} << dn_bits[expert]),
        "packed Top-K MoE bit width cannot represent codebook");
    gu_payload_ptrs[expert] = gu_payload.data_ptr<uint8_t>();
    dn_payload_ptrs[expert] = dn_payload.data_ptr<uint8_t>();
    gu_codebook_ptrs[expert] = gu_codebook.data_ptr<float>();
    dn_codebook_ptrs[expert] = dn_codebook.data_ptr<float>();
    gu_codes[expert] = gu_codebook.size(0);
    gu_dims[expert] = gu_dim;
    dn_codes[expert] = dn_codebook.size(0);
    dn_dims[expert] = dn_dim;
    gu_transposed[expert] =
        cached_transposed_codebook(gu_codebook);
    dn_direct[expert] =
        dn_rows[expert] * dn_dim <
        dn_codes[expert] * dn_dim + dn_rows[expert];
    if (!dn_direct[expert]) {
      dn_transposed[expert] =
          cached_transposed_codebook(dn_codebook);
    }
  }
  TORCH_CHECK(
      intermediate > 0 && result.numel() >= hidden,
      "packed Top-K MoE output size mismatch");

  // The input is shared by all selected experts.  Experts of the same tier
  // normally share a codebook, so calculate one GU score table per unique
  // codebook instead of repeating it for every routed expert.
  std::vector<int64_t> gu_unique_representatives;
  std::vector<int64_t> gu_unique_for_expert(experts);
  for (int64_t expert = 0; expert < experts; ++expert) {
    int64_t unique = -1;
    for (int64_t candidate = 0;
         candidate <
         static_cast<int64_t>(gu_unique_representatives.size());
         ++candidate) {
      const int64_t other = gu_unique_representatives[candidate];
      if (gu_codebook_ptrs[expert] == gu_codebook_ptrs[other] &&
          gu_blocks[expert] == gu_blocks[other] &&
          gu_codes[expert] == gu_codes[other] &&
          gu_dims[expert] == gu_dims[other]) {
        unique = candidate;
        break;
      }
    }
    if (unique < 0) {
      unique =
          static_cast<int64_t>(gu_unique_representatives.size());
      gu_unique_representatives.push_back(expert);
    }
    gu_unique_for_expert[expert] = unique;
  }

  std::vector<int64_t> gu_score_offsets(
      gu_unique_representatives.size());
  int64_t gu_score_count = 0;
  for (int64_t unique = 0;
       unique <
       static_cast<int64_t>(gu_unique_representatives.size());
       ++unique) {
    const int64_t expert = gu_unique_representatives[unique];
    gu_score_offsets[unique] = gu_score_count;
    gu_score_count += gu_blocks[expert] * gu_codes[expert];
  }
  std::vector<int64_t> dn_score_offsets(experts, -1);
  std::vector<int64_t> dn_score_experts;
  std::vector<int64_t> dn_block_offsets(1, 0);
  int64_t dn_score_count = 0;
  for (int64_t expert = 0; expert < experts; ++expert) {
    if (dn_direct[expert]) {
      continue;
    }
    dn_score_offsets[expert] = dn_score_count;
    dn_score_count += dn_blocks[expert] * dn_codes[expert];
    dn_score_experts.push_back(expert);
    dn_block_offsets.push_back(
        dn_block_offsets.back() + dn_blocks[expert]);
  }

  const int64_t gate_offset = gu_score_count;
  const int64_t up_offset =
      gate_offset + experts * intermediate;
  const int64_t activation_offset =
      up_offset + experts * intermediate;
  const int64_t activation_temp_offset =
      activation_offset + experts * intermediate;
  const int64_t dn_score_offset =
      activation_temp_offset + experts * intermediate;
  const int64_t dn_partial_offset =
      dn_score_offset + dn_score_count;
  const int64_t required =
      dn_partial_offset + experts * hidden;
  TORCH_CHECK(
      workspace.numel() >= required,
      "packed Top-K MoE workspace is too small: ",
      workspace.numel(), " < ", required);

  const float* xp = x_row.data_ptr<float>();
  const float* routep = route_weights.data_ptr<float>();
  float* workspacep = workspace.data_ptr<float>();
  float* gu_scorep = workspacep;
  float* gatep = workspacep + gate_offset;
  float* upp = workspacep + up_offset;
  float* activationp = workspacep + activation_offset;
  float* dn_scorep = workspacep + dn_score_offset;
  float* dn_partialp = workspacep + dn_partial_offset;
  float* resultp = result.data_ptr<float>();
  const float activation_limit = static_cast<float>(limit);
  const float situ_beta = static_cast<float>(beta);
  const float situ_linear_beta = static_cast<float>(linear_beta);
  double gu_score_elapsed = 0.0;
  double gu_lookup_elapsed = 0.0;
  double phase_times[5];

  // ATen owns one persistent worker pool for the process.  The whole MoE is
  // still one registered native call.  Score and consume one shared codebook
  // at a time so its 1--30 MiB score table remains hot in LLC; materializing
  // every tier before lookup evicts the first tiers on mixed Top-16 routes.
  for (int64_t unique = 0;
       unique <
       static_cast<int64_t>(gu_unique_representatives.size());
       ++unique) {
    const int64_t expert = gu_unique_representatives[unique];
    const double score_started = wall_seconds();
    at::parallel_for(
        0, gu_blocks[expert], 1,
        [&](int64_t begin, int64_t end) {
      for (int64_t block = begin; block < end; ++block) {
      const int64_t blocks = gu_blocks[expert];
      const int64_t codes = gu_codes[expert];
      const int64_t dim = gu_dims[expert];
      const float* codebook =
          gu_transposed[expert].data_ptr<float>();
      const float* xv = xp + block * dim;
      float* score =
          gu_scorep + gu_score_offsets[unique] + block * codes;
      codebook_scores(xv, codebook, score, codes, dim);
      }
    });
    gu_score_elapsed += wall_seconds() - score_started;
    std::vector<int64_t> members;
    for (int64_t candidate = 0; candidate < experts; ++candidate) {
      if (gu_unique_for_expert[candidate] == unique) {
        members.push_back(candidate);
      }
    }
    const double lookup_started = wall_seconds();
    at::parallel_for(
        0,
        static_cast<int64_t>(members.size()) * intermediate,
        1,
        [&](int64_t begin, int64_t end) {
      for (int64_t item = begin; item < end; ++item) {
      const int64_t expert =
          members[item / intermediate];
      const int64_t row = item % intermediate;
      const int64_t blocks = gu_blocks[expert];
      const int64_t codes = gu_codes[expert];
      const uint8_t* payload = gu_payload_ptrs[expert];
      const float* score = gu_scorep + gu_score_offsets[unique];
      float gate = lookup_sum_packed(
          score,
          payload,
          row * blocks,
          blocks,
          codes,
          gu_bits[expert]);
      float up = lookup_sum_packed(
          score,
          payload,
          (intermediate + row) * blocks,
          blocks,
          codes,
          gu_bits[expert]);
      if (activation_limit != 0.0f) {
        gate = std::min(gate, activation_limit);
        up = std::max(
            -activation_limit, std::min(up, activation_limit));
      }
      const int64_t activation_item =
          expert * intermediate + row;
      gatep[activation_item] = gate;
      upp[activation_item] = up;
      }
    });
    gu_lookup_elapsed += wall_seconds() - lookup_started;
  }
  phase_times[0] = wall_seconds();

  const int64_t activation_count = experts * intermediate;
  auto gate_values =
      workspace.narrow(0, gate_offset, activation_count);
  auto up_values =
      workspace.narrow(0, up_offset, activation_count);
  auto activation_values =
      workspace.narrow(0, activation_offset, activation_count);
  auto activation_temp =
      workspace.narrow(0, activation_temp_offset, activation_count);
  // ATen's persistent CPU pool supplies vectorized exp/tanh.  Scalar libm
  // calls here cost more than all scheduling saved by the fusion on Top-16.
  if (activation == "situ") {
    activation_values.copy_(gate_values);
    activation_values.div_(situ_beta);
    activation_values.tanh_();
    activation_values.mul_(situ_beta);
    activation_temp.copy_(gate_values);
    activation_temp.sigmoid_();
    activation_values.mul_(activation_temp);
    if (situ_linear_beta > 0.0f) {
      up_values.div_(situ_linear_beta);
      up_values.tanh_();
      up_values.mul_(situ_linear_beta);
    }
    activation_values.mul_(up_values);
  } else {
    activation_temp.copy_(gate_values);
    activation_temp.sigmoid_();
    activation_values.copy_(gate_values);
    activation_values.mul_(activation_temp);
    activation_values.mul_(up_values);
  }
  phase_times[1] = wall_seconds();

  at::parallel_for(
      0, dn_block_offsets.back(), 1,
      [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t selected = static_cast<int64_t>(
          std::upper_bound(
              dn_block_offsets.begin(),
              dn_block_offsets.end(),
              item) -
          dn_block_offsets.begin() - 1);
      const int64_t expert = dn_score_experts[selected];
      const int64_t block = item - dn_block_offsets[selected];
      const int64_t codes = dn_codes[expert];
      const int64_t dim = dn_dims[expert];
      const float* xv =
          activationp + expert * intermediate + block * dim;
      const float* codebook =
          dn_transposed[expert].data_ptr<float>();
      float* score =
          dn_scorep + dn_score_offsets[expert] + block * codes;
      codebook_scores(xv, codebook, score, codes, dim);
    }
  });
  phase_times[2] = wall_seconds();

  at::parallel_for(
      0, experts * hidden, 1,
      [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t expert = item / hidden;
      const int64_t row = item - expert * hidden;
      const int64_t blocks = dn_blocks[expert];
      float value;
      if (dn_direct[expert]) {
        value = direct_dot_packed(
            activationp + expert * intermediate,
            dn_codebook_ptrs[expert],
            dn_payload_ptrs[expert],
            row * blocks,
            blocks,
            dn_bits[expert],
            dn_dims[expert]);
      } else {
        value = lookup_sum_packed(
            dn_scorep + dn_score_offsets[expert],
            dn_payload_ptrs[expert],
            row * blocks,
            blocks,
            dn_codes[expert],
            dn_bits[expert]);
      }
      dn_partialp[item] = value * routep[expert];
    }
  });
  phase_times[3] = wall_seconds();

  at::parallel_for(0, hidden, 1, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      float total = 0.0f;
      for (int64_t expert = 0; expert < experts; ++expert) {
        total += dn_partialp[expert * hidden + row];
      }
      resultp[row] = total;
    }
  });
  phase_times[4] = wall_seconds();
  packed_moe_phase_seconds[0] += gu_score_elapsed;
  packed_moe_phase_seconds[1] += gu_lookup_elapsed;
  for (int phase = 0; phase < 4; ++phase) {
    packed_moe_phase_seconds[phase + 2] +=
        phase_times[phase + 1] - phase_times[phase];
  }
  ++packed_moe_phase_calls;
  return result.narrow(0, 0, hidden);
}

void reset_packed_moe_phase_profile_cpu() {
  std::fill(
      std::begin(packed_moe_phase_seconds),
      std::end(packed_moe_phase_seconds),
      0.0);
  packed_moe_phase_calls = 0;
}

std::vector<double> packed_moe_phase_profile_cpu() {
  return {
      static_cast<double>(packed_moe_phase_calls),
      packed_moe_phase_seconds[0],
      packed_moe_phase_seconds[1],
      packed_moe_phase_seconds[2],
      packed_moe_phase_seconds[3],
      packed_moe_phase_seconds[4],
      packed_moe_phase_seconds[5],
  };
}

torch::Tensor qwen35_delta_recurrent_cpu(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor gate,
    torch::Tensor beta,
    torch::Tensor state,
    torch::Tensor output) {
  TORCH_CHECK(
      !query.is_cuda() && query.dim() == 2 && query.is_contiguous() &&
          key.sizes() == query.sizes() && key.is_contiguous(),
      "Qwen3.5 delta query/key must be contiguous CPU [heads,key_dim]");
  TORCH_CHECK(
      query.scalar_type() == at::kFloat ||
          query.scalar_type() == at::kBFloat16,
      "Qwen3.5 delta query/key must be float32 or bfloat16");
  TORCH_CHECK(
      key.scalar_type() == query.scalar_type() &&
          value.scalar_type() == query.scalar_type() &&
          output.scalar_type() == query.scalar_type(),
      "Qwen3.5 delta query/key/value/output dtypes must match");
  const int64_t heads = query.size(0);
  const int64_t key_dim = query.size(1);
  TORCH_CHECK(
      value.dim() == 2 && value.size(0) == heads &&
          value.is_contiguous() && output.sizes() == value.sizes() &&
          output.is_contiguous(),
      "Qwen3.5 delta value/output must be contiguous [heads,value_dim]");
  const int64_t value_dim = value.size(1);
  TORCH_CHECK(
      !state.is_cuda() && state.scalar_type() == at::kFloat &&
          state.sizes() ==
              torch::IntArrayRef({heads, key_dim, value_dim}) &&
          state.is_contiguous(),
      "Qwen3.5 delta state must be contiguous float32 [heads,key,value]");
  TORCH_CHECK(
      !gate.is_cuda() && !beta.is_cuda() && gate.is_contiguous() &&
          beta.is_contiguous() && gate.numel() == heads &&
          beta.numel() == heads &&
          (gate.scalar_type() == at::kFloat ||
           gate.scalar_type() == at::kBFloat16) &&
          (beta.scalar_type() == at::kFloat ||
           beta.scalar_type() == at::kBFloat16),
      "Qwen3.5 delta gate/beta must contain one scalar per head");

  const bool bf16 = query.scalar_type() == at::kBFloat16;
  const float* query_f = bf16 ? nullptr : query.data_ptr<float>();
  const float* key_f = bf16 ? nullptr : key.data_ptr<float>();
  const float* value_f = bf16 ? nullptr : value.data_ptr<float>();
  float* output_f = bf16 ? nullptr : output.data_ptr<float>();
  const at::BFloat16* query_b =
      bf16 ? query.data_ptr<at::BFloat16>() : nullptr;
  const at::BFloat16* key_b =
      bf16 ? key.data_ptr<at::BFloat16>() : nullptr;
  const at::BFloat16* value_b =
      bf16 ? value.data_ptr<at::BFloat16>() : nullptr;
  at::BFloat16* output_b =
      bf16 ? output.data_ptr<at::BFloat16>() : nullptr;
  const bool gate_bf16 = gate.scalar_type() == at::kBFloat16;
  const bool beta_bf16 = beta.scalar_type() == at::kBFloat16;
  const float* gate_f = gate_bf16 ? nullptr : gate.data_ptr<float>();
  const float* beta_f = beta_bf16 ? nullptr : beta.data_ptr<float>();
  const at::BFloat16* gate_b =
      gate_bf16 ? gate.data_ptr<at::BFloat16>() : nullptr;
  const at::BFloat16* beta_b =
      beta_bf16 ? beta.data_ptr<at::BFloat16>() : nullptr;
  float* statep = state.data_ptr<float>();
  const float output_scale =
      1.0f / std::sqrt(static_cast<float>(key_dim));

#pragma omp parallel for schedule(static)
  for (int64_t head = 0; head < heads; ++head) {
    thread_local std::vector<float> scratch;
    scratch.resize(2 * key_dim + 2 * value_dim);
    float* query_norm = scratch.data();
    float* key_norm = query_norm + key_dim;
    float* prediction = key_norm + key_dim;
    float* result = prediction + value_dim;

    float query_square = 0.0f;
    float key_square = 0.0f;
    const int64_t key_base = head * key_dim;
    for (int64_t lane = 0; lane < key_dim; ++lane) {
      const float q = bf16
          ? static_cast<float>(query_b[key_base + lane])
          : query_f[key_base + lane];
      const float k = bf16
          ? static_cast<float>(key_b[key_base + lane])
          : key_f[key_base + lane];
      query_square += q * q;
      key_square += k * k;
    }
    const float query_inverse =
        1.0f / std::max(std::sqrt(query_square), 1.0e-6f);
    const float key_inverse =
        1.0f / std::max(std::sqrt(key_square), 1.0e-6f);
    for (int64_t lane = 0; lane < key_dim; ++lane) {
      query_norm[lane] =
          (bf16 ? static_cast<float>(query_b[key_base + lane])
                : query_f[key_base + lane]) *
          query_inverse;
      key_norm[lane] =
          (bf16 ? static_cast<float>(key_b[key_base + lane])
                : key_f[key_base + lane]) *
          key_inverse;
    }
    std::fill(prediction, prediction + value_dim, 0.0f);
    std::fill(result, result + value_dim, 0.0f);
    const float decay = std::exp(
        gate_bf16 ? static_cast<float>(gate_b[head]) : gate_f[head]);
    const float beta_value = beta_bf16
        ? static_cast<float>(beta_b[head])
        : beta_f[head];
    float* state_head = statep + head * key_dim * value_dim;

    for (int64_t k_lane = 0; k_lane < key_dim; ++k_lane) {
      float* state_row = state_head + k_lane * value_dim;
      const float key_value = key_norm[k_lane];
      for (int64_t v_lane = 0; v_lane < value_dim; ++v_lane) {
        const float current = state_row[v_lane] * decay;
        state_row[v_lane] = current;
        prediction[v_lane] += current * key_value;
      }
    }
    const int64_t value_base = head * value_dim;
    for (int64_t v_lane = 0; v_lane < value_dim; ++v_lane) {
      const float current = bf16
          ? static_cast<float>(value_b[value_base + v_lane])
          : value_f[value_base + v_lane];
      prediction[v_lane] =
          (current - prediction[v_lane]) * beta_value;
    }
    for (int64_t k_lane = 0; k_lane < key_dim; ++k_lane) {
      float* state_row = state_head + k_lane * value_dim;
      const float key_value = key_norm[k_lane];
      const float query_value = query_norm[k_lane] * output_scale;
      for (int64_t v_lane = 0; v_lane < value_dim; ++v_lane) {
        const float current =
            state_row[v_lane] + key_value * prediction[v_lane];
        state_row[v_lane] = current;
        result[v_lane] += current * query_value;
      }
    }
    for (int64_t v_lane = 0; v_lane < value_dim; ++v_lane) {
      if (bf16) {
        output_b[value_base + v_lane] = at::BFloat16(result[v_lane]);
      } else {
        output_f[value_base + v_lane] = result[v_lane];
      }
    }
  }
  return output;
}

torch::Tensor qwen35_conv1d_update_cpu(
    torch::Tensor input,
    torch::Tensor state,
    torch::Tensor weight,
    torch::Tensor output) {
  TORCH_CHECK(
      !input.is_cuda() && input.dim() == 3 && input.size(2) == 1 &&
          input.is_contiguous() &&
          (input.scalar_type() == at::kFloat ||
           input.scalar_type() == at::kBFloat16),
      "Qwen3.5 conv input must be contiguous CPU [batch,channels,1]");
  const int64_t batch = input.size(0);
  const int64_t channels = input.size(1);
  TORCH_CHECK(
      !state.is_cuda() && state.dim() == 3 &&
          state.size(0) == batch && state.size(1) == channels &&
          state.size(2) >= 1 && state.is_contiguous() &&
          state.scalar_type() == input.scalar_type(),
      "Qwen3.5 conv state must be contiguous [batch,channels,kernel]");
  const int64_t kernel = state.size(2);
  TORCH_CHECK(
      !weight.is_cuda() && weight.numel() == channels * kernel &&
          weight.is_contiguous() &&
          (weight.scalar_type() == at::kFloat ||
           weight.scalar_type() == at::kBFloat16),
      "Qwen3.5 conv weight shape mismatch");
  TORCH_CHECK(
      output.sizes() == input.sizes() && output.is_contiguous() &&
          output.scalar_type() == input.scalar_type(),
      "Qwen3.5 conv output shape or dtype mismatch");
  const bool bf16 = input.scalar_type() == at::kBFloat16;
  const bool weight_bf16 = weight.scalar_type() == at::kBFloat16;
  float* input_f = bf16 ? nullptr : input.data_ptr<float>();
  at::BFloat16* input_b =
      bf16 ? input.data_ptr<at::BFloat16>() : nullptr;
  float* state_f = bf16 ? nullptr : state.data_ptr<float>();
  at::BFloat16* state_b =
      bf16 ? state.data_ptr<at::BFloat16>() : nullptr;
  const float* weight_f =
      weight_bf16 ? nullptr : weight.data_ptr<float>();
  const at::BFloat16* weight_b =
      weight_bf16 ? weight.data_ptr<at::BFloat16>() : nullptr;
  float* output_f = bf16 ? nullptr : output.data_ptr<float>();
  at::BFloat16* output_b =
      bf16 ? output.data_ptr<at::BFloat16>() : nullptr;

#pragma omp parallel for schedule(static)
  for (int64_t item = 0; item < batch * channels; ++item) {
    const int64_t channel = item % channels;
    const int64_t state_base = item * kernel;
    const int64_t weight_base = channel * kernel;
    const float current = bf16
        ? static_cast<float>(input_b[item])
        : input_f[item];
    float sum = 0.0f;
    for (int64_t offset = 0; offset < kernel - 1; ++offset) {
      const float previous = bf16
          ? static_cast<float>(state_b[state_base + offset + 1])
          : state_f[state_base + offset + 1];
      const float coefficient = weight_bf16
          ? static_cast<float>(weight_b[weight_base + offset])
          : weight_f[weight_base + offset];
      sum += previous * coefficient;
      if (bf16) {
        state_b[state_base + offset] =
            state_b[state_base + offset + 1];
      } else {
        state_f[state_base + offset] =
            state_f[state_base + offset + 1];
      }
    }
    const float last_coefficient = weight_bf16
        ? static_cast<float>(weight_b[weight_base + kernel - 1])
        : weight_f[weight_base + kernel - 1];
    sum += current * last_coefficient;
    if (bf16) {
      state_b[state_base + kernel - 1] = at::BFloat16(current);
      output_b[item] = at::BFloat16(sum / (1.0f + std::exp(-sum)));
    } else {
      state_f[state_base + kernel - 1] = current;
      output_f[item] = sum / (1.0f + std::exp(-sum));
    }
  }
  return output;
}

torch::Tensor kda_recurrent_cpu(
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
    double lower_bound,
    torch::Tensor output_gate,
    torch::Tensor norm_weight,
    double norm_eps) {
  TORCH_CHECK(
      !query.is_cuda() && query.dim() == 2 &&
          query.is_contiguous() && key.sizes() == query.sizes() &&
          gate.sizes() == query.sizes() && key.is_contiguous() &&
          gate.is_contiguous(),
      "CPU KDA query/key/gate must be contiguous [heads, key_dim]");
  TORCH_CHECK(
      query.scalar_type() == at::kFloat ||
          query.scalar_type() == at::kBFloat16,
      "CPU KDA inputs must be float32 or bfloat16");
  TORCH_CHECK(
      key.scalar_type() == query.scalar_type() &&
          gate.scalar_type() == query.scalar_type() &&
          value.scalar_type() == query.scalar_type() &&
          output.scalar_type() == query.scalar_type(),
      "CPU KDA input and output dtypes must match");
  const int64_t heads = query.size(0);
  const int64_t key_dim = query.size(1);
  TORCH_CHECK(
      value.dim() == 2 && value.size(0) == heads &&
          value.is_contiguous(),
      "CPU KDA value must be contiguous [heads, value_dim]");
  const int64_t value_dim = value.size(1);
  TORCH_CHECK(
      !state.is_cuda() && state.scalar_type() == at::kFloat &&
          state.sizes() ==
              torch::IntArrayRef({heads, value_dim, key_dim}) &&
          state.is_contiguous(),
      "CPU KDA state must be contiguous float32 [heads,value,key]");
  TORCH_CHECK(
      !workspace.is_cuda() &&
          workspace.scalar_type() == at::kFloat &&
          workspace.numel() >= 3 * heads * key_dim &&
          workspace.is_contiguous(),
      "CPU KDA workspace must hold normalized Q/K and decay");
  TORCH_CHECK(
      output.sizes() == value.sizes() && output.is_contiguous(),
      "CPU KDA output shape mismatch");
  const bool fuse_output_norm = output_gate.numel() != 0;
  if (fuse_output_norm) {
    TORCH_CHECK(
        !output_gate.is_cuda() && output_gate.sizes() == value.sizes() &&
            output_gate.scalar_type() == value.scalar_type() &&
            output_gate.is_contiguous() && !norm_weight.is_cuda() &&
            norm_weight.numel() == value.size(1) &&
            norm_weight.is_contiguous() &&
            (norm_weight.scalar_type() == at::kFloat ||
             norm_weight.scalar_type() == at::kBFloat16),
        "fused KDA output norm shape or dtype mismatch");
  }
  TORCH_CHECK(
      beta.numel() >= heads && a_log.numel() >= heads &&
          dt_bias.numel() >= heads * key_dim,
      "CPU KDA beta/A_log/dt_bias shape mismatch");
  TORCH_CHECK(
      (beta.scalar_type() == at::kFloat ||
       beta.scalar_type() == at::kBFloat16) &&
          (a_log.scalar_type() == at::kFloat ||
           a_log.scalar_type() == at::kBFloat16) &&
          (dt_bias.scalar_type() == at::kFloat ||
           dt_bias.scalar_type() == at::kBFloat16),
      "CPU KDA scalar parameters must be float32 or bfloat16");

  const bool bf16 = query.scalar_type() == at::kBFloat16;
  const float* query_f =
      bf16 ? nullptr : query.data_ptr<float>();
  const float* key_f =
      bf16 ? nullptr : key.data_ptr<float>();
  const float* value_f =
      bf16 ? nullptr : value.data_ptr<float>();
  const float* gate_f =
      bf16 ? nullptr : gate.data_ptr<float>();
  const at::BFloat16* query_b =
      bf16 ? query.data_ptr<at::BFloat16>() : nullptr;
  const at::BFloat16* key_b =
      bf16 ? key.data_ptr<at::BFloat16>() : nullptr;
  const at::BFloat16* value_b =
      bf16 ? value.data_ptr<at::BFloat16>() : nullptr;
  const at::BFloat16* gate_b =
      bf16 ? gate.data_ptr<at::BFloat16>() : nullptr;
  const float* output_gate_f =
      !fuse_output_norm || bf16 ? nullptr : output_gate.data_ptr<float>();
  const at::BFloat16* output_gate_b =
      fuse_output_norm && bf16
      ? output_gate.data_ptr<at::BFloat16>()
      : nullptr;
  const bool norm_weight_bf16 =
      fuse_output_norm && norm_weight.scalar_type() == at::kBFloat16;
  const float* norm_weight_f =
      !fuse_output_norm || norm_weight_bf16
      ? nullptr
      : norm_weight.data_ptr<float>();
  const at::BFloat16* norm_weight_b =
      fuse_output_norm && norm_weight_bf16
      ? norm_weight.data_ptr<at::BFloat16>()
      : nullptr;
  const bool beta_bf16 = beta.scalar_type() == at::kBFloat16;
  const bool a_bf16 = a_log.scalar_type() == at::kBFloat16;
  const bool dt_bf16 = dt_bias.scalar_type() == at::kBFloat16;
  const float* betap =
      beta_bf16 ? nullptr : beta.data_ptr<float>();
  const float* ap =
      a_bf16 ? nullptr : a_log.data_ptr<float>();
  const float* dtp =
      dt_bf16 ? nullptr : dt_bias.data_ptr<float>();
  const at::BFloat16* betab =
      beta_bf16 ? beta.data_ptr<at::BFloat16>() : nullptr;
  const at::BFloat16* ab =
      a_bf16 ? a_log.data_ptr<at::BFloat16>() : nullptr;
  const at::BFloat16* dtb =
      dt_bf16 ? dt_bias.data_ptr<at::BFloat16>() : nullptr;

  float* workspacep = workspace.data_ptr<float>();
  float* query_norm = workspacep;
  float* key_norm = query_norm + heads * key_dim;
  float* decay = key_norm + heads * key_dim;
  float* statep = state.data_ptr<float>();
  float* output_f =
      bf16 ? nullptr : output.data_ptr<float>();
  at::BFloat16* output_b =
      bf16 ? output.data_ptr<at::BFloat16>() : nullptr;
  const float lower = static_cast<float>(lower_bound);

  at::parallel_for(0, heads, 1, [&](int64_t begin, int64_t end) {
    for (int64_t head = begin; head < end; ++head) {
      const int64_t base = head * key_dim;
      float query_square = 0.0f;
      float key_square = 0.0f;
      for (int64_t lane = 0; lane < key_dim; ++lane) {
        const int64_t index = base + lane;
        const float q =
            bf16 ? static_cast<float>(query_b[index]) : query_f[index];
        const float k =
            bf16 ? static_cast<float>(key_b[index]) : key_f[index];
        query_square += q * q;
        key_square += k * k;
      }
      const float query_inverse =
          1.0f / std::max(std::sqrt(query_square), 1.0e-6f);
      const float key_inverse =
          1.0f / std::max(std::sqrt(key_square), 1.0e-6f);
      const float a = std::exp(
          a_bf16 ? static_cast<float>(ab[head]) : ap[head]);
      for (int64_t lane = 0; lane < key_dim; ++lane) {
        const int64_t index = base + lane;
        const float q =
            bf16 ? static_cast<float>(query_b[index]) : query_f[index];
        const float k =
            bf16 ? static_cast<float>(key_b[index]) : key_f[index];
        const float g =
            (bf16 ? static_cast<float>(gate_b[index]) : gate_f[index]) +
            (dt_bf16 ? static_cast<float>(dtb[index]) : dtp[index]);
        query_norm[index] = q * query_inverse;
        key_norm[index] = k * key_inverse;
        decay[index] = std::exp(
            lower / (1.0f + std::exp(-a * g)));
      }
    }
  });

  const float output_scale =
      1.0f / std::sqrt(static_cast<float>(key_dim));
  at::parallel_for(0, heads, 1, [&](int64_t begin, int64_t end) {
    for (int64_t head = begin; head < end; ++head) {
      float output_square = 0.0f;
      for (int64_t row = 0; row < value_dim; ++row) {
      float* state_row =
          statep + (head * value_dim + row) * key_dim;
      const float* key_row = key_norm + head * key_dim;
      const float* query_row = query_norm + head * key_dim;
      const float* decay_row = decay + head * key_dim;
      float prediction = 0.0f;
      int64_t lane = 0;
#if defined(__AVX512F__)
      __m512 prediction_vector = _mm512_setzero_ps();
      for (; lane + 16 <= key_dim; lane += 16) {
        const __m512 current = _mm512_loadu_ps(state_row + lane);
        const __m512 decayed = _mm512_mul_ps(
            current, _mm512_loadu_ps(decay_row + lane));
        _mm512_storeu_ps(state_row + lane, decayed);
        prediction_vector = _mm512_fmadd_ps(
            decayed,
            _mm512_loadu_ps(key_row + lane),
            prediction_vector);
      }
      prediction = _mm512_reduce_add_ps(prediction_vector);
#endif
      for (; lane < key_dim; ++lane) {
        state_row[lane] *= decay_row[lane];
        prediction += state_row[lane] * key_row[lane];
      }
      const int64_t value_index = head * value_dim + row;
      const float current_value =
          bf16
          ? static_cast<float>(value_b[value_index])
          : value_f[value_index];
      const float beta_value =
          beta_bf16
          ? static_cast<float>(betab[head])
          : betap[head];
      const float delta =
          (current_value - prediction) /
          (1.0f + std::exp(-beta_value));
      float output_value = 0.0f;
      lane = 0;
#if defined(__AVX512F__)
      __m512 output_vector = _mm512_setzero_ps();
      const __m512 delta_vector = _mm512_set1_ps(delta);
      for (; lane + 16 <= key_dim; lane += 16) {
        const __m512 updated = _mm512_fmadd_ps(
            delta_vector,
            _mm512_loadu_ps(key_row + lane),
            _mm512_loadu_ps(state_row + lane));
        _mm512_storeu_ps(state_row + lane, updated);
        output_vector = _mm512_fmadd_ps(
            updated,
            _mm512_loadu_ps(query_row + lane),
            output_vector);
      }
      output_value = _mm512_reduce_add_ps(output_vector);
#endif
      for (; lane < key_dim; ++lane) {
        state_row[lane] += delta * key_row[lane];
        output_value += state_row[lane] * query_row[lane];
      }
      output_value *= output_scale;
      output_square += output_value * output_value;
      if (bf16) {
        output_b[value_index] = at::BFloat16(output_value);
      } else {
        output_f[value_index] = output_value;
      }
      }
      if (fuse_output_norm) {
        const float inverse = 1.0f / std::sqrt(
            output_square / static_cast<float>(value_dim) +
            static_cast<float>(norm_eps));
        const int64_t base = head * value_dim;
        for (int64_t row = 0; row < value_dim; ++row) {
          const int64_t index = base + row;
          const float current = bf16
              ? static_cast<float>(output_b[index])
              : output_f[index];
          const float gate_value = bf16
              ? static_cast<float>(output_gate_b[index])
              : output_gate_f[index];
          const float coefficient = norm_weight_bf16
              ? static_cast<float>(norm_weight_b[row])
              : norm_weight_f[row];
          const float normalized = current * inverse * coefficient /
              (1.0f + std::exp(-gate_value));
          if (bf16) {
            output_b[index] = at::BFloat16(normalized);
          } else {
            output_f[index] = normalized;
          }
        }
      }
    }
  });
  return output;
}

bool short_conv3_cpu(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    std::vector<torch::Tensor> states,
    std::vector<torch::Tensor> weights) {
  TORCH_CHECK(
      !query.is_cuda() && query.dim() == 1 &&
          query.is_contiguous() && key.sizes() == query.sizes() &&
          value.sizes() == query.sizes() && key.is_contiguous() &&
          value.is_contiguous(),
      "CPU short-conv inputs must be contiguous flattened tensors");
  TORCH_CHECK(
      query.scalar_type() == at::kFloat ||
          query.scalar_type() == at::kBFloat16,
      "CPU short-conv inputs must be float32 or bfloat16");
  TORCH_CHECK(
      key.scalar_type() == query.scalar_type() &&
          value.scalar_type() == query.scalar_type() &&
          states.size() == 3 && weights.size() == 3,
      "CPU short-conv operand count or dtype mismatch");
  const int64_t channels = query.numel();
  std::vector<torch::Tensor> inputs = {query, key, value};
  int64_t history = -1;
  for (int stream = 0; stream < 3; ++stream) {
    TORCH_CHECK(
        !states[stream].is_cuda() &&
            states[stream].scalar_type() == query.scalar_type() &&
            states[stream].dim() == 2 &&
            states[stream].size(0) == channels &&
            states[stream].is_contiguous(),
        "CPU short-conv state shape mismatch");
    if (history < 0) {
      history = states[stream].size(1);
    }
    TORCH_CHECK(
        states[stream].size(1) == history &&
            !weights[stream].is_cuda() &&
            weights[stream].numel() == channels * (history + 1) &&
            weights[stream].is_contiguous() &&
            (weights[stream].scalar_type() == at::kFloat ||
             weights[stream].scalar_type() == at::kBFloat16),
        "CPU short-conv weight shape mismatch");
  }
  const bool bf16 = query.scalar_type() == at::kBFloat16;
  at::parallel_for(
      0, 3 * channels, 1,
      [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t stream = item / channels;
      const int64_t channel = item - stream * channels;
      auto& input = inputs[stream];
      auto& state = states[stream];
      auto& weight = weights[stream];
      const bool weight_bf16 =
          weight.scalar_type() == at::kBFloat16;
      float* state_f =
          bf16 ? nullptr : state.data_ptr<float>();
      at::BFloat16* state_b =
          bf16 ? state.data_ptr<at::BFloat16>() : nullptr;
      float* input_f =
          bf16 ? nullptr : input.data_ptr<float>();
      at::BFloat16* input_b =
          bf16 ? input.data_ptr<at::BFloat16>() : nullptr;
      const float* weight_f =
          weight_bf16 ? nullptr : weight.data_ptr<float>();
      const at::BFloat16* weight_b =
          weight_bf16
          ? weight.data_ptr<at::BFloat16>()
          : nullptr;
      const int64_t state_base = channel * history;
      const int64_t weight_base = channel * (history + 1);
      const float current =
          bf16
          ? static_cast<float>(input_b[channel])
          : input_f[channel];
      float sum = 0.0f;
      for (int64_t offset = 0; offset < history; ++offset) {
        const float previous =
            bf16
            ? static_cast<float>(state_b[state_base + offset])
            : state_f[state_base + offset];
        const float coefficient =
            weight_bf16
            ? static_cast<float>(weight_b[weight_base + offset])
            : weight_f[weight_base + offset];
        sum += previous * coefficient;
        if (offset + 1 < history) {
          if (bf16) {
            state_b[state_base + offset] =
                state_b[state_base + offset + 1];
          } else {
            state_f[state_base + offset] =
                state_f[state_base + offset + 1];
          }
        }
      }
      const float final_coefficient =
          weight_bf16
          ? static_cast<float>(weight_b[weight_base + history])
          : weight_f[weight_base + history];
      sum += current * final_coefficient;
      if (history > 0) {
        if (bf16) {
          state_b[state_base + history - 1] =
              at::BFloat16(current);
        } else {
          state_f[state_base + history - 1] = current;
        }
      }
      const float activated = sum / (1.0f + std::exp(-sum));
      if (bf16) {
        input_b[channel] = at::BFloat16(activated);
      } else {
        input_f[channel] = activated;
      }
    }
  });
  return true;
}

torch::Tensor gated_rmsnorm_cpu(
    torch::Tensor value,
    torch::Tensor gate,
    torch::Tensor weight,
    torch::Tensor output,
    double eps) {
  TORCH_CHECK(
      !value.is_cuda() && value.dim() == 2 &&
          value.is_contiguous() && gate.sizes() == value.sizes() &&
          gate.is_contiguous() && output.sizes() == value.sizes() &&
          output.is_contiguous(),
      "CPU gated RMSNorm operands must be contiguous [rows,dim]");
  TORCH_CHECK(
      value.scalar_type() == at::kFloat ||
          value.scalar_type() == at::kBFloat16,
      "CPU gated RMSNorm values must be float32 or bfloat16");
  TORCH_CHECK(
      gate.scalar_type() == value.scalar_type() &&
          output.scalar_type() == value.scalar_type() &&
          !weight.is_cuda() && weight.is_contiguous() &&
          weight.numel() == value.size(1) &&
          (weight.scalar_type() == at::kFloat ||
           weight.scalar_type() == at::kBFloat16),
      "CPU gated RMSNorm dtype or weight shape mismatch");
  const int64_t rows = value.size(0);
  const int64_t dim = value.size(1);
  const bool bf16 = value.scalar_type() == at::kBFloat16;
  const bool weight_bf16 =
      weight.scalar_type() == at::kBFloat16;
  const float* value_f =
      bf16 ? nullptr : value.data_ptr<float>();
  const float* gate_f =
      bf16 ? nullptr : gate.data_ptr<float>();
  float* output_f =
      bf16 ? nullptr : output.data_ptr<float>();
  const at::BFloat16* value_b =
      bf16 ? value.data_ptr<at::BFloat16>() : nullptr;
  const at::BFloat16* gate_b =
      bf16 ? gate.data_ptr<at::BFloat16>() : nullptr;
  at::BFloat16* output_b =
      bf16 ? output.data_ptr<at::BFloat16>() : nullptr;
  const float* weight_f =
      weight_bf16 ? nullptr : weight.data_ptr<float>();
  const at::BFloat16* weight_b =
      weight_bf16 ? weight.data_ptr<at::BFloat16>() : nullptr;
  const float epsilon = static_cast<float>(eps);
  at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const int64_t base = row * dim;
      float square = 0.0f;
      for (int64_t lane = 0; lane < dim; ++lane) {
        const float current =
            bf16
            ? static_cast<float>(value_b[base + lane])
            : value_f[base + lane];
        square += current * current;
      }
      const float inverse =
          1.0f / std::sqrt(square / static_cast<float>(dim) + epsilon);
      for (int64_t lane = 0; lane < dim; ++lane) {
        const float current =
            bf16
            ? static_cast<float>(value_b[base + lane])
            : value_f[base + lane];
        const float gate_value =
            bf16
            ? static_cast<float>(gate_b[base + lane])
            : gate_f[base + lane];
        const float scale =
            weight_bf16
            ? static_cast<float>(weight_b[lane])
            : weight_f[lane];
        const float normalized =
            current * inverse * scale /
            (1.0f + std::exp(-gate_value));
        if (bf16) {
          output_b[base + lane] = at::BFloat16(normalized);
        } else {
          output_f[base + lane] = normalized;
        }
      }
    }
  });
  return output;
}

torch::Tensor moe_mixed_cpu(
    torch::Tensor x_row,
    std::vector<torch::Tensor> gu_index_list,
    std::vector<torch::Tensor> gu_codebook_list,
    std::vector<torch::Tensor> dn_index_list,
    std::vector<torch::Tensor> dn_codebook_list,
    torch::Tensor route_weights,
    torch::Tensor shared_w1_q,
    torch::Tensor shared_w1_s,
    torch::Tensor shared_w3_q,
    torch::Tensor shared_w3_s,
    torch::Tensor shared_w2_q,
    torch::Tensor shared_w2_s,
    int64_t group_size,
    double limit,
    bool indices_transposed) {
  const int64_t experts = static_cast<int64_t>(gu_index_list.size());
  const char* vq_int8_mode = std::getenv("CCCP_CPU_VQ_INT8");
  const bool use_vq_int8 =
      indices_transposed && vq_int8_mode != nullptr &&
      vq_int8_mode[0] != '\0' && vq_int8_mode[0] != '0';
  const int64_t vq_chunks = use_vq_int8 ? 32 : 0;
  TORCH_CHECK(
      experts > 0 && experts == static_cast<int64_t>(gu_codebook_list.size()) &&
          experts == static_cast<int64_t>(dn_index_list.size()) &&
          experts == static_cast<int64_t>(dn_codebook_list.size()),
      "routed expert operand counts must match");
  TORCH_CHECK(
      !x_row.is_cuda() && x_row.dim() == 2 && x_row.size(0) == 1 &&
          x_row.scalar_type() == at::kFloat && x_row.is_contiguous(),
      "CPU MoE fusion requires one CPU input row");
  TORCH_CHECK(
      group_size > 0 && group_size % 2 == 0,
      "INT4 group size must be a positive even number");

  const auto& x = x_row;
  const auto& weights = route_weights;
  TORCH_CHECK(
      !weights.is_cuda() && weights.scalar_type() == at::kFloat &&
          weights.is_contiguous(),
      "route weights must be contiguous float32 on CPU");
  TORCH_CHECK(weights.numel() == experts, "route weight count mismatch");
  const int64_t hidden = x.size(1);

  std::vector<int64_t> gu_blocks(experts);
  std::vector<int64_t> gu_codes(experts);
  std::vector<int64_t> gu_dims(experts);
  std::vector<int64_t> dn_blocks(experts);
  std::vector<int64_t> dn_codes(experts);
  std::vector<int64_t> dn_dims(experts);
  int64_t intermediate = -1;
  for (int64_t expert = 0; expert < experts; ++expert) {
    const auto& gu_index = gu_index_list[expert];
    const auto& gu_codebook = gu_codebook_list[expert];
    const auto& dn_index = dn_index_list[expert];
    const auto& dn_codebook = dn_codebook_list[expert];
    TORCH_CHECK(
        !gu_index.is_cuda() && !gu_codebook.is_cuda() &&
            !dn_index.is_cuda() && !dn_codebook.is_cuda(),
        "all routed expert operands must be on CPU");
    TORCH_CHECK(
        gu_index.scalar_type() == at::kByte &&
            dn_index.scalar_type() == at::kByte &&
            gu_codebook.scalar_type() == at::kFloat &&
            dn_codebook.scalar_type() == at::kFloat &&
            gu_index.dim() == 2 && dn_index.dim() == 2 &&
            gu_codebook.dim() == 2 && dn_codebook.dim() == 2 &&
            gu_index.is_contiguous() && dn_index.is_contiguous() &&
            gu_codebook.is_contiguous() && dn_codebook.is_contiguous(),
        "invalid routed VQ operand layout");
    const int64_t gu_rows =
        use_vq_int8 ? gu_index.size(1) : gu_index.size(0);
    const int64_t dn_rows =
        indices_transposed ? dn_index.size(1) : dn_index.size(0);
    const int64_t this_intermediate = gu_rows / 2;
    TORCH_CHECK(
        gu_rows == 2 * this_intermediate && dn_rows == hidden,
        "routed expert row count mismatch");
    if (intermediate < 0) {
      intermediate = this_intermediate;
    }
    TORCH_CHECK(
        this_intermediate == intermediate,
        "routed expert intermediate widths must match");
    gu_blocks[expert] =
        use_vq_int8 ? gu_index.size(0) : gu_index.size(1);
    gu_codes[expert] = gu_codebook.size(0);
    gu_dims[expert] = gu_codebook.size(1);
    dn_blocks[expert] =
        indices_transposed ? dn_index.size(0) : dn_index.size(1);
    dn_codes[expert] = dn_codebook.size(0);
    dn_dims[expert] = dn_codebook.size(1);
    TORCH_CHECK(
        gu_blocks[expert] * gu_dims[expert] == hidden &&
            dn_blocks[expert] * dn_dims[expert] == intermediate,
        "routed expert input width mismatch");
  }

  const auto& w1q = shared_w1_q;
  const auto& w1s = shared_w1_s;
  const auto& w3q = shared_w3_q;
  const auto& w3s = shared_w3_s;
  const auto& w2q = shared_w2_q;
  const auto& w2s = shared_w2_s;
  TORCH_CHECK(
      !w1q.is_cuda() && !w3q.is_cuda() && !w2q.is_cuda() &&
          !w1s.is_cuda() && !w3s.is_cuda() && !w2s.is_cuda(),
      "shared expert operands must be on CPU");
  TORCH_CHECK(
      w1q.scalar_type() == at::kByte &&
          w3q.scalar_type() == at::kByte &&
          w2q.scalar_type() == at::kByte &&
          w1s.scalar_type() == at::kHalf &&
          w3s.scalar_type() == at::kHalf &&
          w2s.scalar_type() == at::kHalf &&
          w1q.is_contiguous() && w3q.is_contiguous() &&
          w2q.is_contiguous() && w1s.is_contiguous() &&
          w3s.is_contiguous() && w2s.is_contiguous(),
      "shared expert quantization dtype mismatch");
  TORCH_CHECK(
      w1q.size(0) == intermediate && w3q.size(0) == intermediate &&
          w1q.size(1) * 2 == hidden && w3q.size(1) * 2 == hidden &&
          w2q.size(0) == hidden && w2q.size(1) * 2 == intermediate,
      "shared expert packed weight shape mismatch");
  TORCH_CHECK(
      w1s.size(0) == intermediate && w3s.size(0) == intermediate &&
          w2s.size(0) == hidden &&
          w1s.size(1) * group_size == hidden &&
          w3s.size(1) * group_size == hidden &&
          w2s.size(1) * group_size == intermediate,
      "shared expert scale shape mismatch");

  std::vector<const uint8_t*> gu_index_ptrs(experts);
  std::vector<const float*> gu_codebook_ptrs(experts);
  std::vector<const uint8_t*> dn_index_ptrs(experts);
  std::vector<const float*> dn_codebook_ptrs(experts);
  for (int64_t expert = 0; expert < experts; ++expert) {
    gu_index_ptrs[expert] =
        gu_index_list[expert].data_ptr<uint8_t>();
    gu_codebook_ptrs[expert] =
        gu_codebook_list[expert].data_ptr<float>();
    dn_index_ptrs[expert] =
        dn_index_list[expert].data_ptr<uint8_t>();
    dn_codebook_ptrs[expert] =
        dn_codebook_list[expert].data_ptr<float>();
  }

  // GU consumes the same input for every selected expert.  Reuse one lookup
  // score table whenever experts share the layer/tier codebook.
  std::vector<int64_t> gu_unique_representatives;
  std::vector<int64_t> gu_unique_for_expert(experts);
  for (int64_t expert = 0; expert < experts; ++expert) {
    int64_t unique = -1;
    for (int64_t candidate = 0;
         candidate < static_cast<int64_t>(gu_unique_representatives.size());
         ++candidate) {
      const int64_t other = gu_unique_representatives[candidate];
      if (gu_codebook_ptrs[expert] == gu_codebook_ptrs[other] &&
          gu_blocks[expert] == gu_blocks[other] &&
          gu_codes[expert] == gu_codes[other] &&
          gu_dims[expert] == gu_dims[other]) {
        unique = candidate;
        break;
      }
    }
    if (unique < 0) {
      unique = static_cast<int64_t>(gu_unique_representatives.size());
      gu_unique_representatives.push_back(expert);
    }
    gu_unique_for_expert[expert] = unique;
  }

  std::vector<int64_t> gu_score_offsets(
      gu_unique_representatives.size());
  std::vector<int64_t> gu_block_offsets(
      gu_unique_representatives.size() + 1, 0);
  int64_t gu_score_count = 0;
  for (int64_t unique = 0;
       unique < static_cast<int64_t>(gu_unique_representatives.size());
       ++unique) {
    gu_score_offsets[unique] = gu_score_count;
    const int64_t expert = gu_unique_representatives[unique];
    gu_score_count += gu_blocks[expert] * gu_codes[expert];
    gu_block_offsets[unique + 1] =
        gu_block_offsets[unique] + gu_blocks[expert];
  }
  std::vector<int64_t> dn_score_offsets(experts);
  std::vector<int64_t> dn_block_offsets(experts + 1, 0);
  int64_t dn_score_count = 0;
  for (int64_t expert = 0; expert < experts; ++expert) {
    dn_score_offsets[expert] = dn_score_count;
    dn_score_count += dn_blocks[expert] * dn_codes[expert];
    dn_block_offsets[expert + 1] =
        dn_block_offsets[expert] + dn_blocks[expert];
  }

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  const int64_t shards =
      indices_transposed
      ? std::max<int64_t>(
            1,
            std::min<int64_t>(
                8, at::get_num_threads() / std::max<int64_t>(1, experts)))
      : 0;
  const int64_t gu_partial_offset = gu_score_count;
  const int64_t gu_partial_count = 0;
  const int64_t activation_offset = gu_partial_offset;
  const int64_t shared_offset =
      activation_offset + experts * intermediate;
  const int64_t dn_score_offset = shared_offset + intermediate;
  const int64_t dn_partial_offset = dn_score_offset + dn_score_count;
  const int64_t dn_partial_count =
      indices_transposed ? experts * shards * hidden : 0;
  const int64_t result_offset =
      dn_partial_offset + dn_partial_count;
  auto workspace = torch::empty({result_offset + hidden}, options);
  auto result = workspace.narrow(0, result_offset, hidden).view({1, hidden});
  const float* xp = x.data_ptr<float>();
  const float* routep = weights.data_ptr<float>();
  float* workspacep = workspace.data_ptr<float>();
  float* gu_scorep = workspacep;
  float* gu_partialp = workspacep + gu_partial_offset;
  float* activationp = workspacep + activation_offset;
  float* sharedp = workspacep + shared_offset;
  float* dn_scorep = workspacep + dn_score_offset;
  float* dn_partialp = workspacep + dn_partial_offset;
  float* resultp = workspacep + result_offset;
  torch::Tensor quantized_scores;
  torch::Tensor quantized_partials;
  int8_t* gu_quantizedp = nullptr;
  int8_t* dn_quantizedp = nullptr;
  int16_t* gu_i16_partialp = nullptr;
  int16_t* dn_i16_partialp = nullptr;
  if (use_vq_int8) {
    const int64_t gu_quantized_count =
        gu_block_offsets.back() * 256;
    const int64_t dn_quantized_count =
        dn_block_offsets.back() * 256;
    quantized_scores = torch::empty(
        {gu_quantized_count + dn_quantized_count},
        torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU));
    gu_quantizedp = quantized_scores.data_ptr<int8_t>();
    dn_quantizedp = gu_quantizedp + gu_quantized_count;
    const int64_t gu_partial_i16_count =
        experts * vq_chunks * 2 * intermediate;
    const int64_t dn_partial_i16_count =
        experts * vq_chunks * hidden;
    quantized_partials = torch::empty(
        {gu_partial_i16_count + dn_partial_i16_count},
        torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU));
    gu_i16_partialp = quantized_partials.data_ptr<int16_t>();
    dn_i16_partialp =
        gu_i16_partialp + gu_partial_i16_count;
  }
  std::vector<float> gu_quant_scales(
      use_vq_int8
          ? gu_unique_representatives.size() * vq_chunks
          : 0,
      1.0f);
  std::vector<float> dn_quant_scales(
      use_vq_int8 ? experts * vq_chunks : 0,
      1.0f);
  const float activation_limit = static_cast<float>(limit);

  const uint8_t* w1qp = w1q.data_ptr<uint8_t>();
  const at::Half* w1sp = w1s.data_ptr<at::Half>();
  const uint8_t* w3qp = w3q.data_ptr<uint8_t>();
  const at::Half* w3sp = w3s.data_ptr<at::Half>();
  const uint8_t* w2qp = w2q.data_ptr<uint8_t>();
  const at::Half* w2sp = w2s.data_ptr<at::Half>();
  const int64_t w1_bytes = hidden / 2;
  const int64_t w1_groups = hidden / group_size;
  const int64_t w2_bytes = intermediate / 2;
  const int64_t w2_groups = intermediate / group_size;
  const bool use_w4a8 = cpu_w4a8_enabled() && group_size == 64;
  const bool use_w4abf16 =
      cpu_w4abf16_enabled() && group_size == 64;
  const bool use_expand_bf16 = cpu_expand_bf16_enabled();
  Int8Activation quantized_input;
  Int8Activation quantized_shared;
  Bf16Activation bf16_input;
  Bf16Activation bf16_shared;
  torch::Tensor expanded_w1;
  torch::Tensor expanded_w3;
  torch::Tensor expanded_w2;
  const at::BFloat16* expanded_w1p = nullptr;
  const at::BFloat16* expanded_w3p = nullptr;
  const at::BFloat16* expanded_w2p = nullptr;
  if (use_w4a8) {
    quantized_input =
        quantize_int8_activation(xp, hidden, group_size);
  }
  if (use_w4abf16 || use_expand_bf16) {
    bf16_input =
        quantize_bf16_activation(xp, hidden, group_size);
  }
  if (use_expand_bf16) {
    expanded_w1 =
        expand_int4_bf16(w1q, w1s, hidden, group_size);
    expanded_w3 =
        expand_int4_bf16(w3q, w3s, hidden, group_size);
    expanded_w2 =
        expand_int4_bf16(w2q, w2s, intermediate, group_size);
    expanded_w1p = expanded_w1.data_ptr<at::BFloat16>();
    expanded_w3p = expanded_w3.data_ptr<at::BFloat16>();
    expanded_w2p = expanded_w2.data_ptr<at::BFloat16>();
  }
  static const bool phase_profile = [] {
    const char* value = std::getenv("CCCP_CPU_MOE_PROFILE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  double phase_times[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  if (phase_profile) {
    phase_times[0] = wall_seconds();
  }

#pragma omp parallel
  {
 #pragma omp for schedule(static) nowait
    for (int64_t item = 0; item < gu_block_offsets.back(); ++item) {
      const int64_t unique = static_cast<int64_t>(
          std::upper_bound(
              gu_block_offsets.begin(),
              gu_block_offsets.end(),
              item) -
          gu_block_offsets.begin() - 1);
      const int64_t expert = gu_unique_representatives[unique];
      const int64_t blocks = gu_blocks[expert];
      const int64_t codes = gu_codes[expert];
      const int64_t dim = gu_dims[expert];
      float* score = gu_scorep + gu_score_offsets[unique];
      const float* codebook = gu_codebook_ptrs[expert];
      const int64_t block = item - gu_block_offsets[unique];
      const float* xv = xp + block * dim;
      float* block_score = score + block * codes;
      for (int64_t code = 0; code < codes; ++code) {
        block_score[code] =
            float_dot(xv, codebook + code * dim, dim);
      }
    }

#pragma omp for schedule(static) nowait
    for (int64_t row = 0; row < intermediate; ++row) {
      float gate =
          use_expand_bf16
          ? bf16_row_dot(
                bf16_input.values.data(),
                expanded_w1p + row * hidden,
                hidden)
          : use_w4abf16
          ? int4_row_dot_w4abf16(
                bf16_input,
                w1qp + row * w1_bytes,
                w1sp + row * w1_groups)
          : use_w4a8
          ? int4_row_dot_w4a8(
                quantized_input,
                w1qp + row * w1_bytes,
                w1sp + row * w1_groups)
          : int4_row_dot(
                xp,
                w1qp + row * w1_bytes,
                w1sp + row * w1_groups,
                hidden,
                group_size);
      float up =
          use_expand_bf16
          ? bf16_row_dot(
                bf16_input.values.data(),
                expanded_w3p + row * hidden,
                hidden)
          : use_w4abf16
          ? int4_row_dot_w4abf16(
                bf16_input,
                w3qp + row * w1_bytes,
                w3sp + row * w1_groups)
          : use_w4a8
          ? int4_row_dot_w4a8(
                quantized_input,
                w3qp + row * w1_bytes,
                w3sp + row * w1_groups)
          : int4_row_dot(
                xp,
                w3qp + row * w1_bytes,
                w3sp + row * w1_groups,
                hidden,
                group_size);
      if (activation_limit != 0.0f) {
        gate = std::min(gate, activation_limit);
        up = std::max(-activation_limit, std::min(up, activation_limit));
      }
      sharedp[row] = gate / (1.0f + std::exp(-gate)) * up;
    }

#pragma omp barrier
    if (use_vq_int8) {
#pragma omp for schedule(static)
      for (int64_t task = 0;
           task <
               static_cast<int64_t>(gu_unique_representatives.size()) *
                   vq_chunks;
           ++task) {
        const int64_t unique = task / vq_chunks;
        const int64_t chunk = task - unique * vq_chunks;
        const int64_t expert = gu_unique_representatives[unique];
        const int64_t codes = gu_codes[expert];
        const int64_t blocks = gu_blocks[expert];
        const int64_t block_begin = blocks * chunk / vq_chunks;
        const int64_t block_end =
            blocks * (chunk + 1) / vq_chunks;
        float maximum = 0.0f;
        for (int64_t block = block_begin; block < block_end; ++block) {
          const float* source =
              gu_scorep + gu_score_offsets[unique] + block * codes;
          for (int64_t code = 0; code < codes; ++code) {
            maximum = std::max(maximum, std::abs(source[code]));
          }
        }
        const float scale =
            maximum > 0.0f ? maximum / 127.0f : 1.0f;
        gu_quant_scales[task] = scale;
        const float inverse = 1.0f / scale;
        for (int64_t block = block_begin; block < block_end; ++block) {
          const float* source =
              gu_scorep + gu_score_offsets[unique] + block * codes;
          int8_t* destination =
              gu_quantizedp +
              (gu_block_offsets[unique] + block) * 256;
          for (int64_t code = 0; code < codes; ++code) {
            const int value = static_cast<int>(
                std::nearbyint(source[code] * inverse));
            destination[code] = static_cast<int8_t>(
                std::max(-127, std::min(127, value)));
          }
          std::fill(
              destination + codes, destination + 256, int8_t{0});
        }
      }
    }
    if (phase_profile) {
#pragma omp single nowait
      { phase_times[1] = wall_seconds(); }
    }
    if (use_vq_int8) {
#pragma omp for schedule(static)
      for (int64_t task = 0; task < experts * vq_chunks; ++task) {
        const int64_t expert = task / vq_chunks;
        const int64_t chunk = task - expert * vq_chunks;
        const int64_t blocks = gu_blocks[expert];
        const uint8_t* indices = gu_index_ptrs[expert];
        int16_t* partial =
            gu_i16_partialp + task * (2 * intermediate);
        std::fill(
            partial, partial + 2 * intermediate, int16_t{0});
        const int64_t unique = gu_unique_for_expert[expert];
        const int64_t block_begin = blocks * chunk / vq_chunks;
        const int64_t block_end =
            blocks * (chunk + 1) / vq_chunks;
        for (int64_t block = block_begin; block < block_end; ++block) {
          const int8_t* block_score =
              gu_quantizedp +
              (gu_block_offsets[unique] + block) * 256;
          const uint8_t* block_indices =
              indices + block * (2 * intermediate);
          int64_t row = 0;
#if defined(__AVX512VBMI__)
          for (; row + 64 <= intermediate; row += 64) {
            add_i8_scores_64(
                partial + row,
                lookup_i8_rows_64(
                    block_score, block_indices + row));
            add_i8_scores_64(
                partial + intermediate + row,
                lookup_i8_rows_64(
                    block_score,
                    block_indices + intermediate + row));
          }
#endif
          for (; row < intermediate; ++row) {
            partial[row] += block_score[block_indices[row]];
            partial[intermediate + row] +=
                block_score[block_indices[intermediate + row]];
          }
        }
      }
#pragma omp for schedule(static)
      for (int64_t item = 0; item < experts * intermediate; ++item) {
        const int64_t expert = item / intermediate;
        const int64_t row = item - expert * intermediate;
        float gate = 0.0f;
        float up = 0.0f;
        const int64_t unique = gu_unique_for_expert[expert];
        for (int64_t chunk = 0; chunk < vq_chunks; ++chunk) {
          const int16_t* partial =
              gu_i16_partialp +
              (expert * vq_chunks + chunk) * (2 * intermediate);
          const float scale =
              gu_quant_scales[unique * vq_chunks + chunk];
          gate += static_cast<float>(partial[row]) * scale;
          up += static_cast<float>(partial[intermediate + row]) * scale;
        }
        if (activation_limit != 0.0f) {
          gate = std::min(gate, activation_limit);
          up = std::max(
              -activation_limit, std::min(up, activation_limit));
          }
        activationp[item] =
            gate / (1.0f + std::exp(-gate)) * up;
      }
    } else {
#pragma omp for schedule(static)
      for (int64_t item = 0; item < experts * intermediate; ++item) {
        const int64_t expert = item / intermediate;
        const int64_t row = item - expert * intermediate;
        const int64_t blocks = gu_blocks[expert];
        const int64_t codes = gu_codes[expert];
        const float* score =
            gu_scorep +
            gu_score_offsets[gu_unique_for_expert[expert]];
        const uint8_t* indices = gu_index_ptrs[expert];
        float gate;
        float up;
        lookup_sum_pair(
            score,
            indices + row * blocks,
            indices + (intermediate + row) * blocks,
            blocks,
            codes,
            gate,
            up);
        if (activation_limit != 0.0f) {
          gate = std::min(gate, activation_limit);
          up = std::max(
              -activation_limit, std::min(up, activation_limit));
        }
        activationp[item] =
            gate / (1.0f + std::exp(-gate)) * up;
      }
    }
    if (phase_profile) {
#pragma omp single nowait
      { phase_times[2] = wall_seconds(); }
    }
    if (use_w4a8) {
#pragma omp single
      {
        quantized_shared =
            quantize_int8_activation(
                sharedp, intermediate, group_size);
      }
    }
    if (use_w4abf16 || use_expand_bf16) {
#pragma omp single
      {
        bf16_shared =
            quantize_bf16_activation(
                sharedp, intermediate, group_size);
      }
    }

 #pragma omp for schedule(static) nowait
    for (int64_t item = 0; item < dn_block_offsets.back(); ++item) {
      const int64_t expert = static_cast<int64_t>(
          std::upper_bound(
              dn_block_offsets.begin(),
              dn_block_offsets.end(),
              item) -
          dn_block_offsets.begin() - 1);
      const int64_t blocks = dn_blocks[expert];
      const int64_t codes = dn_codes[expert];
      const int64_t dim = dn_dims[expert];
      const float* activated = activationp + expert * intermediate;
      const float* codebook = dn_codebook_ptrs[expert];
      float* score = dn_scorep + dn_score_offsets[expert];
      const int64_t block = item - dn_block_offsets[expert];
      const float* xv = activated + block * dim;
      float* block_score = score + block * codes;
      for (int64_t code = 0; code < codes; ++code) {
        block_score[code] =
            float_dot(xv, codebook + code * dim, dim);
      }
    }

#pragma omp for schedule(static) nowait
    for (int64_t row = 0; row < hidden; ++row) {
      resultp[row] =
          use_expand_bf16
          ? bf16_row_dot(
                bf16_shared.values.data(),
                expanded_w2p + row * intermediate,
                intermediate)
          : use_w4abf16
          ? int4_row_dot_w4abf16(
                bf16_shared,
                w2qp + row * w2_bytes,
                w2sp + row * w2_groups)
          : use_w4a8
          ? int4_row_dot_w4a8(
                quantized_shared,
                w2qp + row * w2_bytes,
                w2sp + row * w2_groups)
          : int4_row_dot(
                sharedp,
                w2qp + row * w2_bytes,
                w2sp + row * w2_groups,
                intermediate,
                group_size);
    }

#pragma omp barrier
    if (use_vq_int8) {
#pragma omp for schedule(static)
      for (int64_t task = 0; task < experts * vq_chunks; ++task) {
        const int64_t expert = task / vq_chunks;
        const int64_t chunk = task - expert * vq_chunks;
        const int64_t codes = dn_codes[expert];
        const int64_t blocks = dn_blocks[expert];
        const int64_t block_begin = blocks * chunk / vq_chunks;
        const int64_t block_end =
            blocks * (chunk + 1) / vq_chunks;
        float maximum = 0.0f;
        for (int64_t block = block_begin; block < block_end; ++block) {
          const float* source =
              dn_scorep + dn_score_offsets[expert] + block * codes;
          for (int64_t code = 0; code < codes; ++code) {
            maximum = std::max(maximum, std::abs(source[code]));
          }
        }
        const float scale =
            maximum > 0.0f ? maximum / 127.0f : 1.0f;
        dn_quant_scales[task] = scale;
        const float inverse = 1.0f / scale;
        for (int64_t block = block_begin; block < block_end; ++block) {
          const float* source =
              dn_scorep + dn_score_offsets[expert] + block * codes;
          int8_t* destination =
              dn_quantizedp +
              (dn_block_offsets[expert] + block) * 256;
          for (int64_t code = 0; code < codes; ++code) {
            const int value = static_cast<int>(
                std::nearbyint(source[code] * inverse));
            destination[code] = static_cast<int8_t>(
                std::max(-127, std::min(127, value)));
          }
          std::fill(
              destination + codes, destination + 256, int8_t{0});
        }
      }
    }
    if (phase_profile) {
#pragma omp single nowait
      { phase_times[3] = wall_seconds(); }
    }
    if (use_vq_int8) {
#pragma omp for schedule(static)
      for (int64_t task = 0; task < experts * vq_chunks; ++task) {
        const int64_t expert = task / vq_chunks;
        const int64_t chunk = task - expert * vq_chunks;
        const int64_t blocks = dn_blocks[expert];
        const uint8_t* indices = dn_index_ptrs[expert];
        int16_t* partial = dn_i16_partialp + task * hidden;
        std::fill(partial, partial + hidden, int16_t{0});
        const int64_t block_begin = blocks * chunk / vq_chunks;
        const int64_t block_end =
            blocks * (chunk + 1) / vq_chunks;
        for (int64_t block = block_begin; block < block_end; ++block) {
          const int8_t* block_score =
              dn_quantizedp +
              (dn_block_offsets[expert] + block) * 256;
          const uint8_t* block_indices = indices + block * hidden;
          int64_t row = 0;
#if defined(__AVX512VBMI__)
          for (; row + 64 <= hidden; row += 64) {
            add_i8_scores_64(
                partial + row,
                lookup_i8_rows_64(
                    block_score, block_indices + row));
          }
#endif
          for (; row < hidden; ++row) {
            partial[row] += block_score[block_indices[row]];
          }
        }
      }
#pragma omp for schedule(static)
      for (int64_t row = 0; row < hidden; ++row) {
        float value = resultp[row];
        for (int64_t expert = 0; expert < experts; ++expert) {
          float expert_value = 0.0f;
          for (int64_t chunk = 0; chunk < vq_chunks; ++chunk) {
            const int16_t partial =
                dn_i16_partialp[
                    (expert * vq_chunks + chunk) * hidden + row];
            expert_value +=
                static_cast<float>(partial) *
                dn_quant_scales[expert * vq_chunks + chunk];
          }
          value += routep[expert] * expert_value;
        }
        resultp[row] = value;
      }
    } else if (indices_transposed) {
#pragma omp for schedule(static)
      for (int64_t task = 0; task < experts * shards; ++task) {
        const int64_t expert = task / shards;
        const int64_t shard = task - expert * shards;
        const int64_t blocks = dn_blocks[expert];
        const int64_t codes = dn_codes[expert];
        const float* score =
            dn_scorep + dn_score_offsets[expert];
        const uint8_t* indices = dn_index_ptrs[expert];
        float* partial = dn_partialp + task * hidden;
        std::fill(partial, partial + hidden, 0.0f);
        const int64_t block_begin = blocks * shard / shards;
        const int64_t block_end = blocks * (shard + 1) / shards;
        for (int64_t block = block_begin; block < block_end; ++block) {
          const float* block_score = score + block * codes;
          const uint8_t* block_indices = indices + block * hidden;
          int64_t row = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
          for (; row + 16 <= hidden; row += 16) {
            _mm512_storeu_ps(
                partial + row,
                _mm512_add_ps(
                    _mm512_loadu_ps(partial + row),
                    lookup_rows_16(
                        block_score, block_indices + row)));
          }
#endif
          for (; row < hidden; ++row) {
            partial[row] += block_score[block_indices[row]];
          }
        }
      }
#pragma omp for schedule(static)
      for (int64_t row = 0; row < hidden; ++row) {
        float value = resultp[row];
        for (int64_t expert = 0; expert < experts; ++expert) {
          float expert_value = 0.0f;
          for (int64_t shard = 0; shard < shards; ++shard) {
            expert_value +=
                dn_partialp[
                    (expert * shards + shard) * hidden + row];
          }
          value += routep[expert] * expert_value;
        }
        resultp[row] = value;
      }
    } else {
#pragma omp for schedule(static)
      for (int64_t row = 0; row < hidden; ++row) {
        resultp[row] += lookup_weighted_many(
            dn_score_offsets,
            dn_index_ptrs,
            dn_blocks,
            dn_codes,
            dn_scorep,
            routep,
            experts,
            row);
      }
    }
  }
  if (phase_profile) {
    phase_times[4] = wall_seconds();
    for (int64_t phase = 0; phase < 4; ++phase) {
      moe_phase_seconds[phase] += phase_times[phase + 1] - phase_times[phase];
    }
    ++moe_phase_calls;
  }
  return result;
}

void reset_moe_phase_profile_cpu() {
  for (double& phase : moe_phase_seconds) {
    phase = 0.0;
  }
  moe_phase_calls = 0;
}

torch::Tensor moe_phase_profile_cpu() {
  auto result = torch::empty(
      {5},
      torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
  double* values = result.data_ptr<double>();
  for (int64_t phase = 0; phase < 4; ++phase) {
    values[phase] = moe_phase_seconds[phase];
  }
  values[4] = static_cast<double>(moe_phase_calls);
  return result;
}

void transpose_vq_indices_into(
    const torch::Tensor& indices,
    torch::Tensor& output) {
  TORCH_CHECK(
      !indices.is_cuda() && indices.scalar_type() == at::kByte &&
          indices.dim() == 2 && indices.is_contiguous(),
      "CPU VQ transpose requires contiguous uint8 indices");
  const int64_t rows = indices.size(0);
  const int64_t blocks = indices.size(1);
  TORCH_CHECK(
      output.scalar_type() == at::kByte &&
          output.dim() == 2 && output.size(0) == blocks &&
          output.size(1) == rows && output.is_contiguous(),
      "CPU VQ transpose output shape mismatch");
  const uint8_t* source = indices.data_ptr<uint8_t>();
  uint8_t* destination = output.data_ptr<uint8_t>();
  constexpr int64_t tile = 32;
  for (int64_t row0 = 0; row0 < rows; row0 += tile) {
    for (int64_t block0 = 0; block0 < blocks; block0 += tile) {
      const int64_t row_end = std::min(rows, row0 + tile);
      const int64_t block_end = std::min(blocks, block0 + tile);
      for (int64_t row = row0; row < row_end; ++row) {
        for (int64_t block = block0; block < block_end; ++block) {
          destination[block * rows + row] =
              source[row * blocks + block];
        }
      }
    }
  }
}

class CpuMoeLayer {
 public:
  CpuMoeLayer(
      std::vector<torch::Tensor> gu_indices,
      std::vector<torch::Tensor> gu_codebooks,
      std::vector<torch::Tensor> dn_indices,
      std::vector<torch::Tensor> dn_codebooks,
      torch::Tensor valid_experts,
      torch::Tensor shared_w1_q,
      torch::Tensor shared_w1_s,
      torch::Tensor shared_w3_q,
      torch::Tensor shared_w3_s,
      torch::Tensor shared_w2_q,
      torch::Tensor shared_w2_s,
      torch::Tensor gate_q,
      torch::Tensor gate_s,
      torch::Tensor gate_bias,
      torch::Tensor gate_mask,
      int64_t group_size,
      double limit,
      int64_t top_k,
      bool normalize_route,
      double routed_scaling)
      : gu_indices_(std::move(gu_indices)),
        gu_codebooks_(std::move(gu_codebooks)),
        dn_indices_(std::move(dn_indices)),
        dn_codebooks_(std::move(dn_codebooks)),
        valid_experts_(valid_experts.to(torch::kBool).contiguous()),
        shared_w1_q_(std::move(shared_w1_q)),
        shared_w1_s_(std::move(shared_w1_s)),
        shared_w3_q_(std::move(shared_w3_q)),
        shared_w3_s_(std::move(shared_w3_s)),
        shared_w2_q_(std::move(shared_w2_q)),
        shared_w2_s_(std::move(shared_w2_s)),
        gate_q_(std::move(gate_q)),
        gate_s_(std::move(gate_s)),
        gate_bias_(gate_bias.to(torch::kFloat32).contiguous()),
        gate_mask_(gate_mask.to(torch::kBool).contiguous()),
        group_size_(group_size),
        limit_(limit),
        top_k_(top_k),
        normalize_route_(normalize_route),
        routed_scaling_(routed_scaling) {
    const int64_t count = static_cast<int64_t>(gu_indices_.size());
    TORCH_CHECK(
        count > 0 &&
            static_cast<int64_t>(gu_codebooks_.size()) == count &&
            static_cast<int64_t>(dn_indices_.size()) == count &&
            static_cast<int64_t>(dn_codebooks_.size()) == count &&
            valid_experts_.numel() == count,
        "cached CPU MoE layer expert counts must match");
    const char* transpose_mode = std::getenv("CCCP_CPU_DN_BLOCK");
    const char* int8_mode = std::getenv("CCCP_CPU_VQ_INT8");
    vq_int8_ =
        int8_mode != nullptr &&
        int8_mode[0] != '\0' && int8_mode[0] != '0';
    indices_transposed_ =
        vq_int8_ ||
        (transpose_mode != nullptr &&
         transpose_mode[0] != '\0' && transpose_mode[0] != '0');
    if (indices_transposed_) {
      if (vq_int8_) {
        gu_transposed_.resize(count);
      }
      dn_transposed_.resize(count);
      std::vector<int64_t> valid_ids;
      const bool* validp = valid_experts_.data_ptr<bool>();
      valid_ids.reserve(count);
      for (int64_t expert = 0; expert < count; ++expert) {
        if (validp[expert]) {
          if (vq_int8_) {
            gu_transposed_[expert] = torch::empty(
                {gu_indices_[expert].size(1), gu_indices_[expert].size(0)},
                torch::TensorOptions()
                    .dtype(torch::kUInt8)
                    .device(torch::kCPU));
          }
          dn_transposed_[expert] = torch::empty(
              {dn_indices_[expert].size(1), dn_indices_[expert].size(0)},
              torch::TensorOptions()
                  .dtype(torch::kUInt8)
                  .device(torch::kCPU));
          valid_ids.push_back(expert);
        }
      }
#pragma omp parallel for schedule(dynamic)
      for (int64_t item = 0;
           item < static_cast<int64_t>(valid_ids.size());
           ++item) {
        const int64_t expert = valid_ids[item];
        if (vq_int8_) {
          transpose_vq_indices_into(
              gu_indices_[expert], gu_transposed_[expert]);
        }
        transpose_vq_indices_into(
            dn_indices_[expert], dn_transposed_[expert]);
      }
    }
    TORCH_CHECK(
        !gate_q_.is_cuda() && !gate_s_.is_cuda() &&
            gate_q_.scalar_type() == at::kByte &&
            gate_s_.scalar_type() == at::kHalf &&
            gate_q_.dim() == 2 && gate_s_.dim() == 2 &&
            gate_q_.size(0) == count && gate_s_.size(0) == count &&
            gate_q_.size(1) * 2 == shared_w1_q_.size(1) * 2 &&
            gate_s_.size(1) * group_size_ == gate_q_.size(1) * 2 &&
            gate_bias_.numel() == count && gate_mask_.numel() == count &&
            top_k_ > 0 && top_k_ <= count,
        "cached CPU MoE router shape mismatch");
    route_scores_ = torch::empty(
        {count},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  }

  torch::Tensor forward(
      torch::Tensor x_row,
      torch::Tensor route_weights,
      torch::Tensor expert_ids) {
    auto ids = expert_ids.to(torch::kLong).contiguous();
    TORCH_CHECK(
        ids.dim() == 1 && route_weights.numel() == ids.numel(),
        "cached CPU MoE route shape mismatch");
    const int64_t count = ids.numel();
    const int64_t* idp = ids.data_ptr<int64_t>();
    return forward_selected(x_row, route_weights, idp, count);
  }

  torch::Tensor forward_learned(torch::Tensor x_row) {
    TORCH_CHECK(
        !x_row.is_cuda() && x_row.scalar_type() == at::kFloat &&
            x_row.dim() == 2 && x_row.size(0) == 1 &&
            x_row.size(1) == gate_q_.size(1) * 2 &&
            x_row.is_contiguous(),
        "cached CPU MoE learned route input mismatch");
    const float* xp = x_row.data_ptr<float>();
    const uint8_t* qp = gate_q_.data_ptr<uint8_t>();
    const at::Half* sp = gate_s_.data_ptr<at::Half>();
    float* scorep = route_scores_.data_ptr<float>();
    const int64_t experts = gate_q_.size(0);
    const int64_t cols = gate_q_.size(1) * 2;
    const int64_t groups = cols / group_size_;
    const int64_t bytes_per_row = cols / 2;
    at::parallel_for(0, experts, 1, [&](int64_t begin, int64_t end) {
      for (int64_t expert = begin; expert < end; ++expert) {
        const float raw = int4_row_dot(
            xp,
            qp + expert * bytes_per_row,
            sp + expert * groups,
            cols,
            group_size_);
        const float softplus =
            raw > 20.0f ? raw : std::log1p(std::exp(raw));
        scorep[expert] = std::sqrt(softplus);
      }
    });

    const float* biasp = gate_bias_.data_ptr<float>();
    const bool* maskp = gate_mask_.data_ptr<bool>();
    std::vector<int64_t> selected(top_k_, -1);
    std::vector<float> choices(
        top_k_, -std::numeric_limits<float>::infinity());
    for (int64_t expert = 0; expert < experts; ++expert) {
      if (!maskp[expert]) {
        continue;
      }
      const float choice = scorep[expert] + biasp[expert];
      for (int64_t rank = 0; rank < top_k_; ++rank) {
        if (choice > choices[rank]) {
          for (int64_t move = top_k_ - 1; move > rank; --move) {
            choices[move] = choices[move - 1];
            selected[move] = selected[move - 1];
          }
          choices[rank] = choice;
          selected[rank] = expert;
          break;
        }
      }
    }
    auto route_weights = torch::empty(
        {top_k_},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    float* routep = route_weights.data_ptr<float>();
    float denominator = normalize_route_ ? 1.0e-20f : 1.0f;
    for (int64_t rank = 0; rank < top_k_; ++rank) {
      TORCH_CHECK(selected[rank] >= 0, "not enough available routed experts");
      routep[rank] = scorep[selected[rank]];
      if (normalize_route_) {
        denominator += routep[rank];
      }
    }
    const float multiplier =
        static_cast<float>(routed_scaling_) / denominator;
    for (int64_t rank = 0; rank < top_k_; ++rank) {
      routep[rank] *= multiplier;
    }
    return forward_selected(
        x_row,
        route_weights,
        selected.data(),
        top_k_);
  }

 private:
  torch::Tensor forward_selected(
      torch::Tensor x_row,
      torch::Tensor route_weights,
      const int64_t* idp,
      int64_t count) {
    const bool* validp = valid_experts_.data_ptr<bool>();
    std::vector<torch::Tensor> gu_indices;
    std::vector<torch::Tensor> gu_codebooks;
    std::vector<torch::Tensor> dn_indices;
    std::vector<torch::Tensor> dn_codebooks;
    gu_indices.reserve(count);
    gu_codebooks.reserve(count);
    dn_indices.reserve(count);
    dn_codebooks.reserve(count);
    for (int64_t slot = 0; slot < count; ++slot) {
      const int64_t expert = idp[slot];
      TORCH_CHECK(
          expert >= 0 &&
              expert < static_cast<int64_t>(gu_indices_.size()) &&
              validp[expert],
          "route selected an unavailable cached CPU expert");
      if (indices_transposed_) {
        gu_indices.push_back(
            vq_int8_ ? gu_transposed_[expert] : gu_indices_[expert]);
        dn_indices.push_back(dn_transposed_[expert]);
      } else {
        gu_indices.push_back(gu_indices_[expert]);
        dn_indices.push_back(dn_indices_[expert]);
      }
      gu_codebooks.push_back(gu_codebooks_[expert]);
      dn_codebooks.push_back(dn_codebooks_[expert]);
    }
    return moe_mixed_cpu(
        x_row,
        std::move(gu_indices),
        std::move(gu_codebooks),
        std::move(dn_indices),
        std::move(dn_codebooks),
        route_weights,
        shared_w1_q_,
        shared_w1_s_,
        shared_w3_q_,
        shared_w3_s_,
        shared_w2_q_,
        shared_w2_s_,
        group_size_,
        limit_,
        indices_transposed_);
  }

  std::vector<torch::Tensor> gu_indices_;
  std::vector<torch::Tensor> gu_codebooks_;
  std::vector<torch::Tensor> dn_indices_;
  std::vector<torch::Tensor> dn_codebooks_;
  std::vector<torch::Tensor> gu_transposed_;
  std::vector<torch::Tensor> dn_transposed_;
  torch::Tensor valid_experts_;
  torch::Tensor shared_w1_q_;
  torch::Tensor shared_w1_s_;
  torch::Tensor shared_w3_q_;
  torch::Tensor shared_w3_s_;
  torch::Tensor shared_w2_q_;
  torch::Tensor shared_w2_s_;
  torch::Tensor gate_q_;
  torch::Tensor gate_s_;
  torch::Tensor gate_bias_;
  torch::Tensor gate_mask_;
  torch::Tensor route_scores_;
  int64_t group_size_;
  double limit_;
  int64_t top_k_;
  bool normalize_route_;
  bool indices_transposed_;
  bool vq_int8_;
  double routed_scaling_;
};

std::vector<torch::Tensor> int4_gemv_many_cpu(
    torch::Tensor x_row,
    std::vector<torch::Tensor> packed_list,
    std::vector<torch::Tensor> scale_list,
    int64_t group_size) {
  TORCH_CHECK(
      !x_row.is_cuda() && x_row.dim() == 2 && x_row.size(0) == 1,
      "CPU multi-INT4 GEMV requires one CPU input row");
  TORCH_CHECK(
      !packed_list.empty() && packed_list.size() == scale_list.size(),
      "multi-INT4 weight/scale counts must match");
  TORCH_CHECK(
      group_size > 0 && group_size % 2 == 0,
      "INT4 group size must be a positive even number");
  auto x = x_row.to(torch::kFloat32).contiguous();
  const int64_t cols = x.size(1);
  const int64_t groups = cols / group_size;
  const float* xp = x.data_ptr<float>();
  const bool use_w4a8 = cpu_w4a8_enabled() && group_size == 64;
  const bool use_w4abf16 =
      cpu_w4abf16_enabled() && group_size == 64;
  const bool use_expand_bf16 = cpu_expand_bf16_enabled();
  Int8Activation quantized;
  Bf16Activation bf16_input;
  if (use_w4a8) {
    quantized = quantize_int8_activation(xp, cols, group_size);
  }
  if (use_w4abf16 || use_expand_bf16) {
    bf16_input =
        quantize_bf16_activation(xp, cols, group_size);
  }

  std::vector<torch::Tensor> packed;
  std::vector<torch::Tensor> scales;
  std::vector<torch::Tensor> outputs;
  std::vector<const uint8_t*> packed_ptrs;
  std::vector<const at::Half*> scale_ptrs;
  std::vector<float*> output_ptrs;
  std::vector<torch::Tensor> expanded_weights;
  std::vector<const at::BFloat16*> expanded_ptrs;
  std::vector<int64_t> row_offsets(packed_list.size() + 1, 0);
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  for (size_t index = 0; index < packed_list.size(); ++index) {
    auto weight = packed_list[index].contiguous();
    auto scale = scale_list[index].contiguous();
    TORCH_CHECK(
        !weight.is_cuda() && !scale.is_cuda() &&
            weight.scalar_type() == at::kByte &&
            scale.scalar_type() == at::kHalf &&
            weight.dim() == 2 && scale.dim() == 2,
        "invalid multi-INT4 operand");
    TORCH_CHECK(
        weight.size(1) * 2 == cols &&
            scale.size(0) == weight.size(0) &&
            scale.size(1) == groups,
        "multi-INT4 shape mismatch");
    packed.push_back(weight);
    scales.push_back(scale);
    outputs.push_back(torch::empty({1, weight.size(0)}, options));
    packed_ptrs.push_back(packed.back().data_ptr<uint8_t>());
    scale_ptrs.push_back(scales.back().data_ptr<at::Half>());
    output_ptrs.push_back(outputs.back().data_ptr<float>());
    row_offsets[index + 1] = row_offsets[index] + weight.size(0);
  }
  if (use_expand_bf16) {
    expanded_weights.reserve(packed.size());
    expanded_ptrs.reserve(packed.size());
    for (size_t index = 0; index < packed.size(); ++index) {
      expanded_weights.push_back(
          expand_int4_bf16(
              packed[index], scales[index], cols, group_size));
      expanded_ptrs.push_back(
          expanded_weights.back().data_ptr<at::BFloat16>());
    }
  }

  const int64_t bytes_per_row = cols / 2;
#pragma omp parallel for schedule(static)
  for (int64_t item = 0; item < row_offsets.back(); ++item) {
    size_t matrix = 0;
    while (item >= row_offsets[matrix + 1]) {
      ++matrix;
    }
    const int64_t row = item - row_offsets[matrix];
    const uint8_t* weights =
        packed_ptrs[matrix] + row * bytes_per_row;
    const at::Half* row_scales =
        scale_ptrs[matrix] + row * groups;
    output_ptrs[matrix][row] =
        use_expand_bf16
        ? bf16_row_dot(
              bf16_input.values.data(),
              expanded_ptrs[matrix] + row * cols,
              cols)
        : use_w4abf16
        ? int4_row_dot_w4abf16(
              bf16_input, weights, row_scales)
        : use_w4a8
        ? int4_row_dot_w4a8(quantized, weights, row_scales)
        : int4_row_dot(
              xp, weights, row_scales, cols, group_size);
  }
  return outputs;
}

torch::Tensor int4_gemv_cpu(
    torch::Tensor x_row,
    torch::Tensor packed,
    torch::Tensor scales,
    int64_t cols,
    int64_t group_size) {
  TORCH_CHECK(!x_row.is_cuda() && !packed.is_cuda() && !scales.is_cuda(),
              "all INT4 operands must be on CPU");
  TORCH_CHECK(x_row.dim() == 2 && x_row.size(0) == 1,
              "the CPU INT4 decode kernel requires x shape [1,C]");
  TORCH_CHECK(packed.dim() == 2 && packed.scalar_type() == at::kByte,
              "packed weight must be uint8 [R,C/2]");
  TORCH_CHECK(scales.dim() == 2 && scales.scalar_type() == at::kHalf,
              "scales must be float16 [R,C/group]");
  TORCH_CHECK(group_size > 0 && group_size % 2 == 0,
              "group size must be a positive even number");
  TORCH_CHECK(cols == packed.size(1) * 2 && x_row.size(1) == cols,
              "INT4 input width mismatch");
  TORCH_CHECK(scales.size(0) == packed.size(0) &&
                  scales.size(1) * group_size == cols,
              "INT4 scale shape mismatch");

  auto x = x_row.to(torch::kFloat32).contiguous();
  auto q = packed.contiguous();
  auto s = scales.contiguous();
  const int64_t rows = q.size(0);
  const int64_t groups = cols / group_size;
  const int64_t bytes_per_group = group_size / 2;
  const float* xp = x.data_ptr<float>();
  const uint8_t* qp = q.data_ptr<uint8_t>();
  const at::Half* sp = s.data_ptr<at::Half>();
  const bool use_w4a8 = cpu_w4a8_enabled() && group_size == 64;
  const bool use_expand_bf16 = cpu_expand_bf16_enabled();
  Int8Activation quantized;
  Bf16Activation bf16_input;
  torch::Tensor expanded;
  const at::BFloat16* expandedp = nullptr;
  if (use_w4a8) {
    quantized = quantize_int8_activation(xp, cols, group_size);
  }
  if (use_expand_bf16) {
    bf16_input =
        quantize_bf16_activation(xp, cols, group_size);
    expanded =
        expand_int4_bf16(q, s, cols, group_size);
    expandedp = expanded.data_ptr<at::BFloat16>();
  }
  auto out = torch::empty(
      {1, rows},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  float* op = out.data_ptr<float>();

  // Decode is a GEMV.  Parallelising by output row keeps each packed weight
  // row and its scales sequential while x remains shared in cache.
  at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
    for (int64_t r = begin; r < end; ++r) {
      const uint8_t* qrow = qp + r * (cols / 2);
      const at::Half* srow = sp + r * groups;
      if (use_expand_bf16) {
        op[r] = bf16_row_dot(
            bf16_input.values.data(),
            expandedp + r * cols,
            cols);
      } else if (use_w4a8) {
        op[r] = int4_row_dot_w4a8(quantized, qrow, srow);
      } else {
        float total = 0.0f;
        for (int64_t g = 0; g < groups; ++g) {
          const uint8_t* qgroup = qrow + g * bytes_per_group;
          const float* xgroup = xp + g * group_size;
          const float dot = int4_group_dot(xgroup, qgroup, group_size);
          total += dot * static_cast<float>(srow[g]);
        }
        op[r] = total;
      }
    }
  });
  return out;
}

torch::Tensor int4_grouped_gemv_cpu(
    torch::Tensor x_groups,
    torch::Tensor packed,
    torch::Tensor scales,
    int64_t cols,
    int64_t group_size,
    int64_t rows_per_input) {
  TORCH_CHECK(
      !x_groups.is_cuda() && !packed.is_cuda() && !scales.is_cuda(),
      "all grouped INT4 operands must be on CPU");
  TORCH_CHECK(x_groups.dim() == 2 && x_groups.size(1) == cols,
              "grouped INT4 input must be [G,C]");
  TORCH_CHECK(packed.dim() == 2 && packed.scalar_type() == at::kByte,
              "packed weight must be uint8 [R,C/2]");
  TORCH_CHECK(scales.dim() == 2 && scales.scalar_type() == at::kHalf,
              "scales must be float16 [R,C/group]");
  TORCH_CHECK(group_size > 0 && group_size % 2 == 0,
              "group size must be a positive even number");
  TORCH_CHECK(cols == packed.size(1) * 2,
              "grouped INT4 input width mismatch");
  TORCH_CHECK(
      rows_per_input > 0 &&
          packed.size(0) == x_groups.size(0) * rows_per_input,
      "grouped INT4 row partition mismatch");

  auto x = x_groups.to(torch::kFloat32).contiguous();
  auto q = packed.contiguous();
  auto s = scales.contiguous();
  const int64_t input_groups = x.size(0);
  const int64_t rows = q.size(0);
  const int64_t weight_groups = cols / group_size;
  TORCH_CHECK(
      scales.size(0) == rows && scales.size(1) == weight_groups,
      "grouped INT4 scale shape mismatch");
  const float* xp = x.data_ptr<float>();
  const uint8_t* qp = q.data_ptr<uint8_t>();
  const at::Half* sp = s.data_ptr<at::Half>();
  auto out = torch::empty(
      {input_groups, rows_per_input},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  float* op = out.data_ptr<float>();

  at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
    for (int64_t r = begin; r < end; ++r) {
      const int64_t input_group = r / rows_per_input;
      const float* xrow = xp + input_group * cols;
      const uint8_t* qrow = qp + r * (cols / 2);
      const at::Half* srow = sp + r * weight_groups;
      float total = 0.0f;
      for (int64_t g = 0; g < weight_groups; ++g) {
        total += int4_group_dot(
                     xrow + g * group_size,
                     qrow + g * (group_size / 2),
                     group_size) *
                 static_cast<float>(srow[g]);
      }
      op[r] = total;
    }
  });
  return out;
}

torch::Tensor o_proj_int4_cpu(
    torch::Tensor x_groups,
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    int64_t a_cols,
    int64_t a_group_size,
    int64_t rows_per_input,
    torch::Tensor b_packed,
    torch::Tensor b_scales,
    int64_t b_cols,
    int64_t b_group_size) {
  TORCH_CHECK(
      !x_groups.is_cuda() && !a_packed.is_cuda() &&
          !a_scales.is_cuda() && !b_packed.is_cuda() &&
          !b_scales.is_cuda() && x_groups.scalar_type() == at::kFloat &&
          x_groups.dim() == 2 && x_groups.size(1) == a_cols &&
          a_packed.scalar_type() == at::kByte &&
          a_scales.scalar_type() == at::kHalf &&
          b_packed.scalar_type() == at::kByte &&
          b_scales.scalar_type() == at::kHalf &&
          a_packed.size(0) == x_groups.size(0) * rows_per_input &&
          a_packed.size(1) * 2 == a_cols &&
          a_scales.size(0) == a_packed.size(0) &&
          a_scales.size(1) * a_group_size == a_cols &&
          b_cols == x_groups.size(0) * rows_per_input &&
          b_packed.size(1) * 2 == b_cols &&
          b_scales.size(0) == b_packed.size(0) &&
          b_scales.size(1) * b_group_size == b_cols,
      "CPU fused O projection shape mismatch");
  auto x = x_groups.contiguous();
  auto aq = a_packed.contiguous();
  auto as = a_scales.contiguous();
  auto bq = b_packed.contiguous();
  auto bs = b_scales.contiguous();
  auto middle = torch::empty(
      {x.size(0), rows_per_input},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto output = torch::empty(
      {1, bq.size(0)},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  const float* xp = x.data_ptr<float>();
  const uint8_t* aqp = aq.data_ptr<uint8_t>();
  const at::Half* asp = as.data_ptr<at::Half>();
  const uint8_t* bqp = bq.data_ptr<uint8_t>();
  const at::Half* bsp = bs.data_ptr<at::Half>();
  float* mp = middle.data_ptr<float>();
  float* op = output.data_ptr<float>();
  const int64_t a_rows = aq.size(0);
  const int64_t a_bytes = a_cols / 2;
  const int64_t a_groups = a_cols / a_group_size;
  const int64_t b_rows = bq.size(0);
  const int64_t b_bytes = b_cols / 2;
  const int64_t b_groups = b_cols / b_group_size;
  const bool use_w4a8 =
      cpu_w4a8_enabled() &&
      a_group_size == 64 && b_group_size == 64;
  const bool use_w4abf16 =
      cpu_w4abf16_enabled() &&
      a_group_size == 64 && b_group_size == 64;
  const bool use_expand_bf16 = cpu_expand_bf16_enabled();
  std::vector<Int8Activation> quantized_inputs;
  Int8Activation quantized_middle;
  std::vector<Bf16Activation> bf16_inputs;
  Bf16Activation bf16_middle;
  torch::Tensor expanded_a;
  torch::Tensor expanded_b;
  const at::BFloat16* expanded_ap = nullptr;
  const at::BFloat16* expanded_bp = nullptr;
  if (use_w4a8) {
    quantized_inputs.reserve(x.size(0));
    for (int64_t input = 0; input < x.size(0); ++input) {
      quantized_inputs.push_back(
          quantize_int8_activation(
              xp + input * a_cols, a_cols, a_group_size));
    }
  }
  if (use_w4abf16 || use_expand_bf16) {
    bf16_inputs.reserve(x.size(0));
    for (int64_t input = 0; input < x.size(0); ++input) {
      bf16_inputs.push_back(
          quantize_bf16_activation(
              xp + input * a_cols, a_cols, a_group_size));
    }
  }
  if (use_expand_bf16) {
    expanded_a =
        expand_int4_bf16(aq, as, a_cols, a_group_size);
    expanded_b =
        expand_int4_bf16(bq, bs, b_cols, b_group_size);
    expanded_ap = expanded_a.data_ptr<at::BFloat16>();
    expanded_bp = expanded_b.data_ptr<at::BFloat16>();
  }
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for (int64_t row = 0; row < a_rows; ++row) {
      const int64_t input = row / rows_per_input;
      const uint8_t* weights = aqp + row * a_bytes;
      const at::Half* row_scales = asp + row * a_groups;
      mp[row] =
          use_expand_bf16
          ? bf16_row_dot(
                bf16_inputs[input].values.data(),
                expanded_ap + row * a_cols,
                a_cols)
          : use_w4abf16
          ? int4_row_dot_w4abf16(
                bf16_inputs[input], weights, row_scales)
          : use_w4a8
          ? int4_row_dot_w4a8(
                quantized_inputs[input], weights, row_scales)
          : int4_row_dot(
                xp + input * a_cols,
                weights,
                row_scales,
                a_cols,
                a_group_size);
    }
    if (use_w4a8) {
#pragma omp single
      {
        quantized_middle =
            quantize_int8_activation(mp, b_cols, b_group_size);
      }
    }
    if (use_w4abf16 || use_expand_bf16) {
#pragma omp single
      {
        bf16_middle =
            quantize_bf16_activation(mp, b_cols, b_group_size);
      }
    }
#pragma omp for schedule(static)
    for (int64_t row = 0; row < b_rows; ++row) {
      const uint8_t* weights = bqp + row * b_bytes;
      const at::Half* row_scales = bsp + row * b_groups;
      op[row] =
          use_expand_bf16
          ? bf16_row_dot(
                bf16_middle.values.data(),
                expanded_bp + row * b_cols,
                b_cols)
          : use_w4abf16
          ? int4_row_dot_w4abf16(
                bf16_middle, weights, row_scales)
          : use_w4a8
          ? int4_row_dot_w4a8(
                quantized_middle, weights, row_scales)
          : int4_row_dot(
                mp, weights, row_scales, b_cols, b_group_size);
    }
  }
  return output;
}

std::vector<torch::Tensor> hc_pre_norm_cpu(
    torch::Tensor x,
    torch::Tensor mixes,
    torch::Tensor scale,
    torch::Tensor base,
    torch::Tensor norm,
    int64_t sinkhorn_iters,
    double rms_eps,
    double hc_eps,
    torch::Tensor y,
    torch::Tensor post,
    torch::Tensor comb) {
  TORCH_CHECK(
      !x.is_cuda() && !mixes.is_cuda() && !scale.is_cuda() &&
          !base.is_cuda() && !norm.is_cuda(),
      "all Hyper-Connection operands must be on CPU");
  TORCH_CHECK(
      x.scalar_type() == at::kFloat && mixes.scalar_type() == at::kFloat,
      "CPU Hyper-Connection requires float32 x and mixes");
  TORCH_CHECK(
      x.dim() == 4 && x.size(0) * x.size(1) == 1,
      "CPU Hyper-Connection decode requires one token");
  const int64_t hc = x.size(2);
  const int64_t hidden = x.size(3);
  TORCH_CHECK(hc > 0 && hc <= 8, "unsupported Hyper-Connection width");
  TORCH_CHECK(
      mixes.numel() == (2 + hc) * hc,
      "Hyper-Connection mix width mismatch");
  TORCH_CHECK(
      scale.numel() == 3 && base.numel() == (2 + hc) * hc,
      "Hyper-Connection scale/base shape mismatch");
  TORCH_CHECK(norm.numel() == hidden, "RMSNorm width mismatch");
  TORCH_CHECK(sinkhorn_iters > 0, "Sinkhorn iteration count must be positive");
  TORCH_CHECK(
      !y.is_cuda() && y.scalar_type() == at::kFloat &&
          y.is_contiguous() && y.numel() == hidden &&
          !post.is_cuda() && post.scalar_type() == at::kFloat &&
          post.is_contiguous() && post.numel() == hc &&
          !comb.is_cuda() && comb.scalar_type() == at::kFloat &&
          comb.is_contiguous() && comb.numel() == hc * hc,
      "CPU Hyper-Connection output workspaces must be contiguous FP32");

  auto xc = x.contiguous();
  auto mc = mixes.contiguous();
  auto sc = scale.to(torch::kFloat32).contiguous();
  auto bc = base.to(torch::kFloat32).contiguous();
  auto nc = norm.to(torch::kFloat32).contiguous();
  const float* xp = xc.data_ptr<float>();
  const float* mp = mc.data_ptr<float>();
  const float* sp = sc.data_ptr<float>();
  const float* bp = bc.data_ptr<float>();
  const float* np = nc.data_ptr<float>();

  float square_sum = 0.0f;
#if defined(__AVX512F__)
  __m512 square_acc = _mm512_setzero_ps();
  int64_t flat_index = 0;
  const int64_t flat_size = hc * hidden;
  for (; flat_index + 16 <= flat_size; flat_index += 16) {
    const __m512 value = _mm512_loadu_ps(xp + flat_index);
    square_acc = _mm512_fmadd_ps(value, value, square_acc);
  }
  square_sum = _mm512_reduce_add_ps(square_acc);
  for (; flat_index < flat_size; ++flat_index) {
    square_sum += xp[flat_index] * xp[flat_index];
  }
#else
  for (int64_t i = 0; i < hc * hidden; ++i) {
    square_sum += xp[i] * xp[i];
  }
#endif
  const float input_rms = 1.0f / std::sqrt(
      square_sum / static_cast<float>(hc * hidden) +
      static_cast<float>(rms_eps));

  float pre_values[8];
  float post_values[8];
  float comb_values[64];
  for (int64_t j = 0; j < hc; ++j) {
    const float pre_arg = mp[j] * input_rms * sp[0] + bp[j];
    const float post_arg =
        mp[hc + j] * input_rms * sp[1] + bp[hc + j];
    pre_values[j] =
        1.0f / (1.0f + std::exp(-pre_arg)) + static_cast<float>(hc_eps);
    post_values[j] = 2.0f / (1.0f + std::exp(-post_arg));
  }
  for (int64_t row = 0; row < hc; ++row) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (int64_t col = 0; col < hc; ++col) {
      const int64_t index = row * hc + col;
      const float value =
          mp[2 * hc + index] * input_rms * sp[2] + bp[2 * hc + index];
      comb_values[index] = value;
      maximum = std::max(maximum, value);
    }
    float denominator = 0.0f;
    for (int64_t col = 0; col < hc; ++col) {
      const int64_t index = row * hc + col;
      const float value = std::exp(comb_values[index] - maximum);
      comb_values[index] = value;
      denominator += value;
    }
    for (int64_t col = 0; col < hc; ++col) {
      const int64_t index = row * hc + col;
      comb_values[index] =
          comb_values[index] / denominator + static_cast<float>(hc_eps);
    }
  }
  for (int64_t col = 0; col < hc; ++col) {
    float denominator = static_cast<float>(hc_eps);
    for (int64_t row = 0; row < hc; ++row) {
      denominator += comb_values[row * hc + col];
    }
    for (int64_t row = 0; row < hc; ++row) {
      comb_values[row * hc + col] /= denominator;
    }
  }
  for (int64_t iteration = 1; iteration < sinkhorn_iters; ++iteration) {
    for (int64_t row = 0; row < hc; ++row) {
      float denominator = static_cast<float>(hc_eps);
      for (int64_t col = 0; col < hc; ++col) {
        denominator += comb_values[row * hc + col];
      }
      for (int64_t col = 0; col < hc; ++col) {
        comb_values[row * hc + col] /= denominator;
      }
    }
    for (int64_t col = 0; col < hc; ++col) {
      float denominator = static_cast<float>(hc_eps);
      for (int64_t row = 0; row < hc; ++row) {
        denominator += comb_values[row * hc + col];
      }
      for (int64_t row = 0; row < hc; ++row) {
        comb_values[row * hc + col] /= denominator;
      }
    }
  }

  float* yp = y.data_ptr<float>();
  float y_square_sum = 0.0f;
  int64_t y_index = 0;
#if defined(__AVX512F__)
  __m512 y_square_acc = _mm512_setzero_ps();
  for (; y_index + 16 <= hidden; y_index += 16) {
    __m512 value = _mm512_setzero_ps();
    for (int64_t j = 0; j < hc; ++j) {
      value = _mm512_fmadd_ps(
          _mm512_loadu_ps(xp + j * hidden + y_index),
          _mm512_set1_ps(pre_values[j]),
          value);
    }
    _mm512_storeu_ps(yp + y_index, value);
    y_square_acc = _mm512_fmadd_ps(value, value, y_square_acc);
  }
  y_square_sum = _mm512_reduce_add_ps(y_square_acc);
#endif
  for (; y_index < hidden; ++y_index) {
    float value = 0.0f;
    for (int64_t j = 0; j < hc; ++j) {
      value += pre_values[j] * xp[j * hidden + y_index];
    }
    yp[y_index] = value;
    y_square_sum += value * value;
  }
  const float output_rms = 1.0f / std::sqrt(
      y_square_sum / static_cast<float>(hidden) + static_cast<float>(rms_eps));
  int64_t norm_index = 0;
#if defined(__AVX512F__)
  const __m512 output_scale = _mm512_set1_ps(output_rms);
  for (; norm_index + 16 <= hidden; norm_index += 16) {
    _mm512_storeu_ps(
        yp + norm_index,
        _mm512_mul_ps(
            _mm512_mul_ps(
                _mm512_loadu_ps(yp + norm_index),
                output_scale),
            _mm512_loadu_ps(np + norm_index)));
  }
#endif
  for (; norm_index < hidden; ++norm_index) {
    yp[norm_index] *= output_rms * np[norm_index];
  }

  std::copy(post_values, post_values + hc, post.data_ptr<float>());
  std::copy(
      comb_values, comb_values + hc * hc, comb.data_ptr<float>());
  return {y, post, comb};
}

torch::Tensor hc_post_cpu(
    torch::Tensor out,
    torch::Tensor residual,
    torch::Tensor post,
    torch::Tensor comb,
    torch::Tensor result) {
  TORCH_CHECK(
      !out.is_cuda() && !residual.is_cuda() && !post.is_cuda() &&
          !comb.is_cuda(),
      "all Hyper-Connection post operands must be on CPU");
  TORCH_CHECK(
      out.scalar_type() == at::kFloat &&
          residual.scalar_type() == at::kFloat,
      "CPU Hyper-Connection post requires float32 operands");
  TORCH_CHECK(
      residual.dim() == 4 && residual.size(0) * residual.size(1) == 1,
      "CPU Hyper-Connection post requires one token");
  const int64_t hc = residual.size(2);
  const int64_t hidden = residual.size(3);
  TORCH_CHECK(
      out.numel() == hidden && post.numel() == hc &&
          comb.numel() == hc * hc,
      "Hyper-Connection post shape mismatch");
  const auto& oc = out;
  const auto& rc = residual;
  const auto& pc = post;
  const auto& cc = comb;
  TORCH_CHECK(
      post.scalar_type() == at::kFloat &&
          comb.scalar_type() == at::kFloat &&
          out.is_contiguous() && residual.is_contiguous() &&
          post.is_contiguous() && comb.is_contiguous(),
      "CPU Hyper-Connection post requires contiguous float32 tensors");
  TORCH_CHECK(
      !result.is_cuda() && result.scalar_type() == at::kFloat &&
          result.is_contiguous() && result.numel() == residual.numel(),
      "CPU Hyper-Connection post output must be contiguous float32");
  const float* op = oc.data_ptr<float>();
  const float* rp = rc.data_ptr<float>();
  const float* pp = pc.data_ptr<float>();
  const float* cp = cc.data_ptr<float>();
  float* resultp = result.data_ptr<float>();
  for (int64_t channel = 0; channel < hc; ++channel) {
    float* destination = resultp + channel * hidden;
    const float output_weight = pp[channel];
    int64_t d = 0;
#if defined(__AVX512F__)
    const __m512 output_scale = _mm512_set1_ps(output_weight);
    for (; d + 16 <= hidden; d += 16) {
      __m512 value = _mm512_mul_ps(
          _mm512_loadu_ps(op + d), output_scale);
      for (int64_t source = 0; source < hc; ++source) {
        value = _mm512_fmadd_ps(
            _mm512_loadu_ps(rp + source * hidden + d),
            _mm512_set1_ps(cp[source * hc + channel]),
            value);
      }
      _mm512_storeu_ps(destination + d, value);
    }
#endif
    for (; d < hidden; ++d) {
      float value = output_weight * op[d];
      for (int64_t source = 0; source < hc; ++source) {
        value += cp[source * hc + channel] *
                 rp[source * hidden + d];
      }
      destination[d] = value;
    }
  }
  return result;
}

std::vector<torch::Tensor> qkv_pre_cpu(
    torch::Tensor q_rank_raw,
    torch::Tensor kv_raw,
    torch::Tensor q_norm,
    torch::Tensor kv_norm,
    torch::Tensor rope_cos,
    torch::Tensor rope_sin,
    double rms_eps) {
  TORCH_CHECK(
      !q_rank_raw.is_cuda() && !kv_raw.is_cuda(),
      "CPU QKV preprocessing requires CPU tensors");
  TORCH_CHECK(
      q_rank_raw.scalar_type() == at::kFloat &&
          kv_raw.scalar_type() == at::kFloat &&
          q_rank_raw.dim() == 2 && q_rank_raw.size(0) == 1 &&
          kv_raw.dim() == 2 && kv_raw.size(0) == 1,
      "CPU QKV preprocessing requires float32 rows");
  auto qr = q_rank_raw.contiguous();
  auto kv = kv_raw.contiguous();
  auto qn = q_norm.to(torch::kFloat32).contiguous();
  auto kvn = kv_norm.to(torch::kFloat32).contiguous();
  auto cos = rope_cos.to(torch::kFloat32).contiguous();
  auto sin = rope_sin.to(torch::kFloat32).contiguous();
  const int64_t q_width = qr.size(1);
  const int64_t kv_width = kv.size(1);
  const int64_t rope_pairs = cos.numel();
  TORCH_CHECK(
      qn.numel() == q_width && kvn.numel() == kv_width &&
          sin.numel() == rope_pairs && rope_pairs * 2 <= kv_width,
      "CPU QKV preprocessing shape mismatch");
  auto q_out = torch::empty_like(qr);
  auto kv_out = torch::empty_like(kv);
  const float* qrp = qr.data_ptr<float>();
  const float* kvp = kv.data_ptr<float>();
  const float* qnp = qn.data_ptr<float>();
  const float* kvnp = kvn.data_ptr<float>();
  const float* cp = cos.data_ptr<float>();
  const float* sp = sin.data_ptr<float>();
  float* qop = q_out.data_ptr<float>();
  float* kvop = kv_out.data_ptr<float>();

  float q_square = float_dot(qrp, qrp, q_width);
  const float q_scale = 1.0f / std::sqrt(
      q_square / static_cast<float>(q_width) +
      static_cast<float>(rms_eps));
  for (int64_t index = 0; index < q_width; ++index) {
    qop[index] = qrp[index] * q_scale * qnp[index];
  }
  float kv_square = float_dot(kvp, kvp, kv_width);
  const float kv_scale = 1.0f / std::sqrt(
      kv_square / static_cast<float>(kv_width) +
      static_cast<float>(rms_eps));
  for (int64_t index = 0; index < kv_width; ++index) {
    kvop[index] = kvp[index] * kv_scale * kvnp[index];
  }
  const int64_t rope_start = kv_width - rope_pairs * 2;
  for (int64_t pair = 0; pair < rope_pairs; ++pair) {
    const int64_t index = rope_start + 2 * pair;
    const float first = kvop[index];
    const float second = kvop[index + 1];
    kvop[index] = first * cp[pair] - second * sp[pair];
    kvop[index + 1] = first * sp[pair] + second * cp[pair];
  }
  return {q_out, kv_out};
}

torch::Tensor q_post_cpu(
    torch::Tensor query,
    torch::Tensor rope_cos,
    torch::Tensor rope_sin,
    double rms_eps) {
  TORCH_CHECK(
      !query.is_cuda() && query.scalar_type() == at::kFloat &&
          query.dim() == 4 && query.size(0) * query.size(1) == 1,
      "CPU Q postprocessing requires one float32 token");
  auto q = query.contiguous();
  auto cos = rope_cos.to(torch::kFloat32).contiguous();
  auto sin = rope_sin.to(torch::kFloat32).contiguous();
  const int64_t heads = q.size(2);
  const int64_t head_dim = q.size(3);
  const int64_t rope_pairs = cos.numel();
  TORCH_CHECK(
      sin.numel() == rope_pairs && rope_pairs * 2 <= head_dim,
      "CPU Q postprocessing RoPE shape mismatch");
  auto output = torch::empty_like(q);
  const float* qp = q.data_ptr<float>();
  const float* cp = cos.data_ptr<float>();
  const float* sp = sin.data_ptr<float>();
  float* op = output.data_ptr<float>();
  const int64_t rope_start = head_dim - rope_pairs * 2;
#pragma omp parallel for schedule(static)
  for (int64_t head = 0; head < heads; ++head) {
    const float* source = qp + head * head_dim;
    float* destination = op + head * head_dim;
    const float square = float_dot(source, source, head_dim);
    const float scale = 1.0f / std::sqrt(
        square / static_cast<float>(head_dim) +
        static_cast<float>(rms_eps));
    for (int64_t index = 0; index < head_dim; ++index) {
      destination[index] = source[index] * scale;
    }
    for (int64_t pair = 0; pair < rope_pairs; ++pair) {
      const int64_t index = rope_start + 2 * pair;
      const float first = destination[index];
      const float second = destination[index + 1];
      destination[index] = first * cp[pair] - second * sp[pair];
      destination[index + 1] = first * sp[pair] + second * cp[pair];
    }
  }
  return output;
}

torch::Tensor q_int4_post_cpu(
    torch::Tensor q_rank,
    torch::Tensor packed,
    torch::Tensor scales,
    int64_t cols,
    int64_t group_size,
    torch::Tensor rope_cos,
    torch::Tensor rope_sin,
    int64_t heads,
    int64_t head_dim,
    double rms_eps) {
  TORCH_CHECK(
      !q_rank.is_cuda() && !packed.is_cuda() && !scales.is_cuda() &&
          q_rank.scalar_type() == at::kFloat &&
          q_rank.dim() == 2 && q_rank.size(0) == 1 &&
          q_rank.size(1) == cols && packed.scalar_type() == at::kByte &&
          scales.scalar_type() == at::kHalf &&
          packed.size(0) == heads * head_dim &&
          packed.size(1) * 2 == cols &&
          scales.size(0) == packed.size(0) &&
          scales.size(1) * group_size == cols,
      "CPU fused Q INT4 projection shape mismatch");
  auto x = q_rank.contiguous();
  auto q = packed.contiguous();
  auto s = scales.contiguous();
  auto cos = rope_cos.to(torch::kFloat32).contiguous();
  auto sin = rope_sin.to(torch::kFloat32).contiguous();
  const int64_t rope_pairs = cos.numel();
  TORCH_CHECK(
      sin.numel() == rope_pairs && rope_pairs * 2 <= head_dim,
      "CPU fused Q INT4 RoPE shape mismatch");
  auto output = torch::empty(
      {1, 1, heads, head_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  const float* xp = x.data_ptr<float>();
  const uint8_t* qp = q.data_ptr<uint8_t>();
  const at::Half* sp = s.data_ptr<at::Half>();
  const float* cp = cos.data_ptr<float>();
  const float* sinp = sin.data_ptr<float>();
  float* op = output.data_ptr<float>();
  const int64_t rows = heads * head_dim;
  const int64_t bytes_per_row = cols / 2;
  const int64_t groups = cols / group_size;
  const int64_t rope_start = head_dim - rope_pairs * 2;
  const bool use_w4a8 = cpu_w4a8_enabled() && group_size == 64;
  const bool use_w4abf16 =
      cpu_w4abf16_enabled() && group_size == 64;
  const bool use_expand_bf16 = cpu_expand_bf16_enabled();
  Int8Activation quantized;
  Bf16Activation bf16_input;
  torch::Tensor expanded;
  const at::BFloat16* expandedp = nullptr;
  if (use_w4a8) {
    quantized = quantize_int8_activation(xp, cols, group_size);
  }
  if (use_w4abf16 || use_expand_bf16) {
    bf16_input =
        quantize_bf16_activation(xp, cols, group_size);
  }
  if (use_expand_bf16) {
    expanded =
        expand_int4_bf16(q, s, cols, group_size);
    expandedp = expanded.data_ptr<at::BFloat16>();
  }
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
      const uint8_t* weights = qp + row * bytes_per_row;
      const at::Half* row_scales = sp + row * groups;
      op[row] =
          use_expand_bf16
          ? bf16_row_dot(
                bf16_input.values.data(),
                expandedp + row * cols,
                cols)
          : use_w4abf16
          ? int4_row_dot_w4abf16(
                bf16_input, weights, row_scales)
          : use_w4a8
          ? int4_row_dot_w4a8(quantized, weights, row_scales)
          : int4_row_dot(
                xp, weights, row_scales, cols, group_size);
    }
#pragma omp for schedule(static)
    for (int64_t head = 0; head < heads; ++head) {
      float* destination = op + head * head_dim;
      const float square = float_dot(destination, destination, head_dim);
      const float scale = 1.0f / std::sqrt(
          square / static_cast<float>(head_dim) +
          static_cast<float>(rms_eps));
      for (int64_t index = 0; index < head_dim; ++index) {
        destination[index] *= scale;
      }
      for (int64_t pair = 0; pair < rope_pairs; ++pair) {
        const int64_t index = rope_start + 2 * pair;
        const float first = destination[index];
        const float second = destination[index + 1];
        destination[index] = first * cp[pair] - second * sinp[pair];
        destination[index + 1] =
            first * sinp[pair] + second * cp[pair];
      }
    }
  }
  return output;
}

torch::Tensor attention_decode_cpu(
    torch::Tensor query,
    torch::Tensor raw_values,
    torch::Tensor raw_positions,
    torch::Tensor selected_values,
    torch::Tensor sink,
    torch::Tensor rope_cos,
    torch::Tensor rope_sin,
    double scale) {
  TORCH_CHECK(
      !query.is_cuda() && !raw_values.is_cuda() &&
          !raw_positions.is_cuda() && !selected_values.is_cuda(),
      "CPU attention received a CUDA tensor");
  TORCH_CHECK(
      query.scalar_type() == at::kFloat &&
          raw_values.scalar_type() == at::kFloat &&
          selected_values.scalar_type() == at::kFloat &&
          sink.scalar_type() == at::kFloat,
      "CPU attention currently requires float32 operands");
  TORCH_CHECK(
      raw_positions.scalar_type() == at::kLong,
      "raw positions must be int64");
  TORCH_CHECK(query.dim() == 3, "query must be [B,H,D]");
  TORCH_CHECK(raw_values.dim() == 3, "raw values must be [B,W,D]");
  TORCH_CHECK(selected_values.dim() == 3,
              "selected values must be [B,K,D]");

  auto q = query.contiguous();
  auto raw = raw_values.contiguous();
  auto positions = raw_positions.contiguous();
  auto selected = selected_values.contiguous();
  auto sinks = sink.to(torch::kFloat32).contiguous();
  auto cos = rope_cos.to(torch::kFloat32).contiguous();
  auto sin = rope_sin.to(torch::kFloat32).contiguous();
  const int64_t batch = q.size(0);
  const int64_t heads = q.size(1);
  const int64_t dim = q.size(2);
  const int64_t raw_count = raw.size(1);
  const int64_t selected_count = selected.size(1);
  const int64_t total_count = raw_count + selected_count;
  const int64_t rope_pairs = cos.numel();
  TORCH_CHECK(raw.size(0) == batch && raw.size(2) == dim,
              "raw value shape mismatch");
  TORCH_CHECK(selected.size(0) == batch && selected.size(2) == dim,
              "selected value shape mismatch");
  TORCH_CHECK(
              positions.dim() == 2 &&
                  positions.size(0) == batch &&
                  positions.size(1) == raw_count,
              "raw position shape mismatch");
  TORCH_CHECK(sinks.numel() == heads, "attention sink shape mismatch");
  TORCH_CHECK(total_count <= 1024, "CPU attention source limit is 1024");
  TORCH_CHECK(rope_pairs * 2 <= dim, "RoPE width exceeds head width");

  const float* qp = q.data_ptr<float>();
  const float* rp = raw.data_ptr<float>();
  const int64_t* pp = positions.data_ptr<int64_t>();
  const float* vp = selected.data_ptr<float>();
  const float* skp = sinks.data_ptr<float>();
  const float* cp = cos.data_ptr<float>();
  const float* snp = sin.data_ptr<float>();
  auto output = torch::zeros_like(q);
  float* op = output.data_ptr<float>();
  const float score_scale = static_cast<float>(scale);

  at::parallel_for(0, batch * heads, 1, [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t b = item / heads;
      const int64_t h = item - b * heads;
      const float* qrow = qp + item * dim;
      float* out = op + item * dim;
      float scores[1024];
      float maximum = -std::numeric_limits<float>::infinity();
      for (int64_t source = 0; source < raw_count; ++source) {
        if (pp[b * raw_count + source] < 0) {
          scores[source] = -std::numeric_limits<float>::infinity();
          continue;
        }
        const float value = float_dot(
            qrow, rp + (b * raw_count + source) * dim, dim) *
            score_scale;
        scores[source] = value;
        maximum = std::max(maximum, value);
      }
      for (int64_t source = 0; source < selected_count; ++source) {
        const float value = float_dot(
            qrow, vp + (b * selected_count + source) * dim, dim) *
            score_scale;
        scores[raw_count + source] = value;
        maximum = std::max(maximum, value);
      }
      float denominator = std::exp(skp[h] - maximum);
      for (int64_t source = 0; source < total_count; ++source) {
        if (!std::isfinite(scores[source])) {
          continue;
        }
        const float probability = std::exp(scores[source] - maximum);
        denominator += probability;
        const float* value = (
            source < raw_count
            ? rp + (b * raw_count + source) * dim
            : vp + (b * selected_count + source - raw_count) * dim);
        float_axpy(out, value, probability, dim);
      }
      const float inverse_denominator = 1.0f / denominator;
      for (int64_t d = 0; d < dim; ++d) {
        out[d] *= inverse_denominator;
      }
      const int64_t rope_start = dim - rope_pairs * 2;
      for (int64_t pair = 0; pair < rope_pairs; ++pair) {
        const int64_t offset = rope_start + pair * 2;
        const float first = out[offset];
        const float second = out[offset + 1];
        // inverse RoPE: sin is negated relative to the forward rotation.
        out[offset] = first * cp[pair] + second * snp[pair];
        out[offset + 1] = -first * snp[pair] + second * cp[pair];
      }
    }
  });
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<CpuResidentProjectionLayer>(
      module, "CpuResidentProjectionLayer")
      .def(
          pybind11::init<
              std::vector<torch::Tensor>,
              std::vector<torch::Tensor>,
              std::vector<int64_t>,
              std::vector<int64_t>,
              int64_t,
              int64_t>())
      .def("forward", &CpuResidentProjectionLayer::forward)
      .def(
          "forward_combined",
          &CpuResidentProjectionLayer::forward_combined)
      .def(
          "forward_grouped",
          &CpuResidentProjectionLayer::forward_grouped);
  pybind11::class_<CpuPackedThreeLayer>(
      module, "CpuPackedThreeLayer")
      .def(
          pybind11::init<
              std::vector<torch::Tensor>,
              std::vector<torch::Tensor>,
              int64_t,
              int64_t,
              int64_t,
              std::vector<torch::Tensor>,
              std::vector<torch::Tensor>,
              int64_t,
              int64_t,
              int64_t,
              std::vector<torch::Tensor>,
              std::vector<torch::Tensor>,
              int64_t,
              int64_t,
              int64_t>())
      .def("forward", &CpuPackedThreeLayer::forward);
  pybind11::class_<CpuPackedThreeMixedLayer>(
      module, "CpuPackedThreeMixedLayer")
      .def(
          pybind11::init<
              std::vector<torch::Tensor>,
              std::vector<torch::Tensor>,
              std::vector<int64_t>,
              std::vector<int64_t>,
              std::vector<int64_t>,
              std::vector<int64_t>,
              std::vector<torch::Tensor>,
              std::vector<torch::Tensor>,
              std::vector<int64_t>,
              std::vector<int64_t>,
              std::vector<int64_t>,
              std::vector<int64_t>,
              std::vector<torch::Tensor>,
              std::vector<torch::Tensor>,
              std::vector<int64_t>,
              std::vector<int64_t>,
              std::vector<int64_t>,
              std::vector<int64_t>>())
      .def("forward", &CpuPackedThreeMixedLayer::forward)
      .def(
          "configure_fused_moe",
          &CpuPackedThreeMixedLayer::configure_fused_moe)
      .def(
          "forward_fused_moe",
          &CpuPackedThreeMixedLayer::forward_fused_moe)
      .def(
          "configure_latent_moe",
          &CpuPackedThreeMixedLayer::configure_latent_moe)
      .def(
          "forward_latent_moe",
          &CpuPackedThreeMixedLayer::forward_latent_moe);
  pybind11::class_<CpuMoeLayer>(module, "CpuMoeLayer")
      .def(
          pybind11::init<
              std::vector<torch::Tensor>,
              std::vector<torch::Tensor>,
              std::vector<torch::Tensor>,
              std::vector<torch::Tensor>,
              torch::Tensor,
              torch::Tensor,
              torch::Tensor,
              torch::Tensor,
              torch::Tensor,
              torch::Tensor,
              torch::Tensor,
              torch::Tensor,
              torch::Tensor,
              torch::Tensor,
              torch::Tensor,
              int64_t,
              double,
              int64_t,
              bool,
              double>())
      .def("forward", &CpuMoeLayer::forward)
      .def("forward_learned", &CpuMoeLayer::forward_learned);
  module.def(
      "vq_gemv",
      &vq_gemv_cpu,
      "CCCP uint8/uint16 VQ GEMV for CPU");
  module.def(
      "bf16_grouped_gemv",
      &bf16_grouped_gemv_cpu,
      "CCCP grouped BF16 GEMV for one shared CPU input");
  module.def(
      "block_fp8_gemv",
      &block_fp8_gemv_cpu,
      "CCCP compact E4M3FN block128 GEMV for CPU");
  module.def(
      "block_fp8_gemm",
      &block_fp8_gemm_cpu,
      "CCCP compact E4M3FN block128 2..16 token GEMM for CPU");
  module.def(
      "block_fp8_to_block_major",
      &block_fp8_to_block_major_cpu,
      "CCCP compact row-major to 32x128 block-major CPU layout");
  module.def(
      "block_fp8_compile_q4_0",
      &block_fp8_compile_q4_0_cpu,
      "CCCP online block-FP8 to Q4_0 linear CPU execution image");
  module.def(
      "q4_0_gemv",
      &q4_0_gemv_cpu,
      "CCCP Q4_0 by Q8_0 linear CPU GEMV");
  module.def(
      "q4_0_gemm",
      &q4_0_gemm_cpu,
      "CCCP Q4_0 by Q8_0 2..64 token CPU GEMM");
  module.def(
      "block_fp8_grouped_gemv",
      &block_fp8_grouped_gemv_cpu,
      "CCCP grouped compact E4M3FN block128 GEMV for CPU");
  module.def(
      "block_fp8_grouped_rows_gemv",
      &block_fp8_grouped_rows_gemv_cpu,
      "CCCP one-input-per-projection compact E4M3FN GEMV for CPU");
  module.def(
      "block_fp8_grouped_gemm",
      &block_fp8_grouped_gemm_cpu,
      "CCCP grouped compact E4M3FN block128 2..16 token GEMM for CPU");
  module.def(
      "reset_block_fp8_gemv_profile",
      &reset_block_fp8_gemv_profile_cpu,
      "Reset compact block-FP8 CPU GEMV timers");
  module.def(
      "block_fp8_gemv_profile",
      &block_fp8_gemv_profile_cpu,
      "Read compact block-FP8 CPU GEMV timers");
  module.def(
      "reset_resident_moe_phase_profile",
      &reset_resident_moe_phase_profile_cpu,
      "Reset resident Router/shared/routed CPU MoE timers");
  module.def(
      "resident_moe_phase_profile",
      &resident_moe_phase_profile_cpu,
      "Read resident Router/shared/routed CPU MoE timers");
  module.def(
      "reset_latent_moe_phase_profile",
      &reset_latent_moe_phase_profile_cpu,
      "Reset resident latent-MoE CPU timers");
  module.def(
      "latent_moe_phase_profile",
      &latent_moe_phase_profile_cpu,
      "Read resident latent-MoE CPU timers");
  module.def(
      "reset_resident_projection_profile",
      &reset_resident_projection_profile_cpu,
      "Reset fixed-address mixed CPU projection timers");
  module.def(
      "resident_projection_profile",
      &resident_projection_profile_cpu,
      "Read fixed-address mixed CPU projection timers");
  module.def(
      "vq_gemv_list",
      &vq_gemv_list_cpu,
      "CCCP list-backed uint8/uint16 VQ GEMV for CPU");
  module.def(
      "vq_gemv_packed_list",
      &vq_gemv_packed_list_cpu,
      "CCCP list-backed packed 8--16-bit VQ GEMV for CPU");
  module.def(
      "vq_dequant_packed",
      &vq_dequant_packed_cpu,
      "CCCP row-major packed VQ dequant for expert-grouped CPU GEMM");
  module.def(
      "q4_0_dequant",
      &q4_0_dequant_cpu,
      "CCCP runtime Q4 dequant for expert-grouped CPU GEMM");
  module.def(
      "vq_repack_block_major",
      &vq_repack_block_major_cpu,
      "CCCP compact packed VQ row-to-block-major CPU relayout");
  module.def(
      "vq_repack_row_tile",
      &vq_repack_row_tile_cpu,
      "CCCP compact packed VQ row-to-tile-major CPU relayout");
  module.def(
      "vq_compile_u16_row_tile",
      &vq_compile_u16_row_tile_cpu,
      "CCCP online packed VQ to uint16 row-tile CPU compilation");
  module.def(
      "vq_compile_q4_0",
      &vq_compile_q4_0_cpu,
      "CCCP online VQ to Q4_0 linear CPU execution image");
  module.def(
      "moe_packed_three_projection",
      &moe_packed_three_projection_cpu,
      "CCCP packed Gate/Up/activation/Down Top-K MoE for CPU");
  module.def(
      "reset_three_projection_phase_profile",
      &reset_three_projection_phase_profile_cpu,
      "Reset packed three-projection CPU MoE phase timers");
  module.def(
      "three_projection_phase_profile",
      &three_projection_phase_profile_cpu,
      "Read packed three-projection CPU MoE phase timers");
  module.def(
      "route_topk_sigmoid",
      &route_topk_sigmoid_cpu,
      "Stable compact CPU sigmoid Router Top-K");
  module.def(
      "moe_packed_topk",
      &moe_packed_topk_cpu,
      "CCCP persistent-pool mixed packed Top-K MoE for CPU");
  module.def(
      "reset_packed_moe_phase_profile",
      &reset_packed_moe_phase_profile_cpu,
      "Reset mixed packed Top-K CPU MoE phase timers");
  module.def(
      "packed_moe_phase_profile",
      &packed_moe_phase_profile_cpu,
      "Read mixed packed Top-K CPU MoE phase timers");
  module.def(
      "qwen35_delta_recurrent",
      &qwen35_delta_recurrent_cpu,
      "CCCP fused Qwen3.5 gated-delta recurrence for CPU");
  module.def(
      "qwen35_conv1d_update",
      &qwen35_conv1d_update_cpu,
      "CCCP fused Qwen3.5 cached depthwise convolution for CPU");
  module.def(
      "kda_recurrent",
      &kda_recurrent_cpu,
      "CCCP fused AVX-512 KDA recurrence for CPU");
  module.def(
      "short_conv3",
      &short_conv3_cpu,
      "CCCP fused three-stream short convolution for CPU");
  module.def(
      "gated_rmsnorm",
      &gated_rmsnorm_cpu,
      "CCCP fused gated RMSNorm for CPU");
  module.def(
      "moe_mixed",
      &moe_mixed_cpu,
      "CCCP fused routed VQ and shared INT4 MoE for CPU");
  module.def(
      "reset_moe_phase_profile",
      &reset_moe_phase_profile_cpu,
      "Reset CCCP CPU MoE phase timers");
  module.def(
      "moe_phase_profile",
      &moe_phase_profile_cpu,
      "Read CCCP CPU MoE phase timers");
  module.def("int4_gemv", &int4_gemv_cpu, "CCCP packed INT4 GEMV for CPU");
  module.def(
      "int4_gemv_many",
      &int4_gemv_many_cpu,
      "CCCP shared-input packed INT4 GEMVs for CPU");
  module.def(
      "int4_grouped_gemv",
      &int4_grouped_gemv_cpu,
      "CCCP grouped-input packed INT4 GEMV for CPU");
  module.def(
      "o_proj_int4",
      &o_proj_int4_cpu,
      "CCCP fused grouped and dense packed INT4 O projection for CPU");
  module.def(
      "hc_pre_norm",
      &hc_pre_norm_cpu,
      "CCCP fused Hyper-Connection pre and RMSNorm for CPU");
  module.def(
      "hc_post",
      &hc_post_cpu,
      "CCCP fused Hyper-Connection post for CPU");
  module.def(
      "qkv_pre",
      &qkv_pre_cpu,
      "CCCP fused Q-rank/KV RMSNorm and KV RoPE for CPU");
  module.def(
      "q_post",
      &q_post_cpu,
      "CCCP fused per-head Q RMSNorm and RoPE for CPU");
  module.def(
      "q_int4_post",
      &q_int4_post_cpu,
      "CCCP fused packed INT4 Q projection, RMSNorm and RoPE for CPU");
  module.def(
      "attention_decode",
      &attention_decode_cpu,
      "CCCP fused single-token attention for CPU");
}
