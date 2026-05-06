/**
 * topk_v2_remap.cu — same algorithm as topk_v2.cu (8-bit-radix select +
 * 4-round refinement) but applies an element-wise transform
 * `apply_transform_tmpl<MODE>(x, p)` to every score before bucketing.
 * The transform is template-specialized per MODE so it inlines at every
 * load site (zero per-element branching in the inner loops).
 *
 * MAPPING_NONE compiles to the identity (no overhead vs the unmapped
 * topk_output_v2 kernel). Public entry point: topk_output_v2_remap
 * (see register.h). Same select32 fast-path routing as topk_output_v2.
 */

#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <optional>

#include <cub/warp/warp_merge_sort.cuh>

#include "register.h"
#include "topk_mapping.cuh"

namespace {

// =====================================================================
// SELECT32-SORT32 small-K fast path with remap, templated on MODE.
// Mirrors topk_v2_select32::Kernel from topk_v2.cu, with the raw score
// replaced by `apply_transform_tmpl<MODE>(raw, mapping_power)` at every
// bucketing site.
// =====================================================================

namespace topk_v2_remap_select32 {

__device__ __forceinline__ uint32_t score_to_key32_bf16(float x) {
    const uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

struct DescendingUint32 {
    __device__ __forceinline__ bool operator()(const uint32_t& a, const uint32_t& b) const {
        return a > b;
    }
};

template <int NUM_THREADS, int MODE>
__global__ __launch_bounds__(NUM_THREADS)
void Kernel(
    const __nv_bfloat16* __restrict__ score,
    const int*           __restrict__ dense_kv_indptr,
    const int*           __restrict__ sparse_kv_indptr,
    const int*           __restrict__ dense_kv_indices,
    int*                 __restrict__ sparse_kv_indices,
    const int            topk_val,
    const int            page_reserved_bos,
    const int            page_reserved_eos,
    const float          mapping_power)
{
    constexpr int LOCAL_K = 32;
    constexpr int kRadix  = 256;
    constexpr int CAND_MAX = 512;

    alignas(128) __shared__ int s_hist_buf[2][kRadix + 128];
    __shared__ int      s_above_count;
    __shared__ int      s_thresh_above_count;
    __shared__ int      s_thresh_at_count;
    __shared__ int      s_threshold_bin;
    __shared__ int      s_last_remain;
    __shared__ int      s_sub_threshold_bin;
    __shared__ int      s_sub_last_remain;
    __shared__ int      s_strictly_above_sub;
    __shared__ int      s_cand_count;
    __shared__ uint32_t s_top_keys[LOCAL_K];
    __shared__ int32_t  s_top_idx [LOCAL_K];
    __shared__ int32_t  s_cand_idx[CAND_MAX];

    using LocalSortT = cub::WarpMergeSort<uint32_t, 1, 32, int32_t>;
    __shared__ typename LocalSortT::TempStorage local_sort_smem;

    extern __shared__ uint8_t s_bins[];

    const int b  = blockIdx.x;
    const int tx = threadIdx.x;

    const int row_start = dense_kv_indptr[b]     + page_reserved_bos;
    const int row_end   = dense_kv_indptr[b + 1] - page_reserved_eos;
    const int row_len   = row_end - row_start;

    int*                                out_idx    = sparse_kv_indices + sparse_kv_indptr[b]
                                                   + page_reserved_bos;
    const __nv_bfloat16* __restrict__   row_scores = score + row_start;
    const int* __restrict__             row_idxmap = dense_kv_indices + row_start;

    for (int i = tx; i < kRadix + 128; i += NUM_THREADS) {
        s_hist_buf[0][i] = 0;
        s_hist_buf[1][i] = 0;
    }
    if (tx == 0) {
        s_above_count = 0; s_thresh_above_count = 0; s_thresh_at_count = 0;
        s_threshold_bin = -1; s_last_remain = 0;
        s_sub_threshold_bin = -1; s_sub_last_remain = 0; s_strictly_above_sub = 0;
        s_cand_count = 0;
    }
    if (tx < LOCAL_K) { s_top_keys[tx] = 0u; s_top_idx[tx] = -1; }
    __syncthreads();

    if (row_len <= 0) {
        if (tx < topk_val) out_idx[tx] = -1;
        return;
    }

    auto run_cumsum_strided = [&]() {
        if (tx < 32) {
            constexpr int kPerLane = kRadix / 32;
            int locals[kPerLane];
            #pragma unroll
            for (int j = 0; j < kPerLane; ++j) {
                locals[j] = s_hist_buf[0][tx * kPerLane + j];
            }
            #pragma unroll
            for (int j = kPerLane - 2; j >= 0; --j) {
                locals[j] += locals[j + 1];
            }
            int my_sum = locals[0];
            #pragma unroll
            for (int delta = 1; delta < 32; delta *= 2) {
                const int v = __shfl_down_sync(0xFFFFFFFF, my_sum, delta);
                if (tx + delta < 32) {
                    my_sum += v;
                }
            }
            const int suffix = my_sum - locals[0];
            #pragma unroll
            for (int j = 0; j < kPerLane; ++j) {
                s_hist_buf[0][tx * kPerLane + j] = locals[j] + suffix;
            }
        }
        __syncthreads();
    };

    for (int local_rank = tx; local_rank < row_len; local_rank += NUM_THREADS) {
        const float    raw    = __bfloat162float(row_scores[local_rank]);
        const float    mapped = apply_transform_tmpl<MODE>(raw, mapping_power);
        const uint32_t key    = score_to_key32_bf16(mapped);
        const int      bin    = static_cast<int>(key >> 24);
        s_bins[local_rank] = static_cast<uint8_t>(bin);
        ::atomicAdd(&s_hist_buf[0][bin], 1);
    }
    __syncthreads();
    run_cumsum_strided();
    const int total_items = s_hist_buf[0][0];

    for (int bin = tx; bin < kRadix; bin += NUM_THREADS) {
        const int total_at_or_above = s_hist_buf[0][bin];
        const int strictly_above    = (bin + 1 < kRadix) ? s_hist_buf[0][bin + 1] : 0;
        if (total_at_or_above >= LOCAL_K && strictly_above < LOCAL_K) {
            s_threshold_bin = bin;
            s_last_remain   = LOCAL_K - strictly_above;
        }
    }
    __syncthreads();

    if (total_items <= LOCAL_K) {
        for (int local_rank = tx; local_rank < row_len; local_rank += NUM_THREADS) {
            const float    raw    = __bfloat162float(row_scores[local_rank]);
            const float    mapped = apply_transform_tmpl<MODE>(raw, mapping_power);
            const uint32_t key    = score_to_key32_bf16(mapped);
            const int      slot   = ::atomicAdd(&s_above_count, 1);
            if (slot < LOCAL_K) {
                s_top_keys[slot] = key;
                s_top_idx [slot] = row_idxmap[local_rank];
            }
        }
        __syncthreads();
    } else {
        const int threshold_bin = s_threshold_bin;
        for (int i = tx; i < kRadix + 128; i += NUM_THREADS) {
            s_hist_buf[0][i] = 0;
            s_hist_buf[1][i] = 0;
        }
        __syncthreads();

        for (int local_rank = tx; local_rank < row_len; local_rank += NUM_THREADS) {
            const int cached_bin = static_cast<int>(s_bins[local_rank]);
            if (cached_bin < threshold_bin) continue;
            const float    raw    = __bfloat162float(row_scores[local_rank]);
            const float    mapped = apply_transform_tmpl<MODE>(raw, mapping_power);
            const uint32_t key    = score_to_key32_bf16(mapped);
            if (cached_bin > threshold_bin) {
                const int slot = ::atomicAdd(&s_above_count, 1);
                if (slot < LOCAL_K) {
                    s_top_keys[slot] = key;
                    s_top_idx [slot] = row_idxmap[local_rank];
                }
            } else {
                ::atomicAdd(&s_hist_buf[0][static_cast<int>((key >> 16) & 0xFF)], 1);
                const int cslot = ::atomicAdd(&s_cand_count, 1);
                if (cslot < CAND_MAX) {
                    s_cand_idx[cslot] = local_rank;
                }
            }
        }
        __syncthreads();
        run_cumsum_strided();

        const int last_remain = s_last_remain;
        for (int bin = tx; bin < kRadix; bin += NUM_THREADS) {
            const int total_at_or_above = s_hist_buf[0][bin];
            const int strictly_above    = (bin + 1 < kRadix) ? s_hist_buf[0][bin + 1] : 0;
            if (total_at_or_above >= last_remain && strictly_above < last_remain) {
                s_sub_threshold_bin  = bin;
                s_sub_last_remain    = last_remain - strictly_above;
                s_strictly_above_sub = strictly_above;
            }
        }
        __syncthreads();

        const int sub_threshold_bin     = s_sub_threshold_bin;
        const int sub_last_remain       = s_sub_last_remain;
        const int strictly_above_sub_bn = s_strictly_above_sub;
        const int above_base            = s_above_count;

        const int total_cand = s_cand_count;
        if (total_cand <= CAND_MAX) {
            for (int i = tx; i < total_cand; i += NUM_THREADS) {
                const int local_rank = s_cand_idx[i];
                const float    raw    = __bfloat162float(row_scores[local_rank]);
                const float    mapped = apply_transform_tmpl<MODE>(raw, mapping_power);
                const uint32_t key    = score_to_key32_bf16(mapped);
                const int sub_bin = static_cast<int>((key >> 16) & 0xFF);
                if (sub_bin > sub_threshold_bin) {
                    const int rel  = ::atomicAdd(&s_thresh_above_count, 1);
                    const int slot = above_base + rel;
                    if (slot < LOCAL_K) {
                        s_top_keys[slot] = key;
                        s_top_idx [slot] = row_idxmap[local_rank];
                    }
                } else if (sub_bin == sub_threshold_bin) {
                    const int rel = ::atomicAdd(&s_thresh_at_count, 1);
                    if (rel < sub_last_remain) {
                        const int slot = above_base + strictly_above_sub_bn + rel;
                        if (slot < LOCAL_K) {
                            s_top_keys[slot] = key;
                            s_top_idx [slot] = row_idxmap[local_rank];
                        }
                    }
                }
            }
        } else {
            for (int local_rank = tx; local_rank < row_len; local_rank += NUM_THREADS) {
                const int cached_bin = static_cast<int>(s_bins[local_rank]);
                if (cached_bin != threshold_bin) continue;
                const float    raw    = __bfloat162float(row_scores[local_rank]);
                const float    mapped = apply_transform_tmpl<MODE>(raw, mapping_power);
                const uint32_t key    = score_to_key32_bf16(mapped);
                const int sub_bin = static_cast<int>((key >> 16) & 0xFF);
                if (sub_bin > sub_threshold_bin) {
                    const int rel  = ::atomicAdd(&s_thresh_above_count, 1);
                    const int slot = above_base + rel;
                    if (slot < LOCAL_K) {
                        s_top_keys[slot] = key;
                        s_top_idx [slot] = row_idxmap[local_rank];
                    }
                } else if (sub_bin == sub_threshold_bin) {
                    const int rel = ::atomicAdd(&s_thresh_at_count, 1);
                    if (rel < sub_last_remain) {
                        const int slot = above_base + strictly_above_sub_bn + rel;
                        if (slot < LOCAL_K) {
                            s_top_keys[slot] = key;
                            s_top_idx [slot] = row_idxmap[local_rank];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (tx < 32) {
        uint32_t kk[1] = { s_top_keys[tx] };
        int32_t  vv[1] = { s_top_idx [tx] };
        LocalSortT(local_sort_smem).Sort(kk, vv, DescendingUint32{});
        s_top_keys[tx] = kk[0];
        s_top_idx [tx] = vv[0];
    }
    __syncthreads();

    if (tx < topk_val) out_idx[tx] = s_top_idx[tx];
}

}  // namespace topk_v2_remap_select32


// =====================================================================
// Slow-path 4-round refinement kernel with remap, templated on MODE.
// Mirrors fast_topk_vortex / TopKOutput_Kernel from topk_v2.cu with the
// transform applied at every score load.
// =====================================================================

constexpr int kThreadsPerBlock = 1024;

#ifdef USE_ROCM
#ifdef SGL_TOPK_DYNAMIC_SMEM_BYTES
constexpr size_t kSmem = static_cast<size_t>(SGL_TOPK_DYNAMIC_SMEM_BYTES);
#else
constexpr size_t kSmem = 48 * 1024;
#endif
#else
// 32KB → 64KB; matches the bump in topk_v2.cu. With 32KB the at-threshold
// cache (4K entries / round) overflowed on degenerate distributions and
// dropped recall to ~0.55 at uniform L=131k.
constexpr size_t kSmem = 16 * 1024 * sizeof(uint32_t);
#endif

// Upper bound on dynamic SMEM the slow-path kernel may opt into. Capped
// at the per-block opt-in ceiling on the target GPU. At launch we use
// kSmem + bins_bytes when it fits, else dispatch the USE_CACHE=false
// specialization (kSmem only, no s_bins).
//
// Per-arch opt-in ceiling reference (cudaDevAttrMaxSharedMemoryPerBlockOptin):
//   - RTX PRO 6000 / Blackwell SM_120 : 99 KB  → use 96 KB
//   - H100 / H200      / Hopper  SM_90 : 228 KB → use 224 KB
constexpr size_t kSmemMax = 80  * 1024;   // RTX PRO 6000 (Blackwell SM_120, ~99 KB opt-in − ~11 KB static)
// constexpr size_t kSmemMax = 64  * 1024;   // H100 / H200 (Hopper SM_90) — uncomment this line on H200; 80 KB hit `cudaFuncSetAttribute: invalid argument` there. 64 KB matches the actual launch usage so it always validates.

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
    __half h = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32_bits(float x) -> uint32_t {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
    [[maybe_unused]]
    static const auto result = [] {
#ifdef USE_ROCM
        return ::cudaFuncSetAttribute(
            reinterpret_cast<const void*>(f), ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#else
        return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#endif
    }();
    TORCH_CHECK(result == cudaSuccess, "setup_kernel_smem_once failed: ", ::cudaGetErrorString(result));
}

template <typename T>
__device__ __forceinline__ float vortex_to_float(T x);

template <>
__device__ __forceinline__ float vortex_to_float<float>(float x) { return x; }

template <>
__device__ __forceinline__ float vortex_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

constexpr int VORTEX_MAX_TOPK = 2048;

// USE_CACHE: when true, the launcher allocated extra dynamic SMEM after
// vh_input_idx for s_bins (one byte per element) and Pass 1 caches the
// Stage-1 bin so Pass 2 skips global re-read + transform re-apply for
// non-threshold elements. When false (length too large to fit s_bins in
// the device's opt-in SMEM), Pass 2 re-reads as in the original kernel.
template <typename ScoreT, int MODE, bool USE_CACHE>
__device__ void fast_topk_vortex_remap(
    const ScoreT* __restrict__ input,
    int*          __restrict__ index,
    int           row_start,
    int           length,
    int           target_k,
    const float   mapping_power)
{
    int topk = target_k;
    constexpr auto BLOCK_SIZE = 1024;
    constexpr auto RADIX = 256;
    constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

    alignas(128) __shared__ int vh_histogram_buf[2][RADIX + 128];
    alignas(128) __shared__ int vh_counter;
    alignas(128) __shared__ int vh_threshold_bin_id;
    alignas(128) __shared__ int vh_num_input[2];

    auto& vh_histogram = vh_histogram_buf[0];

    extern __shared__ int vh_input_idx[][SMEM_INPUT_SIZE];
    uint8_t* const s_bins = reinterpret_cast<uint8_t*>(&vh_input_idx[2][0]);

    const int tx = threadIdx.x;

    if (tx < RADIX + 1) vh_histogram[tx] = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const float raw    = vortex_to_float(input[idx + row_start]);
        const float mapped = apply_transform_tmpl<MODE>(raw, mapping_power);
        const auto  bin    = convert_to_uint8(mapped);
        if constexpr (USE_CACHE) s_bins[idx] = static_cast<uint8_t>(bin);
        ::atomicAdd(&vh_histogram[bin], 1);
    }
    __syncthreads();

    const auto run_cumsum = [&] {
#pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            static_assert(1 << 8 == RADIX);
            if (C10_LIKELY(tx < RADIX)) {
                const auto j = 1 << i;
                const auto k = i & 1;
                auto value = vh_histogram_buf[k][tx];
                if (tx < RADIX - j) {
                    value += vh_histogram_buf[k][tx + j];
                }
                vh_histogram_buf[k ^ 1][tx] = value;
            }
            __syncthreads();
        }
    };

    run_cumsum();
    if (tx < RADIX && vh_histogram[tx] > topk && vh_histogram[tx + 1] <= topk) {
        vh_threshold_bin_id = tx;
        vh_num_input[0] = 0;
        vh_counter = 0;
    }
    __syncthreads();

    const auto threshold_bin = vh_threshold_bin_id;
    topk -= vh_histogram[threshold_bin + 1];

    // Helper: byte-0 bin for an element. With USE_CACHE we read the byte
    // already stored in SMEM during Pass 1; otherwise we re-read raw + re-
    // apply the transform (the original behavior).
    auto read_bin = [&] (int idx) -> int {
        if constexpr (USE_CACHE) {
            return static_cast<int>(s_bins[idx]);
        } else {
            const float raw    = vortex_to_float(input[idx + row_start]);
            const float mapped = apply_transform_tmpl<MODE>(raw, mapping_power);
            return static_cast<int>(convert_to_uint8(mapped));
        }
    };

    if (topk == 0) {
        // Above-threshold shortcut.
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            const int bin = read_bin(idx);
            if (bin > threshold_bin) {
                const auto pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = idx;
            }
        }
        __syncthreads();
        return;
    } else {
        __syncthreads();
        if (tx < RADIX + 1) vh_histogram[tx] = 0;
        __syncthreads();

        // Pass 2: gate on bin (cached when available). Above-threshold
        // emits immediately; at-threshold re-reads raw + re-applies the
        // transform — only there — to derive the byte-1 sub_bin for
        // Stage-2 round 0.
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            const int bin = read_bin(idx);
            if (bin > threshold_bin) {
                const auto pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = idx;
            } else if (bin == threshold_bin) {
                const float raw    = vortex_to_float(input[idx + row_start]);
                const float mapped = apply_transform_tmpl<MODE>(raw, mapping_power);
                const auto pos = ::atomicAdd(&vh_num_input[0], 1);
                if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                    vh_input_idx[0][pos] = idx;
                    const auto b32 = convert_to_uint32_bits(mapped);
                    const auto sub_bin = (b32 >> 24) & 0xFF;
                    ::atomicAdd(&vh_histogram[sub_bin], 1);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll 4
    for (int round = 0; round < 4; ++round) {
        __shared__ int vh_last_remain;
        const auto r_idx = round % 2;

        const auto _raw_num_input = vh_num_input[r_idx];
        const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE))
                                   ? _raw_num_input
                                   : int(SMEM_INPUT_SIZE);

        run_cumsum();
        if (tx < RADIX && vh_histogram[tx] > topk && vh_histogram[tx + 1] <= topk) {
            vh_threshold_bin_id = tx;
            vh_num_input[r_idx ^ 1] = 0;
            vh_last_remain = topk - vh_histogram[tx + 1];
        }
        __syncthreads();

        const auto threshold_bin = vh_threshold_bin_id;
        topk -= vh_histogram[threshold_bin + 1];

        if (topk == 0) {
            for (int i = tx; i < num_input; i += BLOCK_SIZE) {
                const auto idx = vh_input_idx[r_idx][i];
                const auto offset = 24 - round * 8;
                const float raw    = vortex_to_float(input[idx + row_start]);
                const float mapped = apply_transform_tmpl<MODE>(raw, mapping_power);
                const auto  bin    = (convert_to_uint32_bits(mapped) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const auto pos = ::atomicAdd(&vh_counter, 1);
                    index[pos] = idx;
                }
            }
            __syncthreads();
            break;
        } else {
            __syncthreads();
            if (tx < RADIX + 1) vh_histogram[tx] = 0;
            __syncthreads();
            for (int i = tx; i < num_input; i += BLOCK_SIZE) {
                const auto idx = vh_input_idx[r_idx][i];
                const auto raw_input = apply_transform_tmpl<MODE>(
                    vortex_to_float(input[idx + row_start]), mapping_power);
                const auto offset = 24 - round * 8;
                const auto bin = (convert_to_uint32_bits(raw_input) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const auto pos = ::atomicAdd(&vh_counter, 1);
                    index[pos] = idx;
                } else if (bin == threshold_bin) {
                    if (round == 3) {
                        const auto pos = ::atomicAdd(&vh_last_remain, -1);
                        if (pos > 0) {
                            index[target_k - pos] = idx;
                        }
                    } else {
                        const auto pos = ::atomicAdd(&vh_num_input[r_idx ^ 1], 1);
                        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                            vh_input_idx[r_idx ^ 1][pos] = idx;
                            const auto b32 = convert_to_uint32_bits(raw_input);
                            const auto sub_bin = (b32 >> (offset - 8)) & 0xFF;
                            ::atomicAdd(&vh_histogram[sub_bin], 1);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

template <typename ScoreT, int MODE, bool USE_CACHE>
__global__ __launch_bounds__(kThreadsPerBlock)
void TopKOutputRemap_Kernel(
    const ScoreT* __restrict__ score,
    const int*    __restrict__ dense_kv_indptr,
    const int*    __restrict__ sparse_kv_indptr,
    const int*    __restrict__ dense_kv_indices,
    int*          __restrict__ sparse_kv_indices,
    const int     page_reserved_bos,
    const int     page_reserved_eos,
    const float   mapping_power)
{
    const int bx = blockIdx.x;

    const int start = dense_kv_indptr[bx] + page_reserved_bos;
    const int end   = dense_kv_indptr[bx + 1] - page_reserved_eos;
    const int topk_val = sparse_kv_indptr[bx + 1] - sparse_kv_indptr[bx]
                         - page_reserved_bos - page_reserved_eos;
    const int nblk  = end - start;
    if (nblk <= topk_val) return;

    const ScoreT* __restrict__ score_blk = score + start;
    const int*    __restrict__ idx_blk   = dense_kv_indices + start;
    int*          __restrict__ out_blk   = sparse_kv_indices
                                         + sparse_kv_indptr[bx]
                                         + page_reserved_bos;

    __shared__ int s_indices[VORTEX_MAX_TOPK];
    fast_topk_vortex_remap<ScoreT, MODE, USE_CACHE>(
        score_blk, s_indices, 0, nblk, topk_val, mapping_power);
    __syncthreads();

    const int tx = threadIdx.x;
    for (int i = tx; i < topk_val; i += kThreadsPerBlock) {
        out_blk[i] = idx_blk[s_indices[i]];
    }
}

}  // namespace


void topk_output_v2_remap(
    const at::Tensor& x,
    const at::Tensor& dense_kv_indptr,
    const at::Tensor& sparse_kv_indptr,
    const at::Tensor& dense_kv_indices,
    at::Tensor&       sparse_kv_indices,
    const int64_t     eff_batch_size,
    const int64_t     reserved_bos,
    const int64_t     reserved_eos,
    const int64_t     max_num_pages,
    const int64_t     mapping_mode,
    const double      mapping_power)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const float  power_exp = static_cast<float>(mapping_power);

    if (x.scalar_type() == at::ScalarType::BFloat16
        && eff_batch_size >= 4
        && max_num_pages > 0 && max_num_pages <= 16384) {
        const int64_t inferred_k = sparse_kv_indices.numel() / eff_batch_size;
        if (inferred_k >= 16 && inferred_k <= 32) {
            const size_t bins_bytes =
                (static_cast<size_t>(max_num_pages) + size_t(15)) & ~size_t(15);

            #define VORTEX_DISPATCH_FAST(MODE_VAL)                                          \
                do {                                                                         \
                    topk_v2_remap_select32::Kernel<1024, MODE_VAL>                           \
                        <<<dim3(eff_batch_size), dim3(1024), bins_bytes, stream>>>(          \
                            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),    \
                            dense_kv_indptr.data_ptr<int>(),                                 \
                            sparse_kv_indptr.data_ptr<int>(),                                \
                            dense_kv_indices.data_ptr<int>(),                                \
                            sparse_kv_indices.data_ptr<int>(),                               \
                            static_cast<int>(inferred_k),                                    \
                            static_cast<int>(reserved_bos),                                  \
                            static_cast<int>(reserved_eos),                                  \
                            power_exp);                                                      \
                } while (0)

            switch (static_cast<int>(mapping_mode)) {
                case MAPPING_NONE:        VORTEX_DISPATCH_FAST(MAPPING_NONE); break;
                case MAPPING_POWER:       VORTEX_DISPATCH_FAST(MAPPING_POWER); break;
                case MAPPING_LOG:         VORTEX_DISPATCH_FAST(MAPPING_LOG); break;
                case MAPPING_ASINH:       VORTEX_DISPATCH_FAST(MAPPING_ASINH); break;
                case MAPPING_LOG1P:       VORTEX_DISPATCH_FAST(MAPPING_LOG1P); break;
                case MAPPING_TRUNC8:      VORTEX_DISPATCH_FAST(MAPPING_TRUNC8); break;
                case MAPPING_ERF:         VORTEX_DISPATCH_FAST(MAPPING_ERF); break;
                case MAPPING_TANH:        VORTEX_DISPATCH_FAST(MAPPING_TANH); break;
                case MAPPING_SUBTRACT:    VORTEX_DISPATCH_FAST(MAPPING_SUBTRACT); break;
                case MAPPING_EXP_STRETCH: VORTEX_DISPATCH_FAST(MAPPING_EXP_STRETCH); break;
                case MAPPING_SHIFT_POW2:  VORTEX_DISPATCH_FAST(MAPPING_SHIFT_POW2); break;
                case MAPPING_SHIFT_POW3:  VORTEX_DISPATCH_FAST(MAPPING_SHIFT_POW3); break;
                case MAPPING_LINEAR_STEEP:VORTEX_DISPATCH_FAST(MAPPING_LINEAR_STEEP); break;
                case MAPPING_HALF_SQUARE: VORTEX_DISPATCH_FAST(MAPPING_HALF_SQUARE); break;
                case MAPPING_HALF_CUBE:   VORTEX_DISPATCH_FAST(MAPPING_HALF_CUBE); break;
                default:
                    TORCH_CHECK(false, "topk_output_v2_remap: unsupported mapping_mode ", mapping_mode);
            }
            #undef VORTEX_DISPATCH_FAST

            const auto err_fast = cudaGetLastError();
            TORCH_CHECK(err_fast == cudaSuccess,
                        "topk_output_v2_remap (select32 fast path) kernel failed: ",
                        ::cudaGetErrorString(err_fast));
            return;
        }
    }

    dim3 nblks(eff_batch_size);
    dim3 nthreads(kThreadsPerBlock);

    // s_bins (uint8 per element) was an attempt to cache the Stage-1 bin
    // so Pass 2 could skip the global re-read + transform re-apply. On
    // bf16 inputs with already-coalesced global reads, the uint8 SMEM
    // path adds 4-way bank conflicts that erase the savings on cheap
    // mappings. We keep the templated kernel for future archs with
    // wider SMEM banks but disable the cache path here. Flip
    // `use_cache` back to the size check to re-enable it.
    const size_t bins_bytes = (static_cast<size_t>(max_num_pages) + 15) & ~size_t(15);
    (void)bins_bytes;
    const bool   use_cache  = false;
    const size_t launch_smem = kSmem;

    #define VORTEX_DISPATCH_SLOW(DTYPE, PTR_EXPR, MODE_VAL)                              \
        do {                                                                              \
            if (use_cache) {                                                              \
                setup_kernel_smem_once<                                                   \
                    TopKOutputRemap_Kernel<DTYPE, MODE_VAL, true>, kSmemMax>();           \
                TopKOutputRemap_Kernel<DTYPE, MODE_VAL, true>                             \
                    <<<nblks, nthreads, launch_smem, stream>>>(                           \
                    PTR_EXPR,                                                             \
                    dense_kv_indptr.data_ptr<int>(),                                      \
                    sparse_kv_indptr.data_ptr<int>(),                                     \
                    dense_kv_indices.data_ptr<int>(),                                     \
                    sparse_kv_indices.data_ptr<int>(),                                    \
                    static_cast<int>(reserved_bos),                                       \
                    static_cast<int>(reserved_eos),                                       \
                    power_exp);                                                           \
            } else {                                                                      \
                setup_kernel_smem_once<                                                   \
                    TopKOutputRemap_Kernel<DTYPE, MODE_VAL, false>, kSmem>();             \
                TopKOutputRemap_Kernel<DTYPE, MODE_VAL, false>                            \
                    <<<nblks, nthreads, kSmem, stream>>>(                                 \
                    PTR_EXPR,                                                             \
                    dense_kv_indptr.data_ptr<int>(),                                      \
                    sparse_kv_indptr.data_ptr<int>(),                                     \
                    dense_kv_indices.data_ptr<int>(),                                     \
                    sparse_kv_indices.data_ptr<int>(),                                    \
                    static_cast<int>(reserved_bos),                                       \
                    static_cast<int>(reserved_eos),                                       \
                    power_exp);                                                           \
            }                                                                             \
        } while (0)

    #define VORTEX_DISPATCH_MODE(DTYPE, PTR_EXPR)                                         \
        do {                                                                              \
            switch (static_cast<int>(mapping_mode)) {                                     \
                case MAPPING_NONE:        VORTEX_DISPATCH_SLOW(DTYPE, PTR_EXPR, MAPPING_NONE); break; \
                case MAPPING_POWER:       VORTEX_DISPATCH_SLOW(DTYPE, PTR_EXPR, MAPPING_POWER); break; \
                case MAPPING_LOG:         VORTEX_DISPATCH_SLOW(DTYPE, PTR_EXPR, MAPPING_LOG); break; \
                case MAPPING_ASINH:       VORTEX_DISPATCH_SLOW(DTYPE, PTR_EXPR, MAPPING_ASINH); break; \
                case MAPPING_LOG1P:       VORTEX_DISPATCH_SLOW(DTYPE, PTR_EXPR, MAPPING_LOG1P); break; \
                case MAPPING_TRUNC8:      VORTEX_DISPATCH_SLOW(DTYPE, PTR_EXPR, MAPPING_TRUNC8); break; \
                case MAPPING_ERF:         VORTEX_DISPATCH_SLOW(DTYPE, PTR_EXPR, MAPPING_ERF); break; \
                case MAPPING_TANH:        VORTEX_DISPATCH_SLOW(DTYPE, PTR_EXPR, MAPPING_TANH); break; \
                case MAPPING_SUBTRACT:    VORTEX_DISPATCH_SLOW(DTYPE, PTR_EXPR, MAPPING_SUBTRACT); break; \
                case MAPPING_EXP_STRETCH: VORTEX_DISPATCH_SLOW(DTYPE, PTR_EXPR, MAPPING_EXP_STRETCH); break; \
                case MAPPING_SHIFT_POW2:  VORTEX_DISPATCH_SLOW(DTYPE, PTR_EXPR, MAPPING_SHIFT_POW2); break; \
                case MAPPING_SHIFT_POW3:  VORTEX_DISPATCH_SLOW(DTYPE, PTR_EXPR, MAPPING_SHIFT_POW3); break; \
                case MAPPING_LINEAR_STEEP:VORTEX_DISPATCH_SLOW(DTYPE, PTR_EXPR, MAPPING_LINEAR_STEEP); break; \
                case MAPPING_HALF_SQUARE: VORTEX_DISPATCH_SLOW(DTYPE, PTR_EXPR, MAPPING_HALF_SQUARE); break; \
                case MAPPING_HALF_CUBE:   VORTEX_DISPATCH_SLOW(DTYPE, PTR_EXPR, MAPPING_HALF_CUBE); break; \
                default:                                                                  \
                    TORCH_CHECK(false, "topk_output_v2_remap: unsupported mapping_mode ", mapping_mode); \
            }                                                                             \
        } while (0)

    if (x.scalar_type() == at::ScalarType::BFloat16) {
        VORTEX_DISPATCH_MODE(__nv_bfloat16,
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()));
    } else if (x.scalar_type() == at::ScalarType::Float) {
        VORTEX_DISPATCH_MODE(float, x.data_ptr<float>());
    } else {
        TORCH_CHECK(false, "topk_output_v2_remap: unsupported dtype ", x.scalar_type());
    }

    #undef VORTEX_DISPATCH_MODE
    #undef VORTEX_DISPATCH_SLOW

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "topk_output_v2_remap kernel failed: ", ::cudaGetErrorString(result));
}
