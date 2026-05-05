/**
 * approx_topk_remap.cu — same algorithm as approx_topk.cu (single-pass
 * 8-bit radix approximate top-K with a two-pass refinement) but applies
 * an element-wise transform `apply_transform_tmpl<MODE>(x, p)` to every
 * score before bucketing. The transform is template-specialized per
 * MODE so it inlines at every load site (zero per-element branching).
 *
 * The mapping reshapes a skewed score distribution so the Stage-1 8-bit
 * histogram spreads across more bins, shrinking the threshold bin and
 * letting Stage-2 refinement do less work. MAPPING_NONE compiles to the
 * identity (no overhead vs the unmapped kernel).
 *
 * Public entry point: approx_topk_output_remap (see register.h). The
 * launcher dispatches per (mapping_mode, dtype) into the templated
 * kernels here — same select32 fast-path routing as approx_topk_output
 * so the K∈[16,32] speedup applies in remap mode too.
 */

#include "register.h"
#include "topk_mapping.cuh"
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cub/warp/warp_merge_sort.cuh>

namespace {

// =====================================================================
// SELECT32-SORT32 small-K fast path with remap, templated on MODE.
// Mirrors approx_topk_select32::Kernel from approx_topk.cu, with the
// raw score replaced by `apply_transform_tmpl<MODE>(raw, mapping_power)`
// at every bucketing site so the cached `s_bins` byte already reflects
// the remapped distribution.
// =====================================================================

namespace approx_topk_remap_select32 {

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

    // Pass 1 — byte-0 hist on the *remapped* score; cache the byte.
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
            } else {  // cached_bin == threshold_bin
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

}  // namespace approx_topk_remap_select32


// =====================================================================
// Templated radix approximate kernel for the slow path (any K, any L).
// Mirrors ApproxTopK_Kernel/approx_topk_inner from approx_topk.cu with
// the transform applied at every score load site.
// =====================================================================

constexpr int kThreadsPerBlock = 1024;
constexpr int RADIX = 256;
constexpr int VORTEX_MAX_TOPK = 2048;

// Maximum dynamic SMEM the approx slow-path kernel may opt into. Capped
// at the per-block opt-in ceiling on the target GPU. At launch we use
// bins_bytes when it fits, else dispatch the USE_CACHE=false
// specialization which uses no dynamic SMEM.
//
// Per-arch opt-in ceiling reference (cudaDevAttrMaxSharedMemoryPerBlockOptin):
//   - RTX PRO 6000 / Blackwell SM_120 : 99 KB  → use 96 KB
//   - H100 / H200      / Hopper  SM_90 : 228 KB → use 224 KB
// Switch the active line to match the deployment arch.
constexpr size_t kApproxRemapSmemMax = 80  * 1024;   // RTX PRO 6000 (Blackwell SM_120, ~99 KB opt-in − ~11 KB static)
// constexpr size_t kApproxRemapSmemMax = 224 * 1024;   // H100 / H200 (Hopper SM_90, 228 KB opt-in)

template <auto* f, size_t max_dynamic_smem>
void approx_setup_kernel_smem_once() {
    [[maybe_unused]]
    static const auto result = [] {
#ifdef USE_ROCM
        return ::cudaFuncSetAttribute(
            reinterpret_cast<const void*>(f),
            ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#else
        return ::cudaFuncSetAttribute(
            f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#endif
    }();
    TORCH_CHECK(result == cudaSuccess,
                "approx_setup_kernel_smem_once failed: ",
                ::cudaGetErrorString(result));
}

template <typename T>
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float<float>(float x) { return x; }

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ uint32_t score_to_key32(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}


// Approx-radix top-k with at-threshold index caching.
//
// Original kernel ran 3 full passes over `length` (Pass 1 histogram,
// Pass 2 sub-bin histogram, Stage-2 refinement). At long inputs the third
// pass dominated, making approx slower than the exact radix kernel which
// has only 2 length-passes plus bounded refinement.
//
// This version caches the at-threshold-bin indices in dynamic SMEM during
// Pass 2; the Stage-2 refinement then iterates only the cached subset
// (≤ SMEM_INPUT_SIZE = 4096 entries), matching the radix kernel's
// 2-passes + bounded-refinement scaling. If the at-threshold subset
// overflows SMEM, the kernel falls back to the original full-length
// re-iteration for correctness.
constexpr int kApproxRemapSmemInputSize = 16384;

template <typename ScoreT, int MODE>
__device__ void approx_topk_remap_inner(
    const ScoreT* __restrict__ input,
    int*          __restrict__ index,
    const int     length,
    const int     target_k,
    const int     tolerate_thresh,
    const float   mapping_power)
{
    constexpr int BLOCK_SIZE = kThreadsPerBlock;

    alignas(128) __shared__ int hist_buf[2][RADIX + 128];
    alignas(128) __shared__ int s_threshold_bin;
    alignas(128) __shared__ int s_counter;
    alignas(128) __shared__ int s_last_remain;
    alignas(128) __shared__ int s_at_threshold_count;

    // Dynamic SMEM caches the at-threshold-bin element indices written
    // during Pass 2, so Stage-2 only iterates that subset.
    extern __shared__ int s_at_threshold_idx[];

    auto& hist = hist_buf[0];
    const int tx = threadIdx.x;

    auto run_cumsum = [&] {
        #pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            if (C10_LIKELY(tx < RADIX)) {
                const int j = 1 << i;
                const int k = i & 1;
                int v = hist_buf[k][tx];
                if (tx < RADIX - j) v += hist_buf[k][tx + j];
                hist_buf[k ^ 1][tx] = v;
            }
            __syncthreads();
        }
    };

    if (tx < RADIX + 1) hist[tx] = 0;
    if (tx == 0) s_at_threshold_count = 0;
    __syncthreads();

    // Stage-1 Pass 1: byte-0 histogram.
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const float raw    = to_float<ScoreT>(input[idx]);
        const float mapped = apply_transform_tmpl<MODE>(raw, mapping_power);
        const auto  bin    = (score_to_key32(mapped) >> 24) & 0xFFu;
        ::atomicAdd(&hist[bin], 1);
    }
    __syncthreads();

    run_cumsum();

    if (tx < RADIX && hist[tx] > target_k && hist[tx + 1] <= target_k) {
        s_threshold_bin = tx;
        s_counter       = 0;
        s_last_remain   = target_k - hist[tx + 1];
    }
    __syncthreads();

    const int tbin0        = s_threshold_bin;
    const int last_remain0 = s_last_remain;

    // Degenerate-bin guard: same as in approx_topk.cu. If the threshold
    // bin alone is bigger than the at-threshold SMEM cache, take the
    // stochastic fast path instead of paying for the slow path's full
    // re-iteration + heavy sub-bin atomicAdd contention. This matches the
    // behavior of the unmapped baseline so the remap variant never
    // regresses on degenerate distributions.
    const int hist_at_tbin0       = hist[tbin0];
    const int hist_strictly_above = (tbin0 + 1 < RADIX) ? hist[tbin0 + 1] : 0;
    const int count_at_threshold  = hist_at_tbin0 - hist_strictly_above;
    const bool degenerate_bin     = count_at_threshold > kApproxRemapSmemInputSize;

    if (last_remain0 <= tolerate_thresh || degenerate_bin) {
        // Early-termination: 1 pass over length, no Stage-2 refinement.
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            const float raw    = to_float<ScoreT>(input[idx]);
            const float mapped = apply_transform_tmpl<MODE>(raw, mapping_power);
            const int   bin    = static_cast<int>((score_to_key32(mapped) >> 24) & 0xFFu);
            if (bin > tbin0) {
                const int pos = ::atomicAdd(&s_counter, 1);
                index[pos] = idx;
            } else if (bin == tbin0) {
                const int pos = ::atomicAdd(&s_last_remain, -1);
                if (pos > 0) {
                    index[target_k - pos] = idx;
                }
            }
        }
        __syncthreads();
        return;
    }

    if (tx < RADIX + 1) hist[tx] = 0;
    __syncthreads();

    // Stage-1 Pass 2: emit above-threshold pages, cache at-threshold
    // indices in SMEM, build byte-1 sub-bin histogram for Stage-2.
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const float raw    = to_float<ScoreT>(input[idx]);
        const float mapped = apply_transform_tmpl<MODE>(raw, mapping_power);
        const uint32_t key32 = score_to_key32(mapped);
        const int   bin0   = static_cast<int>((key32 >> 24) & 0xFFu);
        if (bin0 > tbin0) {
            const int pos = ::atomicAdd(&s_counter, 1);
            index[pos] = idx;
        } else if (bin0 == tbin0) {
            const int bin1 = static_cast<int>((key32 >> 16) & 0xFFu);
            ::atomicAdd(&hist[bin1], 1);
            const int slot = ::atomicAdd(&s_at_threshold_count, 1);
            if (slot < kApproxRemapSmemInputSize) {
                s_at_threshold_idx[slot] = idx;
            }
        }
    }
    __syncthreads();

    run_cumsum();

    if (tx < RADIX && hist[tx] > last_remain0 && hist[tx + 1] <= last_remain0) {
        s_threshold_bin = tx;
        s_last_remain   = last_remain0 - hist[tx + 1];
    }
    __syncthreads();

    const int tbin1     = s_threshold_bin;
    const int at_thresh = s_at_threshold_count;

    if (at_thresh <= kApproxRemapSmemInputSize) {
        // Stage-2 fast: iterate only cached at-threshold indices.
        for (int i = tx; i < at_thresh; i += BLOCK_SIZE) {
            const int idx = s_at_threshold_idx[i];
            const float raw    = to_float<ScoreT>(input[idx]);
            const float mapped = apply_transform_tmpl<MODE>(raw, mapping_power);
            const int   bin1   = static_cast<int>((score_to_key32(mapped) >> 16) & 0xFFu);
            if (bin1 > tbin1) {
                const int pos = ::atomicAdd(&s_counter, 1);
                index[pos] = idx;
            } else if (bin1 == tbin1) {
                const int pos = ::atomicAdd(&s_last_remain, -1);
                if (pos > 0) {
                    index[target_k - pos] = idx;
                }
            }
        }
    } else {
        // Overflow fallback: at-threshold count exceeded SMEM cache;
        // re-iterate the full length (original Pass 3 behavior).
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            const float raw    = to_float<ScoreT>(input[idx]);
            const float mapped = apply_transform_tmpl<MODE>(raw, mapping_power);
            const uint32_t key32 = score_to_key32(mapped);
            const int   bin0   = static_cast<int>((key32 >> 24) & 0xFFu);
            if (bin0 != tbin0) continue;
            const int   bin1   = static_cast<int>((key32 >> 16) & 0xFFu);
            if (bin1 > tbin1) {
                const int pos = ::atomicAdd(&s_counter, 1);
                index[pos] = idx;
            } else if (bin1 == tbin1) {
                const int pos = ::atomicAdd(&s_last_remain, -1);
                if (pos > 0) {
                    index[target_k - pos] = idx;
                }
            }
        }
    }
    __syncthreads();
}


template <typename ScoreT, int MODE>
__global__ __launch_bounds__(kThreadsPerBlock)
void ApproxTopKRemap_Kernel(
    const ScoreT* __restrict__ score,
    const int*    __restrict__ dense_kv_indptr,
    const int*    __restrict__ sparse_kv_indptr,
    const int*    __restrict__ dense_kv_indices,
    int*          __restrict__ sparse_kv_indices,
    const int     page_reserved_bos,
    const int     page_reserved_eos,
    const float   tolerate_ratio,
    const float   mapping_power)
{
    const int bx = blockIdx.x;

    const int start    = dense_kv_indptr[bx] + page_reserved_bos;
    const int end      = dense_kv_indptr[bx + 1] - page_reserved_eos;
    const int target_k = sparse_kv_indptr[bx + 1] - sparse_kv_indptr[bx]
                         - page_reserved_bos - page_reserved_eos;
    const int nblk     = end - start;
    if (nblk <= target_k) return;

    int tolerate_thresh =
        static_cast<int>(tolerate_ratio * static_cast<float>(target_k));
    if (tolerate_thresh < 0)        tolerate_thresh = 0;
    if (tolerate_thresh > target_k) tolerate_thresh = target_k;

    const ScoreT* __restrict__ score_blk = score + start;
    const int*    __restrict__ idx_blk   = dense_kv_indices + start;
    int*          __restrict__ out_blk   = sparse_kv_indices
                                         + sparse_kv_indptr[bx]
                                         + page_reserved_bos;

    __shared__ int s_indices[VORTEX_MAX_TOPK];

    approx_topk_remap_inner<ScoreT, MODE>(
        score_blk, s_indices, nblk, target_k, tolerate_thresh, mapping_power);
    __syncthreads();

    const int tx = threadIdx.x;
    for (int i = tx; i < target_k; i += kThreadsPerBlock) {
        out_blk[i] = idx_blk[s_indices[i]];
    }
}

}  // namespace


void approx_topk_output_remap(
    const at::Tensor& x,
    const at::Tensor& dense_kv_indptr,
    const at::Tensor& sparse_kv_indptr,
    const at::Tensor& dense_kv_indices,
    at::Tensor&       sparse_kv_indices,
    const int64_t     eff_batch_size,
    const int64_t     reserved_bos,
    const int64_t     reserved_eos,
    const int64_t     max_num_pages,
    const double      tolerate_ratio,
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
                    approx_topk_remap_select32::Kernel<1024, MODE_VAL>                       \
                        <<<dim3(eff_batch_size), dim3(1024), bins_bytes, stream>>>(          \
                            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),            \
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
                    TORCH_CHECK(false, "approx_topk_output_remap: unsupported mapping_mode ", mapping_mode);
            }
            #undef VORTEX_DISPATCH_FAST

            const auto err_fast = cudaGetLastError();
            TORCH_CHECK(err_fast == cudaSuccess,
                        "approx_topk_output_remap (select32 fast path) kernel failed: ",
                        cudaGetErrorString(err_fast));
            return;
        }
    }

    dim3 nblks(eff_batch_size);
    dim3 nthreads(kThreadsPerBlock);
    const float tol = static_cast<float>(tolerate_ratio);

    // Dynamic SMEM caches at-threshold-bin indices (kApproxRemapSmemInputSize
    // ints = 16 KB). Stage-2 then iterates only that subset instead of the
    // full row.
    constexpr size_t kApproxRemapSmemBytes =
        static_cast<size_t>(kApproxRemapSmemInputSize) * sizeof(int);
    static_assert(kApproxRemapSmemBytes <= kApproxRemapSmemMax,
                  "approx-remap SMEM cache exceeds opt-in ceiling");

    #define VORTEX_DISPATCH_SLOW(DTYPE, PTR_EXPR, MODE_VAL)                              \
        do {                                                                              \
            approx_setup_kernel_smem_once<                                                \
                ApproxTopKRemap_Kernel<DTYPE, MODE_VAL>, kApproxRemapSmemMax>();          \
            ApproxTopKRemap_Kernel<DTYPE, MODE_VAL>                                       \
                <<<nblks, nthreads, kApproxRemapSmemBytes, stream>>>(                     \
                PTR_EXPR,                                                                 \
                dense_kv_indptr.data_ptr<int>(),                                          \
                sparse_kv_indptr.data_ptr<int>(),                                         \
                dense_kv_indices.data_ptr<int>(),                                         \
                sparse_kv_indices.data_ptr<int>(),                                        \
                static_cast<int>(reserved_bos),                                           \
                static_cast<int>(reserved_eos),                                           \
                tol, power_exp);                                                          \
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
                    TORCH_CHECK(false, "approx_topk_output_remap: unsupported mapping_mode ", mapping_mode); \
            }                                                                             \
        } while (0)

    if (x.scalar_type() == at::ScalarType::BFloat16) {
        VORTEX_DISPATCH_MODE(__nv_bfloat16,
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()));
    } else if (x.scalar_type() == at::ScalarType::Float) {
        VORTEX_DISPATCH_MODE(float, x.data_ptr<float>());
    } else {
        TORCH_CHECK(false, "approx_topk_output_remap: unsupported dtype ", x.scalar_type());
    }

    #undef VORTEX_DISPATCH_MODE
    #undef VORTEX_DISPATCH_SLOW

    const auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "approx_topk_output_remap kernel failed: ", cudaGetErrorString(err));
}
