/**
 * approx_topk.cu — single-pass 8-bit radix approximate top-K, with an
 * optional second pass when the threshold bin is too wide.
 *
 *   Pass 1 — histogram on byte 0 of fp32 keys, find threshold bin tbin0,
 *            compute last_remain0 = target_k - hist0[tbin0+1].
 *
 *   If last_remain0 <= tolerate_ratio * target_k:
 *      Single-pass emit: items with bin0 > tbin0 are strict winners,
 *      items with bin0 == tbin0 are filled in atomic-arrival order.
 *
 *   Otherwise (threshold bin too wide):
 *      Pass 2 — emit byte-0 strict winners and build a sub-histogram on
 *               byte 1 of the byte-0 == tbin0 items.
 *      Find tbin1 within that sub-histogram, then a final emit pass:
 *      items with bin0 == tbin0 && bin1 > tbin1 are strict winners,
 *      items with bin0 == tbin0 && bin1 == tbin1 fill the remaining
 *      slots in atomic-arrival order.  No further refinement.
 *
 *   tolerate_ratio = 1.0 -> always single-pass (cheapest, loosest).
 *   tolerate_ratio = 0.0 -> always two-pass (tighter approximation).
 */

#include "register.h"
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cub/warp/warp_merge_sort.cuh>

namespace {

// =====================================================================
// Inlined SELECT32-SORT32 small-K fast path (bf16, K<=32, pages<=16384).
// Port of TopK30_RandomSplit_Select32_Kernel from
// csrc/topk_sglang_merge.cu (vendored from main vortex_torch repo) with
// SPLITS=1, MAPPING_NONE, PART_CONTIGUOUS — no split, no merge, no
// workspace, no done_counter, no remap.
//
// Long-L optimizations on top of the upstream port:
//  1. `s_bins` dynamic-SMEM cache. Pass 1 records (key>>24) per element
//     so Pass 2 can short-circuit the dominant `bin < threshold_bin`
//     branch with a SMEM byte read instead of a global score reload.
//     16 KB per CTA at the L=16384 cap.
//  2. Pass 2 also pushes threshold-bin elements (local_rank) into a
//     small `s_cand_idx` list (CAND_MAX=512 → 2 KB). Pass 3 then
//     iterates only that list (~L/256 entries for random scores)
//     instead of scanning all L. If the cap overflows on a
//     concentrated distribution, Pass 3 falls back to scanning all L
//     with the s_bins gate (correctness preserved).
//  3. Single-warp __shfl-based reverse-cumsum replacing the original
//     8-step strided Hillis-Steele scan. Cuts 8 __syncthreads per
//     cumsum × 2 cumsums = 16 barriers (~2 µs at NT=1024). This is
//     the optimization that extends the sweet spot from L=4K to
//     L=16K — at long L the original cumsum's barrier cost was the
//     dominant fixed overhead per kernel invocation.
//
// NUM_THREADS=1024 mirrors kSelCfg1 from the upstream — bench sweep
// confirmed smaller NT collapses the speedup. Routing decision lives
// in approx_topk_output below.
// =====================================================================

namespace approx_topk_select32 {

__device__ __forceinline__ uint32_t score_to_key32_bf16(float x) {
    const uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

struct DescendingUint32 {
    __device__ __forceinline__ bool operator()(const uint32_t& a, const uint32_t& b) const {
        return a > b;
    }
};

template <int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS)
void Kernel(
    const __nv_bfloat16* __restrict__ score,
    const int*           __restrict__ dense_kv_indptr,
    const int*           __restrict__ sparse_kv_indptr,
    const int*           __restrict__ dense_kv_indices,
    int*                 __restrict__ sparse_kv_indices,
    const int            topk_val,
    const int            page_reserved_bos,
    const int            page_reserved_eos)
{
    constexpr int LOCAL_K = 32;
    constexpr int kRadix  = 256;
    // Max threshold-bin candidates we record in Pass 2 to feed Pass 3.
    // For random L=16K, expected count is ~L/256 = 64; 512 gives ~8x
    // headroom. If exceeded (concentrated distribution), Pass 3 falls
    // back to scanning all L with the s_bins gate.
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

    // Dynamic SMEM cache: s_bins[i] = (key_i >> 24) for i in [0, row_len).
    // Sized to max_num_pages from the host launch.
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

    // Reverse-inclusive cumsum over s_hist_buf[0][0..kRadix). Result lands
    // back in s_hist_buf[0]. The previous strided implementation needed 8
    // __syncthreads barriers per call (~1.2 µs at NT=1024); this version
    // does the entire 256-bin scan inside a single warp using __shfl,
    // costing one trailing __syncthreads to broadcast.
    auto run_cumsum_strided = [&]() {
        if (tx < 32) {
            constexpr int kPerLane = kRadix / 32;  // 8
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

    // Pass 1 — byte-0 hist + write s_bins cache.
    for (int local_rank = tx; local_rank < row_len; local_rank += NUM_THREADS) {
        const float    raw = __bfloat162float(row_scores[local_rank]);
        const uint32_t key = score_to_key32_bf16(raw);
        const int      bin = static_cast<int>(key >> 24);
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
            const float    raw  = __bfloat162float(row_scores[local_rank]);
            const uint32_t key  = score_to_key32_bf16(raw);
            const int      slot = ::atomicAdd(&s_above_count, 1);
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

        // Pass 2 — cached-bin short-circuits the dominant `< threshold_bin`
        // branch. For threshold-bin elements, also push the local_rank into
        // s_cand_idx[] so Pass 3 only iterates the candidate list (not all L).
        for (int local_rank = tx; local_rank < row_len; local_rank += NUM_THREADS) {
            const int cached_bin = static_cast<int>(s_bins[local_rank]);
            if (cached_bin < threshold_bin) continue;
            const float    raw = __bfloat162float(row_scores[local_rank]);
            const uint32_t key = score_to_key32_bf16(raw);
            if (cached_bin > threshold_bin) {
                const int slot = ::atomicAdd(&s_above_count, 1);
                if (slot < LOCAL_K) {
                    s_top_keys[slot] = key;
                    s_top_idx [slot] = row_idxmap[local_rank];
                }
            } else {  // cached_bin == threshold_bin
                // Always update sub-hist (correct sub_threshold_bin even on overflow).
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

        // Pass 3 — fast path iterates only the candidate list (~L/256 entries
        // for random scores). If the cap was exceeded, fall back to scanning
        // all L with the s_bins gate so correctness is preserved for any
        // distribution.
        const int total_cand = s_cand_count;
        if (total_cand <= CAND_MAX) {
            for (int i = tx; i < total_cand; i += NUM_THREADS) {
                const int local_rank = s_cand_idx[i];
                const float    raw = __bfloat162float(row_scores[local_rank]);
                const uint32_t key = score_to_key32_bf16(raw);
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
            // Fallback for pathological concentrated distributions.
            for (int local_rank = tx; local_rank < row_len; local_rank += NUM_THREADS) {
                const int cached_bin = static_cast<int>(s_bins[local_rank]);
                if (cached_bin != threshold_bin) continue;
                const float    raw = __bfloat162float(row_scores[local_rank]);
                const uint32_t key = score_to_key32_bf16(raw);
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

    // Stage D: warp sort 32 candidates.
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

}  // namespace approx_topk_select32



constexpr int kThreadsPerBlock = 1024;
constexpr int RADIX = 256;
constexpr int VORTEX_MAX_TOPK = 2048;

template <typename T>
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float<float>(float x) { return x; }

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// fp32 -> total-order uint32 (sign-flip).  Higher key == higher score.
__device__ __forceinline__ uint32_t score_to_key32(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}


// At-threshold-bin index cache size for the slow-path Stage-2 refinement.
// Must equal kApproxRemapSmemInputSize (see approx_topk_remap.cu).
constexpr int kApproxSmemInputSize = 4096;

template <typename ScoreT>
__device__ void approx_topk_inner(
    const ScoreT* __restrict__ input,
    int*          __restrict__ index,
    const int     length,
    const int     target_k,
    const int     tolerate_thresh)
{
    constexpr int BLOCK_SIZE = kThreadsPerBlock;

    alignas(128) __shared__ int hist_buf[2][RADIX + 128];
    alignas(128) __shared__ int s_threshold_bin;
    alignas(128) __shared__ int s_counter;        // strict-winner write head
    alignas(128) __shared__ int s_last_remain;    // atomic-arrival countdown
    alignas(128) __shared__ int s_at_threshold_count;

    extern __shared__ int s_at_threshold_idx[];   // [kApproxSmemInputSize]

    auto& hist = hist_buf[0];
    const int tx = threadIdx.x;

    // Reverse inclusive cumulative sum; final result lands in hist_buf[0].
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

    // ---------------- Pass 1: histogram on byte 0 ----------------
    if (tx < RADIX + 1) hist[tx] = 0;
    if (tx == 0) s_at_threshold_count = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto bin =
            (score_to_key32(to_float<ScoreT>(input[idx])) >> 24) & 0xFFu;
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

    // ---------------- Single-pass emit ----------------
    if (last_remain0 <= tolerate_thresh) {
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            const auto bin =
                (score_to_key32(to_float<ScoreT>(input[idx])) >> 24) & 0xFFu;
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

    // ---------------- Pass 2: emit byte-0 strict + byte-1 sub-histogram ----------------
    // Cache at-threshold-bin indices in dynamic SMEM so Pass 3 only iterates
    // that subset (typically O(length/256)) instead of the full row again.
    if (tx < RADIX + 1) hist[tx] = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto key32 = score_to_key32(to_float<ScoreT>(input[idx]));
        const auto bin0  = (key32 >> 24) & 0xFFu;
        if (bin0 > tbin0) {
            const int pos = ::atomicAdd(&s_counter, 1);
            index[pos] = idx;
        } else if (bin0 == tbin0) {
            const auto bin1 = (key32 >> 16) & 0xFFu;
            ::atomicAdd(&hist[bin1], 1);
            const int slot = ::atomicAdd(&s_at_threshold_count, 1);
            if (slot < kApproxSmemInputSize) {
                s_at_threshold_idx[slot] = idx;
            }
        }
    }
    __syncthreads();

    run_cumsum();

    // Find byte-1 threshold bin against the new top-k = last_remain0.
    if (tx < RADIX && hist[tx] > last_remain0 && hist[tx + 1] <= last_remain0) {
        s_threshold_bin = tx;
        s_last_remain   = last_remain0 - hist[tx + 1];
    }
    __syncthreads();

    const int tbin1     = s_threshold_bin;
    const int at_thresh = s_at_threshold_count;

    if (at_thresh <= kApproxSmemInputSize) {
        // ---------------- Pass 3: only iterate cached at-threshold subset ----------------
        for (int i = tx; i < at_thresh; i += BLOCK_SIZE) {
            const int idx = s_at_threshold_idx[i];
            const auto key32 = score_to_key32(to_float<ScoreT>(input[idx]));
            const auto bin1 = (key32 >> 16) & 0xFFu;
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
        // Overflow: at-threshold-bin count exceeded SMEM cache; re-iterate
        // the full length (original Pass 3 behavior).
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            const auto key32 = score_to_key32(to_float<ScoreT>(input[idx]));
            const auto bin0  = (key32 >> 24) & 0xFFu;
            if (bin0 != tbin0) continue;
            const auto bin1 = (key32 >> 16) & 0xFFu;
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


template <typename ScoreT>
__global__ __launch_bounds__(kThreadsPerBlock)
void ApproxTopK_Kernel(
    const ScoreT* __restrict__ score,
    const int*    __restrict__ dense_kv_indptr,
    const int*    __restrict__ sparse_kv_indptr,
    const int*    __restrict__ dense_kv_indices,
    int*          __restrict__ sparse_kv_indices,
    const int     page_reserved_bos,
    const int     page_reserved_eos,
    const float   tolerate_ratio)
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

    approx_topk_inner<ScoreT>(score_blk, s_indices, nblk, target_k, tolerate_thresh);
    __syncthreads();

    const int tx = threadIdx.x;
    for (int i = tx; i < target_k; i += kThreadsPerBlock) {
        out_blk[i] = idx_blk[s_indices[i]];
    }
}

}  // namespace


void approx_topk_output(
    const at::Tensor& x,
    const at::Tensor& dense_kv_indptr,
    const at::Tensor& sparse_kv_indptr,
    const at::Tensor& dense_kv_indices,
    at::Tensor&       sparse_kv_indices,
    const int64_t     eff_batch_size,
    const int64_t     reserved_bos,
    const int64_t     reserved_eos,
    const int64_t     max_num_pages,
    const double      tolerate_ratio)
{
    (void)max_num_pages;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // -----------------------------------------------------------------
    // Internal small-K fast path (bf16, K∈[16,32], max_num_pages<=16384,
    // eff_batch_size>=4). Bench sweep evidence:
    //   pages=4096:        SELECT32-SORT32 wins 1.16–1.20x  ✓
    //   pages∈{8192,16384}: statistically tied (≤0.5%)      = no regression
    // The B<=2 / K<=8 windows are excluded because the bench showed
    // 15–25% regressions there. SELECT32 returns EXACT top-K, so when
    // it fires it ignores tolerate_ratio and produces a stricter result
    // than the approximate algorithm — a quality improvement at no
    // latency cost in this bucket.
    //
    // K is inferred from sparse_kv_indices.numel() / eff_batch_size
    // (no GPU sync); assumes uniform K per row.
    // -----------------------------------------------------------------
    if (x.scalar_type() == at::ScalarType::BFloat16
        && eff_batch_size >= 4
        && max_num_pages > 0 && max_num_pages <= 16384) {
        const int64_t inferred_k = sparse_kv_indices.numel() / eff_batch_size;
        if (inferred_k >= 16 && inferred_k <= 32) {
            // NUM_THREADS=1024 mirrors kSelCfg1 from topk_sglang_merge.cu —
            // benchmark showed smaller NT collapses the speedup.
            // Dynamic SMEM holds s_bins (one byte per row element). Aligning
            // to 16 keeps the byte block well-aligned; 16 KB at the cap is
            // comfortably under the 48 KB default per-CTA ceiling.
            const size_t bins_bytes =
                (static_cast<size_t>(max_num_pages) + size_t(15)) & ~size_t(15);
            approx_topk_select32::Kernel<1024>
                <<<dim3(eff_batch_size), dim3(1024), bins_bytes, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                    dense_kv_indptr.data_ptr<int>(),
                    sparse_kv_indptr.data_ptr<int>(),
                    dense_kv_indices.data_ptr<int>(),
                    sparse_kv_indices.data_ptr<int>(),
                    static_cast<int>(inferred_k),
                    static_cast<int>(reserved_bos),
                    static_cast<int>(reserved_eos));
            const auto err_fast = cudaGetLastError();
            TORCH_CHECK(err_fast == cudaSuccess,
                        "approx_topk_output (select32 fast path) kernel failed: ",
                        cudaGetErrorString(err_fast));
            return;
        }
    }

    dim3 nblks(eff_batch_size);
    dim3 nthreads(kThreadsPerBlock);

    const float tol = static_cast<float>(tolerate_ratio);

    // Dynamic SMEM caches at-threshold-bin indices (16 KB) so Stage-2 only
    // iterates that subset, matching the topk_v2 slow-path scaling.
    constexpr size_t kApproxSmemBytes =
        static_cast<size_t>(kApproxSmemInputSize) * sizeof(int);

    if (x.scalar_type() == at::ScalarType::BFloat16) {
        ApproxTopK_Kernel<__nv_bfloat16>
            <<<nblks, nthreads, kApproxSmemBytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            static_cast<int>(reserved_bos),
            static_cast<int>(reserved_eos),
            tol);
    } else if (x.scalar_type() == at::ScalarType::Float) {
        ApproxTopK_Kernel<float>
            <<<nblks, nthreads, kApproxSmemBytes, stream>>>(
            x.data_ptr<float>(),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            static_cast<int>(reserved_bos),
            static_cast<int>(reserved_eos),
            tol);
    } else {
        TORCH_CHECK(false,
                    "approx_topk_output: unsupported dtype ", x.scalar_type());
    }

    const auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "approx_topk_output kernel failed: ", cudaGetErrorString(err));
}
