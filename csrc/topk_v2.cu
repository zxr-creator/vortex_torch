/**
 * @NOTE: This file is adapted from
 * https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/topk_selector.py
 * We:
 * 1. adapt from tilelang to pure cuda
 * 2. optimize the performance a little
 * 3. fix the potential illegal memory access
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
 // in topk_output_v2.
 // =====================================================================

 namespace topk_v2_select32 {

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
     // Random L=16K → expected ~L/256 = 64; 512 gives 8x headroom.
     // Overflow falls back to scan-all-L Pass 3 with the s_bins gate.
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
             // Reverse-inclusive cumsum within the lane's 8 bins.
             #pragma unroll
             for (int j = kPerLane - 2; j >= 0; --j) {
                 locals[j] += locals[j + 1];
             }
             // locals[0] now holds the lane's total. Compute reverse-inclusive
             // warp prefix sum so lane l ends up with sum of locals[0] from
             // lanes l..31. Hillis-Steele on shfl_down_sync.
             int my_sum = locals[0];
             #pragma unroll
             for (int delta = 1; delta < 32; delta *= 2) {
                 const int v = __shfl_down_sync(0xFFFFFFFF, my_sum, delta);
                 if (tx + delta < 32) {
                     my_sum += v;
                 }
             }
             // Exclusive suffix = sum of strictly higher lanes' totals.
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
         // for random scores). Overflow falls back to scanning all L with the
         // s_bins gate (preserves correctness for any distribution).
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

 }  // namespace topk_v2_select32


 constexpr int kThreadsPerBlock = 1024;

 #ifdef USE_ROCM
 // On ROCm, the per-workgroup LDS budget depends on the target arch, so we inject a
 // per-arch value from `setup_rocm.py` via `-DSGL_TOPK_DYNAMIC_SMEM_BYTES=...`.
 #ifdef SGL_TOPK_DYNAMIC_SMEM_BYTES
 constexpr size_t kSmem = static_cast<size_t>(SGL_TOPK_DYNAMIC_SMEM_BYTES);
 #else
 constexpr size_t kSmem = 48 * 1024;  // bytes
 #endif
 #else
 // 32KB → 64KB. With 32KB the at-threshold cache holds 4K entries per
 // round, which overflows on degenerate distributions (e.g., uniform-bf16
 // at L=131k where one giant central bin holds tens of thousands of
 // elements). Overflow elements get dropped by the `pos < SMEM_INPUT_SIZE`
 // guard in Pass 2, which collapses recall from ~1.0 to ~0.55. Doubling to
 // 64KB (8K entries per round) keeps occupancy at 2 blocks/SM on Blackwell
 // and rescues recall on those configs.
 constexpr size_t kSmem = 16 * 1024 * sizeof(uint32_t);  // 64KB (bytes)
 #endif

 __device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
   __half h = __float2half_rn(x);
   uint16_t bits = __half_as_ushort(h);
   uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
   return static_cast<uint8_t>(key >> 8);
 }

 __device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
   uint32_t bits = __float_as_uint(x);
   return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
 }

 template <auto* f, size_t max_dynamic_smem>
 void setup_kernel_smem_once() {
   [[maybe_unused]]
   static const auto result = [] {
 #ifdef USE_ROCM
     // hipify will turn cudaFuncSetAttribute -> hipFuncSetAttribute. On ROCm,
     // hipFuncSetAttribute expects `const void*` and hipcc does not accept passing
     // a function pointer directly, so cast explicitly.
     return ::cudaFuncSetAttribute(
         reinterpret_cast<const void*>(f), ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
 #else
     // CUDA: keep original behavior (no cast needed).
     return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
 #endif
   }();
   TORCH_CHECK(result == cudaSuccess, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
 }

 // ======================================================================
 // Vortex integration: BOS/EOS-aware segmented TopK with index remapping
 // ======================================================================

 template <typename T>
 __device__ __forceinline__ float vortex_to_float(T x);

 template <>
 __device__ __forceinline__ float vortex_to_float<float>(float x) { return x; }

 template <>
 __device__ __forceinline__ float vortex_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
     return __bfloat162float(x);
 }

 constexpr int VORTEX_MAX_TOPK = 2048;

 // Templated version of fast_topk_cuda_tl:
 //   - ScoreT: float or __nv_bfloat16
 //   - target_k: runtime parameter (replaces compile-time TopK)
 template <typename ScoreT>
 __device__ void fast_topk_vortex(
     const ScoreT* __restrict__ input,
     int*          __restrict__ index,
     int           row_start,
     int           length,
     int           target_k)
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

     const int tx = threadIdx.x;

     // Stage 1: 8-bit coarse histogram
     if (tx < RADIX + 1) vh_histogram[tx] = 0;
     __syncthreads();

     for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
         const auto bin = convert_to_uint8(vortex_to_float(input[idx + row_start]));
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

     if (topk == 0) {
         for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
             const auto bin = static_cast<int>(
                 convert_to_uint8(vortex_to_float(input[idx + row_start])));
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

         for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
             const auto raw_input = vortex_to_float(input[idx + row_start]);
             const auto bin = static_cast<int>(convert_to_uint8(raw_input));
             if (bin > threshold_bin) {
                 const auto pos = ::atomicAdd(&vh_counter, 1);
                 index[pos] = idx;
             } else if (bin == threshold_bin) {
                 const auto pos = ::atomicAdd(&vh_num_input[0], 1);
                 if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                     vh_input_idx[0][pos] = idx;
                     const auto b32 = convert_to_uint32(raw_input);
                     const auto sub_bin = (b32 >> 24) & 0xFF;
                     ::atomicAdd(&vh_histogram[sub_bin], 1);
                 }
             }
         }
         __syncthreads();
     }

     // Stage 2: refine with 8-bit radix passes
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
                 const auto bin = (convert_to_uint32(
                     vortex_to_float(input[idx + row_start])) >> offset) & 0xFF;
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
                 const auto raw_input = vortex_to_float(input[idx + row_start]);
                 const auto offset = 24 - round * 8;
                 const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
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
                             const auto b32 = convert_to_uint32(raw_input);
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

 // Wrapper kernel: one CUDA block per batch*head segment
 template <typename ScoreT>
 __global__ __launch_bounds__(kThreadsPerBlock)
 void TopKOutput_Kernel(
     const ScoreT* __restrict__ score,
     const int*    __restrict__ dense_kv_indptr,
     const int*    __restrict__ sparse_kv_indptr,
     const int*    __restrict__ dense_kv_indices,
     int*          __restrict__ sparse_kv_indices,
     const int     page_reserved_bos,
     const int     page_reserved_eos)
 {
     const int bx = blockIdx.x;

     const int start = dense_kv_indptr[bx] + page_reserved_bos;
     const int end   = dense_kv_indptr[bx + 1] - page_reserved_eos;
     const int topk_val = sparse_kv_indptr[bx + 1] - sparse_kv_indptr[bx] - page_reserved_bos - page_reserved_eos;
     const int nblk  = end - start;
     if (nblk <= topk_val) return;

     const ScoreT* __restrict__ score_blk = score + start;
     const int*    __restrict__ idx_blk   = dense_kv_indices + start;
     int*          __restrict__ out_blk   = sparse_kv_indices
                                          + sparse_kv_indptr[bx]
                                          + page_reserved_bos;

     __shared__ int s_indices[VORTEX_MAX_TOPK];
     fast_topk_vortex<ScoreT>(score_blk, s_indices, 0, nblk, topk_val);
     __syncthreads();

     // Remap position indices -> page indices via dense_kv_indices
     const int tx = threadIdx.x;
     for (int i = tx; i < topk_val; i += kThreadsPerBlock) {
         out_blk[i] = idx_blk[s_indices[i]];
     }
 }

 }  // namespace

 // ======================================================================
 // Vortex host entry point — same interface as topk_output in topk.cu
 // ======================================================================
 void topk_output_v2(
     const at::Tensor& x,
     const at::Tensor& dense_kv_indptr,
     const at::Tensor& sparse_kv_indptr,
     const at::Tensor& dense_kv_indices,
     at::Tensor&       sparse_kv_indices,
     const int64_t     eff_batch_size,
     const int64_t     reserved_bos,
     const int64_t     reserved_eos,
     const int64_t     max_num_pages)
 {
     cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

     // -----------------------------------------------------------------
     // Internal small-K fast path (bf16, K∈[16,32], max_num_pages<=16384,
     // eff_batch_size>=4). Bench sweep evidence:
     //   pages=4096:        SELECT32-SORT32 wins 1.16–1.20x  ✓
     //   pages∈{8192,16384}: statistically tied (≤0.5%)      = no regression
     // The B<=2 / K<=8 windows are excluded because the bench showed
     // 15–25% regressions there — the 1024-thread launch under-utilizes
     // the SM at low B, and K=8 lacks the items to amortize the 2-stage
     // radix select setup. K is inferred from
     // sparse_kv_indices.numel() / eff_batch_size — host-side metadata
     // read with NO GPU sync, assumes uniform K. Anything outside this
     // window keeps the existing 4-round refinement path.
     // -----------------------------------------------------------------
     if (x.scalar_type() == at::ScalarType::BFloat16
         && eff_batch_size >= 4
         && max_num_pages > 0 && max_num_pages <= 16384) {
         const int64_t inferred_k = sparse_kv_indices.numel() / eff_batch_size;
         if (inferred_k >= 16 && inferred_k <= 32) {
             // Dynamic SMEM holds s_bins (one byte per row element). 16-byte
             // alignment; 16 KB at the cap is well under the 48 KB default.
             const size_t bins_bytes =
                 (static_cast<size_t>(max_num_pages) + size_t(15)) & ~size_t(15);
             topk_v2_select32::Kernel<1024>
                 <<<dim3(eff_batch_size), dim3(1024), bins_bytes, stream>>>(
                     reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                     dense_kv_indptr.data_ptr<int>(),
                     sparse_kv_indptr.data_ptr<int>(),
                     dense_kv_indices.data_ptr<int>(),
                     sparse_kv_indices.data_ptr<int>(),
                     static_cast<int>(inferred_k),
                     static_cast<int>(reserved_bos),
                     static_cast<int>(reserved_eos));
             const auto err_fast = cudaGetLastError();
             TORCH_CHECK(err_fast == cudaSuccess,
                         "topk_output_v2 (select32 fast path) kernel failed: ",
                         ::cudaGetErrorString(err_fast));
             return;
         }
     }

     dim3 nblks(eff_batch_size);
     dim3 nthreads(kThreadsPerBlock);

     if (x.scalar_type() == at::ScalarType::BFloat16) {
         setup_kernel_smem_once<TopKOutput_Kernel<__nv_bfloat16>, kSmem>();
         TopKOutput_Kernel<__nv_bfloat16><<<nblks, nthreads, kSmem, stream>>>(
             reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
             dense_kv_indptr.data_ptr<int>(),
             sparse_kv_indptr.data_ptr<int>(),
             dense_kv_indices.data_ptr<int>(),
             sparse_kv_indices.data_ptr<int>(),
             reserved_bos,
             reserved_eos);
     } else if (x.scalar_type() == at::ScalarType::Float) {
         setup_kernel_smem_once<TopKOutput_Kernel<float>, kSmem>();
         TopKOutput_Kernel<float><<<nblks, nthreads, kSmem, stream>>>(
             x.data_ptr<float>(),
             dense_kv_indptr.data_ptr<int>(),
             sparse_kv_indptr.data_ptr<int>(),
             dense_kv_indices.data_ptr<int>(),
             sparse_kv_indices.data_ptr<int>(),
             reserved_bos,
             reserved_eos);
     } else {
         TORCH_CHECK(false,
                     "topk_output: unsupported dtype ",
                     x.scalar_type());
     }

     const auto result = cudaGetLastError();
     TORCH_CHECK(result == cudaSuccess,
                 "topk_output kernel failed: ", ::cudaGetErrorString(result));
 }
