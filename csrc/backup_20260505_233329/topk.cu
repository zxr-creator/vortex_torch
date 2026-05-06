#include "register.h"
#include <cub/cub.cuh>
#ifdef VORTEX_ENABLE_FP8
#include <cuda_fp8.h>
#endif


template <typename T>
__device__ __forceinline__ float score_to_float(T v);

template <>
__device__ __forceinline__ float score_to_float<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ float score_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

#ifdef VORTEX_ENABLE_FP8
// fp8 -> half -> float: uses the canonical cuda_fp8 conversion intrinsic.
__device__ __forceinline__ float fp8_to_fp32(__nv_fp8_e4m3 v) {
    return __half2float(static_cast<__half>(
        __nv_cvt_fp8_to_halfraw(v.__x, __NV_E4M3)));
}

__device__ __forceinline__ float fp8_to_fp32(__nv_fp8_e5m2 v) {
    return __half2float(static_cast<__half>(
        __nv_cvt_fp8_to_halfraw(v.__x, __NV_E5M2)));
}

template <>
__device__ __forceinline__ float score_to_float<__nv_fp8_e4m3>(__nv_fp8_e4m3 v) {
    return fp8_to_fp32(v);
}

template <>
__device__ __forceinline__ float score_to_float<__nv_fp8_e5m2>(__nv_fp8_e5m2 v) {
    return fp8_to_fp32(v);
}
#endif


template <typename T>
__device__ __forceinline__ T score_neg_inf();

template <>
__device__ __forceinline__ float score_neg_inf<float>() {
    return -CUDART_INF_F;
}

template <>
__device__ __forceinline__ __nv_bfloat16 score_neg_inf<__nv_bfloat16>() {
    return __float2bfloat16(-CUDART_INF_F);
}

#ifdef VORTEX_ENABLE_FP8
template <>
__device__ __forceinline__ __nv_fp8_e4m3 score_neg_inf<__nv_fp8_e4m3>() {
    // e4m3fn has no infinity; 0xFF is NaN, 0xFE encodes the most negative finite value (-448).
    __nv_fp8_e4m3 v;
    v.__x = 0xFE;
    return v;
}

template <>
__device__ __forceinline__ __nv_fp8_e5m2 score_neg_inf<__nv_fp8_e5m2>() {
    // e5m2: 0xFC encodes -inf.
    __nv_fp8_e5m2 v;
    v.__x = 0xFC;
    return v;
}
#endif


// Chunked, shared-memory-merge-based exact top-k for long inputs.
//
// The single-block in-register `TopKOutput_Kernel` caps at NUM_THREADS *
// ITEM_PER_THREAD = 4096 elements per row; anything past that is silently
// truncated. This kernel processes the row in chunks of CHUNK = NUM_THREADS *
// ITEM_CHUNK and maintains a running "top-CHUNK" buffer in shared memory.
// Each iteration:
//   1. Load up to CHUNK new elements into the second half of the per-thread
//      register array, with the first half holding the running top.
//   2. cub::BlockRadixSort over the combined 2*CHUNK descending.
//   3. Write the top CHUNK back to the shared running buffer.
// At the end, the first `topk_val` entries of the running buffer are the
// exact top-k. Capacity: topk_val must be ≤ CHUNK; the dispatcher enforces
// this and falls back to topk_output_v2 otherwise.
template <typename ScoreT, int NUM_THREADS, int ITEM_CHUNK>
__global__ void TopKOutput_Chunked_Kernel(
const ScoreT* __restrict__ score,
const int*    __restrict__ dense_kv_indptr,
const int*    __restrict__ sparse_kv_indptr,
const int*    __restrict__ dense_kv_indices,
int*          __restrict__ sparse_kv_indices,
const int     page_reserved_bos,
const int     page_reserved_eos)
{
    constexpr int CHUNK     = NUM_THREADS * ITEM_CHUNK;
    constexpr int ITEM_SORT = 2 * ITEM_CHUNK;

    const int bx = blockIdx.x;

    const int start    = dense_kv_indptr[bx] + page_reserved_bos;
    const int end      = dense_kv_indptr[bx + 1] - page_reserved_eos;
    const int topk_val = (sparse_kv_indptr[bx + 1] - sparse_kv_indptr[bx])
                         - page_reserved_bos - page_reserved_eos;
    const int nblk     = end - start;
    if (nblk <= topk_val) return;

    const ScoreT* __restrict__ score_blk = score + start;
    const int*    __restrict__ idx_blk   = dense_kv_indices + start;
    int*          __restrict__ out_blk   = sparse_kv_indices + sparse_kv_indptr[bx] + page_reserved_bos;

    const int tx = threadIdx.x;
    const ScoreT ninf_s = score_neg_inf<ScoreT>();

    using BLF = cub::BlockLoad<ScoreT, NUM_THREADS, ITEM_CHUNK, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BLI = cub::BlockLoad<int,    NUM_THREADS, ITEM_CHUNK, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using Sort = cub::BlockRadixSort<float, NUM_THREADS, ITEM_SORT, int>;

    __shared__ union {
        typename BLF::TempStorage  lf;
        typename BLI::TempStorage  li;
        typename Sort::TempStorage sort;
    } temp;

    __shared__ float run_keys[CHUNK];
    __shared__ int   run_vals[CHUNK];

    // Initialise running top-CHUNK to -inf so the first chunk dominates.
    for (int i = tx; i < CHUNK; i += NUM_THREADS) {
        run_keys[i] = -CUDART_INF_F;
        run_vals[i] = 0;
    }
    __syncthreads();

    ScoreT key_raw[ITEM_CHUNK];
    int    chunk_val[ITEM_CHUNK];
    float  key[ITEM_SORT];
    int    val[ITEM_SORT];

    for (int cstart = 0; cstart < nblk; cstart += CHUNK) {
        const int rem = nblk - cstart;
        const int cnt = (rem < CHUNK) ? rem : CHUNK;

        BLF(temp.lf).Load(score_blk + cstart, key_raw, cnt, ninf_s);
        __syncthreads();
        BLI(temp.li).Load(idx_blk   + cstart, chunk_val, cnt, 0);
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < ITEM_CHUNK; ++i) {
            key[i] = score_to_float<ScoreT>(key_raw[i]);
            val[i] = chunk_val[i];
        }

        // Read the running top-CHUNK out of shared memory into the second
        // half of the per-thread sort array. Layout is "blocked" so thread t
        // owns slots [t * ITEM_CHUNK, (t+1) * ITEM_CHUNK).
        #pragma unroll
        for (int i = 0; i < ITEM_CHUNK; ++i) {
            const int sm_idx = tx * ITEM_CHUNK + i;
            key[ITEM_CHUNK + i] = run_keys[sm_idx];
            val[ITEM_CHUNK + i] = run_vals[sm_idx];
        }
        __syncthreads();

        Sort(temp.sort).SortDescending(key, val);
        __syncthreads();

        // Sorted output is in BLOCKED layout: thread t holds ranks
        // [t * ITEM_SORT, (t+1) * ITEM_SORT). The top CHUNK live in
        // threads 0..NUM_THREADS/2 - 1 (each contributing all ITEM_SORT regs).
        if (tx < NUM_THREADS / 2) {
            #pragma unroll
            for (int i = 0; i < ITEM_SORT; ++i) {
                const int sm_idx = tx * ITEM_SORT + i;
                run_keys[sm_idx] = key[i];
                run_vals[sm_idx] = val[i];
            }
        }
        __syncthreads();
    }

    for (int i = tx; i < topk_val; i += NUM_THREADS) {
        out_blk[i] = run_vals[i];
    }
}


template <typename ScoreT, int NUM_THREADS, int ITEM_PER_THREAD>
__global__ void TopKOutput_Kernel(
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
    const int topk_val = (sparse_kv_indptr[bx + 1] - sparse_kv_indptr[bx]) - page_reserved_bos - page_reserved_eos;
    const int nblk  = end - start;
    if (nblk <= topk_val) return;

    const ScoreT* __restrict__ score_blk = score + start;
    const int*    __restrict__ idx_blk   = dense_kv_indices + start;
    int*          __restrict__ out_blk   = sparse_kv_indices + sparse_kv_indptr[bx] + page_reserved_bos;

    const ScoreT ninf_score = score_neg_inf<ScoreT>();

    ScoreT key_raw[ITEM_PER_THREAD];
    float  key[ITEM_PER_THREAD];
    int    val[ITEM_PER_THREAD];

    using BLF  = cub::BlockLoad<ScoreT, NUM_THREADS, ITEM_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BLI  = cub::BlockLoad<int,    NUM_THREADS, ITEM_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BSI  = cub::BlockStore<int,   NUM_THREADS, ITEM_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using Sort = cub::BlockRadixSort<float, NUM_THREADS, ITEM_PER_THREAD, int>;

    __shared__ union {
        typename BLF::TempStorage  lf;
        typename BLI::TempStorage  li;
        typename BSI::TempStorage  si;
        typename Sort::TempStorage sort;
    } temp;

    BLF(temp.lf).Load(score_blk, key_raw, nblk, ninf_score);

    #pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; ++i){
        key[i] = score_to_float<ScoreT>(key_raw[i]);
    }
    __syncthreads();

    BLI(temp.li).Load(idx_blk,   val, nblk, 0);
    __syncthreads();

    Sort(temp.sort).SortDescending(key, val);
    __syncthreads();

    const int valid_out = min(topk_val, nblk);
    BSI(temp.si).Store(out_blk, /*per-thread regs*/ val, valid_out);
}


template <typename ScoreT>
static void dispatch_topk_output(
const ScoreT*       x_ptr,
const at::Tensor&   dense_kv_indptr,
const at::Tensor&   sparse_kv_indptr,
const at::Tensor&   dense_kv_indices,
at::Tensor&         sparse_kv_indices,
const int64_t       eff_batch_size,
const int64_t       reserved_bos,
const int64_t       reserved_eos,
const int64_t       max_num_pages,
cudaStream_t        stream)
{
    dim3 nblks(eff_batch_size);

    #define LAUNCH_TOPK(THREADS, ITEMS)                                             \
        TopKOutput_Kernel<ScoreT, THREADS, ITEMS><<<nblks, THREADS, 0, stream>>>(   \
            x_ptr,                                                                  \
            dense_kv_indptr.data_ptr<int>(),                                        \
            sparse_kv_indptr.data_ptr<int>(),                                       \
            dense_kv_indices.data_ptr<int>(),                                       \
            sparse_kv_indices.data_ptr<int>(),                                      \
            reserved_bos,                                                           \
            reserved_eos)

    if (max_num_pages <= 128)          { LAUNCH_TOPK(128, 1);  }
    else if (max_num_pages <= 256)     { LAUNCH_TOPK(128, 2);  }
    else if (max_num_pages <= 512)     { LAUNCH_TOPK(128, 4);  }
    else if (max_num_pages <= 1024)    { LAUNCH_TOPK(128, 8);  }
    else if (max_num_pages <= 1536)    { LAUNCH_TOPK(128, 12); }
    else if (max_num_pages <= 2048)    { LAUNCH_TOPK(128, 16); }
    else if (max_num_pages <= 2560)    { LAUNCH_TOPK(256, 10); }
    else if (max_num_pages <= 3072)    { LAUNCH_TOPK(256, 12); }
    else if (max_num_pages <= 3584)    { LAUNCH_TOPK(256, 14); }
    else                               { LAUNCH_TOPK(256, 16); }

    #undef LAUNCH_TOPK
}


// Single-block in-register `TopKOutput_Kernel` is hard-capped at
// NUM_THREADS * ITEM_PER_THREAD = 4096 elements per row — beyond that
// cub::BlockLoad silently reads only the first 4096 entries. For rows
// longer than `kBlockSortMaxPages` we use `TopKOutput_Chunked_Kernel`,
// which reads the row in chunks of 2048 and merges with a running
// top-2048 in shared memory, producing exact top-k at any length so long
// as `topk_val ≤ kChunkedTopkMax`. If topk_val exceeds that limit we fall
// back to `topk_output_v2` (radix-select).
constexpr int64_t kBlockSortMaxPages  = 4096;
constexpr int64_t kChunkedTopkMax     = 2048;
constexpr int     kChunkedNumThreads  = 256;
constexpr int     kChunkedItemPerThr  = 8;     // CHUNK = 256 * 8 = 2048

void topk_output(
const at::Tensor& x,
const at::Tensor& dense_kv_indptr,
const at::Tensor& sparse_kv_indptr,
const at::Tensor& dense_kv_indices,
at::Tensor&       sparse_kv_indices,
const int64_t     eff_batch_size,
const int64_t     reserved_bos,
const int64_t     reserved_eos,
const int64_t     max_num_pages
){
    if (max_num_pages > kBlockSortMaxPages) {
        // Long input: pick the chunked exact kernel when we can fit the
        // running top-K in its shared-memory buffer; else delegate to the
        // radix-select kernel for correctness at any K.
        const int64_t inferred_k =
            (eff_batch_size > 0)
                ? (sparse_kv_indices.numel() / eff_batch_size
                   - reserved_bos - reserved_eos)
                : 0;

        if (inferred_k <= 0 || inferred_k > kChunkedTopkMax) {
            topk_output_v2(x, dense_kv_indptr, sparse_kv_indptr, dense_kv_indices,
                           sparse_kv_indices, eff_batch_size,
                           reserved_bos, reserved_eos, max_num_pages);
            return;
        }

        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        dim3 nblks(eff_batch_size);

        #define LAUNCH_CHUNKED(SCORE_T, PTR_EXPR)                                    \
            TopKOutput_Chunked_Kernel<SCORE_T, kChunkedNumThreads, kChunkedItemPerThr>\
                <<<nblks, kChunkedNumThreads, 0, stream>>>(                          \
                    PTR_EXPR,                                                        \
                    dense_kv_indptr.data_ptr<int>(),                                 \
                    sparse_kv_indptr.data_ptr<int>(),                                \
                    dense_kv_indices.data_ptr<int>(),                                \
                    sparse_kv_indices.data_ptr<int>(),                               \
                    static_cast<int>(reserved_bos),                                  \
                    static_cast<int>(reserved_eos))

        const auto dtype = x.scalar_type();
        if (dtype == at::ScalarType::Float) {
            LAUNCH_CHUNKED(float, x.data_ptr<float>());
        } else if (dtype == at::ScalarType::BFloat16) {
            LAUNCH_CHUNKED(__nv_bfloat16,
                reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()));
        }
#ifdef VORTEX_ENABLE_FP8
        else if (dtype == at::ScalarType::Float8_e4m3fn) {
            LAUNCH_CHUNKED(__nv_fp8_e4m3,
                reinterpret_cast<const __nv_fp8_e4m3*>(x.data_ptr()));
        } else if (dtype == at::ScalarType::Float8_e5m2) {
            LAUNCH_CHUNKED(__nv_fp8_e5m2,
                reinterpret_cast<const __nv_fp8_e5m2*>(x.data_ptr()));
        }
#endif
        else {
            TORCH_CHECK(false, "topk_output (chunked): unsupported dtype ", dtype);
        }
        #undef LAUNCH_CHUNKED

        const auto err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "topk_output (chunked) kernel failed: ", cudaGetErrorString(err));
        return;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const auto dtype = x.scalar_type();

    if (dtype == at::ScalarType::Float) {
        dispatch_topk_output<float>(
            x.data_ptr<float>(),
            dense_kv_indptr, sparse_kv_indptr, dense_kv_indices, sparse_kv_indices,
            eff_batch_size, reserved_bos, reserved_eos, max_num_pages, stream);
    } else if (dtype == at::ScalarType::BFloat16) {
        dispatch_topk_output<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            dense_kv_indptr, sparse_kv_indptr, dense_kv_indices, sparse_kv_indices,
            eff_batch_size, reserved_bos, reserved_eos, max_num_pages, stream);
    }
#ifdef VORTEX_ENABLE_FP8
    else if (dtype == at::ScalarType::Float8_e4m3fn) {
        dispatch_topk_output<__nv_fp8_e4m3>(
            reinterpret_cast<const __nv_fp8_e4m3*>(x.data_ptr()),
            dense_kv_indptr, sparse_kv_indptr, dense_kv_indices, sparse_kv_indices,
            eff_batch_size, reserved_bos, reserved_eos, max_num_pages, stream);
    } else if (dtype == at::ScalarType::Float8_e5m2) {
        dispatch_topk_output<__nv_fp8_e5m2>(
            reinterpret_cast<const __nv_fp8_e5m2*>(x.data_ptr()),
            dense_kv_indptr, sparse_kv_indptr, dense_kv_indices, sparse_kv_indices,
            eff_batch_size, reserved_bos, reserved_eos, max_num_pages, stream);
    }
#endif
    else {
        TORCH_CHECK(false, "topk_output: unsupported dtype ", dtype);
    }
}
