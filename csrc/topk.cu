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


// `TopKOutput_Kernel` is a single-block in-register cub::BlockRadixSort.
// Its capacity is exactly NUM_THREADS * ITEM_PER_THREAD elements per row,
// and the dispatcher tops out at (256, 16) = 4096. For rows longer than
// that, cub::BlockLoad reads only the first 4096 entries; the remainder
// is silently dropped (padded with -inf in the OOB tail). The output is
// then "top-k of the first 4096 elements" — recall ≈ 4096 / length, so
// for length=131072 we measured recall ≈ 0.033.
//
// Bumping NUM_THREADS / ITEM_PER_THREAD doesn't rescue this: each thread
// holds ITEM_PER_THREAD × (key + raw_key + val) register slots, and at
// (256, 16) we're already near the per-thread register ceiling. Pushing
// to (256, 64) = 16384 is borderline; (256, 512) = 131072 would spill so
// hard the kernel would be slower than full DRAM sort.
//
// The correct algorithm at long input is radix-select with bounded
// refinement, which is exactly `topk_output_v2`. So when `max_num_pages`
// exceeds the block-sort capacity, delegate to `topk_output_v2` to keep
// `topk_output` correct over the whole input range. Short inputs still
// take the fast register-sort path.
constexpr int64_t kBlockSortMaxPages = 4096;

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
        topk_output_v2(x, dense_kv_indptr, sparse_kv_indptr, dense_kv_indices,
                       sparse_kv_indices, eff_batch_size,
                       reserved_bos, reserved_eos, max_num_pages);
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
