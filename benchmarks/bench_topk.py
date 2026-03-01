"""
Benchmark: CUB BlockRadixSort (topk_output) vs Radix Selection (topk_output_v2)

Generates synthetic packed-segment data matching the vortex topk interface,
verifies correctness (both kernels select the same pages), and measures
latency across configurations.

Usage:
    python benchmarks/bench_topk.py [--warmup 10] [--iters 100] [--device cuda:0]
"""
import argparse
import time
import torch
import vortex_torch_C


def make_test_data(
    batch_size: int,
    num_pages: int,
    topk_val: int,
    reserved_bos: int = 1,
    reserved_eos: int = 1,
    dtype=torch.bfloat16,
    device: str = "cuda:0",
):
    """Create synthetic tensors matching the vortex topk interface.

    Each of the `batch_size` segments has `num_pages` total pages
    (including reserved BOS/EOS). The dense layout is a packed 1-D
    concatenation of all segments.
    """
    S_pack = batch_size * num_pages
    sparse_per_seg = topk_val + reserved_bos + reserved_eos

    # Dense segment boundaries: [0, num_pages, 2*num_pages, ...]
    dense_kv_indptr = torch.arange(
        0, S_pack + 1, num_pages, dtype=torch.int32, device=device
    )

    # Sparse segment boundaries
    sparse_kv_indptr = torch.arange(
        0,
        batch_size * sparse_per_seg + 1,
        sparse_per_seg,
        dtype=torch.int32,
        device=device,
    )

    # Page indices: sequential within each segment (identity mapping)
    dense_kv_indices = torch.arange(S_pack, dtype=torch.int32, device=device)

    # Random scores
    scores = torch.randn(S_pack, dtype=torch.float32, device=device).to(dtype)

    # Pre-allocate output (fill with sentinel)
    S_sparse = batch_size * sparse_per_seg
    sparse_kv_indices = torch.full(
        (S_sparse,), -1, dtype=torch.int32, device=device
    )

    return (
        scores,
        dense_kv_indptr,
        sparse_kv_indptr,
        dense_kv_indices,
        sparse_kv_indices,
        sparse_per_seg,
    )


def verify_correctness(
    batch_size: int,
    num_pages: int,
    topk_val: int,
    reserved_bos: int = 1,
    reserved_eos: int = 1,
    dtype=torch.bfloat16,
    device: str = "cuda:0",
):
    """Check that both kernels select the same set of top-k pages."""
    (
        scores,
        dense_kv_indptr,
        sparse_kv_indptr,
        dense_kv_indices,
        _,
        sparse_per_seg,
    ) = make_test_data(batch_size, num_pages, topk_val, reserved_bos, reserved_eos, dtype, device)

    S_sparse = batch_size * sparse_per_seg

    # Run baseline (CUB)
    out_baseline = torch.full((S_sparse,), -1, dtype=torch.int32, device=device)
    vortex_torch_C.topk_output(
        scores,
        dense_kv_indptr,
        sparse_kv_indptr,
        dense_kv_indices,
        out_baseline,
        batch_size,
        topk_val,
        reserved_bos,
        reserved_eos,
        num_pages,
    )

    # Run radix selection (v2)
    out_v2 = torch.full((S_sparse,), -1, dtype=torch.int32, device=device)
    vortex_torch_C.topk_output_v2(
        scores,
        dense_kv_indptr,
        sparse_kv_indptr,
        dense_kv_indices,
        out_v2,
        batch_size,
        topk_val,
        reserved_bos,
        reserved_eos,
        num_pages,
    )

    torch.cuda.synchronize()

    # Compare: both should select the same set of pages (order may differ)
    all_match = True
    for b in range(batch_size):
        start = b * sparse_per_seg + reserved_bos
        end = start + topk_val
        set_baseline = set(out_baseline[start:end].cpu().tolist())
        set_v2 = set(out_v2[start:end].cpu().tolist())
        if set_baseline != set_v2:
            all_match = False
            print(f"  MISMATCH at batch {b}:")
            print(f"    baseline: {sorted(set_baseline)[:10]}...")
            print(f"    v2:       {sorted(set_v2)[:10]}...")
            break

    return all_match


def bench_kernel(
    kernel_fn,
    scores,
    dense_kv_indptr,
    sparse_kv_indptr,
    dense_kv_indices,
    sparse_kv_indices,
    batch_size: int,
    topk_val: int,
    reserved_bos: int,
    reserved_eos: int,
    num_pages: int,
    warmup: int,
    iters: int,
):
    """Time a single kernel variant. Returns median latency in microseconds."""
    # Warmup
    for _ in range(warmup):
        kernel_fn(
            scores,
            dense_kv_indptr,
            sparse_kv_indptr,
            dense_kv_indices,
            sparse_kv_indices,
            batch_size,
            topk_val,
            reserved_bos,
            reserved_eos,
            num_pages,
        )
    torch.cuda.synchronize()

    # Timed iterations
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        # Reset output each iteration to avoid caching effects
        sparse_kv_indices.fill_(-1)
        start_events[i].record()
        kernel_fn(
            scores,
            dense_kv_indptr,
            sparse_kv_indptr,
            dense_kv_indices,
            sparse_kv_indices,
            batch_size,
            topk_val,
            reserved_bos,
            reserved_eos,
            num_pages,
        )
        end_events[i].record()

    torch.cuda.synchronize()

    times_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(start_events, end_events)]
    times_us.sort()
    median = times_us[len(times_us) // 2]
    return median, times_us


def main():
    parser = argparse.ArgumentParser(description="Benchmark topk kernels")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device
    warmup = args.warmup
    iters = args.iters

    # Test configurations: (batch_size, num_pages, topk_val)
    configs = [
        # Small: typical vortex decode workload
        (8,    128,   30),
        (8,    256,   30),
        (8,    512,   30),
        (8,   1024,   30),
        (8,   2048,   30),
        (8,   4096,   30),
        # Varying topk
        (8,   1024,   50),
        (8,   1024,  100),
        # Larger batch
        (32,   512,   30),
        (32,  1024,   30),
        (32,  2048,   30),
        (32,  4096,   30),
        # Stress: large pages (baseline caps at 4096, v2 can go beyond)
        (8,   8192,   30),
        (8,  16384,   30),
    ]

    reserved_bos = 1
    reserved_eos = 1
    dtype = torch.bfloat16

    # ---- Correctness check ----
    print("=" * 70)
    print("Correctness verification")
    print("=" * 70)
    for batch_size, num_pages, topk_val in configs:
        # Baseline (CUB) caps at 4096 — skip those for correctness check
        if num_pages > 4096:
            print(f"  B={batch_size:>4d}  pages={num_pages:>6d}  k={topk_val:>4d}  "
                  f"SKIP (baseline caps at 4096)")
            continue
        ok = verify_correctness(
            batch_size, num_pages, topk_val, reserved_bos, reserved_eos, dtype, device
        )
        status = "PASS" if ok else "FAIL"
        print(f"  B={batch_size:>4d}  pages={num_pages:>6d}  k={topk_val:>4d}  {status}")

    # ---- Latency benchmark ----
    print()
    print("=" * 70)
    print(f"Latency benchmark  (warmup={warmup}, iters={iters})")
    print("=" * 70)
    print(f"{'B':>5s}  {'pages':>7s}  {'k':>5s}  "
          f"{'CUB (us)':>10s}  {'Radix (us)':>11s}  {'speedup':>8s}")
    print("-" * 70)

    for batch_size, num_pages, topk_val in configs:
        (
            scores,
            dense_kv_indptr,
            sparse_kv_indptr,
            dense_kv_indices,
            sparse_kv_indices,
            _,
        ) = make_test_data(
            batch_size, num_pages, topk_val, reserved_bos, reserved_eos, dtype, device
        )

        # Benchmark radix selection (v2) — always works
        med_v2, _ = bench_kernel(
            vortex_torch_C.topk_output_v2,
            scores,
            dense_kv_indptr,
            sparse_kv_indptr,
            dense_kv_indices,
            sparse_kv_indices.clone(),
            batch_size,
            topk_val,
            reserved_bos,
            reserved_eos,
            num_pages,
            warmup,
            iters,
        )

        # Benchmark baseline (CUB) — only if within its 4096 page limit
        if num_pages <= 4096:
            med_baseline, _ = bench_kernel(
                vortex_torch_C.topk_output,
                scores,
                dense_kv_indptr,
                sparse_kv_indptr,
                dense_kv_indices,
                sparse_kv_indices.clone(),
                batch_size,
                topk_val,
                reserved_bos,
                reserved_eos,
                num_pages,
                warmup,
                iters,
            )
            speedup = med_baseline / med_v2 if med_v2 > 0 else float("inf")
            print(
                f"{batch_size:>5d}  {num_pages:>7d}  {topk_val:>5d}  "
                f"{med_baseline:>10.1f}  {med_v2:>11.1f}  {speedup:>7.2f}x"
            )
        else:
            print(
                f"{batch_size:>5d}  {num_pages:>7d}  {topk_val:>5d}  "
                f"{'N/A':>10s}  {med_v2:>11.1f}  {'N/A':>8s}"
            )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
