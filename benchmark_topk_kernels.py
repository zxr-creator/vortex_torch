"""Benchmark the three top-K page-selection CUDA kernels exposed by
vortex_torch_C — `topk_output` (csrc/topk.cu, cub::BlockRadixSort),
`topk_output_v2` (csrc/topk_v2.cu, 2-stage 8-bit radix select with
SELECT32 small-K fast path), and `approx_topk_output`
(csrc/approx_topk.cu, single/two-pass 8-bit radix approximate top-K
with the same SELECT32 fast path).

All three share the same row-segmented layout:
  x                  : [B*L]   bf16/fp32 score, one row of length L per batch entry
  dense_kv_indptr    : [B+1]   int32, row offsets into x          → [0, L, 2L, ...]
  dense_kv_indices   : [B*L]   int32, page-id remap (identity here)
  sparse_kv_indptr   : [B+1]   int32, output offsets               → [0, K, 2K, ...]
  sparse_kv_indices  : [B*K]   int32, output buffer
  eff_batch_size     : int     = B (we use num_kv_heads = 1)
  reserved_bos / eos : int     = 0 here
  max_num_pages      : int     = L
  (approx only) tolerate_ratio : float (we sweep a couple of values)

Sweep: batch ∈ {1,2,4,8,16}, length ∈ {2k,4k,8k,16k,32k}.
`topk_output` is skipped above L=4096 (kernel rejects it).

Reported metric is median latency (us) over `--iters` timed runs after
`--warmup` warm-ups, using cudaEvent timing.
"""

import argparse
import statistics
import sys
from typing import Optional

import torch

try:
    from vortex_torch_C import topk_output, topk_output_v2, approx_topk_output
except ImportError as e:
    sys.stderr.write(
        "Failed to import vortex_torch_C — build the extension first "
        "(`pip install -e .` from the repo root). Underlying error: "
        f"{e!r}\n"
    )
    sys.exit(1)


def make_inputs(batch: int, length: int, topk: int, dtype: torch.dtype, device: str):
    total = batch * length
    # Random scores; bf16/fp32 both supported. Use a wide range so byte-0
    # histograms are well-spread (not all in one bin).
    scores = torch.randn(total, dtype=dtype, device=device)

    arange_b1 = torch.arange(batch + 1, device=device, dtype=torch.int32)
    dense_kv_indptr = arange_b1 * length
    sparse_kv_indptr = arange_b1 * topk

    # Identity remap — the kernel writes idx_blk[selected_local_rank].
    dense_kv_indices = torch.arange(total, device=device, dtype=torch.int32)
    sparse_kv_indices = torch.empty(batch * topk, device=device, dtype=torch.int32)

    return (
        scores,
        dense_kv_indptr,
        sparse_kv_indptr,
        dense_kv_indices,
        sparse_kv_indices,
    )


def time_kernel(fn, iters: int, warmup: int) -> float:
    """Return median latency in microseconds across `iters` calls."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Per-call timing — each kernel is short, so individual events are
    # noisy at low B/L, but median over many iters smooths it out and
    # avoids amortizing one slow launch over a batch loop.
    samples_us = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start_evt.record()
        fn()
        end_evt.record()
        end_evt.synchronize()
        samples_us.append(start_evt.elapsed_time(end_evt) * 1e3)  # ms → us
    return statistics.median(samples_us)


def bench_one(
    batch: int,
    length: int,
    topk: int,
    dtype: torch.dtype,
    iters: int,
    warmup: int,
    tolerate_ratios: list[float],
) -> dict[str, Optional[float]]:
    device = "cuda"
    (scores, d_ptr, s_ptr, d_idx, s_idx) = make_inputs(batch, length, topk, dtype, device)

    results: dict[str, Optional[float]] = {}

    # ---- topk (csrc/topk.cu) — only supports max_num_pages <= 4096.
    if length <= 4096:
        def run_topk():
            topk_output(scores, d_ptr, s_ptr, d_idx, s_idx, batch, 0, 0, length)
        results["topk"] = time_kernel(run_topk, iters, warmup)
    else:
        results["topk"] = None

    # ---- topk_v2 (csrc/topk_v2.cu)
    def run_topk_v2():
        topk_output_v2(scores, d_ptr, s_ptr, d_idx, s_idx, batch, 0, 0, length)
    results["topk_v2"] = time_kernel(run_topk_v2, iters, warmup)

    # ---- approx_topk (csrc/approx_topk.cu) at each tolerate ratio
    for tol in tolerate_ratios:
        def run_approx(tol=tol):
            approx_topk_output(
                scores, d_ptr, s_ptr, d_idx, s_idx, batch, 0, 0, length, tol
            )
        results[f"approx_topk(tol={tol})"] = time_kernel(run_approx, iters, warmup)

    return results


def fmt(us: Optional[float]) -> str:
    return "    n/a" if us is None else f"{us:7.2f}"


def run_sweep(
    topk: int,
    batches: list[int],
    lengths: list[int],
    dtype: torch.dtype,
    iters: int,
    warmup: int,
    tolerate_ratios: list[float],
) -> dict[tuple[int, int], dict[str, Optional[float]]]:
    """Run the (B, L) sweep for a single K and return a results map.

    Skipped cells (K >= L) are stored as None for every column so the
    summary can still account for them uniformly.
    """
    cols = ["topk", "topk_v2"] + [
        f"approx_topk(tol={t})" for t in tolerate_ratios
    ]

    print(f"\n{'='*30} K = {topk} {'='*30}")
    header = f"{'B':>3} {'L':>6}  " + "  ".join(f"{c:>22}" for c in cols)
    print(header)
    print("-" * len(header))

    table: dict[tuple[int, int], dict[str, Optional[float]]] = {}
    for batch in batches:
        for length in lengths:
            if topk >= length:
                row = f"{batch:>3} {length:>6}  " + "  ".join(
                    f"{'skip(K>=L)':>22}" for _ in cols
                )
                print(row, flush=True)
                table[(batch, length)] = {c: None for c in cols}
                continue
            res = bench_one(
                batch, length, topk, dtype, iters, warmup, tolerate_ratios,
            )
            row = f"{batch:>3} {length:>6}  " + "  ".join(
                f"{fmt(res[c]):>22}" for c in cols
            )
            print(row, flush=True)
            table[(batch, length)] = res
        print()
    return table


def print_summary(
    all_results: dict[int, dict[tuple[int, int], dict[str, Optional[float]]]],
    cols: list[str],
):
    """Per-K summary: best kernel per (B,L) row, plus the win-count
    breakdown across the whole sweep."""
    print("=" * 78)
    print("SUMMARY — fastest kernel per (K, B, L)")
    print("=" * 78)
    win_counts: dict[int, dict[str, int]] = {}
    for topk, table in all_results.items():
        win_counts[topk] = {c: 0 for c in cols}
        print(f"\n[K = {topk}]")
        print(f"{'B':>3} {'L':>6}  {'best':>22}  {'us':>8}  {'speedup vs topk_v2':>22}")
        print("-" * 70)
        for (batch, length), res in table.items():
            valid = {k: v for k, v in res.items() if v is not None}
            if not valid:
                print(f"{batch:>3} {length:>6}  {'(skipped)':>22}")
                continue
            best_name = min(valid, key=valid.get)
            best_us = valid[best_name]
            win_counts[topk][best_name] += 1
            v2 = res.get("topk_v2")
            speedup = f"{v2 / best_us:6.2f}x" if v2 else "    n/a"
            print(f"{batch:>3} {length:>6}  {best_name:>22}  {best_us:>8.2f}  {speedup:>22}")

    print()
    print("=" * 78)
    print("Win counts (cells where each kernel was fastest)")
    print("=" * 78)
    print(f"{'K':>6}  " + "  ".join(f"{c:>22}" for c in cols))
    for topk, counts in win_counts.items():
        print(f"{topk:>6}  " + "  ".join(f"{counts[c]:>22d}" for c in cols))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topks", type=int, nargs="+", default=[30, 2048],
                   help="One or more K values to sweep. Default: 30 2048.")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--tolerate-ratios", type=float, nargs="+",
                   default=[0.0, 0.1])
    p.add_argument("--batches", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16])
    p.add_argument("--lengths", type=int, nargs="+",
                   default=[2048, 4096, 8192, 16384, 32768])
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.stderr.write("CUDA not available; this benchmark needs a GPU.\n")
        sys.exit(1)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"Device   : {torch.cuda.get_device_name()}")
    print(f"dtype    : {args.dtype}")
    print(f"top-Ks   : {args.topks}")
    print(f"batches  : {args.batches}")
    print(f"lengths  : {args.lengths}")
    print(f"iters    : {args.iters} (warmup {args.warmup})")
    print(f"tol      : {args.tolerate_ratios}")

    cols = ["topk", "topk_v2"] + [
        f"approx_topk(tol={t})" for t in args.tolerate_ratios
    ]

    all_results: dict[int, dict[tuple[int, int], dict[str, Optional[float]]]] = {}
    for topk in args.topks:
        all_results[topk] = run_sweep(
            topk, args.batches, args.lengths, dtype,
            args.iters, args.warmup, args.tolerate_ratios,
        )

    print_summary(all_results, cols)


if __name__ == "__main__":
    main()
