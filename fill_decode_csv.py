"""Fill /root/vortex_torch/example.csv from the nsys CUDA-GPU traces in
/root/vortex_torch/nsys_decode_reports/.

Per-cell metric definitions (durations / latencies in milliseconds):

  indexer          duration of the FIRST `indexSelectSmallIndex*` kernel
  cache            duration of the FIRST kernel whose name starts with
                   `vtxgraphcachepool_cache`
  attention        duration of the SECOND `BatchDecodeWithPagedKVCacheKernel`
                   (= first sparse attention; layer 0 is dense)
  attention_dense  duration of the FIRST `BatchDecodeWithPagedKVCacheKernel`

  batch_dense      SUM of durations of every kernel from the 1st
                   `flashinfer::norm::RMSNormKernel` up to and INCLUDING
                   the 1st `BatchDecodeWithPagedKVCacheKernel`
                   (the dense window).

  batch            SUM of durations of every kernel from the 1st
                   `PersistentVariableLengthMergeStatesKernel` up to and
                   INCLUDING the 2nd `BatchDecodeWithPagedKVCacheKernel`
                   (the sparse window).

`batch_dense > batch` reflects the wall-time savings of replacing dense
attention with sparse attention; the dense BatchDecode kernel alone
typically dominates `batch_dense`.

Failed/missing data points are written as `-`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

REPORT_DIR = Path("/root/vortex_torch/nsys_decode_reports")
CSV_IN     = Path("/root/vortex_torch/example.csv")
CSV_OUT    = Path("/root/vortex_torch/example.csv")
TS_PREFIX  = "decode_20260505_022516"

METHOD_TO_TAG = {
    "Block Sparse": "block_sparse",
    "Quest":        "quest",
}
MODEL_TO_TAG = {
    "Qwen3-0.6B": "qwen3_0p6b",
    "Qwen3-1.7B": "qwen3_1p7b",
    "Qwen3-4B":   "qwen3_4b",
    "Qwen3-8B":   "qwen3_8b",
}
INPUT_LEN_TO_LABEL = {
    4096:  "4k",
    8192:  "8k",
    16384: "16k",
    32768: "32k",
}


def trace_path(method: str, model: str, batch_size: int, input_len: int) -> Path:
    name = (
        f"{TS_PREFIX}_{METHOD_TO_TAG[method]}_{MODEL_TO_TAG[model]}"
        f"_bs{batch_size}_in{INPUT_LEN_TO_LABEL[input_len]}_new64"
        f"_cuda_gpu_trace.txt"
    )
    return REPORT_DIR / name


def parse_trace(path: Path):
    """Yield (start_ns, duration_ns, name) for each kernel row in the trace.

    Trace rows start with `|` and have the form:
       | Start | Dur | CorrId | Grd... | Blk... | ... | Name |
    The Name column is the last `|`-delimited field. Truncated names ending in
    `…` are still matchable by prefix/substring on the visible portion.
    """
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) < 21:
                continue
            try:
                start = int(parts[0])
                dur   = int(parts[1])
            except ValueError:
                continue
            yield start, dur, parts[-1]


def collect_metrics(path: Path) -> Optional[dict]:
    """Walk the trace once, find anchor kernels, return ms metrics."""
    if not path.is_file():
        return None

    rows = list(parse_trace(path))

    def find_first_idx(pred):
        for i, (s, d, n) in enumerate(rows):
            if pred(n):
                return i
        return None

    def find_nth_idx(pred, n):
        seen = 0
        for i, (s, d, nm) in enumerate(rows):
            if pred(nm):
                seen += 1
                if seen == n:
                    return i
        return None

    is_index = lambda n: "indexSelectSmallIndex" in n
    is_cache = lambda n: n.startswith("vtxgraphcachepool_cache")
    is_attn  = lambda n: "BatchDecodeWithPagedKVCacheKernel" in n
    is_merge = lambda n: "PersistentVariableLengthMergeStatesKernel" in n
    # Plain RMSNorm only — substring excludes `FusedAddRMSNormKernel`
    # because the latter has `FusedAdd` between `norm::` and `RMSNormKernel`.
    is_rmsnorm = lambda n: "flashinfer::norm::RMSNormKernel" in n

    i_index1   = find_first_idx(is_index)
    i_cache1   = find_first_idx(is_cache)
    i_attn1    = find_nth_idx(is_attn, 1)
    i_attn2    = find_nth_idx(is_attn, 2)
    i_merge1   = find_first_idx(is_merge)
    i_rmsnorm1 = find_first_idx(is_rmsnorm)

    def dur_at(i):
        return None if i is None else rows[i][1]

    def window_sum(start_idx, end_idx):
        if start_idx is None or end_idx is None or start_idx > end_idx:
            return None
        return sum(rows[i][1] for i in range(start_idx, end_idx + 1))

    indexer_ns      = dur_at(i_index1)
    cache_ns        = dur_at(i_cache1)
    attn_ns         = dur_at(i_attn2)
    attn_dense_ns   = dur_at(i_attn1)
    batch_dense_ns  = window_sum(i_rmsnorm1, i_attn1)  # dense window
    batch_ns        = window_sum(i_merge1, i_attn2)    # sparse window

    def to_ms(ns):
        return None if ns is None else ns / 1e6

    return {
        "indexer":         to_ms(indexer_ns),
        "cache":           to_ms(cache_ns),
        "attention":       to_ms(attn_ns),
        "attention_dense": to_ms(attn_dense_ns),
        "batch":           to_ms(batch_ns),
        "batch_dense":     to_ms(batch_dense_ns),
    }


def fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.4f}"


def main() -> None:
    rows_in: list[dict] = []
    with CSV_IN.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            rows_in.append(r)

    assert fieldnames == [
        "batch_size", "input_len", "model", "method",
        "indexer", "cache", "attention", "attention_dense",
        "batch", "batch_dense", "speedup",
    ], fieldnames

    n_filled = 0
    n_missing_file = 0
    n_partial = 0

    for r in rows_in:
        bs    = int(r["batch_size"])
        ilen  = int(r["input_len"])
        model = r["model"]
        meth  = r["method"]
        path  = trace_path(meth, model, bs, ilen)

        metrics = collect_metrics(path)
        if metrics is None:
            n_missing_file += 1
            for col in ("indexer", "cache", "attention",
                        "attention_dense", "batch", "batch_dense"):
                r[col] = "-"
            continue

        if any(v is None for v in metrics.values()):
            n_partial += 1
        else:
            n_filled += 1

        r["indexer"]         = fmt(metrics["indexer"])
        r["cache"]           = fmt(metrics["cache"])
        r["attention"]       = fmt(metrics["attention"])
        r["attention_dense"] = fmt(metrics["attention_dense"])
        r["batch"]           = fmt(metrics["batch"])
        r["batch_dense"]     = fmt(metrics["batch_dense"])

    with CSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_in:
            writer.writerow(r)

    print(f"[fill_decode_csv] wrote {CSV_OUT}")
    print(f"[fill_decode_csv]   rows fully filled : {n_filled}")
    print(f"[fill_decode_csv]   rows missing trace: {n_missing_file}")
    print(f"[fill_decode_csv]   rows partial      : {n_partial}")


if __name__ == "__main__":
    main()
