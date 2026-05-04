#!/usr/bin/env bash
# Sweep the three top-k CUDA kernels (topk_output / topk_output_v2 /
# approx_topk_output) across Qwen3 model sizes, batch sizes, and input
# lengths. Per-measurement records land in JSONL; a flat CSV-style
# summary is printed at the end and saved next to the JSONL.
#
# Run from anywhere:
#   bash examples/bench_topk_kernels.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATHS=(Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B)
MODEL_LABELS=(qwen3_0p6b   qwen3_1p7b    qwen3_4b     qwen3_8b)
BATCH_SIZES=(4 16)
INPUT_LENS=(4096 8192 16384 32768)
INPUT_LABELS=(4k   8k   16k   32k)

BLOCK_SIZE="${BLOCK_SIZE:-16}"
TOPK_VALS="${TOPK_VALS:-29,61,125,253}"
TOLERATE_RATIOS="${TOLERATE_RATIOS:-0.0,0.05,0.1}"
RESERVED_BOS="${RESERVED_BOS:-1}"
RESERVED_EOS="${RESERVED_EOS:-2}"
DTYPE="${DTYPE:-bfloat16}"
NUM_WARMUP="${NUM_WARMUP:-20}"
NUM_ITERS="${NUM_ITERS:-100}"

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/summary_topk_bench}"
mkdir -p "${OUT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
JSONL="${OUT_DIR}/topk_bench_${TS}.jsonl"
SUMMARY="${OUT_DIR}/topk_bench_${TS}.csv"

echo "[bench_topk_kernels] writing per-measurement records to ${JSONL}"
echo "[bench_topk_kernels] block_size=${BLOCK_SIZE}  topk_vals=${TOPK_VALS}  tolerate_ratios=${TOLERATE_RATIOS}"
echo "[bench_topk_kernels] dtype=${DTYPE}  warmup=${NUM_WARMUP}  iters=${NUM_ITERS}"
echo

for i in "${!MODEL_PATHS[@]}"; do
  model="${MODEL_PATHS[$i]}"
  mlabel="${MODEL_LABELS[$i]}"
  for bs in "${BATCH_SIZES[@]}"; do
    for j in "${!INPUT_LENS[@]}"; do
      ilen="${INPUT_LENS[$j]}"
      ilabel="${INPUT_LABELS[$j]}"
      echo "==== ${mlabel}  bs=${bs}  in=${ilabel} (${ilen}) ===="
      "${PYTHON}" "${SCRIPT_DIR}/bench_topk_kernels.py" \
        --model-name "${model}" \
        --model-label "${mlabel}" \
        --batch-size "${bs}" \
        --input-len "${ilen}" \
        --input-label "${ilabel}" \
        --block-size "${BLOCK_SIZE}" \
        --topk-vals "${TOPK_VALS}" \
        --tolerate-ratios "${TOLERATE_RATIOS}" \
        --reserved-bos "${RESERVED_BOS}" \
        --reserved-eos "${RESERVED_EOS}" \
        --dtype "${DTYPE}" \
        --num-warmup "${NUM_WARMUP}" \
        --num-iters "${NUM_ITERS}" \
        --output-jsonl "${JSONL}"
      echo
    done
  done
done

echo "[bench_topk_kernels] aggregating ${JSONL} -> ${SUMMARY}"
"${PYTHON}" - <<PY
import csv, json, os, sys

jsonl_path = "${JSONL}"
csv_path   = "${SUMMARY}"

cols = [
    "model", "batch_size", "input_label", "input_len", "topk_val",
    "kernel", "tolerate_ratio",
    "num_kv_heads", "eff_batch_size", "blocks_per_row", "per_row_sparse",
    "mean_ms", "p50_ms", "p95_ms", "min_ms",
]

rows = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

# Stable order: model, batch, input_len, topk, kernel.
def sort_key(r):
    return (r["model"], r["batch_size"], r["input_len"], r["topk_val"], r["kernel"])
rows.sort(key=sort_key)

with open(csv_path, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)

# Also print a compact table to stdout.
widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in cols}
header = " ".join(c.ljust(widths[c]) for c in cols)
print(header)
print("-" * len(header))
for r in rows:
    print(" ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
PY

echo
echo "[bench_topk_kernels] done. JSONL=${JSONL}  CSV=${SUMMARY}"
