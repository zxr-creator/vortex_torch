#!/usr/bin/env bash
# Sweep the top-k CUDA kernels reported as 'sort_topk' (csrc/topk.cu),
# 'radix_topk' (csrc/topk_v2.cu), 'approx_radix_topk' (csrc/approx_topk.cu),
# and their autotuned remap variants ('radix_topk_remap',
# 'approx_radix_topk_remap'), across Qwen3 model sizes under four
# score-tensor distributions. The sglang_ori reference is intentionally
# excluded — its compile-time TopK and degenerate recall make it
# uninformative at this scale.
#
# Sweep (block_size = 1, so input_len == blocks_per_row):
#   batch_size = 128
#   (topk_val, blocks_per_row) pairs:
#     ( 32 ,   2048 )   #   2k tokens,  1.6% selected
#     ( 64 ,   2048 )   #   2k tokens,  3.1% selected
#     ( 128,   2048 )   #   2k tokens,  6.3% selected
#     ( 256,   2048 )   #   2k tokens, 12.5% selected
#     ( 2048,  32768)   #  32k tokens,  6.3% selected
#     ( 2048,  65536)   #  64k tokens,  3.1% selected
#     ( 2048, 131072)   # 128k tokens,  1.6% selected
#   distributions: uniform / normal / real (lognormal proxy) / bimodal
#
# Each measurement reports recall@k for k ∈ {32, 64, 128, topk_val}:
# the fraction of the kernel's top-k true blocks (torch.topk over the
# candidate region, excluding reserved BOS/EOS) that the kernel's selected
# set covers. recall@k is naturally bounded above by min(1, topk_val/k),
# since the kernel only emits topk_val candidates per row.
#
# Per-measurement records land in JSONL; a flat CSV-style summary is
# printed at the end and saved next to the JSONL.
#
# Run from anywhere:
#   bash examples/bench_topk_kernels.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATHS=(Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B)
MODEL_LABELS=(qwen3_0p6b   qwen3_1p7b    qwen3_4b     qwen3_8b)

BATCH_SIZE="${BATCH_SIZE:-128}"
BLOCK_SIZE="${BLOCK_SIZE:-1}"

# Parallel arrays: SWEEP_TOPK[i] paired with SWEEP_BPR[i].
# input_len = blocks_per_row * BLOCK_SIZE.
SWEEP_TOPK=(  32   64   128  256  2048  2048  2048  )
SWEEP_BPR=(   2048 2048 2048 2048 32768 65536 131072)

DISTRIBUTIONS=(uniform normal real bimodal)

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

humanize_len() {
  local L=$1
  if   (( L >= 1048576 )); then echo "$((L/1048576))M"
  elif (( L >= 1024 ));    then echo "$((L/1024))k"
  else                          echo "$L"
  fi
}

echo "[bench_topk_kernels] writing per-measurement records to ${JSONL}"
echo "[bench_topk_kernels] batch_size=${BATCH_SIZE}  block_size=${BLOCK_SIZE}"
echo "[bench_topk_kernels] tolerate_ratios=${TOLERATE_RATIOS}  distributions=${DISTRIBUTIONS[*]}"
echo "[bench_topk_kernels] dtype=${DTYPE}  warmup=${NUM_WARMUP}  iters=${NUM_ITERS}"
echo "[bench_topk_kernels] sweep ((topk, blocks_per_row) pairs):"
for j in "${!SWEEP_TOPK[@]}"; do
  ilen=$(( SWEEP_BPR[$j] * BLOCK_SIZE ))
  printf "    topk=%-5s blocks_per_row=%-7s input_len=%-9s (%s)\n" \
    "${SWEEP_TOPK[$j]}" "${SWEEP_BPR[$j]}" "${ilen}" "$(humanize_len "${ilen}")"
done
echo

for i in "${!MODEL_PATHS[@]}"; do
  model="${MODEL_PATHS[$i]}"
  mlabel="${MODEL_LABELS[$i]}"
  for dist in "${DISTRIBUTIONS[@]}"; do
    for j in "${!SWEEP_TOPK[@]}"; do
      topk="${SWEEP_TOPK[$j]}"
      bpr="${SWEEP_BPR[$j]}"
      ilen=$(( bpr * BLOCK_SIZE ))
      ilabel="$(humanize_len "${ilen}")"
      echo "==== ${mlabel}  bs=${BATCH_SIZE}  in=${ilabel} (${ilen})  topk=${topk}  dist=${dist} ===="
      "${PYTHON}" "${SCRIPT_DIR}/bench_topk_kernels.py" \
        --model-name "${model}" \
        --model-label "${mlabel}" \
        --batch-size "${BATCH_SIZE}" \
        --input-len "${ilen}" \
        --input-label "${ilabel}" \
        --block-size "${BLOCK_SIZE}" \
        --topk-vals "${topk}" \
        --tolerate-ratios "${TOLERATE_RATIOS}" \
        --reserved-bos "${RESERVED_BOS}" \
        --reserved-eos "${RESERVED_EOS}" \
        --dtype "${DTYPE}" \
        --distribution "${dist}" \
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
    "model", "batch_size", "distribution", "input_label", "input_len",
    "blocks_per_row", "topk_val", "per_row_sparse",
    "kernel", "tolerate_ratio",
    "num_kv_heads", "eff_batch_size",
    "mean_ms", "p50_ms", "p95_ms", "min_ms",
    "recall_at_32", "recall_at_64", "recall_at_128", "recall_at_topk",
    "mapping_mode", "mapping_power", "mapping_tag",
    "autotune_baseline_recall_at_topk",
]

rows = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

# Stable order: model, dist, input_len, topk, kernel.
def sort_key(r):
    return (
        r["model"], r.get("distribution", ""),
        r["batch_size"], r["input_len"], r["topk_val"], r["kernel"],
    )
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