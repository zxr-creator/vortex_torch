#!/usr/bin/env bash
# Driver for benchmark_topk_kernels.py — sweeps the three top-K page
# selection CUDA kernels (topk_output, topk_output_v2, approx_topk_output)
# across K ∈ {30, 2048} × batch ∈ {1,2,4,8,16} × length ∈ {2k,4k,8k,16k,32k}
# in a single invocation, then prints a per-K best-kernel summary.
#
# Usage:
#   ./benchmark_topk_kernels.sh                       # default sweep
#   ./benchmark_topk_kernels.sh --dtype fp32          # forward extra args to .py
#   TOPKS="30 64 2048" ITERS=500 ./benchmark_topk_kernels.sh
#
# Settings the kernels can't run are auto-skipped:
#   * K >= L            → labelled `skip(K>=L)` (kernel short-circuits)
#   * topk_output, L>4k → labelled `n/a`        (csrc/topk.cu rejects it)
#
# The host has no GPU; run this from a GPU-enabled node (interactive
# salloc, srun, or wrap the call in sbatch).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/benchmark_topk_kernels.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TOPKS="${TOPKS:-30 2048}"
DTYPE="${DTYPE:-bf16}"
ITERS="${ITERS:-200}"
WARMUP="${WARMUP:-20}"
BATCHES="${BATCHES:-1 2 4 8 16}"
LENGTHS="${LENGTHS:-2048 4096 8192 16384 32768}"
TOLERATE_RATIOS="${TOLERATE_RATIOS:-0.0 0.1}"

LOG_DIR="${SCRIPT_DIR}/logs/topk_bench"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
KS_TAG="$(echo "${TOPKS}" | tr ' ' '_')"
LOG_FILE="${LOG_DIR}/topk_bench_${DTYPE}_K${KS_TAG}_${TS}.log"

echo "Logging to: ${LOG_FILE}"
echo "Device    : $(${PYTHON_BIN} -c 'import torch;print(torch.cuda.get_device_name() if torch.cuda.is_available() else "NO CUDA")' 2>/dev/null)"

# shellcheck disable=SC2086  # word-splitting is intentional for the sweep lists
"${PYTHON_BIN}" "${PY_SCRIPT}" \
    --topks ${TOPKS} \
    --dtype "${DTYPE}" \
    --iters "${ITERS}" \
    --warmup "${WARMUP}" \
    --batches ${BATCHES} \
    --lengths ${LENGTHS} \
    --tolerate-ratios ${TOLERATE_RATIOS} \
    "$@" 2>&1 | tee "${LOG_FILE}"
