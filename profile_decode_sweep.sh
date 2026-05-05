#!/usr/bin/env bash
# Sweep decode profiling over batch sizes and input lengths.
# Generate .nsys-rep + readable nsys stats reports for each setting.
# Run from the directory that contains profile_decode.py.

set -u -o pipefail

PYTHON_SCRIPT=${PYTHON_SCRIPT:-profile_decode.py}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
OUT_DIR=${OUT_DIR:-nsys_decode_reports}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
# Set ENABLE_VORTEX=0 to profile plain FlashInfer (debug only).
ENABLE_VORTEX=${ENABLE_VORTEX:-1}
# Set PROFILE_PREFILL=1 to also capture prefill-time cache-construction kernels.
PROFILE_PREFILL=${PROFILE_PREFILL:-0}
VORTEX_TOPK_VAL=${VORTEX_TOPK_VAL:-29}

vortex_flag=()
if [[ "${ENABLE_VORTEX}" == "0" ]]; then
  vortex_flag=(--no-vortex)
fi

prefill_flag=()
if [[ "${PROFILE_PREFILL}" == "1" ]]; then
  prefill_flag=(--profile-prefill)
fi

# Test matrix
MODEL_PATHS=(Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B)
MODEL_LABELS=(qwen3_0p6b qwen3_1p7b qwen3_4b qwen3_8b)
BATCH_SIZES=(4 16)
INPUT_LENS=(4096 8192 16384 32768)
INPUT_LABELS=(4k 8k 16k 32k)
# Sparse-attention flow names (vortex_module_name) to sweep.
# block_sparse: q_mean · centroid → topK
# quest:        max(q*kmax, q*kmin) → sum → max → topK
ATTN_MODULES=(block_sparse_attention gqa_quest_sparse_attention)
ATTN_LABELS=(block_sparse quest)

mkdir -p "${OUT_DIR}"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "ERROR: Cannot find ${PYTHON_SCRIPT}."
  echo "Run this script from the directory containing profile_decode.py,"
  echo "or set PYTHON_SCRIPT=/path/to/profile_decode.py"
  exit 1
fi

if ! command -v nsys >/dev/null 2>&1; then
  echo "ERROR: nsys not found in PATH."
  echo "Please install Nsight Systems or add nsys to PATH."
  exit 1
fi

MANIFEST="${OUT_DIR}/manifest_${RUN_TAG}.csv"
echo "run_tag,attn_module,attn_label,model_path,model_label,batch_size,input_len,max_new_tokens,nsys_rep,kern_sum,trace_report,nvtx_report,log_file,status" > "${MANIFEST}"

for aidx in "${!ATTN_MODULES[@]}"; do
  attn_module="${ATTN_MODULES[$aidx]}"
  attn_label="${ATTN_LABELS[$aidx]}"

  for midx in "${!MODEL_PATHS[@]}"; do
    model_path="${MODEL_PATHS[$midx]}"
    model_label="${MODEL_LABELS[$midx]}"

  for bs in "${BATCH_SIZES[@]}"; do
    for idx in "${!INPUT_LENS[@]}"; do
      input_len="${INPUT_LENS[$idx]}"
      input_label="${INPUT_LABELS[$idx]}"

      name="decode_${RUN_TAG}_${attn_label}_${model_label}_bs${bs}_in${input_label}_new${MAX_NEW_TOKENS}"
      report_base="${OUT_DIR}/${name}"
      log_file="${report_base}.log"

      nsys_rep="${report_base}.nsys-rep"
      kern_sum_report="${report_base}_cuda_gpu_kern_sum.txt"
      trace_report="${report_base}_cuda_gpu_trace.txt"
      nvtx_report="${report_base}_nvtx_sum.txt"

      echo "============================================================"
      echo "Profiling setting:"
      echo "  attn_module   = ${attn_module} (${attn_label})"
      echo "  model         = ${model_path} (${model_label})"
      echo "  batch_size    = ${bs}"
      echo "  input_len     = ${input_len} (${input_label})"
      echo "  max_new_tokens= ${MAX_NEW_TOKENS}"
      echo "  nsys report   = ${nsys_rep}"
      echo "  log file      = ${log_file}"
      echo "============================================================"

      nsys profile \
        -o "${report_base}" \
        --trace=cuda,nvtx \
        --sample=none \
        --cpuctxsw=none \
        --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --force-overwrite true \
        python "${PYTHON_SCRIPT}" \
          --model-path "${model_path}" \
          --batch-size "${bs}" \
          --max-new-tokens "${MAX_NEW_TOKENS}" \
          --input-len "${input_len}" \
          --vortex-topk-val "${VORTEX_TOPK_VAL}" \
          --vortex-module-name "${attn_module}" \
          "${vortex_flag[@]}" \
          "${prefill_flag[@]}" \
        > "${log_file}" 2>&1

      status=$?

      if [[ ${status} -eq 0 && -f "${nsys_rep}" ]]; then
        echo "Generating readable nsys stats reports..."

        # Kernel aggregated summary
        nsys stats \
          --report cuda_gpu_kern_sum \
          --format table \
          --force-overwrite true \
          "${nsys_rep}" \
          > "${kern_sum_report}" 2>> "${log_file}"

        # Kernel timeline trace
        nsys stats \
          --report cuda_gpu_trace \
          --format table \
          --force-overwrite true \
          "${nsys_rep}" \
          > "${trace_report}" 2>> "${log_file}"

        # NVTX summary, useful for decode_loop / decode_step_x ranges
        nsys stats \
          --report nvtx_sum \
          --format table \
          --force-overwrite true \
          "${nsys_rep}" \
          > "${nvtx_report}" 2>> "${log_file}"

        echo "DONE:"
        echo "  ${nsys_rep}"
        echo "  ${kern_sum_report}"
        echo "  ${trace_report}"
        echo "  ${nvtx_report}"
      else
        echo "FAILED: attn=${attn_label}, model=${model_label}, bs=${bs}, input_len=${input_label}; see ${log_file}" >&2
      fi

      echo "${RUN_TAG},${attn_module},${attn_label},${model_path},${model_label},${bs},${input_len},${MAX_NEW_TOKENS},${nsys_rep},${kern_sum_report},${trace_report},${nvtx_report},${log_file},${status}" >> "${MANIFEST}"
    done
  done
  done
done

echo "============================================================"
echo "Sweep complete."
echo "Manifest: ${MANIFEST}"
echo "Reports are under: ${OUT_DIR}/"
echo "============================================================"