#!/usr/bin/env bash
# Sweep RULER accuracy across 4 configurations:
#   dense_bf16  : full attention, bf16 KV cache
#   dense_fp8   : full attention, fp8 KV cache
#   quest_bf16  : gqa_quest_sparse_attention, bf16 KV cache
#   quest_fp8   : gqa_quest_sparse_attention, fp8 KV cache
# Run from anywhere:
#   bash examples/run_ruler.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/data/datasets/xinrui/uv_env/vortex/bin/python}"

model_name="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
summary_dir="${SUMMARY_DIR:-${SCRIPT_DIR}/summary_ruler}"
mkdir -p "${summary_dir}"

# tag : module_name : kv_cache_dtype
configs=(
  "dense_bf16:full_attention:auto"
  "dense_fp8:full_attention:fp8_e4m3"
  "quest_bf16:gqa_quest_sparse_attention:auto"
  "quest_fp8:gqa_quest_sparse_attention:fp8_e4m3"
)

declare -A acc_map
for cfg in "${configs[@]}"; do
  IFS=':' read -r tag module kv_dtype <<< "${cfg}"
  log_file="${summary_dir}/${tag}.log"
  echo "[run_ruler] ${tag}  module=${module}  kv_cache_dtype=${kv_dtype}"
  "${PYTHON}" "${SCRIPT_DIR}/run_ruler.py" \
    --model-name "${model_name}" \
    --vortex-module-name "${module}" \
    --kv-cache-dtype "${kv_dtype}" \
    --max-new-tokens 32768 \
    --mem 0.7 \
    --tag "${tag}" 2>&1 | tee "${log_file}"
  acc=$(grep -oE "Ruler Accuracy: [0-9.]+%" "${log_file}" | tail -1 | awk '{print $3}')
  acc_map[${tag}]="${acc:-N/A}"
done

echo
echo "================ RULER accuracy summary ================"
{
  printf "%-14s %s\n" "config" "accuracy"
  for cfg in "${configs[@]}"; do
    IFS=':' read -r tag _ _ <<< "${cfg}"
    printf "%-14s %s\n" "${tag}" "${acc_map[${tag}]}"
  done
} | tee "${summary_dir}/summary.txt"
