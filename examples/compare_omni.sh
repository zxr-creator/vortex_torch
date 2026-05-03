#!/usr/bin/env bash
# Sweep topk_val for gqa_quest_sparse_attention with fp8 KV cache,
# matching the OmniServe LServe dynamic_sparse_token_budget sweep.
# Run from the vortex_torch repo root:
#   bash examples/compare_omni.sh
topk_vals=(29 61 93 125 157 189 221 253)
export CUDA_VISIBLE_DEVICES=6
vortex_module_name="gqa_quest_sparse_attention"
model_name="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
data_path="${DATA_PATH:-aime24_llama8b.jsonl}"
trials="${TRIALS:-2}"
summary_dir="${SUMMARY_DIR:-summary_compare_omni}"

mkdir -p "${summary_dir}"

for topk_val in "${topk_vals[@]}"; do
  echo "[compare_omni] topk_val=${topk_val}"
  log_file="${summary_dir}/topk_${topk_val}.log"
  python verify_algo.py \
    --topk-val "${topk_val}" \
    --topk-ratio 0 \
    --vortex-module-name "${vortex_module_name}" \
    --model-name "${model_name}" \
    --data-path "${data_path}" \
    --trials "${trials}" \
    --kv-cache-dtype "fp8_e4m3" \
    --max-input-length 4096 \
    --generation-max-new-tokens 8192 \
    --mem 0.7 \
    --summary-dir "${summary_dir}/topk_${topk_val}" 2>&1 | tee "${log_file}"
done
