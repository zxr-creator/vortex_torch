#!/usr/bin/env bash
# Sweep topk_val for gqa_quest_sparse_attention with fp8 KV cache,
# matching the OmniServe LServe dynamic_sparse_token_budget sweep.
# Run from the vortex_torch repo root:
#   bash examples/compare_omni.sh
export CUDA_VISIBLE_DEVICES=4
vortex_module_name="gqa_quest_sparse_attention"
model_name="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
data_path="${DATA_PATH:-aime24_llama8b.jsonl}"
trials="${TRIALS:-2}"
summary_dir="${SUMMARY_DIR:-summary_compare_omni}"

mkdir -p "${summary_dir}"


python verify_algo.py \
    --topk-val 29 \
    --vortex-module-name "${vortex_module_name}" \
    --model-name "${model_name}" \
    --data-path "${data_path}" \
    --trials "${trials}" \
    --kv-cache-dtype "auto" \
    --max-input-length 4096 \
    --generation-max-new-tokens 32768 \
    --mem 0.7 \
    --summary-dir "${summary_dir}" 2>&1 | tee "${summary_dir}/test_algo_20.log"