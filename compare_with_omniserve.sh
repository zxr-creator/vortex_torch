#!/usr/bin/env bash
# Side-by-side decode benchmark: vortex_torch (fp8 / bf16) vs OmniServe LServe (kv8 / kv16).
#
# Same model, batch, prefill length, decode budget on both stacks.
# Reports decode_ms_per_token + decode_tokens_per_sec for each (stack × kv-bits) combo.
#
# Usage:
#   bash compare_with_omniserve.sh            # runs all 4 combos (vortex 8/16, omniserve 8/16)
#   bash compare_with_omniserve.sh fp8        # vortex fp8 only
#   bash compare_with_omniserve.sh int8       # omniserve kv8 only
#   bash compare_with_omniserve.sh bf16       # vortex bf16 only
#   bash compare_with_omniserve.sh fp16kv     # omniserve kv16 only
#
# Env vars:
#   BATCH_SIZE     (default 1)
#   INPUT_LEN      (default 8192)
#   MAX_NEW_TOKENS (default 64)
#   MODEL_PATH_FP16 (default: /root/omniserve/models/llama-3-8b-Instruct-fp16)
#   MODEL_PATH_KV8  (default: /root/omniserve/models/Llama-3-8B-Instruct-Gradient-1048k-w8a8-per-channel-kv8-per-tensor)
#   ATTN_PATH       (default: /root/omniserve/attn_patterns/Llama-3-8B-Instruct-Gradient-1048k)

set -u -o pipefail

BATCH_SIZE="${BATCH_SIZE:-1}"
INPUT_LEN="${INPUT_LEN:-8192}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

MODEL_PATH_FP16="${MODEL_PATH_FP16:-/root/omniserve/models/llama-3-8b-Instruct-fp16}"
MODEL_PATH_KV8="${MODEL_PATH_KV8:-/root/omniserve/models/Llama-3-8B-Instruct-Gradient-1048k-w8a8-per-channel-kv8-per-tensor}"
ATTN_PATH="${ATTN_PATH:-/root/omniserve/attn_patterns/Llama-3-8B-Instruct-Gradient-1048k}"

VORTEX_PY="${VORTEX_PY:-/root/vortex/bin/python}"
OMNI_PY="${OMNI_PY:-/opt/conda/envs/OmniServe/bin/python}"

VORTEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMNI_DIR="${OMNI_DIR:-/root/omniserve}"

OUT_DIR="${OUT_DIR:-${VORTEX_DIR}/lserve_compare_logs}"
mkdir -p "$OUT_DIR"

filter="${1:-all}"

run_vortex() {
    local kv_bits="$1"
    local tag="vortex_kv${kv_bits}"
    local out="${OUT_DIR}/${tag}.log"
    echo
    echo "================ ${tag} ================"
    cd "$VORTEX_DIR"
    "$VORTEX_PY" profile_decode_lserve_setting.py \
        --kv-bits "$kv_bits" \
        --batch-size "$BATCH_SIZE" \
        --input-len "$INPUT_LEN" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --model-path "$MODEL_PATH_FP16" \
        2>&1 | tee "$out" | grep -E "^\[(setting|result)\]|Resolved attention|Cannot find|Error|Traceback" | tail -20
}

run_omniserve_kv8() {
    local out="${OUT_DIR}/omniserve_kv8.log"
    echo
    echo "================ omniserve_kv8 ================"
    cd "$OMNI_DIR"
    if [[ ! -d "$MODEL_PATH_KV8" ]]; then
        echo "[skip] MODEL_PATH_KV8=$MODEL_PATH_KV8 not present"
        return
    fi
    NUM_RETRIEVAL_GPU_PAGE_BLOCKS=3000 NUM_STREAMING_GPU_PAGE_BLOCKS=200 \
    "$OMNI_PY" tests/lserve_kv_smoke.py \
        --model "$MODEL_PATH_KV8" --quant-path "$MODEL_PATH_KV8" \
        --kv-bits 8 --precision w8a8kv8 --kv-quant-granularity per_tensor \
        --group-size -1 --ifb-mode \
        --max-num-batched-tokens 4195000 --max-num-seqs "$BATCH_SIZE" --omit-prompt \
        --chunk-prefill-size 32000 --multiblock-switch 2048 \
        --static-sparse-attn-load-dir "$ATTN_PATH" \
        --static-sparsity 0.5 --sparse-context-mode --sparse-decode-mode 1 \
        --ctx-sink-token 128 --ctx-local-token 8192 \
        --dec-sink-token 128 --dec-local-token 256 \
        --sub-chunk-per-block 4 --dynamic-sparse-token-budget 4096 \
        --selector-update-interval 4 \
        --test-length "$INPUT_LEN" --test-depth 0.5 \
        2>&1 | tee "$out" | grep -E "^\[smoke\]|kernel-invocations|sandwich|Dolores|USE INT8|Error|Traceback" | tail -20
}

run_omniserve_kv16() {
    local out="${OUT_DIR}/omniserve_kv16.log"
    echo
    echo "================ omniserve_kv16 ================"
    cd "$OMNI_DIR"
    if [[ ! -d "$MODEL_PATH_FP16" ]]; then
        echo "[skip] MODEL_PATH_FP16=$MODEL_PATH_FP16 not present"
        return
    fi
    NUM_RETRIEVAL_GPU_PAGE_BLOCKS=3000 NUM_STREAMING_GPU_PAGE_BLOCKS=200 \
    "$OMNI_PY" tests/lserve_kv_smoke.py \
        --model "$MODEL_PATH_FP16" --quant-path "$MODEL_PATH_FP16" \
        --kv-bits 16 --precision w16a16kv16 --kv-quant-granularity per_tensor \
        --group-size -1 --ifb-mode \
        --max-num-batched-tokens 4195000 --max-num-seqs "$BATCH_SIZE" --omit-prompt \
        --chunk-prefill-size 32000 --multiblock-switch 2048 \
        --static-sparse-attn-load-dir "$ATTN_PATH" \
        --static-sparsity 0.5 --sparse-context-mode --sparse-decode-mode 1 \
        --ctx-sink-token 128 --ctx-local-token 8192 \
        --dec-sink-token 128 --dec-local-token 256 \
        --sub-chunk-per-block 4 --dynamic-sparse-token-budget 4096 \
        --selector-update-interval 4 \
        --test-length "$INPUT_LEN" --test-depth 0.5 \
        2>&1 | tee "$out" | grep -E "^\[smoke\]|kernel-invocations|sandwich|Dolores|USE FP16/BF16|Error|Traceback" | tail -20
}

case "$filter" in
    all)
        run_omniserve_kv8
        run_omniserve_kv16
        run_vortex 8
        run_vortex 16
        ;;
    fp8)        run_vortex 8 ;;
    bf16)       run_vortex 16 ;;
    int8)       run_omniserve_kv8 ;;
    fp16kv)     run_omniserve_kv16 ;;
    *)          echo "Unknown filter: $filter"; exit 1 ;;
esac

echo
echo "Logs saved to ${OUT_DIR}/"
ls -la "${OUT_DIR}/"
