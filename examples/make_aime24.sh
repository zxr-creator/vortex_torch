#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/aime24.jsonl}"

python "${SCRIPT_DIR}/make_aime24.py" \
    --model "${MODEL}" \
    --output "${OUTPUT}"
