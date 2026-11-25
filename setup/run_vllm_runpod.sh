#!/usr/bin/env bash
# ==============================================================================
# [RunPod Native] Launch vLLM OpenAI-Compatible Server Directly (No Nested Docker)
# ==============================================================================
# RunPod GPU pods already run inside a container; launch `vllm serve` directly
# and persist Hugging Face weights on the attached 40 GB Network Volume (`/workspace`).
#
# Usage:
#   bash setup/run_vllm_runpod.sh [MODEL_ID] [PORT] [MAX_MODEL_LEN] [PREFIX_CACHE] [LORA_MODULES]
#
# Examples:
#   # Standard RQ1 / Pilot (prefix caching ON):
#   bash setup/run_vllm_runpod.sh Qwen/Qwen3-1.7B 8000 12288 on
#
#   # RQ4 H4 Prefix-Cache OFF Probe:
#   bash setup/run_vllm_runpod.sh Qwen/Qwen3-1.7B 8000 12288 off
#
#   # Milestone 7 LoRA Evaluation:
#   bash setup/run_vllm_runpod.sh Qwen/Qwen3-1.7B 8000 12288 on "lora_r32=/workspace/adapters/qwen3_1.7b_r32_100pct"
# ==============================================================================

set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen3-1.7B}"
PORT="${2:-8000}"
MAX_MODEL_LEN="${3:-12288}"
PREFIX_CACHE="${4:-on}"
LORA_MODULES="${5:-}"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "${HF_HOME}"

EXTRA_ARGS=()
if [[ "${PREFIX_CACHE}" == "on" ]]; then
    EXTRA_ARGS+=(--enable-prefix-caching)
else
    EXTRA_ARGS+=(--no-enable-prefix-caching)
fi

if [[ -n "${LORA_MODULES}" ]]; then
    EXTRA_ARGS+=(--enable-lora --max-lora-rank 64 --lora-modules "${LORA_MODULES}")
fi

echo "============================================================"
echo " Launching RunPod Native vLLM Server"
echo " Model          : ${MODEL_ID}"
echo " Port           : ${PORT}"
echo " Max Model Len  : ${MAX_MODEL_LEN} (Decision D1: P + 8k B_alloc + 512)"
echo " Prefix Caching : ${PREFIX_CACHE}"
echo " LoRA Modules   : ${LORA_MODULES:-none}"
echo " HF_HOME        : ${HF_HOME}"
echo "============================================================"

exec vllm serve "${MODEL_ID}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    "${EXTRA_ARGS[@]}"
