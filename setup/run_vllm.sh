#!/usr/bin/env bash
# ==============================================================================
# [CBA-4] Launch vLLM OpenAI-Compatible Server for Qwen3-1.7B / Qwen3-4B
# ==============================================================================
# Usage:
#   bash setup/run_vllm.sh [MODEL_ID] [PORT] [VLLM_TAG]
# Example:
#   bash setup/run_vllm.sh Qwen/Qwen3-1.7B 8000 v0.8.5
# ==============================================================================

set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen3-1.7B}"
PORT="${2:-8000}"
VLLM_TAG="${3:-v0.8.5}"
CONTAINER_NAME="vllm-cba-server"

echo "============================================================"
echo " Launching vLLM Server"
echo " Model          : ${MODEL_ID}"
echo " Port           : ${PORT}"
echo " Docker Image   : vllm/vllm-openai:${VLLM_TAG}"
echo " Max Model Len  : 8192 (Supports 2k / 4k / 8k budgets)"
echo " Prefix Caching : Enabled (--enable-prefix-caching)"
echo " Tool Parser    : hermes (--enable-auto-tool-choice)"
echo "============================================================"

# Stop existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[INFO] Removing existing container ${CONTAINER_NAME}..."
    docker rm -f "${CONTAINER_NAME}"
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --gpus all \
  --ipc=host \
  -p "${PORT}:8000" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  "vllm/vllm-openai:${VLLM_TAG}" \
  --model "${MODEL_ID}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-prefix-caching \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90

echo "[INFO] Container started. Follow logs with: docker logs -f ${CONTAINER_NAME}"
echo "[INFO] Waiting for http://localhost:${PORT}/health ..."
