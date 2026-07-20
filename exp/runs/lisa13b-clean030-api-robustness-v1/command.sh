#!/usr/bin/env bash
set -euo pipefail

# Remote Linux GPU server only.

MODEL_VERSION="lisa13b-clean030-v1"
MODEL_PATH="./artifacts/lisa-safety-seg/lisa13b-clean030-v1/merged_hf"
CLIP_TOWER="./clip-vit-large-patch14"
SMOKE_IMAGE="./dataset/reason_seg/ReasonSeg/val/val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing.jpg"
SMOKE_PROMPT="标出未按规定佩戴安全帽的作业人员。"
OUTPUT_DIR="./exp/runs/lisa13b-clean030-api-robustness-v1/outputs"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_INDEX="0"
HOST="127.0.0.1"
PORT="8002"
QUEUE_TIMEOUT="0.15"
MAX_QUEUE_SIZE="1"
CLIENT_TIMEOUT="150"
REQUIRED_SHARED_PROCESS="VLLM::EngineCore"

if [ ! -f "$CLIP_TOWER/config.json" ] || [ ! -f "$CLIP_TOWER/preprocessor_config.json" ]; then
  HF_CACHE_ROOT="${HF_HOME:-${HOME}/.cache/huggingface}"
  CLIP_CONFIG="$(find "$HF_CACHE_ROOT" -path "*/models--openai--clip-vit-large-patch14/snapshots/*/config.json" -print -quit)"
  if [ -n "$CLIP_CONFIG" ]; then
    CLIP_TOWER="$(dirname "$CLIP_CONFIG")"
  fi
fi

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "Missing frozen model config: $MODEL_PATH/config.json" >&2
  exit 1
fi
if [ ! -f "$CLIP_TOWER/config.json" ] || [ ! -f "$CLIP_TOWER/preprocessor_config.json" ]; then
  echo "Missing local CLIP files under: $CLIP_TOWER" >&2
  exit 1
fi
if [ ! -f "$SMOKE_IMAGE" ]; then
  echo "Missing fixed smoke image: $SMOKE_IMAGE" >&2
  exit 1
fi
if [ -d "$OUTPUT_DIR" ] && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]; then
  echo "API robustness output directory is not empty: $OUTPUT_DIR" >&2
  echo "Preserve or move the existing outputs before rerunning." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the API robustness verification." >&2
  exit 1
fi

"$PYTHON_BIN" -m production.verify_api_robustness \
  --model-path "$MODEL_PATH" \
  --vision-tower "$CLIP_TOWER" \
  --image "$SMOKE_IMAGE" \
  --prompt "$SMOKE_PROMPT" \
  --output-dir "$OUTPUT_DIR" \
  --model-version "$MODEL_VERSION" \
  --gpu-index "$GPU_INDEX" \
  --host "$HOST" \
  --port "$PORT" \
  --queue-timeout "$QUEUE_TIMEOUT" \
  --max-queue-size "$MAX_QUEUE_SIZE" \
  --client-timeout "$CLIENT_TIMEOUT" \
  --require-existing-process-substring "$REQUIRED_SHARED_PROCESS"

sed -n '1,320p' "$OUTPUT_DIR/summary.md"
