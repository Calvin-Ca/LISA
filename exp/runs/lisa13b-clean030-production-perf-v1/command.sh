#!/usr/bin/env bash
set -euo pipefail

# Remote Linux GPU server only.

MODEL_VERSION="lisa13b-clean030-v1"
MODEL_PATH="./artifacts/lisa-safety-seg/lisa13b-clean030-v1/merged_hf"
CLIP_TOWER="./clip-vit-large-patch14"
SMOKE_IMAGE="./dataset/reason_seg/ReasonSeg/val/val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing.jpg"
SMOKE_PROMPT="标出未按规定佩戴安全帽的作业人员。"
OUTPUT_DIR="./exp/runs/lisa13b-clean030-production-perf-v1/outputs"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_INDEX="0"
HOST="127.0.0.1"
PORT="8001"
WARMUP_REQUESTS="5"
MEASURED_REQUESTS="30"
STABILITY_REQUESTS="100"
MAX_P95_MS="1500"
MAX_PEAK_MEMORY_MIB="36864"
MAX_MEMORY_DRIFT_MIB="500"

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
  echo "Performance output directory is not empty: $OUTPUT_DIR" >&2
  echo "Preserve or move the existing outputs before rerunning." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the GPU performance baseline." >&2
  exit 1
fi

set +e
"$PYTHON_BIN" production/benchmark_api.py \
  --model-path "$MODEL_PATH" \
  --vision-tower "$CLIP_TOWER" \
  --image "$SMOKE_IMAGE" \
  --prompt "$SMOKE_PROMPT" \
  --output-dir "$OUTPUT_DIR" \
  --model-version "$MODEL_VERSION" \
  --precision bf16 \
  --gpu-index "$GPU_INDEX" \
  --host "$HOST" \
  --port "$PORT" \
  --warmup-requests "$WARMUP_REQUESTS" \
  --measured-requests "$MEASURED_REQUESTS" \
  --stability-requests "$STABILITY_REQUESTS" \
  --max-p95-ms "$MAX_P95_MS" \
  --max-peak-memory-mib "$MAX_PEAK_MEMORY_MIB" \
  --max-memory-drift-mib "$MAX_MEMORY_DRIFT_MIB"
BENCHMARK_STATUS="$?"
set -e

if [ -f "$OUTPUT_DIR/summary.md" ]; then
  sed -n '1,240p' "$OUTPUT_DIR/summary.md"
elif [ -f "$OUTPUT_DIR/server.log" ]; then
  echo "No summary was generated. Last server log lines:" >&2
  tail -120 "$OUTPUT_DIR/server.log" >&2
fi

exit "$BENCHMARK_STATUS"
