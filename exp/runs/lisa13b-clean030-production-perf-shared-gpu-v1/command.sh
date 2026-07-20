#!/usr/bin/env bash
set -euo pipefail

# Remote Linux GPU server only.

MODEL_VERSION="lisa13b-clean030-v1"
MODEL_PATH="./artifacts/lisa-safety-seg/lisa13b-clean030-v1/merged_hf"
CLIP_TOWER="./clip-vit-large-patch14"
SMOKE_IMAGE="./dataset/reason_seg/ReasonSeg/val/val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing.jpg"
SMOKE_PROMPT="标出未按规定佩戴安全帽的作业人员。"
OUTPUT_ROOT="./exp/runs/lisa13b-clean030-production-perf-shared-gpu-v1/outputs"
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
REQUIRED_SHARED_PROCESS="VLLM::EngineCore"
ROUNDS="3"

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
if [ -d "$OUTPUT_ROOT" ] && [ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -print -quit)" ]; then
  echo "Shared-GPU output directory is not empty: $OUTPUT_ROOT" >&2
  echo "Preserve or move the existing outputs before rerunning." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the GPU performance baseline." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
OVERALL_STATUS="0"
ROUND_SUMMARY_ARGS=()

for ROUND in $(seq 1 "$ROUNDS"); do
  ROUND_OUTPUT="$OUTPUT_ROOT/round-$ROUND"
  echo "Starting shared-GPU performance round $ROUND/$ROUNDS"
  set +e
  "$PYTHON_BIN" production/benchmark_api.py \
    --model-path "$MODEL_PATH" \
    --vision-tower "$CLIP_TOWER" \
    --image "$SMOKE_IMAGE" \
    --prompt "$SMOKE_PROMPT" \
    --output-dir "$ROUND_OUTPUT" \
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
    --max-memory-drift-mib "$MAX_MEMORY_DRIFT_MIB" \
    --allow-existing-compute-processes \
    --require-existing-process-substring "$REQUIRED_SHARED_PROCESS"
  ROUND_STATUS="$?"
  set -e

  if [ ! -f "$ROUND_OUTPUT/summary.json" ]; then
    echo "Round $ROUND did not generate summary.json." >&2
    if [ -f "$ROUND_OUTPUT/server.log" ]; then
      tail -120 "$ROUND_OUTPUT/server.log" >&2
    fi
    if [ "$ROUND_STATUS" -eq 0 ]; then
      exit 1
    fi
    exit "$ROUND_STATUS"
  fi
  ROUND_SUMMARY_ARGS+=(--round-summary "$ROUND_OUTPUT/summary.json")
  if [ "$ROUND_STATUS" -ne 0 ]; then
    OVERALL_STATUS="$ROUND_STATUS"
  fi
  sleep 5
done

"$PYTHON_BIN" production/summarize_api_benchmarks.py \
  "${ROUND_SUMMARY_ARGS[@]}" \
  --output-dir "$OUTPUT_ROOT"

sed -n '1,260p' "$OUTPUT_ROOT/aggregate_summary.md"
exit "$OVERALL_STATUS"
