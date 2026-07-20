#!/usr/bin/env bash
set -euo pipefail

# Remote Linux GPU server only.

MODEL_VERSION="lisa13b-clean030-v1"
MODEL_ARTIFACT="./artifacts/lisa-safety-seg/lisa13b-clean030-v1"
DOCKERFILE="./production/Dockerfile"
IMAGE_TAG="lisa-safety-seg:lisa13b-clean030-v1-0092463"
CONTAINER_NAME="lisa-clean030-container-smoke-v1"
SMOKE_IMAGE="./dataset/reason_seg/ReasonSeg/val/val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing.jpg"
SMOKE_PROMPT="标出未按规定佩戴安全帽的作业人员。"
OUTPUT_DIR="./exp/runs/lisa13b-clean030-container-smoke-v1/outputs"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_INDEX="0"
HOST="127.0.0.1"
PORT="8004"
MINIMUM_UNIT_TESTS="49"
MAX_PEAK_MEMORY_MIB="36864"
MIN_REMAINING_MEMORY_MIB="4096"
MAX_POST_STOP_DRIFT_MIB="500"
REQUIRED_SHARED_PROCESS="VLLM::EngineCore"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for the container smoke verification." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the container smoke verification." >&2
  exit 1
fi
if ! docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; then
  echo "Docker NVIDIA runtime is not registered." >&2
  exit 1
fi
if [ ! -f "$DOCKERFILE" ]; then
  echo "Missing Dockerfile: $DOCKERFILE" >&2
  exit 1
fi
if [ ! -f "$MODEL_ARTIFACT/merged_hf/config.json" ]; then
  echo "Missing frozen model config under: $MODEL_ARTIFACT" >&2
  exit 1
fi
if [ ! -f "$MODEL_ARTIFACT/SHA256SUMS" ]; then
  echo "Missing frozen model SHA256SUMS under: $MODEL_ARTIFACT" >&2
  exit 1
fi
if [ ! -f "$SMOKE_IMAGE" ]; then
  echo "Missing fixed smoke image: $SMOKE_IMAGE" >&2
  exit 1
fi
if [ -d "$OUTPUT_DIR" ] && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]; then
  echo "Container smoke output directory is not empty: $OUTPUT_DIR" >&2
  echo "Preserve or move the existing outputs before rerunning." >&2
  exit 1
fi
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Container name already exists: $CONTAINER_NAME" >&2
  echo "Inspect and remove the old test container before rerunning." >&2
  exit 1
fi

HF_CACHE_ROOT="${HF_HOME:-${HOME}/.cache/huggingface}"
CLIP_CONFIG="$(find "$HF_CACHE_ROOT" -path "*/models--openai--clip-vit-large-patch14/snapshots/*/config.json" -print -quit)"
if [ -z "$CLIP_CONFIG" ]; then
  echo "CLIP snapshot config was not found under: $HF_CACHE_ROOT" >&2
  exit 1
fi
CLIP_TOWER="$(dirname "$CLIP_CONFIG")"
CLIP_MODEL_DIR="$(dirname "$(dirname "$CLIP_TOWER")")"
CLIP_SNAPSHOT="$(basename "$CLIP_TOWER")"
if [ ! -f "$CLIP_TOWER/preprocessor_config.json" ] || [ ! -d "$CLIP_MODEL_DIR/blobs" ]; then
  echo "CLIP snapshot or blobs directory is incomplete: $CLIP_MODEL_DIR" >&2
  exit 1
fi

(
  cd "$MODEL_ARTIFACT"
  sha256sum \
    --check \
    --quiet \
    SHA256SUMS
)
echo "Frozen model SHA-256 verification passed."

set +e
"$PYTHON_BIN" -m production.verify_container_smoke \
  --dockerfile "$DOCKERFILE" \
  --image-tag "$IMAGE_TAG" \
  --container-name "$CONTAINER_NAME" \
  --model-artifact "$MODEL_ARTIFACT" \
  --vision-model-dir "$CLIP_MODEL_DIR" \
  --vision-snapshot "$CLIP_SNAPSHOT" \
  --image "$SMOKE_IMAGE" \
  --prompt "$SMOKE_PROMPT" \
  --output-dir "$OUTPUT_DIR" \
  --model-version "$MODEL_VERSION" \
  --gpu-index "$GPU_INDEX" \
  --host "$HOST" \
  --port "$PORT" \
  --minimum-unit-tests "$MINIMUM_UNIT_TESTS" \
  --max-peak-memory-mib "$MAX_PEAK_MEMORY_MIB" \
  --min-remaining-memory-mib "$MIN_REMAINING_MEMORY_MIB" \
  --max-post-stop-drift-mib "$MAX_POST_STOP_DRIFT_MIB" \
  --require-existing-process-substring "$REQUIRED_SHARED_PROCESS"
STATUS="$?"
set -e

if [ -f "$OUTPUT_DIR/summary.md" ]; then
  sed -n '1,360p' "$OUTPUT_DIR/summary.md"
else
  echo "Container smoke did not generate summary.md." >&2
  if [ -f "$OUTPUT_DIR/server.log" ]; then
    tail -160 "$OUTPUT_DIR/server.log" >&2
  fi
  if [ -f "$OUTPUT_DIR/build.log" ]; then
    tail -160 "$OUTPUT_DIR/build.log" >&2
  fi
  if [ -f "$OUTPUT_DIR/unit_tests.log" ]; then
    tail -160 "$OUTPUT_DIR/unit_tests.log" >&2
  fi
fi

exit "$STATUS"
