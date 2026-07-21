#!/usr/bin/env bash
set -euo pipefail

# Remote Linux server only. This verification does not attach a GPU or load model weights.

MODEL_VERSION="lisa13b-request-limit-v1"
DOCKERFILE="./production/Dockerfile"
IMAGE_TAG="lisa-safety-seg:lisa13b-request-limit-v1-fix1"
CONTAINER_NAME="lisa-request-limit-v1-fix1"
OUTPUT_DIR="./exp/runs/lisa13b-request-limit-v1/outputs-after-fix"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_INDEX="0"
HOST="127.0.0.1"
PORT="8006"
MAX_REQUEST_BYTES="1024"
MINIMUM_UNIT_TESTS="65"
MAX_MEMORY_DRIFT_MIB="500"
REQUIRED_SHARED_PROCESS="VLLM::EngineCore"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for the request-limit verification." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to verify the shared process remains online." >&2
  exit 1
fi
if [ ! -f "$DOCKERFILE" ]; then
  echo "Missing Dockerfile: $DOCKERFILE" >&2
  exit 1
fi
if ss -ltn | grep -q ":$PORT "; then
  echo "Port is already in use: $HOST:$PORT" >&2
  exit 1
fi
if [ -d "$OUTPUT_DIR" ] && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]; then
  echo "Request-limit output directory is not empty: $OUTPUT_DIR" >&2
  echo "Preserve or move the existing outputs before rerunning." >&2
  exit 1
fi
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Container name already exists: $CONTAINER_NAME" >&2
  echo "Inspect and remove the old test container before rerunning." >&2
  exit 1
fi

set +e
"$PYTHON_BIN" -m production.verify_request_limit \
  --dockerfile "$DOCKERFILE" \
  --image-tag "$IMAGE_TAG" \
  --container-name "$CONTAINER_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --model-version "$MODEL_VERSION" \
  --host "$HOST" \
  --port "$PORT" \
  --max-request-bytes "$MAX_REQUEST_BYTES" \
  --minimum-unit-tests "$MINIMUM_UNIT_TESTS" \
  --max-memory-drift-mib "$MAX_MEMORY_DRIFT_MIB" \
  --gpu-index "$GPU_INDEX" \
  --require-existing-process-substring "$REQUIRED_SHARED_PROCESS"
STATUS="$?"
set -e

if [ -f "$OUTPUT_DIR/summary.md" ]; then
  sed -n '1,300p' "$OUTPUT_DIR/summary.md"
else
  echo "Request-limit verification did not generate summary.md." >&2
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
