#!/usr/bin/env bash
set -euo pipefail

# Remote Linux release host only.
# Authentication must be completed outside this script with docker login and
# an rclone config or environment-backed credentials.

MODEL_VERSION="lisa13b-clean030-v1"
MODEL_ARTIFACT="./artifacts/lisa-safety-seg/lisa13b-clean030-v1"
CONTAINER_SUMMARY="./exp/runs/lisa13b-clean030-container-smoke-v1/outputs/summary.json"
PRECISION_SUMMARY="./exp/runs/lisa13b-clean030-lora-v1/production-preflight/summary.json"
RELEASE_MANIFEST="$MODEL_ARTIFACT/release-manifest.json"

IMAGE_REPOSITORY="${LISA_IMAGE_REPOSITORY:-}"
MODEL_REMOTE="${LISA_MODEL_REMOTE:-}"

if [ -z "$IMAGE_REPOSITORY" ]; then
  echo "LISA_IMAGE_REPOSITORY is required." >&2
  echo "Example: registry.example.com/safety/lisa-safety-seg" >&2
  exit 1
fi
if [ -z "$MODEL_REMOTE" ]; then
  echo "LISA_MODEL_REMOTE is required." >&2
  echo "Example: internal-models:lisa-safety-seg/lisa13b-clean030-v1" >&2
  exit 1
fi
for command in docker rclone python3 sha256sum; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command is missing: $command" >&2
    exit 1
  fi
done
for path in \
  "$MODEL_ARTIFACT/manifest.json" \
  "$MODEL_ARTIFACT/SHA256SUMS" \
  "$MODEL_ARTIFACT/MODEL_CARD.md" \
  "$MODEL_ARTIFACT/merged_hf/config.json" \
  "$CONTAINER_SUMMARY" \
  "$PRECISION_SUMMARY"; do
  if [ ! -f "$path" ]; then
    echo "Required release input is missing: $path" >&2
    exit 1
  fi
done

SOURCE_IMAGE="$(python3 -c "import json,pathlib; \
d=json.loads(pathlib.Path('$CONTAINER_SUMMARY').read_text()); \
assert d['acceptance']['passed'] is True,'容器验收未通过'; \
assert d['model_version']=='$MODEL_VERSION','模型版本不一致'; \
print(d['image_tag'])")"
GIT_COMMIT="$(python3 -c "import json,pathlib; \
d=json.loads(pathlib.Path('$CONTAINER_SUMMARY').read_text()); \
print(d['repo_git_commit'])")"
VERIFIED_IMAGE_ID="$(python3 -c "import json,pathlib; \
d=json.loads(pathlib.Path('$CONTAINER_SUMMARY').read_text()); \
print(str(d['image']['id']).removeprefix('sha256:'))")"
ARTIFACT_VERSION="$(python3 -c "import json,pathlib; \
d=json.loads(pathlib.Path('$MODEL_ARTIFACT/manifest.json').read_text()); \
print(d['model_version'])")"
if [ "$ARTIFACT_VERSION" != "$MODEL_VERSION" ]; then
  echo "Frozen artifact version differs from release version." >&2
  exit 1
fi
GIT_SHORT="${GIT_COMMIT:0:12}"
IMAGE_REF="${IMAGE_REPOSITORY}:${MODEL_VERSION}-${GIT_SHORT}"

(
  cd "$MODEL_ARTIFACT"
  sha256sum \
    --check \
    --quiet \
    SHA256SUMS
)
echo "Frozen model SHA-256 verification passed."

if ! docker image inspect "$SOURCE_IMAGE" >/dev/null 2>&1; then
  echo "Validated source image is missing locally: $SOURCE_IMAGE" >&2
  echo "Rerun the container smoke verification before publishing." >&2
  exit 1
fi
ACTUAL_IMAGE_ID="$(docker image inspect \
  --format '{{.Id}}' \
  "$SOURCE_IMAGE")"
ACTUAL_IMAGE_ID="${ACTUAL_IMAGE_ID#sha256:}"
if [ "$ACTUAL_IMAGE_ID" != "$VERIFIED_IMAGE_ID" ]; then
  echo "Local source image ID differs from the validated image ID." >&2
  echo "Rerun the container smoke verification before publishing." >&2
  exit 1
fi

docker tag \
  "$SOURCE_IMAGE" \
  "$IMAGE_REF"
docker push \
  "$IMAGE_REF"

python3 -m production.prepare_release \
  --model-artifact "$MODEL_ARTIFACT" \
  --image-ref "$IMAGE_REF" \
  --container-summary "$CONTAINER_SUMMARY" \
  --validation-summary "$PRECISION_SUMMARY" \
  --validation-summary "$CONTAINER_SUMMARY" \
  --output "$RELEASE_MANIFEST"

rclone copy \
  "$MODEL_ARTIFACT" \
  "$MODEL_REMOTE" \
  --checksum \
  --immutable \
  --progress

rclone check \
  "$MODEL_ARTIFACT" \
  "$MODEL_REMOTE" \
  --checksum \
  --one-way

echo "Release published."
echo "Image: $IMAGE_REF"
echo "Model remote: $MODEL_REMOTE"
echo "Manifest: $RELEASE_MANIFEST"
