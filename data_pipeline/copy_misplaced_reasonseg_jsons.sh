#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
SOURCE_COMMIT="${SOURCE_COMMIT:-e1bcdc6}"
EXPECTED_COUNT=25
SOURCE_DIR="dataset/reason_seg/ReasonSeg/train"
TARGET_DIR="dataset/reason_seg/ReasonSegRelabel/train"

if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--apply" ]]; then
  echo "Usage: bash data_pipeline/copy_misplaced_reasonseg_jsons.sh [--dry-run|--apply]" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

if ! git cat-file -e "${SOURCE_COMMIT}^{commit}" 2>/dev/null; then
  echo "Source commit does not exist: ${SOURCE_COMMIT}" >&2
  exit 3
fi

if [[ ! -d "${SOURCE_DIR}" || ! -d "${TARGET_DIR}" ]]; then
  echo "ReasonSeg source or ReasonSegRelabel target directory is missing." >&2
  exit 4
fi

SOURCE_FILES=()
while IFS= read -r source_path; do
  SOURCE_FILES+=("${source_path}")
done < <(
  git diff-tree --no-commit-id --name-only -r "${SOURCE_COMMIT}" -- "${SOURCE_DIR}" \
    | sort
)

if [[ "${#SOURCE_FILES[@]}" -ne "${EXPECTED_COUNT}" ]]; then
  echo "Expected ${EXPECTED_COUNT} modified ReasonSeg JSON files in ${SOURCE_COMMIT}, got ${#SOURCE_FILES[@]}." >&2
  exit 5
fi

for source_path in "${SOURCE_FILES[@]}"; do
  filename="${source_path##*/}"
  target_path="${TARGET_DIR}/${filename}"

  if [[ "${source_path}" != *.json ]]; then
    echo "Unexpected non-JSON source: ${source_path}" >&2
    exit 6
  fi
  if [[ ! -f "${source_path}" ]]; then
    echo "Missing source JSON: ${source_path}" >&2
    exit 7
  fi
  if [[ ! -f "${target_path}" ]]; then
    echo "Missing target JSON: ${target_path}" >&2
    exit 8
  fi
  if ! git diff --quiet -- "${source_path}" "${target_path}"; then
    echo "Refusing to overwrite files with uncommitted changes: ${filename}" >&2
    exit 9
  fi
done

echo "[plan] source commit: ${SOURCE_COMMIT}"
echo "[plan] source directory: ${SOURCE_DIR}"
echo "[plan] target directory: ${TARGET_DIR}"
echo "[plan] JSON files: ${#SOURCE_FILES[@]}"

for source_path in "${SOURCE_FILES[@]}"; do
  filename="${source_path##*/}"
  target_path="${TARGET_DIR}/${filename}"
  echo "[copy] ${source_path} -> ${target_path}"
  if [[ "${MODE}" == "--apply" ]]; then
    cp -- "${source_path}" "${target_path}"
  fi
done

if [[ "${MODE}" == "--dry-run" ]]; then
  echo "[dry-run] no files were changed"
  echo "[next] rerun with --apply after reviewing the 25 paths"
  exit 0
fi

for source_path in "${SOURCE_FILES[@]}"; do
  filename="${source_path##*/}"
  target_path="${TARGET_DIR}/${filename}"
  if ! cmp -s -- "${source_path}" "${target_path}"; then
    echo "Copy verification failed: ${filename}" >&2
    exit 10
  fi
done

echo "[done] copied and verified ${#SOURCE_FILES[@]} JSON files"
echo "[warning] ReasonSeg still contains the misplaced edits; restore it in the next recovery step before training."
