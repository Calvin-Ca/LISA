#!/usr/bin/env bash
set -euo pipefail

# Remote Linux GPU server only.
# Rebuild eval outputs without retraining or re-merging LoRA weights.

EXP_NAME="lisa13b-clean030-lora-v1"
MERGED_MODEL="./runs/${EXP_NAME}/merged_hf"
SAM_CKPT="./data_pipeline/sam_vit_h_4b8939.pth"
CLIP_TOWER="./clip-vit-large-patch14"
CLEAN_DATASET="./dataset/reason_seg/ReasonSegClean030"
FULL_DATASET="./dataset/reason_seg/ReasonSeg"
CLEAN_OUTPUT_DIR="./exp/runs/${EXP_NAME}/clean-eval-outputs"
FULL_OUTPUT_DIR="./exp/runs/${EXP_NAME}/full-eval-outputs"
BASE_VAL_METRICS="./exp/runs/lisa13b-local-val/outputs/per_sample_metrics.jsonl"
COMPARE_SCRIPT="./exp/compare_benchmark_metrics.py"
LISA_BENCHMARK_FONT_PATH="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

if [ ! -d "$CLIP_TOWER" ]; then
  CLIP_CONFIG="$(find "$HOME/.cache/huggingface/hub" -path "*/models--openai--clip-vit-large-patch14/snapshots/*/config.json" -print -quit)"
  if [ -n "$CLIP_CONFIG" ]; then
    CLIP_TOWER="$(dirname "$CLIP_CONFIG")"
  fi
fi

if [ ! -f "$MERGED_MODEL/config.json" ]; then
  echo "Missing merged model: $MERGED_MODEL/config.json" >&2
  exit 1
fi
if [ ! -f "$SAM_CKPT" ]; then
  echo "Missing SAM checkpoint: $SAM_CKPT" >&2
  exit 1
fi
if [ ! -f "$CLIP_TOWER/config.json" ] || [ ! -f "$CLIP_TOWER/preprocessor_config.json" ]; then
  echo "Missing CLIP vision tower files under: $CLIP_TOWER" >&2
  exit 1
fi

if [ ! -d "${CLEAN_DATASET}/val" ] || [ ! -f "${CLEAN_DATASET}/clean_subset_summary.json" ]; then
  echo "[prepare] Missing Clean030 dataset; rebuilding from benchmark metrics."
  if [ ! -f "./exp/runs/lisa13b-local-train/outputs/per_sample_metrics.jsonl" ]; then
    echo "Missing train metrics: ./exp/runs/lisa13b-local-train/outputs/per_sample_metrics.jsonl" >&2
    exit 1
  fi
  if [ ! -f "./exp/runs/lisa13b-local-val/outputs/per_sample_metrics.jsonl" ]; then
    echo "Missing val metrics: ./exp/runs/lisa13b-local-val/outputs/per_sample_metrics.jsonl" >&2
    exit 1
  fi
  python data_pipeline/build_clean_subset_from_benchmark.py --overwrite
fi

if [ ! -d "${CLEAN_DATASET}/val" ]; then
  echo "Missing clean val dataset: ${CLEAN_DATASET}/val" >&2
  exit 1
fi
if [ ! -d "${FULL_DATASET}/val" ]; then
  echo "Missing full val dataset: ${FULL_DATASET}/val" >&2
  exit 1
fi
if [ ! -f "$BASE_VAL_METRICS" ]; then
  echo "Missing base val metrics: $BASE_VAL_METRICS" >&2
  exit 1
fi

export LISA_BENCHMARK_FONT_PATH

write_last_command() {
  local path="$1"
  local val_dataset="$2"
  local output_dir="$3"

  mkdir -p "$(dirname "$path")"
  {
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'set -euo pipefail'
    printf '%s\n' ''
    printf '%s\n' '# Remote Linux GPU server.'
    printf 'export LISA_BENCHMARK_FONT_PATH=%q\n' "$LISA_BENCHMARK_FONT_PATH"
    printf '%s\n' ''
    printf 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python benchmark_reason_seg.py \\\n'
    printf '  --version %q \\\n' "$MERGED_MODEL"
    printf '  --vision-tower %q \\\n' "$CLIP_TOWER"
    printf '  --dataset_dir ./dataset \\\n'
    printf '  --val_dataset %q \\\n' "$val_dataset"
    printf '  --vision_pretrained %q \\\n' "$SAM_CKPT"
    printf '  --output_dir %q \\\n' "$output_dir"
    printf '  --precision bf16 \\\n'
    printf '  --workers 4 \\\n'
    printf '  --save_visualizations \\\n'
    printf '  --max_visualizations -1 \\\n'
    printf '  --save_masks\n'
  } > "$path"
  chmod +x "$path"
}

run_eval() {
  local val_dataset="$1"
  local output_dir="$2"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python benchmark_reason_seg.py \
    --version "$MERGED_MODEL" \
    --vision-tower "$CLIP_TOWER" \
    --dataset_dir ./dataset \
    --val_dataset "$val_dataset" \
    --vision_pretrained "$SAM_CKPT" \
    --output_dir "$output_dir" \
    --precision bf16 \
    --workers 4 \
    --save_visualizations \
    --max_visualizations -1 \
    --save_masks

  write_last_command "${output_dir}/last_command.sh" "$val_dataset" "$output_dir"
}

rm -rf \
  "$CLEAN_OUTPUT_DIR" \
  "$FULL_OUTPUT_DIR" \
  "./exp/runs/${EXP_NAME}/eval-clean-val" \
  "./exp/runs/${EXP_NAME}/eval-full-val"

echo "[eval] ReasonSegClean030|val -> $CLEAN_OUTPUT_DIR"
run_eval "ReasonSegClean030|val" "$CLEAN_OUTPUT_DIR"

echo "[eval] ReasonSeg|val -> $FULL_OUTPUT_DIR"
run_eval "ReasonSeg|val" "$FULL_OUTPUT_DIR"

python "$COMPARE_SCRIPT" \
  --quiet \
  --base "$BASE_VAL_METRICS" \
  --tuned "${CLEAN_OUTPUT_DIR}/per_sample_metrics.jsonl" \
  --output "${CLEAN_OUTPUT_DIR}/comparison_by_delta_iou.md"

python "$COMPARE_SCRIPT" \
  --quiet \
  --base "$BASE_VAL_METRICS" \
  --tuned "${FULL_OUTPUT_DIR}/per_sample_metrics.jsonl" \
  --output "${FULL_OUTPUT_DIR}/comparison_by_delta_iou.md"

echo "[done] clean val outputs: $CLEAN_OUTPUT_DIR"
echo "[done] full val outputs: $FULL_OUTPUT_DIR"
