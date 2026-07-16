#!/usr/bin/env bash
set -euo pipefail

# Remote Linux GPU server only.
# Rebuild full ReasonSeg validation outputs without retraining or re-merging.

EXP_NAME="lisa13b-relabel303-lora-v1"
MERGED_MODEL="./runs/${EXP_NAME}/merged_hf"
SAM_CKPT="./data_pipeline/sam_vit_h_4b8939.pth"
CLIP_TOWER="./clip-vit-large-patch14"
FULL_DATASET="./dataset/reason_seg/ReasonSeg"
FULL_OUTPUT_DIR="./exp/runs/${EXP_NAME}/full-eval-outputs"
BASE_VAL_METRICS="./exp/runs/lisa13b-local-val/outputs/per_sample_metrics.jsonl"
CLEAN030_VAL_METRICS="./exp/runs/lisa13b-clean030-lora-v1/full-eval-outputs/per_sample_metrics.jsonl"
COMPARE_SCRIPT="./exp/compare_benchmark_metrics.py"
COMPARISON_PAGE_SCRIPT="./exp/build_annotation_prediction_report.py"
BASE_COMPARE_REPORT="./exp/runs/${EXP_NAME}/base-vs-relabel303.md"
CLEAN030_COMPARE_REPORT="./exp/runs/${EXP_NAME}/clean030-vs-relabel303.md"
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
if [ ! -d "${FULL_DATASET}/val" ]; then
  echo "Missing full validation dataset: ${FULL_DATASET}/val" >&2
  exit 1
fi
if [ ! -f "$BASE_VAL_METRICS" ]; then
  echo "Missing Base validation metrics: $BASE_VAL_METRICS" >&2
  exit 1
fi

export LISA_BENCHMARK_FONT_PATH

rm -rf "$FULL_OUTPUT_DIR"

echo "[eval] ReasonSeg|val -> $FULL_OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python benchmark_reason_seg.py \
  --version "$MERGED_MODEL" \
  --vision-tower "$CLIP_TOWER" \
  --dataset_dir ./dataset \
  --val_dataset "ReasonSeg|val" \
  --vision_pretrained "$SAM_CKPT" \
  --output_dir "$FULL_OUTPUT_DIR" \
  --precision bf16 \
  --workers 4 \
  --save_visualizations \
  --max_visualizations -1 \
  --save_masks

{
  printf '%s\n' '#!/usr/bin/env bash'
  printf '%s\n' 'set -euo pipefail'
  printf '%s\n' ''
  printf '%s\n' '# Remote Linux GPU server only.'
  printf 'export LISA_BENCHMARK_FONT_PATH=%q\n' "$LISA_BENCHMARK_FONT_PATH"
  printf '%s\n' ''
  printf 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python benchmark_reason_seg.py \\\n'
  printf '  --version %q \\\n' "$MERGED_MODEL"
  printf '  --vision-tower %q \\\n' "$CLIP_TOWER"
  printf '  --dataset_dir ./dataset \\\n'
  printf '  --val_dataset %q \\\n' "ReasonSeg|val"
  printf '  --vision_pretrained %q \\\n' "$SAM_CKPT"
  printf '  --output_dir %q \\\n' "$FULL_OUTPUT_DIR"
  printf '  --precision bf16 \\\n'
  printf '  --workers 4 \\\n'
  printf '  --save_visualizations \\\n'
  printf '  --max_visualizations -1 \\\n'
  printf '  --save_masks\n'
} > "${FULL_OUTPUT_DIR}/last_command.sh"
chmod +x "${FULL_OUTPUT_DIR}/last_command.sh"

python "$COMPARISON_PAGE_SCRIPT" \
  --base-metrics "$BASE_VAL_METRICS" \
  --tuned-metrics "${FULL_OUTPUT_DIR}/per_sample_metrics.jsonl" \
  --update-tuned-pages

python "$COMPARE_SCRIPT" \
  --quiet \
  --base "$BASE_VAL_METRICS" \
  --tuned "${FULL_OUTPUT_DIR}/per_sample_metrics.jsonl" \
  --output "$BASE_COMPARE_REPORT"

if [ -f "$CLEAN030_VAL_METRICS" ]; then
  python "$COMPARE_SCRIPT" \
    --quiet \
    --base "$CLEAN030_VAL_METRICS" \
    --tuned "${FULL_OUTPUT_DIR}/per_sample_metrics.jsonl" \
    --output "$CLEAN030_COMPARE_REPORT"
else
  echo "[warn] Clean030 metrics not found; skipping Clean030 vs Relabel303 report."
fi

echo "[done] full validation outputs: $FULL_OUTPUT_DIR"
echo "[done] Base comparison: $BASE_COMPARE_REPORT"
if [ -f "$CLEAN030_COMPARE_REPORT" ]; then
  echo "[done] Clean030 comparison: $CLEAN030_COMPARE_REPORT"
fi
