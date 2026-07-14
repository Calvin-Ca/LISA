#!/usr/bin/env bash
set -euo pipefail

# Remote Linux GPU server only.
# Rebuild eval outputs without retraining or re-merging LoRA weights.

EXP_NAME="lisa13b-clean030-lora-v1"
MERGED_MODEL="./runs/${EXP_NAME}/merged_hf"
SAM_CKPT="./data_pipeline/sam_vit_h_4b8939.pth"
CLIP_TOWER="./clip-vit-large-patch14"
CLEAN_EVAL_RUN="./exp/runs/${EXP_NAME}/eval-clean-val"
FULL_EVAL_RUN="./exp/runs/${EXP_NAME}/eval-full-val"
CLEAN_OUTPUT_DIR="${CLEAN_EVAL_RUN}/outputs"
FULL_OUTPUT_DIR="${FULL_EVAL_RUN}/outputs"
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
if [ ! -d "./dataset/reason_seg/ReasonSegClean030/val" ]; then
  echo "Missing clean val dataset: ./dataset/reason_seg/ReasonSegClean030/val" >&2
  exit 1
fi
if [ ! -d "./dataset/reason_seg/ReasonSeg/val" ]; then
  echo "Missing full val dataset: ./dataset/reason_seg/ReasonSeg/val" >&2
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

rm -rf "$CLEAN_OUTPUT_DIR" "$FULL_OUTPUT_DIR"

echo "[eval] ReasonSegClean030|val -> $CLEAN_OUTPUT_DIR"
run_eval "ReasonSegClean030|val" "$CLEAN_OUTPUT_DIR"

echo "[eval] ReasonSeg|val -> $FULL_OUTPUT_DIR"
run_eval "ReasonSeg|val" "$FULL_OUTPUT_DIR"

echo "[done] clean val outputs: $CLEAN_OUTPUT_DIR"
echo "[done] full val outputs: $FULL_OUTPUT_DIR"
