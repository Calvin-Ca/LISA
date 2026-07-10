#!/usr/bin/env bash
set -euo pipefail

# Remote Linux GPU server only.

BASE_MODEL="./LISA13B"
SAM_CKPT="./data_pipeline/sam_vit_h_4b8939.pth"
CLIP_TOWER="./clip-vit-large-patch14"
LISA_BENCHMARK_FONT_PATH="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

if [ ! -d "$CLIP_TOWER" ]; then
  CLIP_CONFIG="$(find "$HOME/.cache/huggingface/hub" -path "*/models--openai--clip-vit-large-patch14/snapshots/*/config.json" -print -quit)"
  if [ -n "$CLIP_CONFIG" ]; then
    CLIP_TOWER="$(dirname "$CLIP_CONFIG")"
  fi
fi

if [ ! -f "$BASE_MODEL/config.json" ]; then
  echo "Missing LISA model config: $BASE_MODEL/config.json" >&2
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

export LISA_BENCHMARK_FONT_PATH

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python benchmark_reason_seg.py \
  --version "$BASE_MODEL" \
  --vision-tower "$CLIP_TOWER" \
  --dataset_dir ./dataset \
  --val_dataset "ReasonSeg|train" \
  --vision_pretrained "$SAM_CKPT" \
  --output_dir ./exp/runs/lisa13b-local-train/outputs \
  --precision bf16 \
  --workers 4 \
  --save_visualizations \
  --max_visualizations -1 \
  --save_masks
