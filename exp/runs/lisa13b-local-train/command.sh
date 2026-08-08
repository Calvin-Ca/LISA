#!/usr/bin/env bash
set -euo pipefail

# Remote Linux GPU server only.

BASE_MODEL="./LISA13B"
SAM_CKPT="./data_pipeline/sam_vit_h_4b8939.pth"
CLIP_TOWER="${LISA_VISION_TOWER:-./clip-vit-large-patch14}"
MODEL_STORE_CLIP="${MODEL_STORE_ROOT:-${HOME}/MODEL_STORE}/clip/clip-vit-large-patch14/upstream-v1/snapshot"
LISA_BENCHMARK_FONT_PATH="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

clip_tower_is_complete() {
  [ -f "$1/config.json" ] &&
    [ -f "$1/preprocessor_config.json" ] &&
    { [ -f "$1/model.safetensors" ] || [ -f "$1/pytorch_model.bin" ]; }
}

if ! clip_tower_is_complete "$CLIP_TOWER"; then
  if clip_tower_is_complete "$MODEL_STORE_CLIP"; then
    CLIP_TOWER="$MODEL_STORE_CLIP"
  else
    HF_CACHE_ROOT="${HF_HOME:-${HOME}/.cache/huggingface}"
    CLIP_CONFIG=""
    if [ -d "$HF_CACHE_ROOT" ]; then
      CLIP_CONFIG="$(find "$HF_CACHE_ROOT" -path "*/models--openai--clip-vit-large-patch14/snapshots/*/config.json" -print -quit)"
    fi
    if [ -n "$CLIP_CONFIG" ] && clip_tower_is_complete "$(dirname "$CLIP_CONFIG")"; then
      CLIP_TOWER="$(dirname "$CLIP_CONFIG")"
    fi
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
if ! clip_tower_is_complete "$CLIP_TOWER"; then
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
