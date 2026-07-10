#!/usr/bin/env bash
set -euo pipefail

# Remote Linux GPU server.
: "${BASE_MODEL:=./LISA13B}"
: "${SAM_CKPT:=./data_pipeline/sam_vit_h_4b8939.pth}"
: "${CLIP_TOWER:?Set CLIP_TOWER to the local openai/clip-vit-large-patch14 snapshot path}"
: "${LISA_BENCHMARK_FONT_PATH:=/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python benchmark_reason_seg.py \
  --version "$BASE_MODEL" \
  --vision-tower "$CLIP_TOWER" \
  --dataset_dir ./dataset \
  --val_dataset "ReasonSeg|val" \
  --vision_pretrained "$SAM_CKPT" \
  --output_dir ./exp/runs/lisa13b-local-smoke/outputs \
  --precision bf16 \
  --max_samples 3 \
  --workers 0 \
  --save_visualizations \
  --max_visualizations 3
