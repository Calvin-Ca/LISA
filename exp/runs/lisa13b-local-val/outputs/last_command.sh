#!/usr/bin/env bash
set -euo pipefail

# Remote Linux GPU server.
CUDA_VISIBLE_DEVICES=0 python benchmark_reason_seg.py \
  --version "$BASE_MODEL" \
  --vision-tower "$CLIP_TOWER" \
  --dataset_dir ./dataset \
  --val_dataset 'ReasonSeg|val' \
  --vision_pretrained "$SAM_CKPT" \
  --output_dir ./exp/runs/lisa13b-local-val/outputs \
  --precision bf16 \
  --workers 4 \
  --save_visualizations \
  --max_visualizations -1 \
  --save_masks
