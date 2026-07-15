#!/usr/bin/env bash
set -euo pipefail

# Remote Linux GPU server.
export LISA_BENCHMARK_FONT_PATH=/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python benchmark_reason_seg.py \
  --version ./runs/lisa13b-clean030-lora-v1/merged_hf \
  --vision-tower /home/ths/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41 \
  --dataset_dir ./dataset \
  --val_dataset ReasonSeg\|val \
  --vision_pretrained ./data_pipeline/sam_vit_h_4b8939.pth \
  --output_dir ./exp/runs/lisa13b-clean030-lora-v1/full-eval-outputs \
  --precision bf16 \
  --workers 4 \
  --save_visualizations \
  --max_visualizations -1 \
  --save_masks
