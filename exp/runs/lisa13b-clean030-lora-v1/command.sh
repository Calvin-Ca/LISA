#!/usr/bin/env bash
set -euo pipefail

# Remote Linux GPU server only.

EXP_NAME="lisa13b-clean030-lora-v1"
BASE_MODEL="./LISA13B"
SAM_CKPT="./data_pipeline/sam_vit_h_4b8939.pth"
CLIP_TOWER="./clip-vit-large-patch14"
CLEAN_DATASET="./dataset/reason_seg/ReasonSegClean030"
MERGED_MODEL="./runs/${EXP_NAME}/merged_hf"
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

python data_pipeline/build_clean_subset_from_benchmark.py --overwrite

if [ ! -f "$CLEAN_DATASET/clean_subset_summary.json" ]; then
  echo "Missing Clean030 summary: $CLEAN_DATASET/clean_subset_summary.json" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" deepspeed --master_port=24999 train_ds.py \
  --version "$BASE_MODEL" \
  --vision-tower "$CLIP_TOWER" \
  --vision_pretrained "$SAM_CKPT" \
  --dataset_dir ./dataset \
  --dataset "reason_seg" \
  --sample_rates "1" \
  --reason_seg_data "ReasonSegClean030|train" \
  --val_dataset "ReasonSegClean030|val" \
  --explanatory -1 \
  --precision bf16 \
  --epochs 6 \
  --steps_per_epoch 100 \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --workers 4 \
  --lr 0.0001 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,v_proj" \
  --exp_name "$EXP_NAME"

cd "./runs/${EXP_NAME}/ckpt_model"
python zero_to_fp32.py . ../pytorch_model.bin
cd ../../..

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python merge_lora_weights_and_save_hf_model.py \
  --version "$BASE_MODEL" \
  --vision-tower "$CLIP_TOWER" \
  --vision_pretrained "$SAM_CKPT" \
  --weight "./runs/${EXP_NAME}/pytorch_model.bin" \
  --save_path "$MERGED_MODEL" \
  --precision bf16 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,v_proj"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python benchmark_reason_seg.py \
  --version "$MERGED_MODEL" \
  --vision-tower "$CLIP_TOWER" \
  --dataset_dir ./dataset \
  --val_dataset "ReasonSegClean030|val" \
  --vision_pretrained "$SAM_CKPT" \
  --output_dir "./exp/runs/${EXP_NAME}-eval-clean-val/outputs" \
  --precision bf16 \
  --workers 4 \
  --save_visualizations \
  --max_visualizations -1 \
  --save_masks

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python benchmark_reason_seg.py \
  --version "$MERGED_MODEL" \
  --vision-tower "$CLIP_TOWER" \
  --dataset_dir ./dataset \
  --val_dataset "ReasonSeg|val" \
  --vision_pretrained "$SAM_CKPT" \
  --output_dir "./exp/runs/${EXP_NAME}-eval-full-val/outputs" \
  --precision bf16 \
  --workers 4 \
  --save_visualizations \
  --max_visualizations -1 \
  --save_masks
