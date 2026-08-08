#!/usr/bin/env bash
set -euo pipefail

# Remote Linux GPU server only.

EXP_NAME="lisa13b-clean030-lora-v1"
BASE_MODEL="./LISA13B"
SAM_CKPT="./data_pipeline/sam_vit_h_4b8939.pth"
CLIP_TOWER="${LISA_VISION_TOWER:-./clip-vit-large-patch14}"
MODEL_STORE_CLIP="${MODEL_STORE_ROOT:-${HOME}/MODEL_STORE}/clip/clip-vit-large-patch14/upstream-v1/snapshot"
CLEAN_DATASET="./dataset/reason_seg/ReasonSegClean030"
MERGED_MODEL="./runs/${EXP_NAME}/merged_hf"
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
  --deepspeed_torch_adam \
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

bash "./exp/runs/${EXP_NAME}/eval_outputs.sh"
