#!/usr/bin/env bash
set -euo pipefail

# Remote Linux GPU server only.

EXP_NAME="lisa13b-relabel303-lora-v1"
BASE_MODEL="./LISA13B"
SAM_CKPT="./data_pipeline/sam_vit_h_4b8939.pth"
CLIP_TOWER="./clip-vit-large-patch14"
TRAIN_DIR="./dataset/reason_seg/ReasonSegRelabel/train"
VAL_DIR="./dataset/reason_seg/ReasonSeg/val"
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
if [ ! -d "$TRAIN_DIR" ]; then
  echo "Missing Relabel303 train directory: $TRAIN_DIR" >&2
  exit 1
fi
if [ ! -d "$VAL_DIR" ]; then
  echo "Missing full validation directory: $VAL_DIR" >&2
  exit 1
fi

echo "[preflight] Validating ReasonSegRelabel|train and ReasonSeg|val."
python -c "import cv2,json; from pathlib import Path; from utils.data_processing import get_mask_from_json; train=Path('$TRAIN_DIR'); val=Path('$VAL_DIR'); bad=[]; train_json=sorted(train.glob('*.json')); train_jpg=sorted(train.glob('*.jpg')); val_json=sorted(val.glob('*.json')); val_jpg=sorted(val.glob('*.jpg')); train_json_stems={x.stem for x in train_json}; train_jpg_stems={x.stem for x in train_jpg}; val_json_stems={x.stem for x in val_json}; val_jpg_stems={x.stem for x in val_jpg}; bad.append(f'train count JSON={len(train_json)} JPG={len(train_jpg)}') if len(train_json)!=303 or len(train_jpg)!=303 else None; bad.append('train JSON/JPG stems differ') if train_json_stems!=train_jpg_stems else None; bad.append(f'val count JSON={len(val_json)} JPG={len(val_jpg)}') if len(val_json)!=86 or len(val_jpg)!=86 else None; bad.append('val JSON/JPG stems differ') if val_json_stems!=val_jpg_stems else None; data=[]; [(lambda f: data.append((f,json.loads(f.read_text(encoding='utf-8')))))(f) for f in train_json]; [(lambda f,d: (bad.append(f'{f.name}: text count is not 6') if not isinstance(d.get('text'),list) or len(d.get('text',[]))!=6 else None, bad.append(f'{f.name}: invalid prompt') if any(not isinstance(x,str) or not x.strip() for x in d.get('text',[])) else None, bad.append(f'{f.name}: duplicate prompt') if len(set(d.get('text',[])))!=len(d.get('text',[])) else None, bad.append(f'{f.name}: is_sentence is not true') if d.get('is_sentence') is not True else None, bad.append(f'{f.name}: shapes are empty') if not isinstance(d.get('shapes'),list) or not d.get('shapes') else None, bad.append(f'{f.name}: invalid polygon') if any(not isinstance(s,dict) or not isinstance(s.get('points'),list) or len(s.get('points',[]))<3 for s in d.get('shapes',[])) else None, bad.append(f'{f.name}: incomplete source') if not isinstance(d.get('source'),dict) or any(not d.get('source',{}).get(k) for k in ('file_name','sample_key','source_category')) else None))(f,d) for f,d in data]; images={}; [(lambda f,img: (images.update({f.stem:img}), bad.append(f'{f.name}: unreadable image') if img is None else None))(f,cv2.imread(str(f))) for f in train_jpg+val_jpg]; [(lambda f,img: bad.append(f'{f.name}: empty target mask') if img is not None and int((get_mask_from_json(str(f),img)[0]==1).sum())==0 else None)(f,images.get(f.stem)) for f in train_json]; print(f'train JSON={len(train_json)} JPG={len(train_jpg)}; val JSON={len(val_json)} JPG={len(val_jpg)}; errors={len(bad)}'); [print(x) for x in bad[:100]]; assert not bad, f'Preflight failed with {len(bad)} errors'; print('Dataset preflight passed.')"

export LISA_BENCHMARK_FONT_PATH

echo "[train] Starting ${EXP_NAME}."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" deepspeed --master_port=24999 train_ds.py \
  --version "$BASE_MODEL" \
  --vision-tower "$CLIP_TOWER" \
  --vision_pretrained "$SAM_CKPT" \
  --dataset_dir ./dataset \
  --dataset "reason_seg" \
  --sample_rates "1" \
  --reason_seg_data "ReasonSegRelabel|train" \
  --val_dataset "ReasonSeg|val" \
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
  --num_classes_per_sample 3 \
  --exp_name "$EXP_NAME"

if [ ! -f "./runs/${EXP_NAME}/ckpt_model/zero_to_fp32.py" ]; then
  echo "Missing DeepSpeed conversion script under runs/${EXP_NAME}/ckpt_model." >&2
  exit 1
fi

echo "[merge] Exporting fp32 checkpoint."
cd "./runs/${EXP_NAME}/ckpt_model"
python zero_to_fp32.py . ../pytorch_model.bin
cd ../../..

echo "[merge] Merging LoRA weights."
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

echo "[eval] Evaluating merged model on ReasonSeg|val."
bash "./exp/runs/${EXP_NAME}/eval_outputs.sh"

echo "[done] ${EXP_NAME}"
