# LISA 训练数据目录(`--dataset_dir`)

`train_ds.py --dataset_dir ./dataset` 会把本目录当作 `base_image_dir`。
场景 A 只用到 `reason_seg`(推理分割)这一支。

## 目录结构

```
dataset/
└── reason_seg/
    └── ReasonSeg/
        ├── train/              # 训练样本:<name>.jpg + <name>.json 成对
        │   ├── xxx.jpg
        │   └── xxx.json
        ├── val/                # 验证样本(同格式)
        └── explanatory/
            └── train.json      # 可选:带文字解释的样本 [{image, query, outputs}]
```

- `train/` `val/` 里每个样本 = 一张 `.jpg` + 一个同名 `.json`(LabelMe 多边形格式)。
- `.json` 规格(与 `utils/data_processing.py::get_mask_from_json` 对齐):
  ```json
  {
    "shapes": [{"label": "target", "points": [[x, y], ...]}],
    "text": ["圈出图中没有佩戴安全帽的工人。"],
    "is_sentence": true
  }
  ```
  - `label` 含 `ignore` → 该多边形值=255(评测时忽略);否则值=1(目标)。
- `explanatory/train.json` 可选。不想用时,训练加 `--explanatory -1` 跳过。

## 数据从哪来

由 `data_pipeline/` 合成,产物在 `data_pipeline/out/`。部署到这里:

```bash
# 1. 合成(见 data_pipeline/README.md)
cd data_pipeline
python build_dataset.py          # 或先 grounded_ingest.py 再 build_dataset.py

# 2. 按 out/split.json 把样本分发到 train/ 与 val/
python deploy_to_dataset.py      # 见下方脚本
```

## 训练命令(场景 A,仅 reason_seg)

```bash
python train_ds.py \
  --dataset_dir ./dataset \
  --dataset "reason_seg" \
  --reason_seg_data "ReasonSeg|train" \
  --explanatory -1 \
  --exp_name lisa-construction
```

> 若数据量小,建议在通用权重基础上做 **LoRA 微调**,而非从零训。具体 LoRA 参数见 `../MS.md` 微调小节。
