# lisa13b-clean030-lora-v1

## 背景

本实验用 `ReasonSegClean030` 高置信子集对本地 LISA-13B 做第一轮 LoRA 微调,用于验证施工安全场景的领域适配链路。

`ReasonSegClean030` 由已完成的 base benchmark 结果筛选得到:

- train 来源: `exp/runs/lisa13b-local-train/outputs/per_sample_metrics.jsonl`
- val 来源: `exp/runs/lisa13b-local-val/outputs/per_sample_metrics.jsonl`
- 筛选规则: 保留 `IoU >= 0.30` 的原始 jpg/json 样本
- 标签来源: 仍使用原 `ReasonSeg` 同名 JSON,不使用 LISA 预测作为标签

本实验重点不是最终刷分,而是确认:

- `COCO/SAM -> LISA JSON -> Clean030 -> LoRA -> benchmark` 链路可运行。
- 高置信样本是否能让 LISA 学到施工安全 prompt 与 mask 的对应关系。
- LoRA 后在完整 `ReasonSeg|val` 上是否有真实提升。

## 配置

- 模型: `./LISA13B`
- 权重路径: `./LISA13B`
- CLIP vision tower: 优先 `./clip-vit-large-patch14`,不存在时从本机 HuggingFace cache 自动查找 `openai/clip-vit-large-patch14`
- SAM 权重: `./data_pipeline/sam_vit_h_4b8939.pth`
- 训练数据集: `ReasonSegClean030|train`
- 训练样本数: 202
- 训练数据筛选阈值: base IoU `>= 0.30`
- 训练精度: `bf16`
- LoRA: `r=8`, `alpha=16`, `dropout=0.05`, target modules `q_proj,v_proj`
- 训练轮数: 6
- 每轮 step: 100
- batch size: 1
- grad accumulation steps: 8
- effective batch size: 8
- 学习率: `0.0001`
- explanatory: `-1`
- 训练时验证集: `ReasonSegClean030|val`,42 张
- 正式评估集: `ReasonSeg|val`,86 张
- 掩码阈值: `0.0`
- 是否保存可视化: 是
- 是否保存预测掩码: 是
- 运行设备: 远程 Linux GPU 服务器
- 运行日期: 待执行后补充

## 执行命令

见 `command.sh`。

```bash
bash exp/runs/lisa13b-clean030-lora-v1/command.sh
```

## 输出文件

训练与权重:

- `runs/lisa13b-clean030-lora-v1/ckpt_model/`
- `runs/lisa13b-clean030-lora-v1/pytorch_model.bin`
- `runs/lisa13b-clean030-lora-v1/merged_hf/`
- `runs/lisa13b-clean030-lora-v1/meta_log_giou*_ciou*.pth`

Clean030 数据清单:

- `dataset/reason_seg/ReasonSegClean030/clean_subset_manifest.json`
- `dataset/reason_seg/ReasonSegClean030/clean_subset_summary.json`

Clean val 评估:

- `exp/runs/lisa13b-clean030-lora-v1-eval-clean-val/outputs/summary.json`
- `exp/runs/lisa13b-clean030-lora-v1-eval-clean-val/outputs/summary.md`
- `exp/runs/lisa13b-clean030-lora-v1-eval-clean-val/outputs/per_sample_metrics.csv`
- `exp/runs/lisa13b-clean030-lora-v1-eval-clean-val/outputs/per_sample_metrics.jsonl`
- `exp/runs/lisa13b-clean030-lora-v1-eval-clean-val/outputs/samples_by_iou.md`
- `exp/runs/lisa13b-clean030-lora-v1-eval-clean-val/outputs/visualizations/`
- `exp/runs/lisa13b-clean030-lora-v1-eval-clean-val/outputs/pred_masks/`

Full val 评估:

- `exp/runs/lisa13b-clean030-lora-v1-eval-full-val/outputs/summary.json`
- `exp/runs/lisa13b-clean030-lora-v1-eval-full-val/outputs/summary.md`
- `exp/runs/lisa13b-clean030-lora-v1-eval-full-val/outputs/per_sample_metrics.csv`
- `exp/runs/lisa13b-clean030-lora-v1-eval-full-val/outputs/per_sample_metrics.jsonl`
- `exp/runs/lisa13b-clean030-lora-v1-eval-full-val/outputs/samples_by_iou.md`
- `exp/runs/lisa13b-clean030-lora-v1-eval-full-val/outputs/visualizations/`
- `exp/runs/lisa13b-clean030-lora-v1-eval-full-val/outputs/pred_masks/`

## 核心指标

Base benchmark 参考:

- Clean val 样本数: 42
- Clean val base gIoU: 0.6435
- Clean val base cIoU: 0.6495
- Clean val base 平均 Dice: 0.7645
- Full val 样本数: 86
- Full val base gIoU: 0.3408
- Full val base cIoU: 0.3177
- Full val base 平均 Dice: 0.4180

LoRA 后待补充:

- Clean val gIoU:
- Clean val cIoU:
- Clean val 平均 Dice:
- Full val gIoU:
- Full val cIoU:
- Full val 平均 Dice:

## 结论

待远程训练和评估完成后补充。

判断标准:

- 如果 clean val 提升,说明 Clean030 高置信样本可被 LoRA 有效吸收。
- 如果 full val 同时提升,说明对真实验证集有迁移收益。
- 如果只提升 clean val 而 full val 不提升,说明模型只学到了 easy subset,下一轮需要补 `guardrail_missing`、`unsafe`、`equipment_proximity`、`poor_housekeeping` 的人工核验 hard cases。

## 备注

- 本实验使用 Clean030 作为第一轮链路验证集,不把筛选后的 clean val 作为唯一正式指标。
- 正式结论必须以完整 `ReasonSeg|val` 的 LoRA 前后对比为准。
- `ReasonSegClean030` 是从原始 jpg/json 复制出的子集,不包含新生成标签,也不使用模型预测 mask 作为标签。
