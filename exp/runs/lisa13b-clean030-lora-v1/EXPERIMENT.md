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
- 优化器: DeepSpeed + PyTorch AdamW (`--deepspeed_torch_adam`),避免远程环境 JIT 编译 `fused_adam`
- explanatory: `-1`
- 训练时验证集: `ReasonSegClean030|val`,42 张
- 正式评估集: `ReasonSeg|val`,86 张
- 掩码阈值: `0.0`
- 是否保存可视化: 是
- 是否保存预测掩码: 是
- 运行设备: 远程 Linux GPU 服务器
- 训练日期: 2026-07-13
- 评估产物时间: 2026-07-14 11:11-11:12

## 执行命令

完整训练、合并和评估:

```bash
bash exp/runs/lisa13b-clean030-lora-v1/command.sh
```

只重新生成 clean/full 两个评估 outputs,不重新训练、不重新合并:

```bash
bash exp/runs/lisa13b-clean030-lora-v1/eval_outputs.sh
```

如果 `dataset/reason_seg/ReasonSegClean030/` 不存在,该脚本会先根据 `exp/runs/lisa13b-local-{train,val}/outputs/per_sample_metrics.jsonl` 自动重建 Clean030 子集。

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

- `exp/runs/lisa13b-clean030-lora-v1/clean-eval-outputs/last_command.sh`
- `exp/runs/lisa13b-clean030-lora-v1/clean-eval-outputs/summary.json`
- `exp/runs/lisa13b-clean030-lora-v1/clean-eval-outputs/summary.md`
- `exp/runs/lisa13b-clean030-lora-v1/clean-eval-outputs/per_sample_metrics.csv`
- `exp/runs/lisa13b-clean030-lora-v1/clean-eval-outputs/per_sample_metrics_by_iou.csv`
- `exp/runs/lisa13b-clean030-lora-v1/clean-eval-outputs/per_sample_metrics.jsonl`
- `exp/runs/lisa13b-clean030-lora-v1/clean-eval-outputs/samples_by_iou.md`
- `exp/runs/lisa13b-clean030-lora-v1/clean-eval-outputs/visualizations/`
- `exp/runs/lisa13b-clean030-lora-v1/clean-eval-outputs/pred_masks/`
- `exp/runs/lisa13b-clean030-lora-v1/clean-eval-outputs/comparison_by_delta_iou.md` (Base/LoRA 逐样本对比,按 IoU 变化从好到坏排序)

Full val 评估:

- `exp/runs/lisa13b-clean030-lora-v1/full-eval-outputs/last_command.sh`
- `exp/runs/lisa13b-clean030-lora-v1/full-eval-outputs/summary.json`
- `exp/runs/lisa13b-clean030-lora-v1/full-eval-outputs/summary.md`
- `exp/runs/lisa13b-clean030-lora-v1/full-eval-outputs/per_sample_metrics.csv`
- `exp/runs/lisa13b-clean030-lora-v1/full-eval-outputs/per_sample_metrics_by_iou.csv`
- `exp/runs/lisa13b-clean030-lora-v1/full-eval-outputs/per_sample_metrics.jsonl`
- `exp/runs/lisa13b-clean030-lora-v1/full-eval-outputs/samples_by_iou.md`
- `exp/runs/lisa13b-clean030-lora-v1/full-eval-outputs/visualizations/`
- `exp/runs/lisa13b-clean030-lora-v1/full-eval-outputs/pred_masks/`
- `exp/runs/lisa13b-clean030-lora-v1/full-eval-outputs/comparison_by_delta_iou.md` (Base/LoRA 逐样本对比,按 IoU 变化从好到坏排序)

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

LoRA 后:

- Clean val 样本数: 42
- Clean val gIoU: 0.7119
- Clean val cIoU: 0.6642
- Clean val 平均 Dice: 0.7868
- Clean val 平均精确率: 0.8300
- Clean val 平均召回率: 0.7687
- Full val 样本数: 86
- Full val gIoU: 0.4494
- Full val cIoU: 0.3858
- Full val 平均 Dice: 0.5156
- Full val 平均精确率: 0.5332
- Full val 平均召回率: 0.5416

LoRA 相对 base 提升:

- Clean val gIoU: +0.0684
- Clean val cIoU: +0.0147
- Clean val 平均 Dice: +0.0223
- Full val gIoU: +0.1086
- Full val cIoU: +0.0681
- Full val 平均 Dice: +0.0976
- Full val 平均精确率: +0.1261
- Full val 平均召回率: +0.0284

Full val 样本级分布:

| 指标 | Base | LoRA 后 | 变化 |
| --- | ---: | ---: | ---: |
| IoU = 0 | 22 | 16 | -6 |
| IoU < 0.1 | 33 | 28 | -5 |
| IoU < 0.3 | 44 | 38 | -6 |
| IoU >= 0.5 | 29 | 39 | +10 |
| False Positive Area | 2,321,866 | 1,623,031 | -698,835 (-30.1%) |
| False Negative Area | 1,821,927 | 1,677,931 | -143,996 (-7.9%) |

Full val 分类型 Mean IoU:

| 类别 | 样本数 | Base | LoRA 后 | 变化 | LoRA 后零 IoU 数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| equipment_proximity | 3 | 0.2611 | 0.4039 | +0.1428 | 1 |
| guardrail_missing | 4 | 0.0410 | 0.2341 | +0.1931 | 1 |
| harness_missing | 4 | 0.5083 | 0.5382 | +0.0299 | 0 |
| helmet_missing | 4 | 0.3794 | 0.5689 | +0.1895 | 0 |
| no helmet | 15 | 0.5712 | 0.5841 | +0.0129 | 2 |
| no jacket | 12 | 0.3109 | 0.3583 | +0.0474 | 2 |
| opening_unprotected | 8 | 0.2326 | 0.4065 | +0.1740 | 1 |
| poor_housekeeping | 4 | 0.2215 | 0.5622 | +0.3407 | 1 |
| safe | 13 | 0.6412 | 0.7693 | +0.1281 | 0 |
| unsafe | 19 | 0.0750 | 0.1849 | +0.1099 | 8 |

评估效率:

- Clean val: 17.84 秒,0.425 秒/样本。
- Full val: 35.83 秒,0.417 秒/样本。

## 结论

本轮 Clean030 LoRA 微调有效。

在完整 `ReasonSeg|val` 上,模型从 base 的 `gIoU=0.3408 / cIoU=0.3177 / Dice=0.4180` 提升到 `gIoU=0.4494 / cIoU=0.3858 / Dice=0.5156`。其中 gIoU 提升 10.86 个点,且 10 个类别的 Mean IoU 均有提升,说明收益并非只来自少数大目标或单一类别。`IoU >= 0.5` 的样本增加 10 个,零 IoU 样本减少 6 个,进一步支持完整验证集上的迁移收益成立。

Mean Precision 提升 12.61 个点,对应 False Positive Area 减少 30.1%,说明误检明显减少。Mean Recall 只提升 2.84 个点,False Negative Area 仅减少 7.9%,说明漏检仍是下一轮主要问题。

Clean val 从 `gIoU=0.6435` 提升到 `0.7119`,说明高置信样本可被 LoRA 有效吸收。但 clean val 仍有 1 个严格零 IoU 样本(`poor_housekeeping`)和 2 个接近零 IoU 的 `no_helmet` 样本,说明即使在筛选后的 easy subset 上,部分 prompt/mask 或实例选择仍不稳定。

分类型结果中 `poor_housekeeping`、`guardrail_missing`、`helmet_missing`、`opening_unprotected` 提升较明显;但这些类别样本数仅 4-8 个,不能据此判断类别能力已经稳定。`unsafe` 虽从 0.0750 提升到 0.1849,仍有 8/19 个零 IoU,是当前最明确的主要短板。`no helmet` 和 `harness_missing` 的提升较小,下一轮也需要通过新增独立样本确认是否已接近瓶颈。

判断标准:

- 如果 clean val 提升,说明 Clean030 高置信样本可被 LoRA 有效吸收。
- 如果 full val 同时提升,说明对真实验证集有迁移收益。
- 如果只提升 clean val 而 full val 不提升,说明模型只学到了 easy subset,下一轮需要补 `guardrail_missing`、`unsafe`、`equipment_proximity`、`poor_housekeeping` 的人工核验 hard cases。

## 备注

- 本实验使用 Clean030 作为第一轮链路验证集,不把筛选后的 clean val 作为唯一正式指标。
- 正式结论必须以完整 `ReasonSeg|val` 的 LoRA 前后对比为准。
- `ReasonSegClean030` 是从原始 jpg/json 复制出的子集,不包含新生成标签,也不使用模型预测 mask 作为标签。
- Full val 的严格零 IoU 样本包括 `equipment_proximity` 1 个、`guardrail_missing` 1 个、`opening_unprotected` 1 个、`poor_housekeeping` 1 个、`no helmet` 2 个、`no jacket` 2 个和 `unsafe` 8 个。
- 下一轮建议优先补充并人工核验 `unsafe` hard cases,其次覆盖 `equipment_proximity`、`guardrail_missing`、`opening_unprotected`、`poor_housekeeping`、`no helmet` 和 `no jacket`,并继续保留完整 `ReasonSeg|val` 作为正式对比集。
- 分类型 Mean IoU 来自 `per_sample_metrics.csv` 的算术平均;小样本类别的数值仅用于定位问题,不作为稳定的类别性能结论。
