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

<!-- benchmark-comparison:clean-val-samples:start -->
## Clean Val 逐样本微调前后对比

- Base: `exp/runs/lisa13b-local-val/outputs/per_sample_metrics.jsonl`
- Tuned: `exp/runs/lisa13b-clean030-lora-v1/clean-eval-outputs/per_sample_metrics.jsonl`
- Matched samples: `42`
- Improved: `32`
- Regressed: `10`
- Unchanged: `0`
- Mean delta IoU: `+0.0684`

### All Samples Sorted by IoU Change (Best to Worst)

| Rank | Change | Delta IoU | Base IoU | Tuned IoU | Delta Dice | Label | Image | Base View | Tuned View | Prompt |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 1 | Improved | +0.5149 | 0.3122 | 0.8270 | +0.4295 | poor_housekeeping | `val__002__-poor_housekeeping-9-_jpg.rf.36f9a475b127b85f69ba023fc29a16e6__poor_housekeeping.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00025_val__002__-poor_housekeeping-9-_jpg.rf.36f9a475b127b85f69ba023fc29a16e6__poor_housekeeping_mask0.md) | [tuned](clean-eval-outputs/visualizations/00011_val__002__-poor_housekeeping-9-_jpg.rf.36f9a475b127b85f69ba023fc29a16e6__poor_housekeeping_mask0.md) | 标出现场材料堆放混乱或文明施工不到位的区域。 |
| 2 | Improved | +0.5101 | 0.4134 | 0.9235 | +0.3752 | no jacket | `val__004__IMG20241113162223_jpg.rf.433b0cb6b59de8d7728e3ee60894c40f__no_jacket.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00059_val__004__IMG20241113162223_jpg.rf.433b0cb6b59de8d7728e3ee60894c40f__no_jacket_mask0.md) | [tuned](clean-eval-outputs/visualizations/00026_val__004__IMG20241113162223_jpg.rf.433b0cb6b59de8d7728e3ee60894c40f__no_jacket_mask0.md) | 把缺少反光衣防护的作业人员分割出来。 |
| 3 | Improved | +0.4288 | 0.5132 | 0.9420 | +0.2918 | no helmet | `val__004__IMG20241113162223_jpg.rf.433b0cb6b59de8d7728e3ee60894c40f__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00058_val__004__IMG20241113162223_jpg.rf.433b0cb6b59de8d7728e3ee60894c40f__no_helmet_mask0.md) | [tuned](clean-eval-outputs/visualizations/00025_val__004__IMG20241113162223_jpg.rf.433b0cb6b59de8d7728e3ee60894c40f__no_helmet_mask0.md) | 现场哪些人员存在未戴安全帽的安全隐患?请分割出来。 |
| 4 | Improved | +0.4274 | 0.5328 | 0.9602 | +0.2845 | helmet_missing | `val__002__-helmet_missing-238-_jpg.rf.d793369e91ce496d9e82d59c120a433a__helmet_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00012_val__002__-helmet_missing-238-_jpg.rf.d793369e91ce496d9e82d59c120a433a__helmet_missing_mask0.md) | [tuned](clean-eval-outputs/visualizations/00006_val__002__-helmet_missing-238-_jpg.rf.d793369e91ce496d9e82d59c120a433a__helmet_missing_mask0.md) | 现场哪些人员存在未戴安全帽的安全隐患?请分割出来。 |
| 5 | Improved | +0.4269 | 0.4897 | 0.9166 | +0.2990 | no helmet | `val__004__IMG_20241113_162221784_jpg.rf.34ad0c7c7fdf4fc5716db91f378fc362__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00082_val__004__IMG_20241113_162221784_jpg.rf.34ad0c7c7fdf4fc5716db91f378fc362__no_helmet_mask0.md) | [tuned](clean-eval-outputs/visualizations/00039_val__004__IMG_20241113_162221784_jpg.rf.34ad0c7c7fdf4fc5716db91f378fc362__no_helmet_mask0.md) | 现场哪些人员存在未戴安全帽的安全隐患?请分割出来。 |
| 6 | Improved | +0.4025 | 0.5204 | 0.9229 | +0.2754 | no helmet | `val__004__IMG20241113161357_jpg.rf.8d057aafc970f94c8f2f31eaae4e159c__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00050_val__004__IMG20241113161357_jpg.rf.8d057aafc970f94c8f2f31eaae4e159c__no_helmet_mask0.md) | [tuned](clean-eval-outputs/visualizations/00021_val__004__IMG20241113161357_jpg.rf.8d057aafc970f94c8f2f31eaae4e159c__no_helmet_mask0.md) | 把没有做好头部防护、未戴安全帽的人分割出来。 |
| 7 | Improved | +0.3747 | 0.5693 | 0.9440 | +0.2456 | no helmet | `val__004__IMG_20241113_161405_jpg.rf.7b78a6ff625a0c910f3c1a1839dae955__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00062_val__004__IMG_20241113_161405_jpg.rf.7b78a6ff625a0c910f3c1a1839dae955__no_helmet_mask0.md) | [tuned](clean-eval-outputs/visualizations/00028_val__004__IMG_20241113_161405_jpg.rf.7b78a6ff625a0c910f3c1a1839dae955__no_helmet_mask0.md) | 把没有做好头部防护、未戴安全帽的人分割出来。 |
| 8 | Improved | +0.3691 | 0.5302 | 0.8994 | +0.2540 | no jacket | `val__004__IMG_20241113_162221784_jpg.rf.34ad0c7c7fdf4fc5716db91f378fc362__no_jacket.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00083_val__004__IMG_20241113_162221784_jpg.rf.34ad0c7c7fdf4fc5716db91f378fc362__no_jacket_mask0.md) | [tuned](clean-eval-outputs/visualizations/00040_val__004__IMG_20241113_162221784_jpg.rf.34ad0c7c7fdf4fc5716db91f378fc362__no_jacket_mask0.md) | 圈出没有穿反光衣或安全背心的作业人员。 |
| 9 | Improved | +0.2747 | 0.5571 | 0.8318 | +0.1926 | helmet_missing | `val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00011_val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing_mask0.md) | [tuned](clean-eval-outputs/visualizations/00005_val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing_mask0.md) | 标出未按规定佩戴安全帽的作业人员。 |
| 10 | Improved | +0.2622 | 0.6766 | 0.9389 | +0.1613 | opening_unprotected | `val__002__-opening_unprotected-47-_JPG.rf.4a2d026cab550ae1a0c3efad740dba3a__opening_unprotected.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00018_val__002__-opening_unprotected-47-_JPG.rf.4a2d026cab550ae1a0c3efad740dba3a__opening_unprotected_mask0.md) | [tuned](clean-eval-outputs/visualizations/00009_val__002__-opening_unprotected-47-_JPG.rf.4a2d026cab550ae1a0c3efad740dba3a__opening_unprotected_mask0.md) | 圈出没有防护的洞口或临边区域。 |
| 11 | Improved | +0.2401 | 0.6403 | 0.8804 | +0.1557 | no jacket | `val__004__IMG_20241113_161503177_jpg.rf.9caaa1a0ebdcc82457a97f52e8e41dd1__no_jacket.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00067_val__004__IMG_20241113_161503177_jpg.rf.9caaa1a0ebdcc82457a97f52e8e41dd1__no_jacket_mask0.md) | [tuned](clean-eval-outputs/visualizations/00030_val__004__IMG_20241113_161503177_jpg.rf.9caaa1a0ebdcc82457a97f52e8e41dd1__no_jacket_mask0.md) | 把缺少反光衣防护的作业人员分割出来。 |
| 12 | Improved | +0.2353 | 0.7113 | 0.9466 | +0.1413 | safe | `val__004__IMG20241113161357_jpg.rf.8d057aafc970f94c8f2f31eaae4e159c__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00051_val__004__IMG20241113161357_jpg.rf.8d057aafc970f94c8f2f31eaae4e159c__safe_mask0.md) | [tuned](clean-eval-outputs/visualizations/00022_val__004__IMG20241113161357_jpg.rf.8d057aafc970f94c8f2f31eaae4e159c__safe_mask0.md) | 圈出图中被标注为安全状态的作业人员或区域。 |
| 13 | Improved | +0.2098 | 0.7369 | 0.9467 | +0.1241 | opening_unprotected | `val__002__-opening_unprotected-14-_jpg.rf.f93c1e97c9f6ccaf8d87d079db200754__opening_unprotected.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00016_val__002__-opening_unprotected-14-_jpg.rf.f93c1e97c9f6ccaf8d87d079db200754__opening_unprotected_mask0.md) | [tuned](clean-eval-outputs/visualizations/00008_val__002__-opening_unprotected-14-_jpg.rf.f93c1e97c9f6ccaf8d87d079db200754__opening_unprotected_mask0.md) | 图中哪些洞口或临边没有做防护?请分割出来。 |
| 14 | Improved | +0.1834 | 0.5710 | 0.7544 | +0.1331 | no helmet | `val__004__IMG20241113160529_jpg.rf.01cc514ff77c809eb7eb263d362448cd__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00029_val__004__IMG20241113160529_jpg.rf.01cc514ff77c809eb7eb263d362448cd__no_helmet_mask0.md) | [tuned](clean-eval-outputs/visualizations/00012_val__004__IMG20241113160529_jpg.rf.01cc514ff77c809eb7eb263d362448cd__no_helmet_mask0.md) | 标出未按规定佩戴安全帽的作业人员。 |
| 15 | Improved | +0.1756 | 0.3275 | 0.5031 | +0.1760 | harness_missing | `val__002__-harness_missing-199-_jpg.rf.613784a2a3a88e15b68ace04501a1544__harness_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00007_val__002__-harness_missing-199-_jpg.rf.613784a2a3a88e15b68ace04501a1544__harness_missing_mask0.md) | [tuned](clean-eval-outputs/visualizations/00001_val__002__-harness_missing-199-_jpg.rf.613784a2a3a88e15b68ace04501a1544__harness_missing_mask0.md) | 把缺少安全带防护的作业人员分割出来。 |
| 16 | Improved | +0.1465 | 0.4242 | 0.5707 | +0.1310 | safe | `val__004__IMG_20241113_161405_jpg.rf.7b78a6ff625a0c910f3c1a1839dae955__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00063_val__004__IMG_20241113_161405_jpg.rf.7b78a6ff625a0c910f3c1a1839dae955__safe_mask0.md) | [tuned](clean-eval-outputs/visualizations/00029_val__004__IMG_20241113_161405_jpg.rf.7b78a6ff625a0c910f3c1a1839dae955__safe_mask0.md) | 标出现场处于安全状态的目标。 |
| 17 | Improved | +0.1429 | 0.6013 | 0.7441 | +0.1023 | equipment_proximity | `val__002__-equipment_proximity-203-_jpg.rf.f10bae0078f7b417b8da3c3f67acc70f__equipment_proximity.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00001_val__002__-equipment_proximity-203-_jpg.rf.f10bae0078f7b417b8da3c3f67acc70f__equipment_proximity_mask0.md) | [tuned](clean-eval-outputs/visualizations/00000_val__002__-equipment_proximity-203-_jpg.rf.f10bae0078f7b417b8da3c3f67acc70f__equipment_proximity_mask0.md) | 图中哪些位置存在设备靠近人员的安全隐患?请分割出来。 |
| 18 | Improved | +0.0913 | 0.8846 | 0.9759 | +0.0490 | no helmet | `val__004__IMG20241113161529_jpg.rf.e77d13689e9d29512af57438dfc09685__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00053_val__004__IMG20241113161529_jpg.rf.e77d13689e9d29512af57438dfc09685__no_helmet_mask0.md) | [tuned](clean-eval-outputs/visualizations/00023_val__004__IMG20241113161529_jpg.rf.e77d13689e9d29512af57438dfc09685__no_helmet_mask0.md) | 现场哪些人员存在未戴安全帽的安全隐患?请分割出来。 |
| 19 | Improved | +0.0706 | 0.4977 | 0.5683 | +0.0601 | no helmet | `val__004__IMG20241113160624_jpg.rf.4b539618ccf9c3ab60c656685e98e24e__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00034_val__004__IMG20241113160624_jpg.rf.4b539618ccf9c3ab60c656685e98e24e__no_helmet_mask0.md) | [tuned](clean-eval-outputs/visualizations/00015_val__004__IMG20241113160624_jpg.rf.4b539618ccf9c3ab60c656685e98e24e__no_helmet_mask0.md) | 标出未按规定佩戴安全帽的作业人员。 |
| 20 | Improved | +0.0662 | 0.4135 | 0.4797 | +0.0633 | helmet_missing | `val__002__-helmet_missing-245-_jpg.rf.dcbe61ef31ed007835863b76354c7261__helmet_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00013_val__002__-helmet_missing-245-_jpg.rf.dcbe61ef31ed007835863b76354c7261__helmet_missing_mask0.md) | [tuned](clean-eval-outputs/visualizations/00007_val__002__-helmet_missing-245-_jpg.rf.dcbe61ef31ed007835863b76354c7261__helmet_missing_mask0.md) | 标出未按规定佩戴安全帽的作业人员。 |
| 21 | Improved | +0.0636 | 0.8913 | 0.9550 | +0.0344 | no helmet | `val__004__IMG_20241113_161758877_HDR_jpg.rf.362f3e155a81c2679a7690b2ef961abb__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00068_val__004__IMG_20241113_161758877_HDR_jpg.rf.362f3e155a81c2679a7690b2ef961abb__no_helmet_mask0.md) | [tuned](clean-eval-outputs/visualizations/00031_val__004__IMG_20241113_161758877_HDR_jpg.rf.362f3e155a81c2679a7690b2ef961abb__no_helmet_mask0.md) | 标出未按规定佩戴安全帽的作业人员。 |
| 22 | Improved | +0.0602 | 0.4602 | 0.5203 | +0.0542 | no helmet | `val__004__IMG20241113160638_jpg.rf.f032f3f45329778ae3108a9bbab937d1__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00037_val__004__IMG20241113160638_jpg.rf.f032f3f45329778ae3108a9bbab937d1__no_helmet_mask0.md) | [tuned](clean-eval-outputs/visualizations/00016_val__004__IMG20241113160638_jpg.rf.f032f3f45329778ae3108a9bbab937d1__no_helmet_mask0.md) | 把没有做好头部防护、未戴安全帽的人分割出来。 |
| 23 | Improved | +0.0492 | 0.3824 | 0.4315 | +0.0497 | no jacket | `val__004__IMG20241113160838_jpg.rf.a3a315c1e66d2e5d0f19bf139df5b5d6__no_jacket.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00045_val__004__IMG20241113160838_jpg.rf.a3a315c1e66d2e5d0f19bf139df5b5d6__no_jacket_mask0.md) | [tuned](clean-eval-outputs/visualizations/00019_val__004__IMG20241113160838_jpg.rf.a3a315c1e66d2e5d0f19bf139df5b5d6__no_jacket_mask0.md) | 标出未按要求穿戴反光背心的工人。 |
| 24 | Improved | +0.0176 | 0.9328 | 0.9504 | +0.0094 | safe | `val__004__IMG_20241113_161954_jpg.rf.6d1a5633986ccbdc7de79340bce250ae__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00077_val__004__IMG_20241113_161954_jpg.rf.6d1a5633986ccbdc7de79340bce250ae__safe_mask0.md) | [tuned](clean-eval-outputs/visualizations/00036_val__004__IMG_20241113_161954_jpg.rf.6d1a5633986ccbdc7de79340bce250ae__safe_mask0.md) | 圈出图中被标注为安全状态的作业人员或区域。 |
| 25 | Improved | +0.0164 | 0.9116 | 0.9280 | +0.0089 | unsafe | `val__004__IMG_20241113_162203623_jpg.rf.e05b826226143833b9ab04f2d13ce7a1__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00081_val__004__IMG_20241113_162203623_jpg.rf.e05b826226143833b9ab04f2d13ce7a1__unsafe_mask0.md) | [tuned](clean-eval-outputs/visualizations/00038_val__004__IMG_20241113_162203623_jpg.rf.e05b826226143833b9ab04f2d13ce7a1__unsafe_mask0.md) | 圈出图中被标注为不安全状态的作业人员或区域。 |
| 26 | Improved | +0.0161 | 0.9378 | 0.9539 | +0.0085 | safe | `val__004__IMG_20241113_161953162_jpg.rf.2fc5157cdd82110c6541270dcb7e5f82__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00076_val__004__IMG_20241113_161953162_jpg.rf.2fc5157cdd82110c6541270dcb7e5f82__safe_mask0.md) | [tuned](clean-eval-outputs/visualizations/00035_val__004__IMG_20241113_161953162_jpg.rf.2fc5157cdd82110c6541270dcb7e5f82__safe_mask0.md) | 标出现场处于安全状态的目标。 |
| 27 | Improved | +0.0138 | 0.9132 | 0.9269 | +0.0075 | safe | `val__004__IMG_20241113_162228212_jpg.rf.1021918a81b951a030f665662fb11a58__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00084_val__004__IMG_20241113_162228212_jpg.rf.1021918a81b951a030f665662fb11a58__safe_mask0.md) | [tuned](clean-eval-outputs/visualizations/00041_val__004__IMG_20241113_162228212_jpg.rf.1021918a81b951a030f665662fb11a58__safe_mask0.md) | 圈出图中被标注为安全状态的作业人员或区域。 |
| 28 | Improved | +0.0134 | 0.9167 | 0.9301 | +0.0072 | safe | `val__004__IMG_20241113_161941847_jpg.rf.ce72086317700d7f52ce01bf415ed037__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00070_val__004__IMG_20241113_161941847_jpg.rf.ce72086317700d7f52ce01bf415ed037__safe_mask0.md) | [tuned](clean-eval-outputs/visualizations/00032_val__004__IMG_20241113_161941847_jpg.rf.ce72086317700d7f52ce01bf415ed037__safe_mask0.md) | 圈出图中被标注为安全状态的作业人员或区域。 |
| 29 | Improved | +0.0087 | 0.9341 | 0.9428 | +0.0046 | safe | `val__004__IMG_20241113_161952050_jpg.rf.7c0b52ecb477f8f7ef8b71489dac19cf__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00074_val__004__IMG_20241113_161952050_jpg.rf.7c0b52ecb477f8f7ef8b71489dac19cf__safe_mask0.md) | [tuned](clean-eval-outputs/visualizations/00034_val__004__IMG_20241113_161952050_jpg.rf.7c0b52ecb477f8f7ef8b71489dac19cf__safe_mask0.md) | 标出现场处于安全状态的目标。 |
| 30 | Improved | +0.0081 | 0.9296 | 0.9377 | +0.0044 | safe | `val__004__IMG20241113161931_jpg.rf.a8d69015746e42c5014b2422eaf72c7b__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00056_val__004__IMG20241113161931_jpg.rf.a8d69015746e42c5014b2422eaf72c7b__safe_mask0.md) | [tuned](clean-eval-outputs/visualizations/00024_val__004__IMG20241113161931_jpg.rf.a8d69015746e42c5014b2422eaf72c7b__safe_mask0.md) | 标出现场处于安全状态的目标。 |
| 31 | Improved | +0.0052 | 0.4646 | 0.4699 | +0.0049 | harness_missing | `val__002__-harness_missing-23-_JPG.rf.9e5a10303ac02c86c71739d108101ea4__harness_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00008_val__002__-harness_missing-23-_JPG.rf.9e5a10303ac02c86c71739d108101ea4__harness_missing_mask0.md) | [tuned](clean-eval-outputs/visualizations/00002_val__002__-harness_missing-23-_JPG.rf.9e5a10303ac02c86c71739d108101ea4__harness_missing_mask0.md) | 把缺少安全带防护的作业人员分割出来。 |
| 32 | Improved | +0.0006 | 0.9080 | 0.9086 | +0.0003 | no jacket | `val__004__IMG_20241113_162029941_jpg.rf.9cb9cb44acb2667b768970c0c5b37320__no_jacket.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00078_val__004__IMG_20241113_162029941_jpg.rf.9cb9cb44acb2667b768970c0c5b37320__no_jacket_mask0.md) | [tuned](clean-eval-outputs/visualizations/00037_val__004__IMG_20241113_162029941_jpg.rf.9cb9cb44acb2667b768970c0c5b37320__no_jacket_mask0.md) | 现场哪些人员没有穿安全背心?请分割出来。 |
| 33 | Regressed | -0.0169 | 0.9569 | 0.9400 | -0.0089 | no helmet | `val__004__IMG20241113160936_jpg.rf.162c6c777d07be2ca5c077d6906547f6__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00047_val__004__IMG20241113160936_jpg.rf.162c6c777d07be2ca5c077d6906547f6__no_helmet_mask0.md) | [tuned](clean-eval-outputs/visualizations/00020_val__004__IMG20241113160936_jpg.rf.162c6c777d07be2ca5c077d6906547f6__no_helmet_mask0.md) | 把没有做好头部防护、未戴安全帽的人分割出来。 |
| 34 | Regressed | -0.0171 | 0.5368 | 0.5197 | -0.0146 | harness_missing | `val__002__-harness_missing-45-_JPG.rf.8bddeb686fc5fada6004b77add73011f__harness_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00009_val__002__-harness_missing-45-_JPG.rf.8bddeb686fc5fada6004b77add73011f__harness_missing_mask0.md) | [tuned](clean-eval-outputs/visualizations/00003_val__002__-harness_missing-45-_JPG.rf.8bddeb686fc5fada6004b77add73011f__harness_missing_mask0.md) | 标出未按要求佩戴安全带的作业人员。 |
| 35 | Regressed | -0.0442 | 0.7045 | 0.6603 | -0.0312 | harness_missing | `val__002__-harness_missing-69-_JPG.rf.1205c382c7eb97a239309f883a43c734__harness_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00010_val__002__-harness_missing-69-_JPG.rf.1205c382c7eb97a239309f883a43c734__harness_missing_mask0.md) | [tuned](clean-eval-outputs/visualizations/00004_val__002__-harness_missing-69-_JPG.rf.1205c382c7eb97a239309f883a43c734__harness_missing_mask0.md) | 把缺少安全带防护的作业人员分割出来。 |
| 36 | Regressed | -0.1981 | 0.4585 | 0.2603 | -0.2156 | safe | `val__004__IMG20241113160613_jpg.rf.028bfffcd39729b2b7c7833027bb05f6__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00032_val__004__IMG20241113160613_jpg.rf.028bfffcd39729b2b7c7833027bb05f6__safe_mask0.md) | [tuned](clean-eval-outputs/visualizations/00014_val__004__IMG20241113160613_jpg.rf.028bfffcd39729b2b7c7833027bb05f6__safe_mask0.md) | 圈出图中被标注为安全状态的作业人员或区域。 |
| 37 | Regressed | -0.2367 | 0.9058 | 0.6691 | -0.1488 | safe | `val__004__IMG_20241113_161944298_jpg.rf.8b7de1452bddd5e3744643b6b17d30e1__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00072_val__004__IMG_20241113_161944298_jpg.rf.8b7de1452bddd5e3744643b6b17d30e1__safe_mask0.md) | [tuned](clean-eval-outputs/visualizations/00033_val__004__IMG_20241113_161944298_jpg.rf.8b7de1452bddd5e3744643b6b17d30e1__safe_mask0.md) | 圈出图中被标注为安全状态的作业人员或区域。 |
| 38 | Regressed | -0.2504 | 0.4456 | 0.1952 | -0.2898 | no jacket | `val__004__IMG20241113160529_jpg.rf.01cc514ff77c809eb7eb263d362448cd__no_jacket.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00030_val__004__IMG20241113160529_jpg.rf.01cc514ff77c809eb7eb263d362448cd__no_jacket_mask0.md) | [tuned](clean-eval-outputs/visualizations/00013_val__004__IMG20241113160529_jpg.rf.01cc514ff77c809eb7eb263d362448cd__no_jacket_mask0.md) | 圈出没有穿反光衣或安全背心的作业人员。 |
| 39 | Regressed | -0.4428 | 0.4428 | 0.0000 | -0.6138 | no helmet | `val__004__IMG20241113160817_jpg.rf.9c7cd48d2297fc816a026d117f0df99b__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00042_val__004__IMG20241113160817_jpg.rf.9c7cd48d2297fc816a026d117f0df99b__no_helmet_mask0.md) | [tuned](clean-eval-outputs/visualizations/00018_val__004__IMG20241113160817_jpg.rf.9c7cd48d2297fc816a026d117f0df99b__no_helmet_mask0.md) | 把没有做好头部防护、未戴安全帽的人分割出来。 |
| 40 | Regressed | -0.5302 | 0.5302 | 0.0000 | -0.6930 | poor_housekeeping | `val__002__-poor_housekeeping-132-_jpg.rf.c6a2c42ceb4d1eae5cd2307fedce5fe6__poor_housekeeping.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00023_val__002__-poor_housekeeping-132-_jpg.rf.c6a2c42ceb4d1eae5cd2307fedce5fe6__poor_housekeeping_mask0.md) | [tuned](clean-eval-outputs/visualizations/00010_val__002__-poor_housekeeping-132-_jpg.rf.c6a2c42ceb4d1eae5cd2307fedce5fe6__poor_housekeeping_mask0.md) | 图中哪些区域存在场地整理不到位的安全隐患?请分割出来。 |
| 41 | Regressed | -0.5643 | 0.6591 | 0.0948 | -0.6214 | no helmet | `val__004__IMG20241113160658_jpg.rf.509371336267fc48e02387bc78859d8f__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00039_val__004__IMG20241113160658_jpg.rf.509371336267fc48e02387bc78859d8f__no_helmet_mask0.md) | [tuned](clean-eval-outputs/visualizations/00017_val__004__IMG20241113160658_jpg.rf.509371336267fc48e02387bc78859d8f__no_helmet_mask0.md) | 标出未按规定佩戴安全帽的作业人员。 |
| 42 | Regressed | -0.6543 | 0.8814 | 0.2271 | -0.5668 | no helmet | `val__004__IMG_20241113_161340_jpg.rf.d2e6016a68c883c9a28b149919ca2c5c__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00060_val__004__IMG_20241113_161340_jpg.rf.d2e6016a68c883c9a28b149919ca2c5c__no_helmet_mask0.md) | [tuned](clean-eval-outputs/visualizations/00027_val__004__IMG_20241113_161340_jpg.rf.d2e6016a68c883c9a28b149919ca2c5c__no_helmet_mask0.md) | 标出未按规定佩戴安全帽的作业人员。 |
<!-- benchmark-comparison:clean-val-samples:end -->

<!-- benchmark-comparison:full-val-samples:start -->
## Full Val 逐样本微调前后对比

- Base: `exp/runs/lisa13b-local-val/outputs/per_sample_metrics.jsonl`
- Tuned: `exp/runs/lisa13b-clean030-lora-v1/full-eval-outputs/per_sample_metrics.jsonl`
- Matched samples: `86`
- Improved: `57`
- Regressed: `18`
- Unchanged: `11`
- Mean delta IoU: `+0.1087`

### All Samples Sorted by IoU Change (Best to Worst)

| Rank | Change | Delta IoU | Base IoU | Tuned IoU | Delta Dice | Label | Image | Base View | Tuned View | Prompt |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 1 | Improved | +0.9596 | 0.0000 | 0.9596 | +0.9794 | poor_housekeeping | `val__002__-poor_housekeeping-93-_jpg.rf.893e8f8298279e8c46d0fa8a503d58ff__poor_housekeeping.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00026_val__002__-poor_housekeeping-93-_jpg.rf.893e8f8298279e8c46d0fa8a503d58ff__poor_housekeeping_mask0.md) | [tuned](full-eval-outputs/visualizations/00026_val__002__-poor_housekeeping-93-_jpg.rf.893e8f8298279e8c46d0fa8a503d58ff__poor_housekeeping_mask0.md) | 圈出施工现场杂乱、可能影响通行安全的位置。 |
| 2 | Improved | +0.9299 | 0.0023 | 0.9322 | +0.9604 | unsafe | `val__004__IMG20241113161129_jpg.rf.3a3917985cacdef53fdb706f8f961778__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00048_val__004__IMG20241113161129_jpg.rf.3a3917985cacdef53fdb706f8f961778__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00048_val__004__IMG20241113161129_jpg.rf.3a3917985cacdef53fdb706f8f961778__unsafe_mask0.md) | 请分割图中存在安全风险的目标区域。 |
| 3 | Improved | +0.7725 | 0.0004 | 0.7729 | +0.8711 | safe | `val__004__IMG20241113160817_jpg.rf.9c7cd48d2297fc816a026d117f0df99b__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00043_val__004__IMG20241113160817_jpg.rf.9c7cd48d2297fc816a026d117f0df99b__safe_mask0.md) | [tuned](full-eval-outputs/visualizations/00043_val__004__IMG20241113160817_jpg.rf.9c7cd48d2297fc816a026d117f0df99b__safe_mask0.md) | 请分割图中符合安全要求的目标区域。 |
| 4 | Improved | +0.7201 | 0.0000 | 0.7201 | +0.8373 | safe | `val__004__IMG20241113161258_jpg.rf.786b1786c44a64de7b328ef8c5a34e93__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00049_val__004__IMG20241113161258_jpg.rf.786b1786c44a64de7b328ef8c5a34e93__safe_mask0.md) | [tuned](full-eval-outputs/visualizations/00049_val__004__IMG20241113161258_jpg.rf.786b1786c44a64de7b328ef8c5a34e93__safe_mask0.md) | 请分割图中符合安全要求的目标区域。 |
| 5 | Improved | +0.7082 | 0.0000 | 0.7082 | +0.8292 | unsafe | `val__004__IMG20241113161910_jpg.rf.7330e0f90826aafd4c8a75a7cfdfaa87__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00055_val__004__IMG20241113161910_jpg.rf.7330e0f90826aafd4c8a75a7cfdfaa87__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00055_val__004__IMG20241113161910_jpg.rf.7330e0f90826aafd4c8a75a7cfdfaa87__unsafe_mask0.md) | 标出现场存在不安全状态的目标。 |
| 6 | Improved | +0.5149 | 0.3122 | 0.8270 | +0.4295 | poor_housekeeping | `val__002__-poor_housekeeping-9-_jpg.rf.36f9a475b127b85f69ba023fc29a16e6__poor_housekeeping.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00025_val__002__-poor_housekeeping-9-_jpg.rf.36f9a475b127b85f69ba023fc29a16e6__poor_housekeeping_mask0.md) | [tuned](full-eval-outputs/visualizations/00025_val__002__-poor_housekeeping-9-_jpg.rf.36f9a475b127b85f69ba023fc29a16e6__poor_housekeeping_mask0.md) | 标出现场材料堆放混乱或文明施工不到位的区域。 |
| 7 | Improved | +0.5138 | 0.0627 | 0.5765 | +0.6134 | opening_unprotected | `val__002__-opening_unprotected-66-_jpg.rf.7727a02d7de525d6c814c405244519a0__opening_unprotected.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00022_val__002__-opening_unprotected-66-_jpg.rf.7727a02d7de525d6c814c405244519a0__opening_unprotected_mask0.md) | [tuned](full-eval-outputs/visualizations/00022_val__002__-opening_unprotected-66-_jpg.rf.7727a02d7de525d6c814c405244519a0__opening_unprotected_mask0.md) | 图中哪些洞口或临边没有做防护?请分割出来。 |
| 8 | Improved | +0.5101 | 0.4134 | 0.9235 | +0.3752 | no jacket | `val__004__IMG20241113162223_jpg.rf.433b0cb6b59de8d7728e3ee60894c40f__no_jacket.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00059_val__004__IMG20241113162223_jpg.rf.433b0cb6b59de8d7728e3ee60894c40f__no_jacket_mask0.md) | [tuned](full-eval-outputs/visualizations/00059_val__004__IMG20241113162223_jpg.rf.433b0cb6b59de8d7728e3ee60894c40f__no_jacket_mask0.md) | 把缺少反光衣防护的作业人员分割出来。 |
| 9 | Improved | +0.4715 | 0.0000 | 0.4715 | +0.6408 | guardrail_missing | `val__002__-guardrail_missing-61-_jpg.rf.c0dd9f521c735b672d1318895a1aaffa__guardrail_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00005_val__002__-guardrail_missing-61-_jpg.rf.c0dd9f521c735b672d1318895a1aaffa__guardrail_missing_mask0.md) | [tuned](full-eval-outputs/visualizations/00005_val__002__-guardrail_missing-61-_jpg.rf.c0dd9f521c735b672d1318895a1aaffa__guardrail_missing_mask0.md) | 指出没有设置防护栏杆的临边区域。 |
| 10 | Improved | +0.4288 | 0.5132 | 0.9420 | +0.2918 | no helmet | `val__004__IMG20241113162223_jpg.rf.433b0cb6b59de8d7728e3ee60894c40f__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00058_val__004__IMG20241113162223_jpg.rf.433b0cb6b59de8d7728e3ee60894c40f__no_helmet_mask0.md) | [tuned](full-eval-outputs/visualizations/00058_val__004__IMG20241113162223_jpg.rf.433b0cb6b59de8d7728e3ee60894c40f__no_helmet_mask0.md) | 现场哪些人员存在未戴安全帽的安全隐患?请分割出来。 |
| 11 | Improved | +0.4274 | 0.5328 | 0.9602 | +0.2845 | helmet_missing | `val__002__-helmet_missing-238-_jpg.rf.d793369e91ce496d9e82d59c120a433a__helmet_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00012_val__002__-helmet_missing-238-_jpg.rf.d793369e91ce496d9e82d59c120a433a__helmet_missing_mask0.md) | [tuned](full-eval-outputs/visualizations/00012_val__002__-helmet_missing-238-_jpg.rf.d793369e91ce496d9e82d59c120a433a__helmet_missing_mask0.md) | 现场哪些人员存在未戴安全帽的安全隐患?请分割出来。 |
| 12 | Improved | +0.4269 | 0.4897 | 0.9166 | +0.2990 | no helmet | `val__004__IMG_20241113_162221784_jpg.rf.34ad0c7c7fdf4fc5716db91f378fc362__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00082_val__004__IMG_20241113_162221784_jpg.rf.34ad0c7c7fdf4fc5716db91f378fc362__no_helmet_mask0.md) | [tuned](full-eval-outputs/visualizations/00082_val__004__IMG_20241113_162221784_jpg.rf.34ad0c7c7fdf4fc5716db91f378fc362__no_helmet_mask0.md) | 现场哪些人员存在未戴安全帽的安全隐患?请分割出来。 |
| 13 | Improved | +0.4185 | 0.0438 | 0.4622 | +0.5484 | poor_housekeeping | `val__002__-poor_housekeeping-79-_JPG.rf.9c14f23630279126bb089ae474f6af94__poor_housekeeping.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00024_val__002__-poor_housekeeping-79-_JPG.rf.9c14f23630279126bb089ae474f6af94__poor_housekeeping_mask0.md) | [tuned](full-eval-outputs/visualizations/00024_val__002__-poor_housekeeping-79-_JPG.rf.9c14f23630279126bb089ae474f6af94__poor_housekeeping_mask0.md) | 圈出施工现场杂乱、可能影响通行安全的位置。 |
| 14 | Improved | +0.4025 | 0.5204 | 0.9229 | +0.2754 | no helmet | `val__004__IMG20241113161357_jpg.rf.8d057aafc970f94c8f2f31eaae4e159c__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00050_val__004__IMG20241113161357_jpg.rf.8d057aafc970f94c8f2f31eaae4e159c__no_helmet_mask0.md) | [tuned](full-eval-outputs/visualizations/00050_val__004__IMG20241113161357_jpg.rf.8d057aafc970f94c8f2f31eaae4e159c__no_helmet_mask0.md) | 把没有做好头部防护、未戴安全帽的人分割出来。 |
| 15 | Improved | +0.3747 | 0.5693 | 0.9440 | +0.2456 | no helmet | `val__004__IMG_20241113_161405_jpg.rf.7b78a6ff625a0c910f3c1a1839dae955__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00062_val__004__IMG_20241113_161405_jpg.rf.7b78a6ff625a0c910f3c1a1839dae955__no_helmet_mask0.md) | [tuned](full-eval-outputs/visualizations/00062_val__004__IMG_20241113_161405_jpg.rf.7b78a6ff625a0c910f3c1a1839dae955__no_helmet_mask0.md) | 把没有做好头部防护、未戴安全帽的人分割出来。 |
| 16 | Improved | +0.3691 | 0.5302 | 0.8994 | +0.2540 | no jacket | `val__004__IMG_20241113_162221784_jpg.rf.34ad0c7c7fdf4fc5716db91f378fc362__no_jacket.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00083_val__004__IMG_20241113_162221784_jpg.rf.34ad0c7c7fdf4fc5716db91f378fc362__no_jacket_mask0.md) | [tuned](full-eval-outputs/visualizations/00083_val__004__IMG_20241113_162221784_jpg.rf.34ad0c7c7fdf4fc5716db91f378fc362__no_jacket_mask0.md) | 圈出没有穿反光衣或安全背心的作业人员。 |
| 17 | Improved | +0.3221 | 0.1396 | 0.4617 | +0.3868 | opening_unprotected | `val__002__-opening_unprotected-11-_jpg.rf.b006631ca767138c74d5c79ee727de86__opening_unprotected.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00015_val__002__-opening_unprotected-11-_jpg.rf.b006631ca767138c74d5c79ee727de86__opening_unprotected_mask0.md) | [tuned](full-eval-outputs/visualizations/00015_val__002__-opening_unprotected-11-_jpg.rf.b006631ca767138c74d5c79ee727de86__opening_unprotected_mask0.md) | 圈出没有防护的洞口或临边区域。 |
| 18 | Improved | +0.2936 | 0.1739 | 0.4675 | +0.3409 | equipment_proximity | `val__002__-equipment_proximity-114-_jpg.rf.07492924a04683042df5548fae45af02__equipment_proximity.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00000_val__002__-equipment_proximity-114-_jpg.rf.07492924a04683042df5548fae45af02__equipment_proximity_mask0.md) | [tuned](full-eval-outputs/visualizations/00000_val__002__-equipment_proximity-114-_jpg.rf.07492924a04683042df5548fae45af02__equipment_proximity_mask0.md) | 圈出施工现场设备邻近作业人员的危险区域。 |
| 19 | Improved | +0.2747 | 0.5571 | 0.8318 | +0.1926 | helmet_missing | `val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00011_val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing_mask0.md) | [tuned](full-eval-outputs/visualizations/00011_val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing_mask0.md) | 标出未按规定佩戴安全帽的作业人员。 |
| 20 | Improved | +0.2622 | 0.6766 | 0.9389 | +0.1613 | opening_unprotected | `val__002__-opening_unprotected-47-_JPG.rf.4a2d026cab550ae1a0c3efad740dba3a__opening_unprotected.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00018_val__002__-opening_unprotected-47-_JPG.rf.4a2d026cab550ae1a0c3efad740dba3a__opening_unprotected_mask0.md) | [tuned](full-eval-outputs/visualizations/00018_val__002__-opening_unprotected-47-_JPG.rf.4a2d026cab550ae1a0c3efad740dba3a__opening_unprotected_mask0.md) | 圈出没有防护的洞口或临边区域。 |
| 21 | Improved | +0.2593 | 0.0001 | 0.2594 | +0.4118 | unsafe | `val__004__IMG20241113160838_jpg.rf.a3a315c1e66d2e5d0f19bf139df5b5d6__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00046_val__004__IMG20241113160838_jpg.rf.a3a315c1e66d2e5d0f19bf139df5b5d6__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00046_val__004__IMG20241113160838_jpg.rf.a3a315c1e66d2e5d0f19bf139df5b5d6__unsafe_mask0.md) | 圈出图中被标注为不安全状态的作业人员或区域。 |
| 22 | Improved | +0.2401 | 0.6403 | 0.8804 | +0.1557 | no jacket | `val__004__IMG_20241113_161503177_jpg.rf.9caaa1a0ebdcc82457a97f52e8e41dd1__no_jacket.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00067_val__004__IMG_20241113_161503177_jpg.rf.9caaa1a0ebdcc82457a97f52e8e41dd1__no_jacket_mask0.md) | [tuned](full-eval-outputs/visualizations/00067_val__004__IMG_20241113_161503177_jpg.rf.9caaa1a0ebdcc82457a97f52e8e41dd1__no_jacket_mask0.md) | 把缺少反光衣防护的作业人员分割出来。 |
| 23 | Improved | +0.2353 | 0.7113 | 0.9466 | +0.1413 | safe | `val__004__IMG20241113161357_jpg.rf.8d057aafc970f94c8f2f31eaae4e159c__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00051_val__004__IMG20241113161357_jpg.rf.8d057aafc970f94c8f2f31eaae4e159c__safe_mask0.md) | [tuned](full-eval-outputs/visualizations/00051_val__004__IMG20241113161357_jpg.rf.8d057aafc970f94c8f2f31eaae4e159c__safe_mask0.md) | 圈出图中被标注为安全状态的作业人员或区域。 |
| 24 | Improved | +0.2098 | 0.7369 | 0.9467 | +0.1241 | opening_unprotected | `val__002__-opening_unprotected-14-_jpg.rf.f93c1e97c9f6ccaf8d87d079db200754__opening_unprotected.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00016_val__002__-opening_unprotected-14-_jpg.rf.f93c1e97c9f6ccaf8d87d079db200754__opening_unprotected_mask0.md) | [tuned](full-eval-outputs/visualizations/00016_val__002__-opening_unprotected-14-_jpg.rf.f93c1e97c9f6ccaf8d87d079db200754__opening_unprotected_mask0.md) | 图中哪些洞口或临边没有做防护?请分割出来。 |
| 25 | Improved | +0.1834 | 0.5710 | 0.7544 | +0.1331 | no helmet | `val__004__IMG20241113160529_jpg.rf.01cc514ff77c809eb7eb263d362448cd__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00029_val__004__IMG20241113160529_jpg.rf.01cc514ff77c809eb7eb263d362448cd__no_helmet_mask0.md) | [tuned](full-eval-outputs/visualizations/00029_val__004__IMG20241113160529_jpg.rf.01cc514ff77c809eb7eb263d362448cd__no_helmet_mask0.md) | 标出未按规定佩戴安全帽的作业人员。 |
| 26 | Improved | +0.1832 | 0.1640 | 0.3472 | +0.2336 | guardrail_missing | `val__002__-guardrail_missing-64-_jpg.rf.ce53035dccc9877ee6e3ffad7fe23cd1__guardrail_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00006_val__002__-guardrail_missing-64-_jpg.rf.ce53035dccc9877ee6e3ffad7fe23cd1__guardrail_missing_mask0.md) | [tuned](full-eval-outputs/visualizations/00006_val__002__-guardrail_missing-64-_jpg.rf.ce53035dccc9877ee6e3ffad7fe23cd1__guardrail_missing_mask0.md) | 图中哪些位置存在防护栏杆缺失隐患?请分割出来。 |
| 27 | Improved | +0.1756 | 0.3275 | 0.5031 | +0.1760 | harness_missing | `val__002__-harness_missing-199-_jpg.rf.613784a2a3a88e15b68ace04501a1544__harness_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00007_val__002__-harness_missing-199-_jpg.rf.613784a2a3a88e15b68ace04501a1544__harness_missing_mask0.md) | [tuned](full-eval-outputs/visualizations/00007_val__002__-harness_missing-199-_jpg.rf.613784a2a3a88e15b68ace04501a1544__harness_missing_mask0.md) | 把缺少安全带防护的作业人员分割出来。 |
| 28 | Improved | +0.1481 | 0.2710 | 0.4190 | +0.1642 | safe | `val__004__IMG20241113160501_jpg.rf.e2fc6e55324dcda33f674df60a4c0547__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00027_val__004__IMG20241113160501_jpg.rf.e2fc6e55324dcda33f674df60a4c0547__safe_mask0.md) | [tuned](full-eval-outputs/visualizations/00027_val__004__IMG20241113160501_jpg.rf.e2fc6e55324dcda33f674df60a4c0547__safe_mask0.md) | 圈出图中被标注为安全状态的作业人员或区域。 |
| 29 | Improved | +0.1465 | 0.4242 | 0.5707 | +0.1310 | safe | `val__004__IMG_20241113_161405_jpg.rf.7b78a6ff625a0c910f3c1a1839dae955__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00063_val__004__IMG_20241113_161405_jpg.rf.7b78a6ff625a0c910f3c1a1839dae955__safe_mask0.md) | [tuned](full-eval-outputs/visualizations/00063_val__004__IMG_20241113_161405_jpg.rf.7b78a6ff625a0c910f3c1a1839dae955__safe_mask0.md) | 标出现场处于安全状态的目标。 |
| 30 | Improved | +0.1429 | 0.6013 | 0.7441 | +0.1023 | equipment_proximity | `val__002__-equipment_proximity-203-_jpg.rf.f10bae0078f7b417b8da3c3f67acc70f__equipment_proximity.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00001_val__002__-equipment_proximity-203-_jpg.rf.f10bae0078f7b417b8da3c3f67acc70f__equipment_proximity_mask0.md) | [tuned](full-eval-outputs/visualizations/00001_val__002__-equipment_proximity-203-_jpg.rf.f10bae0078f7b417b8da3c3f67acc70f__equipment_proximity_mask0.md) | 图中哪些位置存在设备靠近人员的安全隐患?请分割出来。 |
| 31 | Improved | +0.1274 | 0.0000 | 0.1274 | +0.2260 | unsafe | `val__004__IMG_20241113_161450_jpg.rf.b72ade8fc935970192f5e4c70fc7249e__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00066_val__004__IMG_20241113_161450_jpg.rf.b72ade8fc935970192f5e4c70fc7249e__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00066_val__004__IMG_20241113_161450_jpg.rf.b72ade8fc935970192f5e4c70fc7249e__unsafe_mask0.md) | 请分割图中存在安全风险的目标区域。 |
| 32 | Improved | +0.1178 | 0.0000 | 0.1178 | +0.2108 | guardrail_missing | `val__002__-guardrail_missing-58-_jpg.rf.391f56af8166b7a9978f49176685f505__guardrail_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00004_val__002__-guardrail_missing-58-_jpg.rf.391f56af8166b7a9978f49176685f505__guardrail_missing_mask0.md) | [tuned](full-eval-outputs/visualizations/00004_val__002__-guardrail_missing-58-_jpg.rf.391f56af8166b7a9978f49176685f505__guardrail_missing_mask0.md) | 把缺少栏杆防护、存在坠落风险的部位分割出来。 |
| 33 | Improved | +0.0913 | 0.8846 | 0.9759 | +0.0490 | no helmet | `val__004__IMG20241113161529_jpg.rf.e77d13689e9d29512af57438dfc09685__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00053_val__004__IMG20241113161529_jpg.rf.e77d13689e9d29512af57438dfc09685__no_helmet_mask0.md) | [tuned](full-eval-outputs/visualizations/00053_val__004__IMG20241113161529_jpg.rf.e77d13689e9d29512af57438dfc09685__no_helmet_mask0.md) | 现场哪些人员存在未戴安全帽的安全隐患?请分割出来。 |
| 34 | Improved | +0.0764 | 0.1233 | 0.1997 | +0.1134 | opening_unprotected | `val__002__-opening_unprotected-63-_jpg.rf.502fc6e8122ebaa5d0a95498b20b0794__opening_unprotected.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00021_val__002__-opening_unprotected-63-_jpg.rf.502fc6e8122ebaa5d0a95498b20b0794__opening_unprotected_mask0.md) | [tuned](full-eval-outputs/visualizations/00021_val__002__-opening_unprotected-63-_jpg.rf.502fc6e8122ebaa5d0a95498b20b0794__opening_unprotected_mask0.md) | 图中哪些洞口或临边没有做防护?请分割出来。 |
| 35 | Improved | +0.0706 | 0.4977 | 0.5683 | +0.0601 | no helmet | `val__004__IMG20241113160624_jpg.rf.4b539618ccf9c3ab60c656685e98e24e__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00034_val__004__IMG20241113160624_jpg.rf.4b539618ccf9c3ab60c656685e98e24e__no_helmet_mask0.md) | [tuned](full-eval-outputs/visualizations/00034_val__004__IMG20241113160624_jpg.rf.4b539618ccf9c3ab60c656685e98e24e__no_helmet_mask0.md) | 标出未按规定佩戴安全帽的作业人员。 |
| 36 | Improved | +0.0662 | 0.4135 | 0.4797 | +0.0633 | helmet_missing | `val__002__-helmet_missing-245-_jpg.rf.dcbe61ef31ed007835863b76354c7261__helmet_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00013_val__002__-helmet_missing-245-_jpg.rf.dcbe61ef31ed007835863b76354c7261__helmet_missing_mask0.md) | [tuned](full-eval-outputs/visualizations/00013_val__002__-helmet_missing-245-_jpg.rf.dcbe61ef31ed007835863b76354c7261__helmet_missing_mask0.md) | 标出未按规定佩戴安全帽的作业人员。 |
| 37 | Improved | +0.0636 | 0.8913 | 0.9550 | +0.0344 | no helmet | `val__004__IMG_20241113_161758877_HDR_jpg.rf.362f3e155a81c2679a7690b2ef961abb__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00068_val__004__IMG_20241113_161758877_HDR_jpg.rf.362f3e155a81c2679a7690b2ef961abb__no_helmet_mask0.md) | [tuned](full-eval-outputs/visualizations/00068_val__004__IMG_20241113_161758877_HDR_jpg.rf.362f3e155a81c2679a7690b2ef961abb__no_helmet_mask0.md) | 标出未按规定佩戴安全帽的作业人员。 |
| 38 | Improved | +0.0602 | 0.4602 | 0.5203 | +0.0542 | no helmet | `val__004__IMG20241113160638_jpg.rf.f032f3f45329778ae3108a9bbab937d1__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00037_val__004__IMG20241113160638_jpg.rf.f032f3f45329778ae3108a9bbab937d1__no_helmet_mask0.md) | [tuned](full-eval-outputs/visualizations/00037_val__004__IMG20241113160638_jpg.rf.f032f3f45329778ae3108a9bbab937d1__no_helmet_mask0.md) | 把没有做好头部防护、未戴安全帽的人分割出来。 |
| 39 | Improved | +0.0492 | 0.3824 | 0.4315 | +0.0497 | no jacket | `val__004__IMG20241113160838_jpg.rf.a3a315c1e66d2e5d0f19bf139df5b5d6__no_jacket.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00045_val__004__IMG20241113160838_jpg.rf.a3a315c1e66d2e5d0f19bf139df5b5d6__no_jacket_mask0.md) | [tuned](full-eval-outputs/visualizations/00045_val__004__IMG20241113160838_jpg.rf.a3a315c1e66d2e5d0f19bf139df5b5d6__no_jacket_mask0.md) | 标出未按要求穿戴反光背心的工人。 |
| 40 | Improved | +0.0405 | 0.0000 | 0.0405 | +0.0778 | unsafe | `val__004__IMG20241113160624_jpg.rf.4b539618ccf9c3ab60c656685e98e24e__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00036_val__004__IMG20241113160624_jpg.rf.4b539618ccf9c3ab60c656685e98e24e__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00036_val__004__IMG20241113160624_jpg.rf.4b539618ccf9c3ab60c656685e98e24e__unsafe_mask0.md) | 圈出图中被标注为不安全状态的作业人员或区域。 |
| 41 | Improved | +0.0250 | 0.2401 | 0.2651 | +0.0318 | unsafe | `val__004__IMG20241113160501_jpg.rf.e2fc6e55324dcda33f674df60a4c0547__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00028_val__004__IMG20241113160501_jpg.rf.e2fc6e55324dcda33f674df60a4c0547__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00028_val__004__IMG20241113160501_jpg.rf.e2fc6e55324dcda33f674df60a4c0547__unsafe_mask0.md) | 圈出图中被标注为不安全状态的作业人员或区域。 |
| 42 | Improved | +0.0176 | 0.9328 | 0.9504 | +0.0094 | safe | `val__004__IMG_20241113_161954_jpg.rf.6d1a5633986ccbdc7de79340bce250ae__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00077_val__004__IMG_20241113_161954_jpg.rf.6d1a5633986ccbdc7de79340bce250ae__safe_mask0.md) | [tuned](full-eval-outputs/visualizations/00077_val__004__IMG_20241113_161954_jpg.rf.6d1a5633986ccbdc7de79340bce250ae__safe_mask0.md) | 圈出图中被标注为安全状态的作业人员或区域。 |
| 43 | Improved | +0.0164 | 0.9116 | 0.9280 | +0.0089 | unsafe | `val__004__IMG_20241113_162203623_jpg.rf.e05b826226143833b9ab04f2d13ce7a1__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00081_val__004__IMG_20241113_162203623_jpg.rf.e05b826226143833b9ab04f2d13ce7a1__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00081_val__004__IMG_20241113_162203623_jpg.rf.e05b826226143833b9ab04f2d13ce7a1__unsafe_mask0.md) | 圈出图中被标注为不安全状态的作业人员或区域。 |
| 44 | Improved | +0.0161 | 0.9378 | 0.9539 | +0.0085 | safe | `val__004__IMG_20241113_161953162_jpg.rf.2fc5157cdd82110c6541270dcb7e5f82__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00076_val__004__IMG_20241113_161953162_jpg.rf.2fc5157cdd82110c6541270dcb7e5f82__safe_mask0.md) | [tuned](full-eval-outputs/visualizations/00076_val__004__IMG_20241113_161953162_jpg.rf.2fc5157cdd82110c6541270dcb7e5f82__safe_mask0.md) | 标出现场处于安全状态的目标。 |
| 45 | Improved | +0.0138 | 0.9132 | 0.9269 | +0.0075 | safe | `val__004__IMG_20241113_162228212_jpg.rf.1021918a81b951a030f665662fb11a58__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00084_val__004__IMG_20241113_162228212_jpg.rf.1021918a81b951a030f665662fb11a58__safe_mask0.md) | [tuned](full-eval-outputs/visualizations/00084_val__004__IMG_20241113_162228212_jpg.rf.1021918a81b951a030f665662fb11a58__safe_mask0.md) | 圈出图中被标注为安全状态的作业人员或区域。 |
| 46 | Improved | +0.0134 | 0.9167 | 0.9301 | +0.0072 | safe | `val__004__IMG_20241113_161941847_jpg.rf.ce72086317700d7f52ce01bf415ed037__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00070_val__004__IMG_20241113_161941847_jpg.rf.ce72086317700d7f52ce01bf415ed037__safe_mask0.md) | [tuned](full-eval-outputs/visualizations/00070_val__004__IMG_20241113_161941847_jpg.rf.ce72086317700d7f52ce01bf415ed037__safe_mask0.md) | 圈出图中被标注为安全状态的作业人员或区域。 |
| 47 | Improved | +0.0092 | 0.1162 | 0.1254 | +0.0146 | opening_unprotected | `val__002__-opening_unprotected-5-_jpg.rf.44631bfc35cdebbd5142d77b296256d3__opening_unprotected.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00019_val__002__-opening_unprotected-5-_jpg.rf.44631bfc35cdebbd5142d77b296256d3__opening_unprotected_mask0.md) | [tuned](full-eval-outputs/visualizations/00019_val__002__-opening_unprotected-5-_jpg.rf.44631bfc35cdebbd5142d77b296256d3__opening_unprotected_mask0.md) | 标出存在洞口未防护隐患的位置。 |
| 48 | Improved | +0.0087 | 0.9341 | 0.9428 | +0.0046 | safe | `val__004__IMG_20241113_161952050_jpg.rf.7c0b52ecb477f8f7ef8b71489dac19cf__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00074_val__004__IMG_20241113_161952050_jpg.rf.7c0b52ecb477f8f7ef8b71489dac19cf__safe_mask0.md) | [tuned](full-eval-outputs/visualizations/00074_val__004__IMG_20241113_161952050_jpg.rf.7c0b52ecb477f8f7ef8b71489dac19cf__safe_mask0.md) | 标出现场处于安全状态的目标。 |
| 49 | Improved | +0.0081 | 0.9296 | 0.9377 | +0.0044 | safe | `val__004__IMG20241113161931_jpg.rf.a8d69015746e42c5014b2422eaf72c7b__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00056_val__004__IMG20241113161931_jpg.rf.a8d69015746e42c5014b2422eaf72c7b__safe_mask0.md) | [tuned](full-eval-outputs/visualizations/00056_val__004__IMG20241113161931_jpg.rf.a8d69015746e42c5014b2422eaf72c7b__safe_mask0.md) | 标出现场处于安全状态的目标。 |
| 50 | Improved | +0.0052 | 0.4646 | 0.4699 | +0.0049 | harness_missing | `val__002__-harness_missing-23-_JPG.rf.9e5a10303ac02c86c71739d108101ea4__harness_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00008_val__002__-harness_missing-23-_JPG.rf.9e5a10303ac02c86c71739d108101ea4__harness_missing_mask0.md) | [tuned](full-eval-outputs/visualizations/00008_val__002__-harness_missing-23-_JPG.rf.9e5a10303ac02c86c71739d108101ea4__harness_missing_mask0.md) | 把缺少安全带防护的作业人员分割出来。 |
| 51 | Improved | +0.0045 | 0.0547 | 0.0592 | +0.0080 | no jacket | `val__004__IMG_20241113_161450_jpg.rf.b72ade8fc935970192f5e4c70fc7249e__no_jacket.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00065_val__004__IMG_20241113_161450_jpg.rf.b72ade8fc935970192f5e4c70fc7249e__no_jacket_mask0.md) | [tuned](full-eval-outputs/visualizations/00065_val__004__IMG_20241113_161450_jpg.rf.b72ade8fc935970192f5e4c70fc7249e__no_jacket_mask0.md) | 现场哪些人员没有穿安全背心?请分割出来。 |
| 52 | Improved | +0.0035 | 0.0000 | 0.0035 | +0.0069 | opening_unprotected | `val__002__-opening_unprotected-26-_jpg.rf.8a97f7db61ee9371c7a662776cd884b2__opening_unprotected.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00017_val__002__-opening_unprotected-26-_jpg.rf.8a97f7db61ee9371c7a662776cd884b2__opening_unprotected_mask0.md) | [tuned](full-eval-outputs/visualizations/00017_val__002__-opening_unprotected-26-_jpg.rf.8a97f7db61ee9371c7a662776cd884b2__opening_unprotected_mask0.md) | 圈出没有防护的洞口或临边区域。 |
| 53 | Improved | +0.0034 | 0.0006 | 0.0040 | +0.0068 | unsafe | `val__004__IMG20241113160638_jpg.rf.f032f3f45329778ae3108a9bbab937d1__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00038_val__004__IMG20241113160638_jpg.rf.f032f3f45329778ae3108a9bbab937d1__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00038_val__004__IMG20241113160638_jpg.rf.f032f3f45329778ae3108a9bbab937d1__unsafe_mask0.md) | 请分割图中存在安全风险的目标区域。 |
| 54 | Improved | +0.0014 | 0.0000 | 0.0014 | +0.0028 | no jacket | `val__004__IMG20241113160658_jpg.rf.509371336267fc48e02387bc78859d8f__no_jacket.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00040_val__004__IMG20241113160658_jpg.rf.509371336267fc48e02387bc78859d8f__no_jacket_mask0.md) | [tuned](full-eval-outputs/visualizations/00040_val__004__IMG20241113160658_jpg.rf.509371336267fc48e02387bc78859d8f__no_jacket_mask0.md) | 标出未按要求穿戴反光背心的工人。 |
| 55 | Improved | +0.0010 | 0.0000 | 0.0010 | +0.0021 | unsafe | `val__004__IMG20241113160658_jpg.rf.509371336267fc48e02387bc78859d8f__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00041_val__004__IMG20241113160658_jpg.rf.509371336267fc48e02387bc78859d8f__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00041_val__004__IMG20241113160658_jpg.rf.509371336267fc48e02387bc78859d8f__unsafe_mask0.md) | 标出现场存在不安全状态的目标。 |
| 56 | Improved | +0.0006 | 0.9080 | 0.9086 | +0.0003 | no jacket | `val__004__IMG_20241113_162029941_jpg.rf.9cb9cb44acb2667b768970c0c5b37320__no_jacket.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00078_val__004__IMG_20241113_162029941_jpg.rf.9cb9cb44acb2667b768970c0c5b37320__no_jacket_mask0.md) | [tuned](full-eval-outputs/visualizations/00078_val__004__IMG_20241113_162029941_jpg.rf.9cb9cb44acb2667b768970c0c5b37320__no_jacket_mask0.md) | 现场哪些人员没有穿安全背心?请分割出来。 |
| 57 | Improved | +0.0000 | 0.0000 | 0.0000 | +0.0001 | no jacket | `val__004__IMG_20241113_161340_jpg.rf.d2e6016a68c883c9a28b149919ca2c5c__no_jacket.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00061_val__004__IMG_20241113_161340_jpg.rf.d2e6016a68c883c9a28b149919ca2c5c__no_jacket_mask0.md) | [tuned](full-eval-outputs/visualizations/00061_val__004__IMG_20241113_161340_jpg.rf.d2e6016a68c883c9a28b149919ca2c5c__no_jacket_mask0.md) | 标出未按要求穿戴反光背心的工人。 |
| 58 | Unchanged | +0.0000 | 0.0000 | 0.0000 | +0.0000 | guardrail_missing | `val__002__-guardrail_missing-14-_jpg.rf.317f592ca167baf04b16296d87650912__guardrail_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00003_val__002__-guardrail_missing-14-_jpg.rf.317f592ca167baf04b16296d87650912__guardrail_missing_mask0.md) | [tuned](full-eval-outputs/visualizations/00003_val__002__-guardrail_missing-14-_jpg.rf.317f592ca167baf04b16296d87650912__guardrail_missing_mask0.md) | 指出没有设置防护栏杆的临边区域。 |
| 59 | Unchanged | +0.0000 | 0.0000 | 0.0000 | +0.0000 | unsafe | `val__004__IMG20241113160529_jpg.rf.01cc514ff77c809eb7eb263d362448cd__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00031_val__004__IMG20241113160529_jpg.rf.01cc514ff77c809eb7eb263d362448cd__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00031_val__004__IMG20241113160529_jpg.rf.01cc514ff77c809eb7eb263d362448cd__unsafe_mask0.md) | 请分割图中存在安全风险的目标区域。 |
| 60 | Unchanged | +0.0000 | 0.0000 | 0.0000 | +0.0000 | no jacket | `val__004__IMG20241113160624_jpg.rf.4b539618ccf9c3ab60c656685e98e24e__no_jacket.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00035_val__004__IMG20241113160624_jpg.rf.4b539618ccf9c3ab60c656685e98e24e__no_jacket_mask0.md) | [tuned](full-eval-outputs/visualizations/00035_val__004__IMG20241113160624_jpg.rf.4b539618ccf9c3ab60c656685e98e24e__no_jacket_mask0.md) | 标出未按要求穿戴反光背心的工人。 |
| 61 | Unchanged | +0.0000 | 0.0000 | 0.0000 | +0.0000 | unsafe | `val__004__IMG20241113161357_jpg.rf.8d057aafc970f94c8f2f31eaae4e159c__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00052_val__004__IMG20241113161357_jpg.rf.8d057aafc970f94c8f2f31eaae4e159c__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00052_val__004__IMG20241113161357_jpg.rf.8d057aafc970f94c8f2f31eaae4e159c__unsafe_mask0.md) | 圈出图中被标注为不安全状态的作业人员或区域。 |
| 62 | Unchanged | +0.0000 | 0.0000 | 0.0000 | +0.0000 | unsafe | `val__004__IMG_20241113_161405_jpg.rf.7b78a6ff625a0c910f3c1a1839dae955__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00064_val__004__IMG_20241113_161405_jpg.rf.7b78a6ff625a0c910f3c1a1839dae955__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00064_val__004__IMG_20241113_161405_jpg.rf.7b78a6ff625a0c910f3c1a1839dae955__unsafe_mask0.md) | 请分割图中存在安全风险的目标区域。 |
| 63 | Unchanged | +0.0000 | 0.0000 | 0.0000 | +0.0000 | unsafe | `val__004__IMG_20241113_161758877_HDR_jpg.rf.362f3e155a81c2679a7690b2ef961abb__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00069_val__004__IMG_20241113_161758877_HDR_jpg.rf.362f3e155a81c2679a7690b2ef961abb__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00069_val__004__IMG_20241113_161758877_HDR_jpg.rf.362f3e155a81c2679a7690b2ef961abb__unsafe_mask0.md) | 圈出图中被标注为不安全状态的作业人员或区域。 |
| 64 | Unchanged | +0.0000 | 0.0000 | 0.0000 | +0.0000 | unsafe | `val__004__IMG_20241113_161941847_jpg.rf.ce72086317700d7f52ce01bf415ed037__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00071_val__004__IMG_20241113_161941847_jpg.rf.ce72086317700d7f52ce01bf415ed037__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00071_val__004__IMG_20241113_161941847_jpg.rf.ce72086317700d7f52ce01bf415ed037__unsafe_mask0.md) | 标出现场存在不安全状态的目标。 |
| 65 | Unchanged | +0.0000 | 0.0000 | 0.0000 | +0.0000 | unsafe | `val__004__IMG_20241113_161944298_jpg.rf.8b7de1452bddd5e3744643b6b17d30e1__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00073_val__004__IMG_20241113_161944298_jpg.rf.8b7de1452bddd5e3744643b6b17d30e1__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00073_val__004__IMG_20241113_161944298_jpg.rf.8b7de1452bddd5e3744643b6b17d30e1__unsafe_mask0.md) | 标出现场存在不安全状态的目标。 |
| 66 | Unchanged | +0.0000 | 0.0000 | 0.0000 | +0.0000 | unsafe | `val__004__IMG_20241113_161952050_jpg.rf.7c0b52ecb477f8f7ef8b71489dac19cf__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00075_val__004__IMG_20241113_161952050_jpg.rf.7c0b52ecb477f8f7ef8b71489dac19cf__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00075_val__004__IMG_20241113_161952050_jpg.rf.7c0b52ecb477f8f7ef8b71489dac19cf__unsafe_mask0.md) | 标出现场存在不安全状态的目标。 |
| 67 | Unchanged | +0.0000 | 0.0000 | 0.0000 | +0.0000 | no helmet | `val__004__IMG_20241113_162132082_jpg.rf.391adeed04f63ff7aa322dc4173f4a3a__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00079_val__004__IMG_20241113_162132082_jpg.rf.391adeed04f63ff7aa322dc4173f4a3a__no_helmet_mask0.md) | [tuned](full-eval-outputs/visualizations/00079_val__004__IMG_20241113_162132082_jpg.rf.391adeed04f63ff7aa322dc4173f4a3a__no_helmet_mask0.md) | 标出未按规定佩戴安全帽的作业人员。 |
| 68 | Unchanged | +0.0000 | 0.0000 | 0.0000 | +0.0000 | unsafe | `val__004__IMG_20241113_162228212_jpg.rf.1021918a81b951a030f665662fb11a58__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00085_val__004__IMG_20241113_162228212_jpg.rf.1021918a81b951a030f665662fb11a58__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00085_val__004__IMG_20241113_162228212_jpg.rf.1021918a81b951a030f665662fb11a58__unsafe_mask0.md) | 圈出图中被标注为不安全状态的作业人员或区域。 |
| 69 | Regressed | -0.0001 | 0.0001 | 0.0000 | -0.0002 | unsafe | `val__004__IMG20241113161931_jpg.rf.a8d69015746e42c5014b2422eaf72c7b__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00057_val__004__IMG20241113161931_jpg.rf.a8d69015746e42c5014b2422eaf72c7b__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00057_val__004__IMG20241113161931_jpg.rf.a8d69015746e42c5014b2422eaf72c7b__unsafe_mask0.md) | 标出现场存在不安全状态的目标。 |
| 70 | Regressed | -0.0053 | 0.0053 | 0.0000 | -0.0105 | opening_unprotected | `val__002__-opening_unprotected-51-_jpg.rf.7db855a017ceaddec7cf0044a0a8b7e2__opening_unprotected.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00020_val__002__-opening_unprotected-51-_jpg.rf.7db855a017ceaddec7cf0044a0a8b7e2__opening_unprotected_mask0.md) | [tuned](full-eval-outputs/visualizations/00020_val__002__-opening_unprotected-51-_jpg.rf.7db855a017ceaddec7cf0044a0a8b7e2__opening_unprotected_mask0.md) | 图中哪些洞口或临边没有做防护?请分割出来。 |
| 71 | Regressed | -0.0081 | 0.0081 | 0.0000 | -0.0160 | equipment_proximity | `val__002__-equipment_proximity-278-_jpg.rf.28a6b911d2ab42e0b0dbffd5f8d5e5db__equipment_proximity.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00002_val__002__-equipment_proximity-278-_jpg.rf.28a6b911d2ab42e0b0dbffd5f8d5e5db__equipment_proximity_mask0.md) | [tuned](full-eval-outputs/visualizations/00002_val__002__-equipment_proximity-278-_jpg.rf.28a6b911d2ab42e0b0dbffd5f8d5e5db__equipment_proximity_mask0.md) | 标出人员或设备距离过近、存在碰撞风险的区域。 |
| 72 | Regressed | -0.0104 | 0.0144 | 0.0039 | -0.0205 | helmet_missing | `val__002__-helmet_missing-249-_JPG.rf.161f677eff339180405f881cbd790a96__helmet_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00014_val__002__-helmet_missing-249-_JPG.rf.161f677eff339180405f881cbd790a96__helmet_missing_mask0.md) | [tuned](full-eval-outputs/visualizations/00014_val__002__-helmet_missing-249-_JPG.rf.161f677eff339180405f881cbd790a96__helmet_missing_mask0.md) | 现场哪些人员存在未戴安全帽的安全隐患?请分割出来。 |
| 73 | Regressed | -0.0169 | 0.9569 | 0.9400 | -0.0089 | no helmet | `val__004__IMG20241113160936_jpg.rf.162c6c777d07be2ca5c077d6906547f6__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00047_val__004__IMG20241113160936_jpg.rf.162c6c777d07be2ca5c077d6906547f6__no_helmet_mask0.md) | [tuned](full-eval-outputs/visualizations/00047_val__004__IMG20241113160936_jpg.rf.162c6c777d07be2ca5c077d6906547f6__no_helmet_mask0.md) | 把没有做好头部防护、未戴安全帽的人分割出来。 |
| 74 | Regressed | -0.0171 | 0.5368 | 0.5197 | -0.0146 | harness_missing | `val__002__-harness_missing-45-_JPG.rf.8bddeb686fc5fada6004b77add73011f__harness_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00009_val__002__-harness_missing-45-_JPG.rf.8bddeb686fc5fada6004b77add73011f__harness_missing_mask0.md) | [tuned](full-eval-outputs/visualizations/00009_val__002__-harness_missing-45-_JPG.rf.8bddeb686fc5fada6004b77add73011f__harness_missing_mask0.md) | 标出未按要求佩戴安全带的作业人员。 |
| 75 | Regressed | -0.0232 | 0.2705 | 0.2473 | -0.0292 | unsafe | `val__004__IMG20241113160613_jpg.rf.028bfffcd39729b2b7c7833027bb05f6__unsafe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00033_val__004__IMG20241113160613_jpg.rf.028bfffcd39729b2b7c7833027bb05f6__unsafe_mask0.md) | [tuned](full-eval-outputs/visualizations/00033_val__004__IMG20241113160613_jpg.rf.028bfffcd39729b2b7c7833027bb05f6__unsafe_mask0.md) | 圈出图中被标注为不安全状态的作业人员或区域。 |
| 76 | Regressed | -0.0442 | 0.7045 | 0.6603 | -0.0312 | harness_missing | `val__002__-harness_missing-69-_JPG.rf.1205c382c7eb97a239309f883a43c734__harness_missing.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00010_val__002__-harness_missing-69-_JPG.rf.1205c382c7eb97a239309f883a43c734__harness_missing_mask0.md) | [tuned](full-eval-outputs/visualizations/00010_val__002__-harness_missing-69-_JPG.rf.1205c382c7eb97a239309f883a43c734__harness_missing_mask0.md) | 把缺少安全带防护的作业人员分割出来。 |
| 77 | Regressed | -0.1513 | 0.1514 | 0.0001 | -0.2627 | no jacket | `val__004__IMG20241113161910_jpg.rf.7330e0f90826aafd4c8a75a7cfdfaa87__no_jacket.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00054_val__004__IMG20241113161910_jpg.rf.7330e0f90826aafd4c8a75a7cfdfaa87__no_jacket_mask0.md) | [tuned](full-eval-outputs/visualizations/00054_val__004__IMG20241113161910_jpg.rf.7330e0f90826aafd4c8a75a7cfdfaa87__no_jacket_mask0.md) | 现场哪些人员没有穿安全背心?请分割出来。 |
| 78 | Regressed | -0.1981 | 0.4585 | 0.2603 | -0.2156 | safe | `val__004__IMG20241113160613_jpg.rf.028bfffcd39729b2b7c7833027bb05f6__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00032_val__004__IMG20241113160613_jpg.rf.028bfffcd39729b2b7c7833027bb05f6__safe_mask0.md) | [tuned](full-eval-outputs/visualizations/00032_val__004__IMG20241113160613_jpg.rf.028bfffcd39729b2b7c7833027bb05f6__safe_mask0.md) | 圈出图中被标注为安全状态的作业人员或区域。 |
| 79 | Regressed | -0.2051 | 0.2051 | 0.0000 | -0.3403 | no jacket | `val__004__IMG_20241113_162132082_jpg.rf.391adeed04f63ff7aa322dc4173f4a3a__no_jacket.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00080_val__004__IMG_20241113_162132082_jpg.rf.391adeed04f63ff7aa322dc4173f4a3a__no_jacket_mask0.md) | [tuned](full-eval-outputs/visualizations/00080_val__004__IMG_20241113_162132082_jpg.rf.391adeed04f63ff7aa322dc4173f4a3a__no_jacket_mask0.md) | 标出未按要求穿戴反光背心的工人。 |
| 80 | Regressed | -0.2302 | 0.2302 | 0.0000 | -0.3743 | no helmet | `val__004__IMG20241113160838_jpg.rf.a3a315c1e66d2e5d0f19bf139df5b5d6__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00044_val__004__IMG20241113160838_jpg.rf.a3a315c1e66d2e5d0f19bf139df5b5d6__no_helmet_mask0.md) | [tuned](full-eval-outputs/visualizations/00044_val__004__IMG20241113160838_jpg.rf.a3a315c1e66d2e5d0f19bf139df5b5d6__no_helmet_mask0.md) | 把没有做好头部防护、未戴安全帽的人分割出来。 |
| 81 | Regressed | -0.2367 | 0.9058 | 0.6691 | -0.1488 | safe | `val__004__IMG_20241113_161944298_jpg.rf.8b7de1452bddd5e3744643b6b17d30e1__safe.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00072_val__004__IMG_20241113_161944298_jpg.rf.8b7de1452bddd5e3744643b6b17d30e1__safe_mask0.md) | [tuned](full-eval-outputs/visualizations/00072_val__004__IMG_20241113_161944298_jpg.rf.8b7de1452bddd5e3744643b6b17d30e1__safe_mask0.md) | 圈出图中被标注为安全状态的作业人员或区域。 |
| 82 | Regressed | -0.2504 | 0.4456 | 0.1952 | -0.2898 | no jacket | `val__004__IMG20241113160529_jpg.rf.01cc514ff77c809eb7eb263d362448cd__no_jacket.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00030_val__004__IMG20241113160529_jpg.rf.01cc514ff77c809eb7eb263d362448cd__no_jacket_mask0.md) | [tuned](full-eval-outputs/visualizations/00030_val__004__IMG20241113160529_jpg.rf.01cc514ff77c809eb7eb263d362448cd__no_jacket_mask0.md) | 圈出没有穿反光衣或安全背心的作业人员。 |
| 83 | Regressed | -0.4428 | 0.4428 | 0.0000 | -0.6138 | no helmet | `val__004__IMG20241113160817_jpg.rf.9c7cd48d2297fc816a026d117f0df99b__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00042_val__004__IMG20241113160817_jpg.rf.9c7cd48d2297fc816a026d117f0df99b__no_helmet_mask0.md) | [tuned](full-eval-outputs/visualizations/00042_val__004__IMG20241113160817_jpg.rf.9c7cd48d2297fc816a026d117f0df99b__no_helmet_mask0.md) | 把没有做好头部防护、未戴安全帽的人分割出来。 |
| 84 | Regressed | -0.5302 | 0.5302 | 0.0000 | -0.6930 | poor_housekeeping | `val__002__-poor_housekeeping-132-_jpg.rf.c6a2c42ceb4d1eae5cd2307fedce5fe6__poor_housekeeping.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00023_val__002__-poor_housekeeping-132-_jpg.rf.c6a2c42ceb4d1eae5cd2307fedce5fe6__poor_housekeeping_mask0.md) | [tuned](full-eval-outputs/visualizations/00023_val__002__-poor_housekeeping-132-_jpg.rf.c6a2c42ceb4d1eae5cd2307fedce5fe6__poor_housekeeping_mask0.md) | 图中哪些区域存在场地整理不到位的安全隐患?请分割出来。 |
| 85 | Regressed | -0.5643 | 0.6591 | 0.0948 | -0.6214 | no helmet | `val__004__IMG20241113160658_jpg.rf.509371336267fc48e02387bc78859d8f__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00039_val__004__IMG20241113160658_jpg.rf.509371336267fc48e02387bc78859d8f__no_helmet_mask0.md) | [tuned](full-eval-outputs/visualizations/00039_val__004__IMG20241113160658_jpg.rf.509371336267fc48e02387bc78859d8f__no_helmet_mask0.md) | 标出未按规定佩戴安全帽的作业人员。 |
| 86 | Regressed | -0.6543 | 0.8814 | 0.2271 | -0.5668 | no helmet | `val__004__IMG_20241113_161340_jpg.rf.d2e6016a68c883c9a28b149919ca2c5c__no_helmet.jpg` | [base](../lisa13b-local-val/outputs/visualizations/00060_val__004__IMG_20241113_161340_jpg.rf.d2e6016a68c883c9a28b149919ca2c5c__no_helmet_mask0.md) | [tuned](full-eval-outputs/visualizations/00060_val__004__IMG_20241113_161340_jpg.rf.d2e6016a68c883c9a28b149919ca2c5c__no_helmet_mask0.md) | 标出未按规定佩戴安全帽的作业人员。 |
<!-- benchmark-comparison:full-val-samples:end -->
