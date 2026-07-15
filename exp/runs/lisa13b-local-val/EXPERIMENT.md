# lisa13b-local-val

## 背景

本实验在 `ReasonSeg|val` 上评估本地 LISA-13B 权重,作为当前施工安全数据集的 base benchmark。

这组结果主要用于:

- 判断 base LISA 是否适合当前施工安全隐患场景。
- 作为后续 LoRA 微调前后的正式对比基线。
- 支撑面试中对数据质量、domain gap、bad case 的解释。

## 配置

- 模型: `./LISA13B`
- 权重路径: `./LISA13B`
- CLIP vision tower: `/home/ths/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41`
- SAM 权重: `./data_pipeline/sam_vit_h_4b8939.pth`
- 数据集: ReasonSeg
- 数据划分: `ReasonSeg|val`
- 最大样本数: 全量 86 张
- 精度: `bf16`
- 掩码阈值: `0.0`
- 是否保存可视化: 是
- 最大可视化数量: -1
- 是否保存预测掩码: 是
- 字体: `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`
- 运行设备: 远程 Linux GPU 服务器
- 运行日期: 2026-07-10

## 执行命令

见 `command.sh`。

## 输出文件

- `outputs/summary.json`
- `outputs/summary.md`
- `outputs/per_sample_metrics.csv`
- `outputs/per_sample_metrics.jsonl`
- `outputs/per_sample_metrics_by_iou.csv`
- `outputs/samples_by_iou.md`
- `outputs/visualizations/`
- `outputs/pred_masks/`

## 核心指标

- 样本数: 86
- gIoU: 0.3408
- cIoU: 0.3177
- 平均 Dice: 0.4180
- 平均精确率: 0.4071
- 平均召回率: 0.5132

## 结论

评估已完成，本实验已作为正式 Base LISA-13B 验证集基线。Base 在完整 86 个样本上的结果为 `gIoU=0.3408 / cIoU=0.3177 / Dice=0.4180`，其中 22 个样本 IoU 为 0，33 个样本 IoU < 0.1，29 个样本 IoU >= 0.5。这说明 base 模型具有一定的通用语义分割能力，但对施工隐患概念的稳定定位仍不足。

Clean030 LoRA 后，同一完整 `ReasonSeg|val` 上的指标提升到 `gIoU=0.4494 / cIoU=0.3858 / Dice=0.5156`。gIoU 提升 0.1086，零 IoU 样本减少 6 个，`IoU >= 0.5` 样本增加 10 个，证明高置信数据上的 LoRA 领域适配对完整验证集有真实迁移收益。

收益主要来自误检降低：Mean Precision 从 0.4071 提升到 0.5332，False Positive Area 下降 30.1%；Mean Recall 仅从 0.5132 提升到 0.5416，False Negative Area 下降 7.9%。因此漏检和 `unsafe` 等抽象隐患类别仍是后续主要短板。

## 备注

- 本实验是正式 base benchmark。
- 评估产物、核心指标和 Base/LoRA 结论已回填；后续修改验证集标注时，必须在同一冻结版本上重新评估 Base 和 LoRA，不得直接与当前指标混用。
