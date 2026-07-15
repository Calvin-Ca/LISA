# lisa13b-local-train

## 背景

本实验在 `ReasonSeg|train` 上评估本地 LISA-13B 权重,用于数据审计和 bad case 分析。

这组结果主要回答两个问题:

- 训练集里的 prompt/mask 是否存在明显错配、错位或类别混乱。
- base LISA 对施工安全类别是否具备基本定位能力,低分样本集中在哪些隐患类型。

注意:训练集结果不作为正式泛化指标。后续如果用该训练集做 LoRA 微调,训练集分数只能用于观察拟合和数据问题,正式对比仍以验证集为准。

## 配置

- 模型: `./LISA13B`
- 权重路径: `./LISA13B`
- CLIP vision tower: `/home/ths/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41`
- SAM 权重: `./data_pipeline/sam_vit_h_4b8939.pth`
- 数据集: ReasonSeg
- 数据划分: `ReasonSeg|train`
- 最大样本数: 全量 415 张
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

- 样本数: 415
- gIoU: 0.3432
- cIoU: 0.2938
- 平均 Dice: 0.4163
- 平均精确率: 0.4069
- 平均召回率: 0.5148

## 结论

评估已完成。Base LISA-13B 在 415 个训练样本上取得 `gIoU=0.3432 / cIoU=0.2938 / Dice=0.4163`。其中 91 个样本 IoU 为 0，170 个样本 IoU < 0.1，132 个样本 IoU >= 0.5，说明 base 模型对部分具体目标已有定位能力，但低分长尾仍很明显。

train 与 val 的 gIoU（0.3432 vs 0.3408）和 Dice（0.4163 vs 0.4180）非常接近。由于这里评估的 base LISA 并未在当前 ReasonSeg 上微调，这不是传统意义上的训练集拟合对比；它更说明当前两个划分对 base 模型具有相近难度，主要瓶颈是施工安全语义的领域 gap、prompt/mask 一致性和 SAM 辅助标注质量。

本结果已用于构建 Clean030 高置信子集，并通过 `outputs/samples_by_iou.md` 和 `dataset/reason_seg/ReasonSegRelabel/samples_by_iou.md` 继续开展低分样本审核。

## 备注

- 本实验用于数据审计,不是正式泛化指标。
- 评估产物和核心指标已回填；bad case 的人工分类和 Relabel 修正属于后续数据治理工作。
