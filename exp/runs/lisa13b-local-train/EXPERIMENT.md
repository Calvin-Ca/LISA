# lisa13b-local-train

## 背景

本实验在 ReasonSeg 训练集划分 上评估本地 LISA-13B 权重,用于观察训练集拟合情况,并排查 LoRA 训练或数据构造过程中可能存在的标注/模型问题。

这组结果不能作为主要泛化指标,正式汇报仍以验证集结果为准。

## 配置

- 模型: 本地 `./LISA13B`
- 权重路径: `./LISA13B`
- 数据集: ReasonSeg
- 数据划分: `ReasonSeg|train`
- 最大样本数: 全量 415 张
- 精度: `bf16`
- 掩码阈值: `0.0`
- 是否保存可视化: 是
- 是否保存预测掩码: 以实际输出目录为准
- 运行设备: 远程 Linux GPU 服务器
- 运行日期: 2026-07-09

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

训练集和验证集指标接近,说明当前瓶颈更可能来自数据质量、类别边界或基座模型对施工风险概念的理解,而不是单纯训练集过拟合。

## 备注

- 原始远程结果来自 `benchmark_outputs/lisa13b-local-train/`。
- 低分样本优先看 `outputs/samples_by_iou.md`。
