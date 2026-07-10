# lisa13b-local-val

## 背景

本实验在 ReasonSeg 验证集划分 上评估本地 LISA-13B 权重,是当前可用于文档、面试材料和后续对比的正式验证集指标来源。

## 配置

- 模型: 本地 `./LISA13B`
- 权重路径: `./LISA13B`
- 数据集: ReasonSeg
- 数据划分: `ReasonSeg|val`
- 最大样本数: 全量 86 张
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

- 样本数: 86
- gIoU: 0.3408
- cIoU: 0.3177
- 平均 Dice: 0.4180
- 平均精确率: 0.4071
- 平均召回率: 0.5132

## 结论

这是当前正式验证集基线。后续如果有 LoRA 微调后结果,应以同一验证集重新评估并在 `comparisons/` 中对比。

## 备注

- 原始远程结果来自 `benchmark_outputs/lisa13b-local-val/`。
- 旧的 `reason_seg` 归档与本实验指标重复,已删除。
