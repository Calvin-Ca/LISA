# lisa13b-local-smoke

## 背景

本实验是本地 LISA-13B 评测链路的冒烟测试,只用于确认模型权重、视觉塔、数据路径、CUDA 环境和掩码输出链路是否能跑通。

不要把这组结果作为正式效果指标。

## 配置

- 模型: 本地 `./LISA13B`
- 权重路径: `./LISA13B`
- 数据集: ReasonSeg
- 数据划分: `ReasonSeg|val`
- 最大样本数: 3
- 精度: `bf16`
- 掩码阈值: `0.0`
- 是否保存可视化: 以实际输出目录为准
- 是否保存预测掩码: 以实际输出目录为准
- 运行设备: 远程 Linux GPU 服务器
- 运行日期: 2026-07-09

## 执行命令

见 `command.sh`。

## 输出文件

- `outputs/summary.json`
- `outputs/summary.md`
- `outputs/per_sample_metrics.csv`
- `outputs/samples_by_iou.md`
- `outputs/visualizations/`
- `outputs/pred_masks/`

## 核心指标

- 样本数: 3
- gIoU: 0.2611
- cIoU: 0.3104
- 平均 Dice: 0.3544
- 平均精确率: 0.3631
- 平均召回率: 0.6200

## 结论

链路可以跑通,但样本数太少,只作为链路检查。正式验证集指标使用 `lisa13b-local-val`。

## 备注

- 原始远程结果来自 `benchmark_outputs/lisa13b-local-smoke/`。
