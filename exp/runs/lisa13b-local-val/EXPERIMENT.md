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

待评估完成后,以本实验作为正式验证集基线。

重点关注:

- base LISA 在施工隐患概念上的总体定位能力。
- 低 IoU 样本中的数据问题和真实困难样本比例。
- 后续 LoRA 微调是否能在同一 `ReasonSeg|val` 上提升 gIoU/cIoU/Dice。

## 备注

- 本实验是正式 base benchmark。
- 执行完成后,`benchmark_reason_seg.py` 会自动更新配置和核心指标;结论和 bad case 分析需要人工补充。
