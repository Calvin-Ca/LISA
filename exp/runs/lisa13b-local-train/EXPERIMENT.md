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

待评估完成后,结合 `outputs/summary.md` 和 `outputs/samples_by_iou.md` 分析。

重点关注:

- 低 IoU 样本是否属于标注错误、prompt 错配、目标不可见等数据问题。
- 低 IoU 是否集中在 `guardrail_missing`、`opening_unprotected`、`equipment_proximity` 等施工领域概念。
- train 与 val 指标是否接近,用于判断瓶颈是数据质量/领域 gap 还是泛化问题。

## 备注

- 本实验用于数据审计,不是正式泛化指标。
- 执行完成后,`benchmark_reason_seg.py` 会自动更新配置和核心指标;结论和 bad case 分析需要人工补充。
