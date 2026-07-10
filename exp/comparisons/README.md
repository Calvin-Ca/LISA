# 实验对比

本目录用于记录跨实验对比、表格和图表。

建议优先整理以下对比:

- `lisa13b-local-train` vs `lisa13b-local-val`: 训练集拟合表现与验证集泛化表现。
- Base LISA vs LoRA LISA: 领域 LoRA 微调带来的增益。
- Qwen-VL 复核开/关: Agent 链路中的误报降低效果。
- `samples_by_iou.md` 中的低分样本: 归纳错误类型,反推下一轮数据修正方向。
