# lisa13b-relabel303-lora-v1

## 背景

本实验使用人工审核后的 `ReasonSegRelabel|train` 对本地 LISA-13B 进行 LoRA 微调，验证具体化、严格等价的多 Prompt 标注能否改善施工安全场景的语言理解和目标分割。

训练集只保留已经具备 6 条 Prompt 的样本，并删除已识别的 prompt/mask 错配、mask 越界和 mask 漏标等待确认 bad case。当前数据状态：

- JSON/JPG 样本对：303
- 独立源图片：270
- 每个 JSON 的 Prompt 数：6
- 每次加载随机抽取 Prompt 数：3
- 类别数：10
- 与 `ReasonSegClean030|train` 的重合样本：150

本实验从与 Clean030 LoRA 相同的 `./LISA13B` 基础模型开始，不继续加载 Clean030 LoRA 权重。训练超参数保持一致，只替换训练数据，以尽量隔离人工 Relabel 数据带来的影响。

## 实验目标

- 验证 `ReasonSegRelabel|train -> LoRA -> merge -> ReasonSeg|val benchmark` 链路。
- 与 Base LISA-13B、Clean030 LoRA 在同一完整验证集上公平比较。
- 重点观察人工 Prompt 重写是否改善零 IoU、漏检和困难类别表现。
- 检查多样化 Prompt 是否在不改变 mask 真值的前提下增强自然语言查询适应性。

## 数据划分

- 训练集：`ReasonSegRelabel|train`，303 个 JSON/JPG 样本对
- 训练时验证集：`ReasonSeg|val`，86 个 JSON/JPG 样本对
- 正式评估集：`ReasonSeg|val`，86 个 JSON/JPG 样本对
- Base 对照结果：`exp/runs/lisa13b-local-val/outputs/per_sample_metrics.jsonl`
- Clean030 LoRA 对照结果：`exp/runs/lisa13b-clean030-lora-v1/full-eval-outputs/per_sample_metrics.jsonl`

`ReasonSegRelabel/val` 当前没有对应 JPG，因此本实验不将其作为验证集。完整 `ReasonSeg|val` 与既有 Base、Clean030 LoRA 的正式评估集合一致。

## 模型与权重

- 基础模型：`./LISA13B`
- CLIP vision tower：优先 `./clip-vit-large-patch14`，不存在时从服务器 HuggingFace cache 查找
- SAM 权重：`./data_pipeline/sam_vit_h_4b8939.pth`
- 初始化方式：从 Base LISA-13B 独立开始
- 运行环境：远程 Linux + GPU

所有权重均由远程服务器本地提供，不下载权重，不在仓库记录服务器私有绝对路径。

## 训练配置

- precision：`bf16`
- epochs：6
- steps per epoch：100
- optimizer steps：600
- batch size：1
- gradient accumulation steps：8
- effective batch size：8（单 GPU）
- 每轮训练采样数：800
- 总图片采样次数：4800
- 每样本随机 Prompt 数：3
- 总 Prompt-mask 训练对约：14400
- learning rate：`0.0001`
- optimizer：DeepSpeed + PyTorch AdamW
- LoRA rank：8
- LoRA alpha：16
- LoRA dropout：0.05
- LoRA target modules：`q_proj,v_proj`
- explanatory：`-1`
- DeepSpeed ZeRO stage：2（由 `train_ds.py` 配置）

## 执行前检查

`command.sh` 在训练前自动检查：

- 训练集恰好包含 303 个 JSON 和 303 个 JPG。
- 完整验证集恰好包含 86 个 JSON 和 86 个 JPG。
- JSON/JPG 文件名去除扩展名后一一对应。
- 每个训练 JSON 包含 6 条非空、不重复 Prompt。
- `is_sentence=true`、`shapes` 非空、多边形至少包含 3 个点。
- `source.file_name`、`source.sample_key`、`source.source_category` 完整。
- 所有图片可由 OpenCV 读取。
- 所有训练标注可生成非空目标 mask。
- Base、CLIP、SAM 权重存在。

任一检查失败时脚本立即停止，不进入训练。

## 远程执行

完整执行数据检查、训练、权重导出、LoRA 合并和评估：

```bash
bash exp/runs/lisa13b-relabel303-lora-v1/command.sh
```

只使用已经合并的模型重新评估，不重新训练：

```bash
bash exp/runs/lisa13b-relabel303-lora-v1/eval_outputs.sh
```

如需指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash exp/runs/lisa13b-relabel303-lora-v1/command.sh
```

## 预期产物

训练与合并权重：

- `runs/lisa13b-relabel303-lora-v1/ckpt_model/`
- `runs/lisa13b-relabel303-lora-v1/pytorch_model.bin`
- `runs/lisa13b-relabel303-lora-v1/merged_hf/`
- `runs/lisa13b-relabel303-lora-v1/meta_log_giou*_ciou*.pth`

完整验证集评估：

- `exp/runs/lisa13b-relabel303-lora-v1/full-eval-outputs/summary.json`
- `exp/runs/lisa13b-relabel303-lora-v1/full-eval-outputs/summary.md`
- `exp/runs/lisa13b-relabel303-lora-v1/full-eval-outputs/per_sample_metrics.jsonl`
- `exp/runs/lisa13b-relabel303-lora-v1/full-eval-outputs/per_sample_metrics.csv`
- `exp/runs/lisa13b-relabel303-lora-v1/full-eval-outputs/per_sample_metrics_by_iou.csv`
- `exp/runs/lisa13b-relabel303-lora-v1/full-eval-outputs/visualizations/`
- `exp/runs/lisa13b-relabel303-lora-v1/full-eval-outputs/pred_masks/`

模型对比：

- `exp/runs/lisa13b-relabel303-lora-v1/base-vs-relabel303.md`
- `exp/runs/lisa13b-relabel303-lora-v1/clean030-vs-relabel303.md`

## 验收指标

正式结论以完整 `ReasonSeg|val` 为准，比较：

- gIoU、cIoU、平均 Dice
- 平均 Precision、平均 Recall
- IoU 为 0 的样本数
- IoU 小于 0.1、0.3 的样本数
- IoU 大于等于 0.5 的样本数
- False Positive / False Negative Area
- 10 个类别的 Mean IoU 和零 IoU 数

重点关注 `unsafe`、`guardrail_missing`、`opening_unprotected`、`equipment_proximity` 等困难类别。训练集结果不能作为泛化性能结论。

## 当前状态

- 实验方案：已确认
- 数据完整性：已在远程服务器人工核验通过
- 实验脚本：已准备
- 训练：待远程执行
- 评估：待远程执行
- 结论：待回填 `summary.json`、日志和 bad case 分析
