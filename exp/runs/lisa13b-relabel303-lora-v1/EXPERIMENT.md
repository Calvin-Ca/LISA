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

## 输出产物

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

## 核心指标

三组模型均在同一个完整 `ReasonSeg|val` 上评估，共 86 个样本：

| 模型 | gIoU | cIoU | Mean Dice | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base LISA-13B | 0.3408 | 0.3177 | 0.4180 | 0.4071 | 0.5132 |
| Clean030 LoRA | **0.4494** | **0.3858** | **0.5156** | **0.5332** | **0.5416** |
| Relabel303 LoRA | 0.4112 | 0.3263 | 0.4752 | 0.5102 | 0.4883 |

Relabel303 相对 Base：

- gIoU：`+0.0704`
- cIoU：`+0.0086`
- Mean Dice：`+0.0573`
- Precision：`+0.1031`
- Recall：`-0.0249`
- 49 个样本提升，27 个样本退化，10 个样本不变

Relabel303 相对 Clean030 LoRA：

- gIoU：`-0.0383`
- cIoU：`-0.0595`
- Mean Dice：`-0.0404`
- Precision：`-0.0230`
- Recall：`-0.0533`
- 36 个样本提升，40 个样本退化，10 个样本不变

## 样本级分布

| 指标 | Base | Clean030 LoRA | Relabel303 LoRA |
| --- | ---: | ---: | ---: |
| IoU = 0 | 22 | 16 | **14** |
| IoU < 0.1 | 33 | **28** | 33 |
| IoU < 0.3 | 44 | **38** | 39 |
| IoU >= 0.5 | 29 | **39** | 33 |
| False Positive Area | 2,321,866 | 1,623,031 | **1,463,719** |
| False Negative Area | 1,821,927 | **1,677,931** | 2,049,496 |

Relabel303 的零 IoU 数和误检面积最低，但低 IoU 样本仍多于 Clean030，且漏检面积最高。模型整体变得更保守：减少了无关区域预测，同时漏掉了更多真实目标。

## 分类别结果

| 类别 | 样本数 | Base | Clean030 LoRA | Relabel303 LoRA | Relabel vs Base | Relabel vs Clean030 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| equipment_proximity | 3 | 0.2611 | **0.4039** | 0.2925 | +0.0315 | -0.1113 |
| guardrail_missing | 4 | 0.0410 | **0.2341** | 0.1450 | +0.1040 | -0.0891 |
| harness_missing | 4 | 0.5083 | **0.5382** | 0.5087 | +0.0004 | -0.0295 |
| helmet_missing | 4 | 0.3794 | 0.5689 | **0.5873** | +0.2079 | +0.0184 |
| no helmet | 15 | 0.5712 | **0.5841** | 0.5691 | -0.0021 | -0.0149 |
| no jacket | 12 | 0.3109 | 0.3583 | **0.4879** | +0.1770 | +0.1296 |
| opening_unprotected | 8 | 0.2326 | **0.4065** | 0.3643 | +0.1317 | -0.0423 |
| poor_housekeeping | 4 | 0.2215 | **0.5622** | 0.5184 | +0.2968 | -0.0439 |
| safe | 13 | 0.6412 | **0.7693** | 0.6619 | +0.0207 | -0.1074 |
| unsafe | 19 | 0.0750 | **0.1849** | 0.0807 | +0.0057 | -0.1042 |

Relabel303 相比 Base 的主要收益集中在 `poor_housekeeping`、`helmet_missing`、`no jacket`、`opening_unprotected` 和 `guardrail_missing`。相比 Clean030，只有 `no jacket` 和 `helmet_missing` 的分类别均值更高。

## 实验结论

1. Relabel303 微调有效，但没有超过 Clean030 LoRA。Relabel303 在完整验证集上将 Base gIoU 从 `0.3408` 提升到 `0.4112`，证明人工审核数据可以产生真实收益；但当前最佳总体结果仍是 Clean030 LoRA 的 `0.4494`。
2. Relabel303 的训练难度显著高于 Clean030。Clean030 的 202 个训练样本全部满足 Base IoU `>= 0.30`，其 Base 平均 IoU 为 `0.6534`；Relabel303 的 Base 平均 IoU 只有 `0.3686`，其中 123 个样本 IoU 小于 `0.1`、67 个样本 IoU 为 0。Prompt 虽已人工改写，但困难 mask、目标粒度和类别语义问题仍然存在。
3. 当前验证集对 Clean030 存在 Prompt 模板优势。Clean030 训练集的 31 种唯一 Prompt 与 `ReasonSeg|val` 的 27 种唯一 Prompt 完全重合 26 种，覆盖率为 `96.3%`；Relabel303 有 1691 种唯一人工 Prompt，与验证集没有完全相同的句子。因此当前结果同时受到样本质量和语言分布偏移影响，不能单独归因于人工 Prompt 的优劣。
4. Relabel303 更倾向于保守预测。相对 Clean030，其 False Positive Area 从 `1,623,031` 降至 `1,463,719`，但 False Negative Area 从 `1,677,931` 增至 `2,049,496`。这说明具体化 Prompt 有助于减少泛化到无关区域，但模型对困难目标的召回不足。
5. `unsafe` 和 `safe` 仍是主要语义瓶颈。这两类的 mask 可能分别对应人员、设备、构件或区域，目标定义不统一。仅扩写 Prompt 无法消除底层标签粒度差异，需要进一步统一类别规则或拆分为具体隐患类别。
6. 本实验不能独立证明“6 条人工 Prompt 优于 1 条原 Prompt”，因为 Relabel303 与 Clean030 的样本集合和难度分布不同。下一步应使用二者重合的 150 个样本，保持图片、mask、训练步数和每次 Prompt 数完全一致，只改变 `text`，并在冻结的独立 golden test 上进行受控消融。

## 未超过 Clean030 的原因分析

严格来说，人工重新标注后并非没有提升：Relabel303 将 Base gIoU 从 `0.3408` 提升到 `0.4112`，但没有超过 Clean030 LoRA 的 `0.4494`。当前结果更适合解释为“人工 Prompt 有效，但收益被其他变量抵消”，主要原因如下。

### 1. 重新标注只修改 Prompt，没有修改 mask

本轮 Relabel 严格限制为只修改 JSON 的 `text`，原有 `shapes` 和 SAM mask 保持不变。因此 Prompt 变得更准确、具体，但 mask 越界、漏标、目标粒度不一致等像素监督噪声仍可能存在。

语言标签变好不等于像素监督同步变好。Prompt 越具体，也越容易暴露 mask 与语义之间的细微不一致。

### 2. Relabel303 的训练样本显著难于 Clean030

| 训练集 | 样本数 | Base 平均 IoU | IoU < 0.1 | IoU = 0 |
| --- | ---: | ---: | ---: | ---: |
| Clean030 | 202 | 0.6534 | 0 | 0 |
| Relabel303 | 303 | 0.3686 | 123 | 67 |

Clean030 是按 Base IoU `>= 0.30` 筛选的高置信样本；Relabel303 则包含大量原模型难以识别的样本。两次实验不仅改变了 Prompt，还改变了样本集合、难度和类别分布，因此不是严格的 Prompt 单变量对比。

### 3. 当前验证集偏向原始 Prompt 模板

`ReasonSeg|val` 共有 27 种唯一 Prompt。Clean030 训练集只有 31 种唯一 Prompt，其中 26 种与验证集完全相同，对验证 Prompt 的模板覆盖率为 `96.3%`。

Relabel303 有 1691 种唯一人工 Prompt，与验证集完全相同的句子为 0。Clean030 更接近模板内评估，Relabel303 则需要跨表达方式泛化。当前 benchmark 可能高估 Clean030 的模板匹配收益，同时低估人工多 Prompt 的语言泛化价值。

### 4. 相同训练步数下，Relabel303 的覆盖不足

两次实验均训练 600 个 optimizer steps，共采样约 4800 次图片：

- Clean030：每个样本平均约被采样 `23.8` 次。
- Relabel303：每个样本平均约被采样 `15.8` 次。

Relabel303 每个样本包含 6 条 Prompt，每次随机抽取 3 条。模型需要在更少的单样本覆盖次数内学习更大的语言表达空间，当前训练预算可能不足以让所有 Prompt 和困难目标稳定收敛。

### 5. `safe`、`unsafe` 等类别的目标定义仍不统一

`unsafe` 的 mask 可能对应人员、设备、构件、行为关联目标或整片空间区域；`safe` 也可能对应人员或区域。即使单条 Prompt 已与对应 mask 对齐，同一类别内部的对象类型和分割粒度仍可能冲突。

Relabel303 的 `unsafe` Mean IoU 相比 Clean030 从 `0.1849` 降至 `0.0807`，说明仅扩写 Prompt 无法解决底层类别语义和目标粒度不统一的问题。

### 6. Relabel303 更保守，误检减少但漏检增加

| 模型 | False Positive Area | False Negative Area |
| --- | ---: | ---: |
| Clean030 LoRA | 1,623,031 | 1,677,931 |
| Relabel303 LoRA | 1,463,719 | 2,049,496 |

人工 Prompt 更具体后，模型减少了无关区域预测，但也更容易只预测较小的确定区域或完全漏掉困难目标：

- IoU 为 0：Clean030 16，Relabel303 14。
- IoU 小于 0.1：Clean030 28，Relabel303 33。
- IoU 大于等于 0.5：Clean030 39，Relabel303 33。

因此 Relabel303 救回了少量完全失败样本，却增加了严重漏检和低 IoU 样本。

### 7. 单次训练存在随机性干扰

当前训练入口没有固定完整随机种子。训练中的图片随机采样、6 条 Prompt 随机抽取、LoRA 参数初始化和优化过程都会产生波动。

本实验每种配置只运行一次，无法确定 Relabel303 相对 Clean030 的 gIoU 差值 `-0.0383` 中有多少来自数据与 Prompt 差异，多少来自随机训练波动。后续受控消融应固定 seed，并在条件允许时运行至少 3 个随机种子，报告 `mean ± std`。

综合来看，人工 Prompt 确实让模型优于 Base，并降低了误检；但困难样本、未修正的 mask 噪声、验证模板偏差、训练覆盖不足、类别语义不统一和随机性共同抵消了收益，因此没有超过高置信筛选的 Clean030。

## 局限与后续

- 当前正式评估集仍是已多次使用的 `ReasonSeg|val`，不是独立冻结的人工 golden test。
- `ReasonSeg|val` 的 Prompt 与 Clean030 训练模板高度重合，可能高估 Clean030 的模板内表现。
- Relabel303 只修改 Prompt，没有重新标注或修正 mask。
- 训练入口未固定随机种子，单次实验可能包含随机采样波动。
- 两组训练均使用 600 optimizer steps；Relabel303 样本更多、Prompt 更多，单个样本和单条 Prompt 的平均覆盖次数更低。

下一实验优先执行 Original150 vs Relabel150 Prompt 受控消融，并至少固定相同随机种子；条件允许时运行 3 个 seed，报告 `mean ± std`。

## 当前状态

- 实验方案：已确认
- 数据完整性：已在远程服务器人工核验通过
- 实验脚本：已执行
- 训练：已完成
- 权重合并：已完成
- 正式评估：已完成，`ReasonSeg|val` 共 86 个样本
- 评估结果时间：2026-07-16
- 当前最佳模型：Clean030 LoRA
- Relabel303 结论：优于 Base，但未超过 Clean030；人工 Prompt 的独立收益需通过重合 150 样本受控消融继续验证
