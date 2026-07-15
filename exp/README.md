# 实验记录

本目录用于集中管理 LISA 施工安全场景的实验结果。每次实验单独建目录,保留执行说明、命令和可追溯指标。

每个常规 benchmark 实验目录约定如下:

- `EXPERIMENT.md`: 实验背景、模型/数据配置、核心指标和结论。
- `command.sh`: 在远程 Linux GPU 服务器上执行的命令。
- `outputs/`: 从 `benchmark_reason_seg.py` 复制过来的评测产物,包括 `summary.json`、`summary.md`、CSV/JSONL 指标、掩码和可视化结果。

`outputs/` 下的图片和可视化目录不入库,只跟踪指标文件和说明文档。大量图片保留在远程服务器或本地归档中。

`lisa13b-clean030-lora-v1` 不走这个 `outputs/` 约定,它把评估结果写到扁平目录:

- `exp/runs/lisa13b-clean030-lora-v1/clean-eval-outputs/`
- `exp/runs/lisa13b-clean030-lora-v1/full-eval-outputs/`

## 实验列表

| 实验 | 用途 | 模型 | 数据划分 | 状态 | 关键输出 |
|---|---|---|---|---|---|
| `lisa13b-local-train` | 在训练集上评估,观察域适配情况和标注问题 | 本地 `./LISA13B` | `ReasonSeg|train` | 已完成（2026-07-10） | `outputs/summary.json` |
| `lisa13b-local-val` | 在验证集上评估,作为正式汇报指标来源 | 本地 `./LISA13B` | `ReasonSeg|val` | 已完成（2026-07-10） | `outputs/summary.json` |
| `lisa13b-clean030-lora-v1` | 用 Clean030 子集做 LoRA 微调,并在 clean/full val 上做正式对比 | `./LISA13B` | `ReasonSegClean030|train` / `ReasonSegClean030|val` / `ReasonSeg|val` | 已完成训练与评估（2026-07-13～14） | `clean-eval-outputs/`, `full-eval-outputs/` |

> Base train/val benchmark 和 Clean030 LoRA 实验均已完成。正式泛化结论以同一个完整 `ReasonSeg|val` 上的 Base/LoRA 对比为准，`ReasonSegClean030|val` 只用于观察高置信子集的学习情况。

## Base vs Clean030 LoRA 最终对比

两组结果均在完整 `ReasonSeg|val` 的同一批 86 个样本上评估。Base 来自 `lisa13b-local-val/outputs/`，LoRA 来自 `lisa13b-clean030-lora-v1/full-eval-outputs/`。

| 指标 | Base LISA-13B | Clean030 LoRA | 变化 |
| --- | ---: | ---: | ---: |
| gIoU | 0.3408 | 0.4494 | +0.1086 |
| cIoU | 0.3177 | 0.3858 | +0.0681 |
| Mean Dice | 0.4180 | 0.5156 | +0.0976 |
| Mean Precision | 0.4071 | 0.5332 | +0.1261 |
| Mean Recall | 0.5132 | 0.5416 | +0.0284 |
| IoU = 0 样本数 | 22 | 16 | -6 |
| IoU >= 0.5 样本数 | 29 | 39 | +10 |
| False Positive Area | 2,321,866 | 1,623,031 | -698,835（-30.1%） |
| False Negative Area | 1,821,927 | 1,677,931 | -143,996（-7.9%） |

Clean030 LoRA 在完整验证集上的主要收益是降低误检：Precision 提升 12.61 个点，False Positive Area 下降 30.1%。Recall 只提升 2.84 个点，False Negative Area 下降 7.9%，因此漏检仍是下一轮数据治理和训练的主要问题。

## 自动记录远程结果

仅在远程服务器执行。常规 benchmark 评测时把 `--output_dir` 直接指向对应实验目录的 `outputs/`:

```bash
--output_dir ./exp/runs/<experiment-name>/outputs
```

`benchmark_reason_seg.py` 检测到输出目录符合 `exp/runs/<experiment-name>/outputs` 约定后,会自动:

- 写入评测产物到 `outputs/`;
- 更新 `EXPERIMENT.md` 的配置和核心指标;
- 若 `command.sh` 不存在则创建;若已存在,不覆盖已确认脚本,只把本次实际命令记录到 `outputs/last_command.sh`。

Clean030 实验由 `exp/runs/lisa13b-clean030-lora-v1/eval_outputs.sh` 管理,它会把结果写到 `clean-eval-outputs/` 和 `full-eval-outputs/`,再自行维护 `last_command.sh`。

推荐直接运行对应实验目录下的 `command.sh`:

```bash
bash exp/runs/lisa13b-local-train/command.sh
bash exp/runs/lisa13b-local-val/command.sh
```

如果输出目录不在 `exp/runs/<experiment-name>/outputs`,但仍想更新实验记录,可以额外传 `--record_exp`。

## 实验闭环约定

每次新增或重跑实验按以下闭环执行:

1. **实验前整理**
   - 先整理实验背景、实验目标、模型/权重来源、数据划分、关键参数、输出目录、预期产物和完整执行脚本。
   - 在用户确认前,不写入或覆盖对应实验目录的 `EXPERIMENT.md` 和 `command.sh`。

2. **用户确认后落盘**
   - 用户确认实验方案后,再把背景、配置、预期输出写入 `EXPERIMENT.md`。
   - 同时把完整执行脚本写入 `command.sh`。
   - 用户确认实验方案即视为允许代理直接提交并推送相关实验文件,无需再次确认提交/推送。
   - `command.sh` 必须自包含,不能依赖用户提前 `export BASE_MODEL`、`SAM_CKPT`、`CLIP_TOWER` 等环境变量。
   - 脚本内部应显式定义所需路径和参数;优先使用仓库相对路径,不要把服务器私有绝对路径、密钥、令牌或大文件写入仓库。

3. **远程执行**
   - 用户在远程 Linux GPU 服务器执行 `command.sh`。
   - 本地不运行训练、推理、SAM/LISA 权重加载或长时间评估。

4. **结果回填**
   - 用户把实验输出、日志或 `outputs/summary.json` 反馈给代理。
   - 代理根据结果完善 `EXPERIMENT.md` 的核心指标、结论、备注和 bad case 分析。
   - 若评估命令输出到 `exp/runs/<experiment-name>/outputs`,脚本会自动更新配置和核心指标;人工分析仍需补充到结论和备注中。
