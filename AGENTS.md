# AGENTS.md

本仓库的协作与执行约束如下，适用于所有编码代理和本地开发流程。

## 执行环境

- 本地 Windows 机器只用于编写代码、修改文档、阅读项目结构和做轻量级静态检查。
- 模型训练、GPU 推理、SAM / GroundingDINO / LISA 权重加载、数据集下载和长时间运行任务全部在远程 Linux + GPU 服务器执行。
- 本地不要运行 `train_ds.py`、`chat.py`、`app.py`，不要加载大模型权重，不要下载大型数据集或模型文件。
- 可以在本地运行不依赖 GPU、权重和大数据的纯逻辑校验，例如数据格式检查、路径检查、`--dry-run` 流程或小型单元测试。

## 开发要求

- 写代码时默认目标运行环境是远程 Linux + GPU，注意路径兼容、CUDA 依赖和大小写敏感文件名。
- 本地提交的变更应保证服务器同步后可直接运行，运行命令默认提供给远程服务器执行。
- 训练相关命令、推理命令和依赖安装命令应明确标注为“远程执行”。
- 不要把服务器私有绝对路径、密钥、令牌、权重文件或数据集大文件写入仓库。
- 给服务器粘贴执行的短命令优先写成单行 `python -c "..."` 或普通 shell 命令，避免使用 heredoc（如 `python - <<'PY' ... PY`），因为缩进或结束标记不顶格会让 shell 卡在 `>` 续行提示。
- 给服务器粘贴执行的长命令需要换行时，必须使用行末反斜杠 `\` 显式续行；`\` 必须是该行最后一个字符，后面不得有空格或注释。
- 不得在命令参数、文件名或路径中间直接换行；换行应放在参数边界，并保证每个路径在同一行内完整呈现。

## 实验执行闭环

- 执行任何训练、推理或评估实验前，先整理实验背景、实验配置、数据划分、模型/权重来源、输出目录、预期产物和完整执行脚本。
- 在用户确认前，不直接修改对应实验目录的 `EXPERIMENT.md` 和 `command.sh`，也不要求用户在服务器上执行实验。
- 用户确认后，再把实验背景和配置写入 `EXPERIMENT.md` 的对应位置，把完整执行脚本写入 `command.sh`。
- 用户确认实验方案后，代理可直接将相关实验文件提交并推送到远端仓库，不需要再次询问是否提交或推送。
- 任何实验执行脚本必须自包含：脚本内部要显式写出所需变量、路径和参数，不依赖用户提前在 shell 中 `export` 环境变量。
- 自包含脚本仍不得把服务器私有绝对路径、密钥、令牌、权重文件或数据集大文件写入仓库；优先使用仓库相对路径，必要时使用用户确认过的可迁移占位路径。
- 用户在远程服务器执行实验后，将结果或日志反馈给代理；代理再根据 `outputs/summary.json`、日志和 bad case 分析完善 `EXPERIMENT.md` 的核心指标、结论和备注。

## 操作脚本文档

COCO 到 LISA 的 pipeline 脚本清单、训练/评估命令和当前实验入口已迁移到 `docs_caich/readme`。

## 项目待办（面试准备）

当前已完成 COCO bbox -> SAM mask -> LISA ReasonSeg -> LoRA -> benchmark 的完整链路。Base LISA-13B 在完整 `ReasonSeg|val` 上的指标为 `gIoU=0.3408 / cIoU=0.3177 / Dice=0.4180`，Clean030 LoRA 后为 `gIoU=0.4494 / cIoU=0.3858 / Dice=0.5156`。后续优先完善证据链、数据质量和实验设计，不急于盲目扩大模型或堆叠超参数实验。

### P0：实验结论与数据可信度

- [x] 修正 `exp/README.md`、`lisa13b-local-train/EXPERIMENT.md` 和 `lisa13b-local-val/EXPERIMENT.md` 中已有结果但仍标记为“待执行/待评估”的内容。
- [x] 整理一张 Base LISA-13B vs Clean030 LoRA 的最终对比表，包含 gIoU、cIoU、Dice、Precision、Recall、零 IoU 数、`IoU >= 0.5` 数和误检/漏检面积变化。
- [ ] 建立人工核验的小型 golden test，优先覆盖 30～60 张独立图片，人工检查 prompt、目标语义和 mask 边界。
- [ ] golden test 按原始视频、拍摄序列或工地分组划分，避免相邻帧跨 train/test；冻结后不再根据模型结果修改。
- [ ] 在 golden test 上重新评估 Base 和当前最佳 LoRA，明确区分“接近 SAM 伪标签”与“接近人工认定的真实目标”。

### P0：Relabel 与 bad case 闭环

- [ ] 使用 `dataset/reason_seg/ReasonSegRelabel/samples_by_iou.md` 优先审核 IoU 为 0、IoU < 0.1、LoRA 后严重退化以及 Target Area 异常的样本。
- [ ] 为 bad case 建立固定错误类型：prompt/mask 错配、prompt 过于抽象、SAM 越界、SAM 漏标、多实例选择不一致、目标不可辨认、语义理解失败、边界分割不准。
- [ ] Relabel 时同时核验 prompt 和 mask，不只做同义句替换；记录每条标签的修改类型和原因。
- [ ] 优先制定 `safe/unsafe` 的统一标注规则，明确分割对象是人员、设备、危险行为还是整片危险区域；必要时将 `unsafe` 拆分为具体隐患类别。
- [ ] 统计 Relabel 前后的错误类型、修改数量和类别分布，形成可追溯的数据治理结论。

### P1：关键消融与稳定性

- [ ] 在相同 LoRA 参数、训练步数、随机种子和评估集下对比 Base、Clean030 LoRA 和 Relabel Full LoRA。
- [ ] GPU 时间允许时增加 Original Full LoRA，用于判断收益主要来自 LoRA、Clean030 筛选还是 Relabel 数据清洗。
- [ ] 如困难类别仍明显落后，增加 Relabel + 类别均衡/hard-case 采样实验，优先关注 `unsafe`、`guardrail_missing`、`opening_unprotected` 和 `equipment_proximity`。
- [ ] 对最终配置至少运行 3 个随机种子，报告 `mean ± std`，不只报最好单次结果。
- [ ] 增加 Base/LoRA paired bootstrap 置信区间；按独立 `source_file_name` 聚类采样，不将同一图片的多个标签视为完全独立样本。
- [ ] 统一输出总体、分类别、分目标尺寸和分错误类型指标，并保留回归样本分析。

### P2：面试展示与工程化

- [ ] 准备一个图片 + 自然语言 prompt -> mask overlay 的推理 Demo，支持 Base/LoRA 并排对比。
- [ ] Demo 中展示模型版本、GPU、显存占用、纯推理时间和端到端延迟，不将当前约 0.42 秒/样本的 benchmark 时间直接等同于完整产品延迟。
- [ ] 准备 3 个成功样例和 2 个可解释的失败样例，说明模型能力边界和改进方向。
- [ ] 整理一页项目总览，包含任务定义、数据流程、模型改造、实验表、bad case、局限性和下一步。
- [ ] 准备“数据 -> 训练 -> 评估 -> 失败分析 -> 迭代”的五段式面试叙述，能在 2 分钟内介绍项目主线。
