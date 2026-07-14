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

## 实验执行闭环

- 执行任何训练、推理或评估实验前，先整理实验背景、实验配置、数据划分、模型/权重来源、输出目录、预期产物和完整执行脚本。
- 在用户确认前，不直接修改对应实验目录的 `EXPERIMENT.md` 和 `command.sh`，也不要求用户在服务器上执行实验。
- 用户确认后，再把实验背景和配置写入 `EXPERIMENT.md` 的对应位置，把完整执行脚本写入 `command.sh`。
- 用户确认实验方案后，代理可直接将相关实验文件提交并推送到远端仓库，不需要再次询问是否提交或推送。
- 任何实验执行脚本必须自包含：脚本内部要显式写出所需变量、路径和参数，不依赖用户提前在 shell 中 `export` 环境变量。
- 自包含脚本仍不得把服务器私有绝对路径、密钥、令牌、权重文件或数据集大文件写入仓库；优先使用仓库相对路径，必要时使用用户确认过的可迁移占位路径。
- 用户在远程服务器执行实验后，将结果或日志反馈给代理；代理再根据 `outputs/summary.json`、日志和 bad case 分析完善 `EXPERIMENT.md` 的核心指标、结论和备注。

## COCO 原始数据到 LISA Pipeline 脚本清单

默认从给定 COCO 原始数据集开始，输入优先按 `data/<dataset_id>/train/_annotations.coco.json` 或 `data/<dataset_id>/_annotations.coco.json` 组织，图片放在标注同级目录或常见子目录中。当前主线是 COCO bbox → SAM mask → LISA ReasonSeg jpg/json → LoRA 微调/评估。

来源标记：

- `[本项目新增]`：围绕施工安全数据、COCO 转 LISA、评估分析新增的脚本。
- `[论文原有]`：LISA 论文作者原仓库已有脚本/模块，本项目继承使用。
- `[论文原有/本项目改造]`：原仓库已有文件，但本项目历史中做过训练或路径兼容适配。

### 主线脚本

- `[本项目新增]` `data_pipeline/organize_coco_by_category.py`：可选的 COCO 原始数据预检查脚本，按类别整理图片链接/拷贝并输出 `manifest.csv`、`summary.json`，用于了解类别分布和缺图情况。
- `[本项目新增]` `data_pipeline/build_phase1_feasibility_subset.py`：从一个或多个 COCO 数据集均衡抽取 train/val 子集，默认读 `data/002`、`data/004`，输出 `data/phase1_feasibility/{train,val}/_annotations.coco.json`、`manifest.csv`、`summary.json`。
- `[本项目新增]` `data_pipeline/visualize_coco_bboxes.py`：可视化 COCO bbox，既可看原始 COCO，也可看 `data/phase1_feasibility`，用于人工抽检框和类别是否合理。
- `[本项目新增]` `data_pipeline/build_lisa_from_coco.py`：COCO 到 LISA 的主转换脚本，读取 phase1 COCO 子集，调用 SAM 把 bbox 转成 mask，再转 LabelMe 多边形，直接写入 `dataset/reason_seg/ReasonSeg/{train,val}/<name>.jpg/.json`，并生成 `phase1_manifest.csv`、`phase1_build_summary.json`。
- `[本项目新增]` `data_pipeline/visualize_lisa_annotations.py`：可视化生成后的 LISA jpg/json，并可同时对照 COCO 全部类别框和当前目标类别框，用于确认“训练实际读取的 mask”。
- `[论文原有/本项目改造]` `train_ds.py`：远程执行 LoRA 微调；当前施工场景默认使用 `--dataset "reason_seg"`、`--reason_seg_data "ReasonSeg|train"`、`--explanatory -1`。
- `[本项目新增]` `benchmark_reason_seg.py`：远程执行 ReasonSeg 评估，输出 `summary.json`、`summary.md`、`per_sample_metrics.*`、预测 mask 和可视化结果。

### 评估后派生脚本

- `[本项目新增]` `data_pipeline/build_clean_subset_from_benchmark.py`：根据 `benchmark_reason_seg.py` 的 per-sample IoU 结果筛选 clean 子集，默认阈值 `--threshold 0.30`，默认输出 `dataset/reason_seg/ReasonSegClean030`。
- `[本项目新增]` `exp/compare_benchmark_metrics.py`：对比 base 与 LoRA 后的 benchmark 指标，生成指标差异和 bad case 分析。
- `[本项目新增]` `exp/build_annotation_prediction_report.py`：基于标注、base 预测、LoRA 预测生成对照报告，用于定位标注问题、预测退化和真实提升样本。
- `[论文原有]` `merge_lora_weights_and_save_hf_model.py`：可选，远程执行，用于把 LoRA 权重合并并保存成 Hugging Face 格式模型。

### 支撑模块

- `[本项目新增]` `data_pipeline/config.py`：pipeline 配置，包含路径、SAM checkpoint、隐患分类体系和 mask 质检阈值；COCO 主转换会复用其中的 SAM 与 QC 配置。
- `[本项目新增]` `data_pipeline/box_to_mask.py`：SAM box prompt 封装，提供 bbox → mask 和 box/mask IoU 质检函数；正式运行需要远程 GPU 和 SAM 权重。
- `[本项目新增]` `data_pipeline/instruction_bank.py`：旧检测框流程的指令模板库；COCO 主转换脚本目前内置 `INSTRUCTION_BANK`。
- `[论文原有]` `utils/data_processing.py`：LISA json → mask 的统一读取逻辑，训练、评估和可视化必须与这里保持一致。
- `[论文原有]` `utils/reason_seg_dataset.py`：训练时读取 `dataset/reason_seg/<dataset>/<split>/*.jpg` 并匹配同名 `.json` 的数据集实现。

### 旧流程/替代入口

- `[本项目新增]` `data_pipeline/grounded_ingest.py`：无 COCO 标注时的替代入口，原始图/视频 → 开放词表检测/规则推导 → 检测框原料。
- `[本项目新增]` `data_pipeline/build_dataset.py`：旧的通用检测框 txt → LISA `data_pipeline/out/` 主编排脚本，输入为 `data_pipeline/raw/*.jpg + .txt`，不是当前 COCO 主线。
- `[本项目新增]` `data_pipeline/quality_check.py`：旧流程 `data_pipeline/out/` 的 mask 叠加质检可视化。
- `[本项目新增]` `data_pipeline/deploy_to_dataset.py`：旧流程把 `data_pipeline/out/` 按 `split.json` 分发到 `dataset/reason_seg/ReasonSeg/{train,val}`；COCO 主线的 `build_lisa_from_coco.py` 已直接写入训练目录，通常不需要再执行它。

## 推荐远程训练命令

```bash
python train_ds.py \
  --dataset_dir ./dataset \
  --dataset "reason_seg" \
  --reason_seg_data "ReasonSeg|train" \
  --explanatory -1 \
  --exp_name lisa-construction
```
