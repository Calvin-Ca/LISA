# Data Pipeline Scripts

本目录包含两类数据脚本：

- 当前主线：COCO bbox -> SAM mask -> LISA ReasonSeg jpg/json。
- legacy 旧流程：`raw/*.jpg + .txt` bbox -> `data_pipeline/out/` -> 手动分发到 ReasonSeg。

默认优先使用当前 COCO 主线。legacy 脚本保留用于无 COCO 标注、txt bbox 原料或历史复现实验。

## 当前主线

| 脚本 | 远程/本地 | 执行方式 | 作用 |
|---|---|---|---|
| `organize_coco_by_category.py` | 本地可跑 | `python data_pipeline/organize_coco_by_category.py --data-root data` | 可选预检查；按类别整理/链接 COCO 图片，输出类别分布和缺图清单。 |
| `build_phase1_feasibility_subset.py` | 本地可跑 | `python data_pipeline/build_phase1_feasibility_subset.py --overwrite` | 从 `data/002,data/004` 等 COCO 数据集中均衡抽取 train/val 子集，生成 `data/phase1_feasibility/`。 |
| `visualize_coco_bboxes.py` | 本地可跑 | `python data_pipeline/visualize_coco_bboxes.py --data-root data/phase1_feasibility --output-root data/phase1_feasibility/vis_bboxes` | 可视化 COCO bbox，人工检查类别和框质量。 |
| `build_lisa_from_coco.py` | 远程执行 | `python data_pipeline/build_lisa_from_coco.py --overwrite` | 调用 SAM 将 COCO bbox 转 mask，再写成 LISA/ReasonSeg jpg/json。需要 GPU 和 SAM 权重。 |
| `visualize_lisa_annotations.py` | 本地可跑 | `python data_pipeline/visualize_lisa_annotations.py --input-dir dataset/reason_seg/ReasonSeg/train` | 可视化 LISA json 多边形，确认训练实际读取的 mask。 |
| `rephrase_reason_seg_instructions.py` | 本地/远程均可 | `python data_pipeline/rephrase_reason_seg_instructions.py --dry-run` | 调用 OpenAI API 为 Relabel train prompt 生成严格等价改写；先写审计 manifest，确认后才用 `--apply` 更新 JSON。 |

当前主线典型顺序：

```bash
python data_pipeline/build_phase1_feasibility_subset.py --overwrite
python data_pipeline/visualize_coco_bboxes.py --data-root data/phase1_feasibility --output-root data/phase1_feasibility/vis_bboxes
python data_pipeline/build_lisa_from_coco.py --overwrite
python data_pipeline/visualize_lisa_annotations.py --input-dir dataset/reason_seg/ReasonSeg/train
```

其中 `build_lisa_from_coco.py` 需要远程 Linux GPU 服务器执行；其他脚本只做文件整理或可视化，数据量不大时本地可跑。

### ReasonSeg 训练指令改写

`rephrase_reason_seg_instructions.py` 默认只读取 `ReasonSegRelabel/train/*.json`，不处理 val，也不加载任何视觉模型。先在无网络、无费用的 dry-run 中检查范围：

```bash
python data_pipeline/rephrase_reason_seg_instructions.py --dry-run
```

正式生成前由当前 shell 提供 `OPENAI_API_KEY`，不得将 key 写入仓库。首先用少量样本验证 prompt 约束和账号模型权限：

```bash
python data_pipeline/rephrase_reason_seg_instructions.py --limit 10
```

`--limit` 默认使用固定 `--seed 42` 做分类别轮询抽样，使首批 10 条尽量覆盖全部类别，而不是按文件名连续抽到同一类。

默认按用户指定的官方方案使用 `gpt-3.5-turbo`，但该模型已在 OpenAI 当前目录中标记为 deprecated。可通过 `--model` 显式选择当前账号可用且支持 Structured Outputs 的新模型：

```bash
python data_pipeline/rephrase_reason_seg_instructions.py --model gpt-5-mini --limit 10
```

生成结果先写入 `instruction_rewrites.jsonl`，不会立即修改标注。人工检查语义等价性后，复用 manifest 并写入每个 JSON 的 `text` 列表：

```bash
python data_pipeline/rephrase_reason_seg_instructions.py --apply
```

`--apply` 只会应用已缓存的成功记录，绝不调用 API；没有审核记录的样本会被跳过。默认每个样本保留 1 条标准指令并增加 4 条改写，共 5 条。脚本支持断点复用；除非显式加 `--force`，已有合法 manifest 记录的样本不会重复调用 API。

为避免误发全量付费请求，生成模式必须显式指定 `--limit N` 或 `--all`。小批量验证通过后才可全量生成：

```bash
python data_pipeline/rephrase_reason_seg_instructions.py --all
```

## 评估后数据派生

| 脚本 | 远程/本地 | 执行方式 | 作用 |
|---|---|---|---|
| `build_clean_subset_from_benchmark.py` | 本地可跑 | `python data_pipeline/build_clean_subset_from_benchmark.py --overwrite` | 根据 base benchmark 的 per-sample IoU 筛出 `ReasonSegClean030`；只复制原始 jpg/json，不使用预测 mask 作为标签。 |

默认输入：

```text
exp/runs/lisa13b-local-train/outputs/per_sample_metrics.jsonl
exp/runs/lisa13b-local-val/outputs/per_sample_metrics.jsonl
```

默认输出：

```text
dataset/reason_seg/ReasonSegClean030/
```

## 支撑模块

| 文件 | 是否直接执行 | 作用 |
|---|---|---|
| `config.py` | 否 | 数据路径、SAM checkpoint、隐患分类、mask 质检阈值等配置。 |
| `box_to_mask.py` | 否 | SAM box prompt 封装，被 `build_lisa_from_coco.py` 和 legacy 流程复用。 |
| `instruction_bank.py` | 否 | legacy txt bbox 流程的指令模板库；当前 COCO 主线主要使用 `build_lisa_from_coco.py` 内置模板。 |

## Legacy 旧流程

这些脚本不是当前 COCO 主线默认入口，只在没有 COCO 标注、只有图片/txt bbox 或需要复现实验旧流程时使用。

| 脚本 | 远程/本地 | 执行方式 | 作用 |
|---|---|---|---|
| `grounded_ingest.py` | 本地/远程视依赖而定 | `python data_pipeline/grounded_ingest.py --frames-only` | 无框数据前端；抽帧或用 GroundingDINO 生成基础实体框，输出 legacy `raw/*.txt`。检测模式依赖 GroundingDINO。 |
| `build_dataset.py` | 远程执行 | `python data_pipeline/build_dataset.py --dry-run` / `python data_pipeline/build_dataset.py` | legacy 主编排；读取 `data_pipeline/raw/*.jpg + .txt`，用 SAM 生成 mask 和 LISA json 到 `data_pipeline/out/`。正式运行需要 GPU 和 SAM 权重。 |
| `quality_check.py` | 本地可跑 | `python data_pipeline/quality_check.py` | legacy `data_pipeline/out/` 的 mask 叠加可视化。 |
| `deploy_to_dataset.py` | 本地可跑 | `python data_pipeline/deploy_to_dataset.py` | legacy 部署脚本；按 `data_pipeline/out/split.json` 把 jpg/json 复制到 `dataset/reason_seg/ReasonSeg/{train,val}`。当前 COCO 主线通常不需要执行。 |

## 产物对齐

当前主线和 legacy 流程最终都应生成 LISA/ReasonSeg 兼容的 jpg/json：

```text
dataset/reason_seg/ReasonSeg/<split>/<name>.jpg
dataset/reason_seg/ReasonSeg/<split>/<name>.json
```

JSON 读取逻辑必须与 `utils/data_processing.py::get_mask_from_json` 保持一致。
