# 实验记录

本目录用于集中管理 LISA 施工安全场景的实验结果。每次实验单独建目录,保留执行说明、命令和可追溯指标。

每个实验目录约定如下:

- `EXPERIMENT.md`: 实验背景、模型/数据配置、核心指标和结论。
- `command.sh`: 在远程 Linux GPU 服务器上执行的命令。
- `outputs/`: 从 `benchmark_reason_seg.py` 复制过来的评测产物,包括 `summary.json`、`summary.md`、CSV/JSONL 指标、掩码和可视化结果。

`outputs/` 下的图片和可视化目录不入库,只跟踪指标文件和说明文档。大量图片保留在远程服务器或本地归档中。

## 实验列表

| 实验 | 用途 | 模型 | 数据划分 | 状态 | 关键输出 |
|---|---|---|---|---|---|
| `lisa13b-local-smoke` | LISA-13B 本地评测链路冒烟测试 | 本地 `./LISA13B` | `ReasonSeg|val` 小样本 | 已导入指标 | `outputs/summary.json` |
| `lisa13b-local-train` | 在训练集上评估,观察拟合情况和标注问题 | 本地 `./LISA13B` | `ReasonSeg|train` | 已导入指标 | `outputs/summary.json` |
| `lisa13b-local-val` | 在验证集上评估,作为正式汇报指标来源 | 本地 `./LISA13B` | `ReasonSeg|val` | 已导入指标 | `outputs/summary.json` |

> 旧目录 `benchmark_outputs/reason_seg` 与 `lisa13b-local-val` 是同一组验证集结果,已删除重复归档,正式引用 `lisa13b-local-val`。

## 导入远程结果

仅在远程服务器执行。把新的 评测结果复制到对应实验目录:

```bash
cp -a benchmark_outputs/lisa13b-local-smoke/. exp/runs/lisa13b-local-smoke/outputs/
cp -a benchmark_outputs/lisa13b-local-train/. exp/runs/lisa13b-local-train/outputs/
cp -a benchmark_outputs/lisa13b-local-val/. exp/runs/lisa13b-local-val/outputs/
```

导入后根据 `outputs/summary.json` 更新对应 `EXPERIMENT.md` 的命令、样本数和指标。
