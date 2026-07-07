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

## 推荐远程训练命令

```bash
python train_ds.py \
  --dataset_dir ./dataset \
  --dataset "reason_seg" \
  --reason_seg_data "ReasonSeg|train" \
  --explanatory -1 \
  --exp_name lisa-construction
```
