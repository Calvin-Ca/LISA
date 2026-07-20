# lisa13b-clean030-production-perf-v1

## 状态

方案已确认，脚本已准备，等待远程 Linux GPU 服务器执行。

## 背景

Clean030 bf16 冻结制品已经完成 SHA-256、FastAPI 单请求冒烟、完整
`ReasonSeg|val` benchmark 和逐样本回归检查。本实验补齐首版生产候选的
API 性能与显存基线，为以下决策提供依据：

- A100 40GB、单GPU、单worker、并发1时，bf16是否有足够显存余量。
- 生产端到端延迟是否满足首版准入阈值。
- 连续请求后是否存在明显显存增长、请求失败或 CUDA OOM。
- 是否有必要进入P1的8bit量化实验；4bit不在本实验范围内。

本实验不训练模型、不修改权重、不修改数据集，也不替代独立 golden
test。

## 模型与环境

- 模型版本：`lisa13b-clean030-v1`
- 模型制品：`artifacts/lisa-safety-seg/lisa13b-clean030-v1/merged_hf`
- 制品完整性：`SHA256SUMS` 已全部校验通过
- 推理精度：bf16
- 量化：关闭8bit、关闭4bit
- CLIP：本地 `openai/clip-vit-large-patch14`
- GPU：NVIDIA A100-PCIE-40GB，40,960 MiB
- 驱动：`580.159.03`
- PyTorch：`2.1.0+cu121`
- Transformers：`4.31.0`
- DeepSpeed：`0.12.6`
- bitsandbytes：`0.41.1`
- FastAPI：`0.100.1`
- Uvicorn：`0.23.2`
- OpenCV：`4.8.0.74`

## 固定输入

- 图片：
  `dataset/reason_seg/ReasonSeg/val/val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing.jpg`
- Prompt：`标出未按规定佩戴安全帽的作业人员。`
- 输入来源：现有 `ReasonSeg|val`，只作为固定性能请求，不用于重新报告精度
- API：`POST /v1/segment`
- 服务地址：`127.0.0.1:8001`
- worker：1
- GPU并发：1
- mask threshold：`0.0`
- 模型加载：eager

## 请求阶段

脚本按顺序执行：

1. 检查端口、冻结模型、CLIP和固定图片。
2. 检查GPU上是否存在其他计算进程；如存在则退出，避免显存和延迟被污染。
3. 启动200毫秒间隔的 `nvidia-smi` 采样。
4. 启动单worker Uvicorn并等待 `/ready`。
5. 记录进程启动到模型ready的时间和模型加载后显存。
6. 执行1次首次API请求。
7. 执行5次预热请求。
8. 执行30次正式计时请求。
9. 执行100次连续稳定性请求。
10. 校验每次响应的 request ID、尺寸、mask数量、PNG Base64格式。
11. 汇总P50/P95/P99、吞吐、加载显存、峰值显存和预热后显存漂移。
12. 停止Uvicorn和显存采样进程，确保不遗留服务。

总请求数为136次。所有请求串行执行，因此本实验只形成并发1基线；并发
2和4必须在修复“超时后底层GPU任务仍运行但并发槽提前释放”问题后执行。

## 准入阈值

- 136次请求全部成功。
- 所有成功响应的mask协议检查通过。
- 30次正式请求客户端P95不超过 `1500 ms`。
- GPU峰值显存不超过 `36,864 MiB`，至少为40GB GPU保留约4GB余量。
- 100次稳定性阶段结束后，相对预热完成时的显存增长不超过 `500 MiB`。
- 不出现 CUDA OOM、模型重载或服务进程异常退出。

任一阈值失败都保留完整产物并标记实验失败，不自动启用量化。先分析失败
原因；只有确认是显存容量问题时才准备独立8bit实验。

## 执行命令

远程 Linux GPU 服务器执行：

```bash
bash exp/runs/lisa13b-clean030-production-perf-v1/command.sh
```

脚本不依赖用户提前 `export` 环境变量。CLIP优先使用仓库相对路径，不存在
时自动查找当前用户的Hugging Face本地缓存。脚本拒绝覆盖已有非空输出目录。

## 预期产物

```text
exp/runs/lisa13b-clean030-production-perf-v1/outputs/
├── runtime_config.json
├── server.log
├── gpu_metrics.csv
├── requests.csv
├── summary.json
└── summary.md
```

- `runtime_config.json`：Git commit、模型、CLIP、输入、请求数和准入阈值。
- `server.log`：不启用访问日志，只保留启动、模型加载和错误信息。
- `gpu_metrics.csv`：时间、显存、GPU利用率和温度。
- `requests.csv`：逐请求阶段、客户端/服务端延迟、mask数和错误。
- `summary.json`：机器可读性能汇总和准入判定。
- `summary.md`：人工可读性能报告。

输出不保存原始图片、请求Base64或返回mask内容，避免重复提交数据和大文件。

## 结果

等待远程执行后填写。

## 结论

等待远程执行后填写。
