# lisa13b-clean030-api-concurrency-v1

## 状态

已准备，等待远程共享 GPU 执行。

## 背景

生产 API 已使用有界等待队列和固定 GPU worker，并通过了两类真实 GPU 验证：

- HTTP 推理超时后，同步 GPU 任务继续由原 worker 串行完成，历史最大 GPU
  在途数保持为 1。
- 队列满和排队超时能够正确拒绝或取消任务，不会产生事后隐性 GPU 推理。

现有性能基线只覆盖客户端并发 1。本实验在不增加 GPU worker 的前提下，将
客户端并发提高到 2 和 4，验证正式队列配置下的端到端延迟、吞吐、排队时间、
GPU 推理时间、显存和长请求序列稳定性。

这里的“并发 2/4”是同时到达 API 的 HTTP 请求数，不是同时执行 2/4 个 GPU
推理。生产服务始终只有 1 个 GPU worker，其他请求进入最多 8 个任务的等待
队列。

目标 A100 40GB 上长期运行不可停止的 `bge-m3` vLLM pooling 服务。本实验
保持其在线，要求实验前已有的 `VLLM::EngineCore` PID 在实验结束时仍存在。

## 模型、数据与配置

- LISA 版本：`lisa13b-clean030-v1`
- 冻结制品：`artifacts/lisa-safety-seg/lisa13b-clean030-v1/merged_hf`
- 精度：bf16，未启用 8bit/4bit
- GPU：索引 0，共享 A100 40GB
- LISA Uvicorn worker：1
- LISA GPU worker：1
- 最大等待队列：8
- 排队超时：30 秒
- 服务端推理超时：120 秒
- 客户端超时：150 秒
- 服务地址：`127.0.0.1:8003`
- 共存服务进程名要求：`VLLM::EngineCore`
- 固定图片：
  `dataset/reason_seg/ReasonSeg/val/val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing.jpg`
- 固定 Prompt：`标出未按规定佩戴安全帽的作业人员。`

实验通过独立子进程注入配置，不修改正式 `production/.env`，不停止或修改
`bge-m3`。

## 实验阶段

同一个 LISA 进程依次执行：

| 阶段 | 客户端并发 | 请求数 | 用途 |
| --- | ---: | ---: | --- |
| `warmup-c1` | 1 | 5 | 排除首次实际推理开销 |
| `measured-c1` | 1 | 30 | 串行对照 |
| `measured-c2` | 2 | 30 | 两个同时到达请求 |
| `measured-c4` | 4 | 30 | 四个同时到达请求 |
| `stability-c4` | 4 | 100 | 高并发连续稳定性 |

总请求数为 195。每个响应都校验 request ID、原图尺寸、mask 数量和 mask 的
PNG Base64，并要求至少返回一个 mask。

每个阶段前后分别读取 `/metrics`，用指标增量计算：

- 进入运行时和 GPU 的请求数。
- 平均排队时间。
- 平均 GPU 推理时间。
- 排队超时、队列满拒绝和取消任务数。

同时以 200 ms 间隔采集整张 GPU 的显存、利用率和温度。由于 GPU 与
`bge-m3` 共享，GPU 指标代表整卡，不将其误记为 LISA 独占数据。

## 准入条件

全部条件必须通过：

- 195 个请求全部 HTTP 成功并返回至少一个有效 PNG mask。
- `measured-c1` 客户端 P95 不超过 1,000 ms。
- `measured-c2` 客户端 P95 不超过 2,000 ms。
- `measured-c4` 和 `stability-c4` 客户端 P95 不超过 4,000 ms。
- 四个正式/稳定性阶段吞吐均不低于 2.0 req/s。
- 运行时收到、启动、成功以及 GPU 成功数均为 195。
- `gpu_inference_in_flight_max=1`，结束时在途 GPU 任务为 0。
- 排队超时、队列满拒绝、取消、推理超时、GPU 失败、unexpected error 和
  CUDA OOM 均为 0。
- GPU 总峰值不超过 36,864 MiB，峰值时剩余显存不少于 4,096 MiB。
- 稳定性请求结束后，相对预热后的显存增长不超过 500 MiB。
- 最终 `/ready` 为 ready。
- 实验前已有共享 GPU 计算进程 PID 没有消失。
- 服务日志中没有 CUDA OOM。

P95 阈值随排队深度放宽，但吞吐阈值不随客户端并发提高。单 GPU worker 不会
因为客户端并发增加而提高底层模型并行度；本实验主要验证排队行为和服务容量
边界。

## 执行命令

远程 Linux GPU 服务器执行：

```bash
bash exp/runs/lisa13b-clean030-api-concurrency-v1/command.sh
```

脚本自包含模型、CLIP、固定输入、端口、请求数、正式队列参数和准入阈值，
不依赖用户提前 `export` 环境变量。非空输出目录不会被覆盖。

## 预期产物

```text
exp/runs/lisa13b-clean030-api-concurrency-v1/outputs/
├── runtime_config.json
├── server.log
├── gpu_metrics.csv
├── requests.csv
├── requests.json
├── metrics_snapshots.json
├── summary.json
└── summary.md
```

结果文件不保存请求图片 Base64。外部 CLIP 绝对路径会转换为 Hugging Face
模型及 snapshot 标识，避免把服务器私有路径写入结构化报告。

## 结果

等待远程执行。

## 局限

- 固定单图片和单 Prompt 适合验证排队、延迟及显存稳定性，不代表全部线上
  图片尺寸、Prompt 长度和类别分布。
- 只确认 `bge-m3` GPU 进程存活，未接入它的业务健康、P95 和吞吐指标。
- 不主动制造 CUDA OOM，避免影响共享 GPU 上的常驻服务。
- 本实验不提高 GPU worker 数，因此不能用于证明多个 LISA GPU 推理可以安全
  并行。
