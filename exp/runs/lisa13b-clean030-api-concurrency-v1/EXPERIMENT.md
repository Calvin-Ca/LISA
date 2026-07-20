# lisa13b-clean030-api-concurrency-v1

## 状态

已完成，真实共享 GPU 并发与稳定性验证全部通过。

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

执行时间：2026-07-20。

195 个请求全部成功，每个请求均返回一个有效 mask。四个计时阶段结果：

| 阶段 | 客户端并发 | 请求数 | P95 ms | 吞吐 req/s | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| `measured-c1` | 1 | 30 | 383.547 | 2.622934 | PASS |
| `measured-c2` | 2 | 30 | 775.983 | 2.608311 | PASS |
| `measured-c4` | 4 | 30 | 1,557.151 | 2.593920 | PASS |
| `stability-c4` | 4 | 100 | 1,567.105 | 2.568395 | PASS |

P95 从并发 1 的 383.547 ms 增加到并发 2 的 775.983 ms、并发 4 的
1,557.151 ms，基本随等待中的串行请求数线性增长。并发 4 连续 100 请求的
P95 为 1,567.105 ms，与 30 请求阶段只相差 9.954 ms，没有随请求序列增长
而持续恶化。

吞吐在四个阶段保持为 2.568395～2.622934 req/s。客户端并发从 1 增加到 4
没有显著提高吞吐，且并发 4 稳定性阶段比并发 1 低约 2.1%。这符合单 GPU
worker 串行推理的设计：并发请求通过队列吸收，增加的是等待时间，而不是底层
GPU 并行度。

运行时计数全部一致：

| 指标 | 实际值 | 准入值 |
| --- | ---: | ---: |
| Requests received | 195 | 195 |
| GPU requests started | 195 | 195 |
| Requests succeeded | 195 | 195 |
| GPU inference succeeded | 195 | 195 |
| Masks returned | 195 | >= 195 |
| Maximum GPU in flight | 1 | 1 |
| Final GPU in flight | 0 | 0 |
| Queue timeout | 0 | 0 |
| Queue rejected | 0 | 0 |
| Queue cancelled | 0 | 0 |
| Request timeout | 0 | 0 |
| GPU inference failed | 0 | 0 |
| Unexpected errors | 0 | 0 |
| CUDA OOM | 0 | 0 |

共享 GPU 总峰值为 31,770 MiB，峰值剩余 9,190 MiB，预热后显存漂移为
0 MiB。相较此前三轮串行基线的 31,740 MiB 峰值只增加 30 MiB，未出现由
客户端并发引起的模型副本或持续显存增长。

最终服务状态为 ready，实验前已有的共享 GPU 计算进程没有消失，服务日志中
没有 CUDA OOM。29 项准入检查全部通过，最终结果为 `PASS`。

## 结论

当前 `bf16 + 单 Uvicorn worker + 单 GPU worker + 等待队列 8` 配置通过
客户端并发 1、2、4 和并发 4 连续 100 请求验证，可作为当前共享 A100
部署的容量基线。

并发 4 下 P95 约 1.57 秒，仍低于 4 秒准入线；195 个请求零失败、零超时、
零拒绝、零 OOM，GPU 历史最大在途数始终为 1。这同时验证了超时并发修复后
没有新的隐性 GPU 并发。

当前瓶颈是单 GPU worker 的串行服务能力，稳定吞吐约 2.6 req/s。提高客户端
并发只能提高排队占用，不能提高吞吐。若未来业务要求显著高于 2.6 req/s，
优先采用一 GPU 一进程的多 GPU 横向扩展和网关负载均衡；不应直接在同一张
共享 A100 上增加 GPU worker 数量。

## 局限

- 固定单图片和单 Prompt 适合验证排队、延迟及显存稳定性，不代表全部线上
  图片尺寸、Prompt 长度和类别分布。
- 只确认 `bge-m3` GPU 进程存活，未接入它的业务健康、P95 和吞吐指标。
- 不主动制造 CUDA OOM，避免影响共享 GPU 上的常驻服务。
- 本实验不提高 GPU worker 数，因此不能用于证明多个 LISA GPU 推理可以安全
  并行。
