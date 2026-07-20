# lisa13b-clean030-timeout-guard-v1

## 状态

已完成，真实共享 GPU 超时串行化回归通过。

## 背景

旧版服务使用 `asyncio.wait_for(asyncio.to_thread(...))` 和 semaphore。
HTTP 等待超时后，底层同步 GPU 推理仍会继续，但 semaphore 上下文可能提前
退出，使下一请求进入 GPU，形成配置之外的隐性并发。

当前实现已改为有界等待队列和固定数量的 GPU worker。本实验验证：HTTP
请求可以提前返回超时，但单 worker 必须继续占有 GPU 执行权，直到后台推理
真正结束，之后才允许下一任务开始。

目标 GPU 上长期运行不可停止的 `bge-m3` vLLM pooling 服务。本实验保持该
服务在线，只启动一个临时 LISA 服务，并检查实验前已有的
`VLLM::EngineCore` PID 在实验结束前仍然存在。

## 模型、数据与环境

- LISA 版本：`lisa13b-clean030-v1`
- 冻结制品：`artifacts/lisa-safety-seg/lisa13b-clean030-v1/merged_hf`
- 精度：bf16，未启用 8bit/4bit
- GPU：索引 0，共享 GPU
- LISA Uvicorn worker：1
- LISA GPU worker：1
- 最大等待队列：8
- 临时推理超时：0.1 秒
- 排队超时：5 秒
- 服务地址：`127.0.0.1:8002`
- 共存服务进程名要求：`VLLM::EngineCore`
- 固定图片：
  `dataset/reason_seg/ReasonSeg/val/val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing.jpg`
- 固定 Prompt：`标出未按规定佩戴安全帽的作业人员。`

0.1 秒仅用于稳定制造 HTTP 超时，不是正式生产超时配置。实验脚本通过独立
子进程环境注入该值，不修改 `production/.env`；正式配置仍保持 120 秒。

## 实验步骤

1. 校验冻结模型、CLIP 配置、固定图片和输出目录。
2. 确认 GPU 上存在 `VLLM::EngineCore`，记录其 PID 和显存。
3. 确认 `127.0.0.1:8002` 可用。
4. 使用 bf16、单 worker、单 GPU worker 启动临时 LISA 服务。
5. 等待 `/ready` 通过。
6. 使用线程屏障同时发送两个相同图片、不同 request ID 的请求。
7. 验证两个请求均返回 HTTP 504 和 `inference_timeout`。
8. HTTP 响应结束后继续轮询 `/metrics`，等待两个底层 GPU 任务完成。
9. 验证历史最大 GPU 在途任务数始终为 1，最终在途数回到 0。
10. 验证两个底层推理均成功、没有 CUDA OOM，且原有共享 GPU 进程未消失。
11. 保存请求、指标、服务日志和验收报告，停止临时 LISA 服务。

## 准入条件

必须同时满足：

- 两个请求均返回 HTTP 504。
- 两个错误码均为 `inference_timeout`。
- `requests_received_total=2`。
- `requests_started_total=2`。
- `requests_timeout_total=2`。
- `gpu_inference_succeeded_total=2`。
- `gpu_inference_failed_total=0`。
- `gpu_inference_in_flight_max=1`。
- 最终 `gpu_inference_in_flight=0`。
- 累计排队时间大于 0。
- 两个 HTTP 请求超时后，两个后台 GPU 任务均最终完成。
- 实验前已有的 GPU 计算进程 PID 没有消失。
- 服务日志中没有 CUDA OOM。

任一条件不满足，实验判定为失败，并保留输出用于分析。

## 执行命令

远程 Linux GPU 服务器执行：

```bash
bash exp/runs/lisa13b-clean030-timeout-guard-v1/command.sh
```

脚本自包含模型、CLIP、输入、Prompt、端口、超时和准入条件，不依赖用户
提前导出环境变量，不修改正式 `production/.env`，也不停止 `bge-m3`。

## 预期产物

```text
exp/runs/lisa13b-clean030-timeout-guard-v1/outputs/
├── runtime_config.json
├── server.log
├── requests.json
├── metrics.json
├── summary.json
└── summary.md
```

## 结果

执行时间：2026-07-20。

临时 LISA 服务成功加载，ready 时间为 `14,288.142 ms`。两个请求通过线程
屏障并发释放，均按预期返回 HTTP 504 和 `inference_timeout`：

| Request | HTTP | Error code | Client latency ms | 结果 |
| --- | ---: | --- | ---: | --- |
| `timeout-guard-1` | 504 | `inference_timeout` | 729.018 | PASS |
| `timeout-guard-2` | 504 | `inference_timeout` | 118.723 | PASS |

线程调度顺序不由 request ID 决定。两个请求的客户端延迟相差约
`610.295 ms`，与累计队列等待 `0.613056 s` 接近，说明其中一个请求先进入
GPU，另一个在单 GPU worker 后排队；排队请求只有在前一个底层推理真正完成
后才开始自己的 0.1 秒 HTTP 推理等待。

运行时指标：

| 指标 | 实际值 | 准入值 |
| --- | ---: | ---: |
| Requests received | 2 | 2 |
| Requests started | 2 | 2 |
| Requests timed out | 2 | 2 |
| GPU inference succeeded | 2 | 2 |
| GPU inference failed | 0 | 0 |
| Maximum GPU in flight | 1 | 1 |
| Final GPU in flight | 0 | 0 |
| Total queue wait | 0.613056 s | > 0 |

全部 12 项准入检查通过：

- 两个 HTTP 响应均为预期超时。
- 两个 HTTP 请求结束后，两个后台 GPU 推理均成功完成。
- 历史最大 GPU 在途任务数为 1，最终回到 0。
- 实验前已有的共享 GPU 计算进程没有消失。
- 服务日志没有 CUDA OOM。

最终结果：`PASS`。

## 结论

本次真实 A100 共享 GPU 验证确认了修复目标：HTTP 超时不会强制取消已经
提交的同步 CUDA 推理，但也不会提前释放 GPU worker。前一个后台推理真正
结束前，后一个请求只能排队，因此生产配置
`LISA_MAX_CONCURRENCY=1` 能够保证实际 GPU 在途任务上限为 1，不再出现
旧版 semaphore 提前释放造成的隐性并发。

实验使用独立子进程注入 0.1 秒超时，没有修改正式
`production/.env`；正式推理超时仍为 120 秒。临时 LISA 服务在结果保存后
自动停止，`bge-m3` 保持在线。

## 局限

- 本实验验证两个并发请求和一个固定输入，不替代并发 1、2、4 及长时间稳定性压测。
- 验证目标是防止超时后的 GPU 任务重叠，不代表同步 CUDA 推理已经支持强制中断。
- 只确认 `bge-m3` GPU 进程存活，没有验证其业务接口延迟和吞吐是否受影响。
- 尚未覆盖队列满、排队超时、CUDA OOM 恢复和异常输入的真实 GPU/API 场景。
