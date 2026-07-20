# lisa13b-clean030-api-robustness-v1

## 状态

已完成，真实共享 GPU API robustness 验证全部通过。

## 背景

生产 API 已增加有界 GPU 队列、排队超时、队列满保护、JPEG/PNG 文件签名
白名单、图片头部尺寸预检、推理错误脱敏以及 CUDA OOM 恢复状态机。

本实验验证这些保护在真实 FastAPI、OpenCV、LISA 和共享 A100 环境中的行为：

- 非法鉴权、Prompt 和图片必须在进入 GPU 前被拒绝。
- 真实 JPEG 和 PNG 必须仍能正常推理并返回 mask。
- 单 GPU worker 下必须稳定产生一个排队超时和一个队列满拒绝。
- 所有异常完成后服务必须继续 ready。
- 哨兵凭据、Prompt、图片字符串和私有路径不得出现在响应或服务日志中。

目标 GPU 长期运行不可停止的 `bge-m3` vLLM pooling 服务。本实验保持其在线，
并验证实验前已有的 `VLLM::EngineCore` PID 没有消失。

## 模型、数据与配置

- LISA 版本：`lisa13b-clean030-v1`
- 冻结制品：`artifacts/lisa-safety-seg/lisa13b-clean030-v1/merged_hf`
- 精度：bf16，未启用 8bit/4bit
- GPU：索引 0，共享 GPU
- LISA Uvicorn worker：1
- LISA GPU worker：1
- 最大等待队列：1
- 排队超时：0.15 秒
- 推理超时：120 秒
- 最大图片字节：20 MiB
- 最大图片像素：25,000,000
- 最大 Prompt：1,000 字符
- 服务地址：`127.0.0.1:8002`
- 共存服务进程名要求：`VLLM::EngineCore`
- 固定图片：
  `dataset/reason_seg/ReasonSeg/val/val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing.jpg`
- 固定 Prompt：`标出未按规定佩戴安全帽的作业人员。`

队列大小 1 和 0.15 秒排队超时只用于稳定构造验收场景。实验通过独立子进程
环境注入配置，不修改正式 `production/.env`。

## 用例

### 不进入 GPU 的非法输入

1. 错误 API Key：HTTP 401，`unauthorized`。
2. 空 Prompt：HTTP 422，`validation_error`。
3. 纯空格 Prompt：HTTP 400，`invalid_request`。
4. 超长 Prompt：HTTP 400，`invalid_request`。
5. 非法 Base64：HTTP 400，`invalid_request`。
6. GIF：HTTP 400，`invalid_request`。
7. WebP：HTTP 400，`invalid_request`。
8. 损坏 PNG：HTTP 400，`invalid_request`。
9. 损坏 JPEG：HTTP 400，`invalid_request`。
10. 头部声明 10000×10000 的 PNG：HTTP 400，`invalid_request`。

执行完上述用例后，`requests_received_total` 必须仍为 0，证明它们没有进入
运行时 GPU 队列。

### 正常格式

11. 固定原始 JPEG：HTTP 200，至少返回一个 mask。
12. 将同一图片无损编码为 PNG：HTTP 200，至少返回一个 mask。

### 队列保护

13. 启动一个正常主请求并等待 `gpu_inference_in_flight=1`。
14. 在主请求运行期间提交第二个请求，使其进入唯一等待位并在 0.15 秒后返回
    HTTP 504 和 `inference_queue_timeout`。
15. 确认等待位占用后提交第三个请求，立即返回 HTTP 503 和
    `inference_queue_full`。

主请求必须成功；排队超时任务不得事后进入 GPU。

## 敏感信息检查

实验使用五类合成哨兵：

- 正确 API Key
- 错误 API Key
- 超长 Prompt 标识
- 非法图片字符串
- 合成私有路径

只在结果中保存是否泄漏及哨兵类别，不把真实凭据、真实私有图片或服务器私有
路径写入仓库。响应正文和 `server.log` 均不得出现哨兵值。

## 准入条件

20 项检查必须全部通过：

- 15 个 API 用例全部符合预期。
- 10 个非法用例后运行时请求数仍为 0。
- 最终运行时收到 5 个请求，其中只有 3 个进入 GPU。
- 3 个正常 GPU 请求全部成功并返回 mask。
- 排队超时、队列拒绝、取消等待任务均各 1 次。
- GPU 推理成功 3 次、失败 0 次。
- `gpu_inference_in_flight_max=1`，最终在途数为 0。
- CUDA OOM 和 unexpected error 均为 0。
- 异常处理后 `/ready` 仍为 ready。
- 响应和服务日志中均没有敏感哨兵。
- 实验前已有 GPU 计算进程 PID 没有消失。
- 服务日志没有 CUDA OOM。

本实验不主动制造 CUDA OOM，避免影响共享 GPU 上的 `bge-m3`。

## 执行命令

远程 Linux GPU 服务器执行：

```bash
bash exp/runs/lisa13b-clean030-api-robustness-v1/command.sh
```

脚本自包含模型、CLIP、固定输入、鉴权哨兵、端口、队列参数和准入条件，不依赖
用户提前导出环境变量，不修改正式 `.env`，也不停止 `bge-m3`。

## 预期产物

```text
exp/runs/lisa13b-clean030-api-robustness-v1/outputs/
├── runtime_config.json
├── server.log
├── cases.json
├── metrics_after_invalid.json
├── metrics_final.json
├── summary.json
└── summary.md
```

## 结果

执行时间：2026-07-20。

临时 LISA 服务成功加载，ready 时间为 `14,286.349 ms`。15 个用例全部通过：

| Case | HTTP | Code | Masks | Latency ms | 结果 |
| --- | ---: | --- | ---: | ---: | --- |
| `auth-invalid` | 401 | `unauthorized` | 0 | 1.911 | PASS |
| `prompt-empty` | 422 | `validation_error` | 0 | 4.211 | PASS |
| `prompt-blank` | 400 | `invalid_request` | 0 | 1.030 | PASS |
| `prompt-too-long` | 400 | `invalid_request` | 0 | 0.840 | PASS |
| `image-invalid-base64` | 400 | `invalid_request` | 0 | 0.970 | PASS |
| `image-gif` | 400 | `invalid_request` | 0 | 0.804 | PASS |
| `image-webp` | 400 | `invalid_request` | 0 | 0.776 | PASS |
| `image-corrupt-png` | 400 | `invalid_request` | 0 | 0.785 | PASS |
| `image-corrupt-jpeg` | 400 | `invalid_request` | 0 | 0.791 | PASS |
| `image-oversized-header` | 400 | `invalid_request` | 0 | 0.773 | PASS |
| `valid-jpeg` | 200 | - | 1 | 662.416 | PASS |
| `valid-png` | 200 | - | 1 | 381.536 | PASS |
| `queue-primary` | 200 | - | 1 | 389.602 | PASS |
| `queue-timeout` | 504 | `inference_queue_timeout` | 0 | 155.171 | PASS |
| `queue-full` | 503 | `inference_queue_full` | 0 | 3.146 | PASS |

10 个非法请求中最慢的空 Prompt 验证为 `4.211 ms`，全部完成后
`requests_received_total=0`，证明鉴权、Prompt 和图片问题均在进入运行时 GPU
队列前被拒绝。

真实 JPEG、PNG 和队列主请求均返回一个 mask。JPEG 的 `662.416 ms` 包含
本轮首次实际推理开销；后续 PNG 和主请求分别为 `381.536 ms` 和
`389.602 ms`，与此前预热后的约 0.4 秒基线一致。

队列保护符合配置：等待任务在 `155.171 ms` 返回，与 0.15 秒排队超时接近；
队列满请求在 `3.146 ms` 内拒绝，没有进入 GPU。

最终运行时指标：

| 指标 | 实际值 | 准入值 |
| --- | ---: | ---: |
| Requests received | 5 | 5 |
| GPU requests started | 3 | 3 |
| Requests succeeded | 3 | 3 |
| Queue timeout | 1 | 1 |
| Queue rejected | 1 | 1 |
| Queue cancelled | 1 | 1 |
| GPU inference succeeded | 3 | 3 |
| GPU inference failed | 0 | 0 |
| Maximum GPU in flight | 1 | 1 |
| Final GPU in flight | 0 | 0 |
| Masks returned | 3 | >= 3 |
| CUDA OOM | 0 | 0 |
| Unexpected errors | 0 | 0 |

全部 20 项准入检查通过：

- 异常完成后服务仍为 ready。
- 响应和服务日志均未发现敏感哨兵。
- 实验前已有共享 GPU 计算进程没有消失。
- 服务日志没有 CUDA OOM。

最终结果：`PASS`。

## 结论

JPEG/PNG 白名单、头部尺寸预检、Prompt 校验、API Key、错误脱敏、有界队列、
排队超时和队列满保护均已在真实 FastAPI、OpenCV、LISA 和共享 A100 环境
通过验证。

无效输入能够在毫秒级拒绝且不进入 GPU，正常 JPEG/PNG 没有功能回归。单
GPU worker 在队列压力下保持最大在途任务数 1，排队超时任务被取消且不会
事后执行，队列满请求快速失败。异常矩阵执行后服务仍然 ready。

实验使用独立子进程注入队列大小 1 和 0.15 秒排队超时，没有修改正式
`production/.env`。正式配置仍为等待队列 8、排队超时 30 秒和推理超时
120 秒。

## 局限

- 使用一个固定图像，不代表全部图片尺寸、编码器和业务 Prompt 分布。
- 未覆盖控制字符等异常 Prompt 字符集。
- 未覆盖空 mask、多 mask 和极小目标。
- 未主动制造 CUDA OOM，OOM 恢复目前只有纯逻辑测试。
- 队列实验验证的是单 GPU worker 的保护行为，不替代客户端并发 1、2、4 的吞吐和长时间稳定性压测。
- 只确认 `bge-m3` GPU 进程存活，没有验证其业务延迟和吞吐。
