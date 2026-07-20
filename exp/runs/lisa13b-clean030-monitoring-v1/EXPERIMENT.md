# lisa13b-clean030-monitoring-v1

## 状态

实验方案已确认，等待远程 Linux GPU 服务器执行。

## 背景与目的

`lisa13b-clean030-v1` 已完成精度回归、shared-GPU 性能、超时串行化、异常
输入、并发稳定性和最终 Docker 容器验收。生产服务随后新增了：

- `/metrics` JSON 指标快照；
- `/metrics/prometheus` Prometheus 文本指标；
- `/alerts` 进程内阈值告警；
- `/v1/segment` 完整 2xx/4xx/5xx、在途数和滚动延迟；
- 队列等待与 GPU 推理 P50/P95/P99；
- 鉴权、请求、Prompt、图片校验失败计数；
- 空 mask、多 mask、CUDA OOM 和 unexpected error 指标。

本实验验证这些能力在真实生产容器和 GPU 推理下工作，并确认新增监控不会
破坏原有串行化、显存安全、日志脱敏和共享 GPU 服务。外部 Prometheus、
Alertmanager、Grafana、DCGM Exporter 和告警接收渠道本轮不部署。

## 固定配置

- 模型版本：`lisa13b-clean030-v1`
- 冻结制品：`artifacts/lisa-safety-seg/lisa13b-clean030-v1`
- 模型制品校验：执行现有 `SHA256SUMS`
- CLIP：服务器 Hugging Face 缓存中的固定 snapshot，完整模型缓存只读挂载
- 精度：bf16
- 量化：不启用 8bit/4bit
- Dockerfile：`production/Dockerfile`
- 镜像标签：`lisa-safety-seg:lisa13b-clean030-monitoring-v1`
- 容器名：`lisa-clean030-monitoring-v1`
- 宿主机监听：`127.0.0.1:8005`
- 容器服务端口：8000
- Uvicorn worker：1
- GPU worker：1
- 最大队列：8
- 排队超时：30 秒
- 推理超时：120 秒
- 指标滚动窗口：100
- 实验告警最小样本数：2
- 4xx 告警阈值：20%
- 5xx 告警阈值：1%
- HTTP P95 告警阈值：2000 ms
- 队列利用率告警阈值：80%
- API Key：工具运行时随机生成，不写入仓库和结构化输出
- GPU：0
- 共享 GPU 前置进程：必须存在 `VLLM::EngineCore`
- 固定图片：
  `dataset/reason_seg/ReasonSeg/val/val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing.jpg`
- 固定 Prompt：`标出未按规定佩戴安全帽的作业人员。`

最小告警样本数只在本实验降为 2，用于用少量请求验证告警生命周期。正式
默认值仍为 20。

## 数据与请求设计

实验不修改训练集、验证集、Prompt 或 mask，只使用一个已固定的
`ReasonSeg|val` 图片做 API 链路验证：

1. 容器 ready 后先读取 JSON、Prometheus 和 alerts，预期 alerts 为 `ok`。
2. 发送 1 次携带正确 API Key 的有效推理请求，预期 HTTP 200 和有效 PNG
   mask。
3. 发送 1 次错误 API Key 请求，预期 HTTP 401、错误码 `unauthorized`，
   且不得进入模型运行时。
4. 此时 2 个 HTTP 样本中 4xx 比例为 50%，预期
   `http_4xx_rate_high` 进入 `firing`。
5. 再发送 4 次有效请求，最终 6 个 HTTP 样本中 4xx 比例为 1/6，低于
   20%，预期告警恢复为 `ok`。
6. 最终验证 5 次真实 GPU 推理均成功，GPU 最大在途数仍为 1。

监控接口使用 `Authorization: Bearer` 验证 Prometheus 鉴权路径；JSON
指标继续使用 `X-API-Key`，从而同时覆盖两种受支持的鉴权方式。

## 执行流程

1. 检查 Docker、NVIDIA Runtime、端口、固定图片、模型和 CLIP 文件。
2. 校验冻结模型全部文件 SHA-256。
3. 确认共享 GPU 上原有 `VLLM::EngineCore` 存在并记录 PID。
4. 使用当前 Git commit 构建生产镜像，复用 Docker 缓存。
5. 在容器内运行当前不少于 59 项纯逻辑测试。
6. 以非特权单 worker 容器启动 LISA，模型和 CLIP 只读挂载。
7. 采集初始 JSON、Prometheus 和 alerts。
8. 执行 1 个正常请求和 1 个鉴权失败请求，确认 4xx 告警触发。
9. 再执行 4 个正常请求，确认 4xx 告警恢复。
10. 比较 JSON 与 Prometheus 中的关键指标值。
11. 检查延迟、队列、GPU 推理窗口样本和 P50/P95/P99。
12. 检查 JSON/Prometheus/alerts 监控接口输出和容器日志不存在 API Key、
    图片 Base64、Prompt 或宿主机私有路径。
13. 正常停止并删除测试容器，保留镜像和输出。
14. 检查共享 GPU 原有进程仍存在，以及停止后的显存漂移。

工具在失败路径也会尽力保存容器日志、停止并删除测试容器，不停止共享的
`bge-m3` 服务，不删除模型、数据或已有镜像。

## 准入条件

全部条件必须通过：

- Docker 镜像构建成功，容器内纯逻辑测试不少于 59 项且全部通过。
- 镜像配置用户为 `lisa`，运行 UID 为 10001、GID 非 root，模型和 CLIP
  挂载均为只读。
- 初始 alerts 为 `ok`。
- 5 个有效请求均返回 HTTP 200、正确模型版本和至少一个有效 PNG mask。
- 错误 API Key 请求返回 HTTP 401 / `unauthorized`，响应不泄露密钥。
- 一次鉴权失败后 `http_4xx_rate_high` 告警进入 `firing`。
- 追加 4 个有效请求后 alerts 恢复为 `ok`。
- 最终 HTTP 请求数为 6：2xx=5、4xx=1、5xx=0。
- 鉴权失败数为 1，进入运行时、开始、成功和 GPU 成功数均为 5。
- HTTP、队列等待和 GPU 推理窗口样本分别为 6、5、5。
- HTTP、队列等待和 GPU 推理 P50/P95/P99 均存在且非负。
- JSON 和 Prometheus 中的 ready、请求、错误、GPU 串行化和关键 P95 指标
  一致。
- GPU 推理失败、排队超时、推理超时、队列拒绝、取消、CUDA OOM 和
  unexpected error 均为 0。
- `gpu_inference_in_flight_max=1`，最终在途数为 0。
- GPU 总峰值不超过 36,864 MiB，峰值剩余不少于 4,096 MiB。
- 停止后显存相对基线漂移绝对值不超过 500 MiB。
- 原有共享 GPU 计算进程 PID 没有消失。
- JSON/Prometheus/alerts 监控接口输出和容器日志中不存在 API Key、错误
  API Key、完整图片 Base64、Prompt 或宿主机私有路径。
- 容器保持 ready，正常停止且退出码为 0。

## 远程执行命令

先拉取本实验提交，再在远程 Linux GPU 服务器仓库根目录执行：

```bash
bash exp/runs/lisa13b-clean030-monitoring-v1/command.sh
```

脚本自包含所有路径、端口、模型、输入、告警阈值和验收参数，不依赖用户提前
`export` 环境变量，也不读取或修改正式 `production/.env`。

## 预期产物

```text
exp/runs/lisa13b-clean030-monitoring-v1/outputs/
├── runtime_config.json
├── build.log
├── unit_tests.log
├── gpu_metrics.csv
├── server.log
├── container_inspect.json
├── metrics-initial.json
├── metrics-final.json
├── prometheus-initial.txt
├── prometheus-final.txt
├── alerts-initial.json
├── alerts-firing.json
├── alerts-recovered.json
├── request-results.json
├── smoke-cycle-1-response.json
├── smoke-cycle-1-mask-0.png
├── ...
├── smoke-cycle-5-response.json
├── smoke-cycle-5-mask-0.png
├── summary.json
└── summary.md
```

结构化输出不会保存 API Key、错误 API Key 或完整输入图片 Base64。

## 结果

等待远程执行后，根据 `outputs/summary.json`、`summary.md` 和日志补充。

## 局限

- 本实验验证应用内指标和告警生命周期，不等于外部 Prometheus 持久化、
  Alertmanager 通知、Grafana 展示或 DCGM GPU 指标已经部署。
- 只使用一个固定图片，不替代 86 样本精度回归和正式并发压测。
- 不主动制造 CUDA OOM、模型加载失败或共享服务退出，避免影响服务器上的
  常驻业务。
- 告警阈值使用小样本验证逻辑，正式阈值仍需根据真实线上流量校准。
