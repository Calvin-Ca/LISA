# LISA 监控与告警

## 当前提供的能力

- `/metrics`：保留原有 JSON 快照，供现有验收脚本使用。
- `/metrics/prometheus`：Prometheus 文本格式。
- `/alerts`：按当前进程内阈值返回 `ok` 或 `firing`。
- `/v1/segment` 完整 HTTP 请求、2xx/4xx/5xx、在途请求和延迟窗口。
- 请求体校验、Prompt 校验、图片校验、鉴权失败和服务错误计数。
- 队列长度、利用率、等待时间 P50/P95/P99。
- GPU 推理时间 P50/P95/P99、GPU 在途数、超时和 CUDA OOM。
- 空 mask、多 mask 和总 mask 数。

滚动分位数只保留最近 `LISA_METRICS_WINDOW_SIZE` 个进程内样本，默认 1000。
进程重启后计数和窗口会清零。长期趋势必须由外部 Prometheus 保存。

## 默认应用告警阈值

```text
最小HTTP样本数：20
4xx比例：20%
5xx比例：1%
HTTP P95：2000 ms
队列利用率：80%
CUDA OOM：任何一次立即告警
unexpected error：任何一次立即告警
模型 unavailable：立即告警
```

这些阈值可通过 `production/.env.example` 中的环境变量调整。首轮阈值基于
当前 shared-GPU 并发测试结果设置，上线后应结合真实流量重新校准。

## 鉴权

当设置 `LISA_API_KEY` 时，JSON、Prometheus 和 alerts 接口均要求鉴权。
支持原有：

```text
X-API-Key: <token>
```

也支持 Prometheus 使用：

```text
Authorization: Bearer <token>
```

Token 通过只读 secret 文件提供给 Prometheus，不写入
`prometheus.yml`、Git、Dockerfile 或镜像。

## 外部接入步骤

以下步骤在远程 Linux 监控环境执行，不在本地开发机启动：

1. 部署 Prometheus。
2. 将 `prometheus.yml` 和 `lisa-alert-rules.yml` 挂载到 Prometheus。
3. 将 LISA API Key 写入权限为 0400 的 secret 文件，并挂载到
   `/run/secrets/lisa_metrics_api_key`。
4. 将示例目标 `lisa-api:8000` 替换为监控网络中可解析的 LISA 地址。
5. 执行 `promtool check config` 和 `promtool check rules`。
6. 部署 Alertmanager，并配置接收渠道与告警抑制规则。
7. 部署 NVIDIA DCGM Exporter 后启用 GPU scrape target。
8. 在 Grafana 导入 Prometheus 数据源并建立 API、队列、GPU、mask 面板。
9. 人工触发一个测试告警，确认告警产生、通知、恢复和静默流程。

## 仍未完成

- Prometheus、Alertmanager、Grafana 和 DCGM Exporter 尚未在服务器部署。
- 告警接收渠道尚未选择。
- 生产网络地址、TLS、反向代理和 secret 挂载尚未配置。
- 输入图片尺寸与 Prompt 长度分布尚未采集，避免默认产生不必要的业务数据。
- 当前没有持久化审计日志和线上 bad case 自动回流。
