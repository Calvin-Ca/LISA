# LISA HTTP Request Body Limit Verification

## 状态

初次远程执行发现 chunked 超限异常被 FastAPI 转换为 HTTP 400；代码已修复，
等待使用独立输出目录重新执行。

## 背景与目的

生产 API 已在 ASGI 层增加完整 HTTP 请求体上限。该限制需要覆盖两种入口：

1. 客户端声明的 `Content-Length` 已超过上限时，在读取正文前立即拒绝。
2. 请求没有可用的长度声明、采用 chunked 传输时，按实际累计接收字节拒绝。

本实验使用真实 Uvicorn、真实 Docker 容器和原始 TCP HTTP 请求验证两条路径，
并确认超限请求不会进入 JSON、Base64、模型运行时或 GPU。实验不加载 LISA、
CLIP 或 SAM 权重，不给容器挂载 GPU，因此不会占用额外模型显存，也不会停止
服务器上已有的 `bge-m3`。

## 固定配置

- 实验目录：`exp/runs/lisa13b-request-limit-v1`
- Dockerfile：`production/Dockerfile`
- 修复后镜像：`lisa-safety-seg:lisa13b-request-limit-v1-fix1`
- 修复后容器：`lisa-request-limit-v1-fix1`
- 服务地址：`127.0.0.1:8006`
- 模型版本标识：`lisa13b-request-limit-v1`
- `LISA_EAGER_LOAD=false`
- 容器 GPU：不挂载
- 实验请求体上限：1,024 bytes
- 生产默认请求体上限：31,457,280 bytes（30 MiB），本实验不修改该默认值
- 容器纯逻辑测试门槛：不少于 65 项
- GPU 显存漂移上限：绝对值不超过 500 MiB
- 必须保持存活的共享进程：`VLLM::EngineCore`

使用 1 KiB 仅为了用很小的测试请求可靠触发边界，验证的是同一套中间件逻辑。
正式环境仍由 `LISA_MAX_REQUEST_BYTES=31457280` 使用 30 MiB 上限。

## HTTP 用例

1. `request-limit-content-length`
   - 声明 `Content-Length=1025`，不发送正文。
   - 预期 HTTP 413 / `request_too_large`。
   - 用于证明服务器不会等待或读取超限正文。
2. `request-limit-chunked`
   - 不发送 `Content-Length`，通过 `Transfer-Encoding: chunked` 发送超过
     1,024 bytes 的正文。
   - 预期 HTTP 413 / `request_too_large`。
   - 用于证明实际接收字节会累计计数，不能用 chunked 绕过限制。
3. `request-limit-small-validation`
   - 发送限制内的 `{}`。
   - 预期 HTTP 422 / `validation_error`。
   - 用于证明正常小请求仍进入 FastAPI 校验，限制没有误伤合法大小的请求。

每个响应都检查 request ID、模型版本响应头和错误码，并检查随机 API Key 与
超限正文哨兵没有进入响应、日志或结构化输出。

## 预期指标

请求结束后必须精确满足：

- `http_requests_total=3`
- `http_responses_4xx_total=3`
- `http_responses_5xx_total=0`
- `request_body_too_large_total=2`
- `request_validation_failed_total=1`
- `http_request_latency_ms_window_samples=3`
- `requests_received_total=0`
- `requests_started_total=0`
- `requests_succeeded_total=0`
- `gpu_inference_succeeded_total=0`
- `gpu_inference_failed_total=0`
- `gpu_inference_in_flight_max=0`
- `model_loads_total=0`
- `cuda_oom_total=0`
- `unexpected_errors_total=0`

## 准入条件

- Docker 镜像构建成功。
- 镜像内不少于 65 项纯逻辑测试全部通过。
- 三个 HTTP 用例的状态码、错误码、request ID 和响应头全部正确。
- 上述指标精确匹配，三个请求均未进入模型运行时或 GPU。
- 镜像配置用户为 `lisa`，运行 UID 为 10001、GID 非 root。
- 容器结束前 `/health=ok`，停止退出码为 0。
- 实验前已有 `VLLM::EngineCore` PID 在实验后仍存在。
- 实验前后整卡显存漂移绝对值不超过 500 MiB。
- 日志没有 CUDA OOM、Traceback、ERROR、API Key、正文哨兵或宿主机私有路径。

## 执行命令

远程 Linux 服务器执行：

```bash
bash exp/runs/lisa13b-request-limit-v1/command.sh
```

脚本自包含镜像、容器、端口、阈值和准入参数，不依赖用户预先 `export`
环境变量，不读取或修改 `production/.env`，不加载模型，不停止已有服务。

## 初次执行记录与修复

- 初次执行时间：2026-07-20
- 容器测试：65/65 通过
- `Content-Length` 超限：HTTP 413，符合预期
- chunked 累计超限：HTTP 400，不符合预期
- 小请求校验：HTTP 422，符合预期
- 失败指标：`request_body_too_large_total=1`，预期为 2
- GPU 显存漂移：0 MiB
- 日志、权限、健康检查和共享进程：全部通过

chunked 报文是合法 HTTP 请求。累计正文超过上限时，接收包装器抛出的内部
异常发生在 FastAPI 请求体解析阶段，被 FastAPI 转换成了 48 字节的通用
HTTP 400 响应 `There was an error parsing the body`，因此原中间件没有机会
输出约定的 413。修复后，中间件会标记已经确认的超限状态，抑制该内部 400，
再统一返回 HTTP 413 / `request_too_large` 并增加超限指标。

修复后的运行使用新镜像标签、容器名和 `outputs-after-fix/`，保留初次失败的
`outputs/` 作为回归证据。

## 预期产物

```text
exp/runs/lisa13b-request-limit-v1/outputs-after-fix/
├── runtime_config.json
├── build.log
├── unit_tests.log
├── container_inspect.json
├── request-results.json
├── metrics.json
├── server.log
├── summary.json
└── summary.md
```

服务器执行后，将 `outputs-after-fix/summary.md` 返回用于补充本页的实测结果
和结论。
