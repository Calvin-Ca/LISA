# LISA 生产推理服务

该目录提供 LISA 施工安全分割模型的生产化基础实现。服务使用自定义 LISA 推理链路，不是普通文本生成接口。

## 已实现

- 模型延迟加载和进程内单例
- 图片 Base64 输入
- Prompt 和图片大小校验
- PNG Base64 mask 输出
- 多 mask 和空 mask 协议
- 有界 GPU 任务队列和固定数量的 GPU worker
- 排队超时、推理超时和队列满保护
- HTTP 推理超时后继续占用原 GPU worker，直到底层同步推理真正结束
- CUDA OOM 专用错误、一次缓存清理和恢复失败后的 unavailable 状态
- JPEG/PNG 文件签名、编码长度和头部像素尺寸预检
- 不记录 API Key、完整 Base64 图片、Prompt 或底层推理异常详情
- 可选 API Key
- `/health`、`/ready`、`/metrics`
- 当前队列长度、GPU 在途任务数及其历史最大值
- request ID 和模型版本响应头
- Dockerfile 和环境变量配置
- 不加载模型的纯逻辑单元测试

## 远程启动

以下命令仅在 Linux GPU 服务器执行。

先冻结模型制品：

```bash
python3 production/freeze_model_artifact.py \
  --source runs/lisa13b-clean030-lora-v1/merged_hf \
  --output-root artifacts/lisa-safety-seg \
  --version lisa13b-clean030-v1
```

该命令会复制模型并生成：

```text
artifacts/lisa-safety-seg/lisa13b-clean030-v1/
├── merged_hf/
├── manifest.json
├── SHA256SUMS
└── MODEL_CARD.md
```

如果只希望先计算校验值和生成元数据，不复制模型：

```bash
python3 production/freeze_model_artifact.py \
  --source runs/lisa13b-clean030-lora-v1/merged_hf \
  --output-root artifacts/lisa-safety-seg \
  --version lisa13b-clean030-v1-manifest \
  --manifest-only
```

`artifacts/` 已被 Git 忽略，模型制品需要通过对象存储或内部制品平台传输。

配置服务：

```bash
cp production/.env.example production/.env
```

按服务器模型挂载位置修改 `production/.env`，然后：

```bash
set -a \
&& source production/.env \
&& set +a \
&& CUDA_VISIBLE_DEVICES=0 python -m uvicorn production.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1
```

生产服务必须保持单 worker。增加进程数会重复加载13B模型并占用多份显存。多GPU部署采用一GPU一容器或一GPU一进程。

当前生产配置使用一个 GPU worker 和最多八个等待任务：

```text
LISA_MAX_CONCURRENCY=1
LISA_MAX_QUEUE_SIZE=8
LISA_QUEUE_TIMEOUT_SECONDS=30
LISA_REQUEST_TIMEOUT_SECONDS=120
```

`LISA_QUEUE_TIMEOUT_SECONDS` 限制任务等待 GPU worker 的时间；尚未开始的
任务超时后会被取消。`LISA_REQUEST_TIMEOUT_SECONDS` 从任务实际开始执行时
计时；如果 HTTP 等待超时，底层同步 GPU 推理会继续完成，但该 worker 在
任务真正结束前不会处理下一个任务。

## OOM 恢复规则

推理阶段发生 CUDA OOM 时，接口返回 HTTP 503 和
`cuda_out_of_memory`，不会自动重试当前请求。运行时会断开可能引用临时
CUDA tensor 的异常链，执行 Python 垃圾回收和一次
`torch.cuda.empty_cache()`：

- 缓存清理成功：服务恢复为 ready，后续新请求可以继续进入队列。
- 缓存清理失败：服务保持 unavailable，`/ready` 返回 503，后续请求被
  拒绝，需要运维重启进程。

`/metrics` 会记录 OOM 次数、恢复成功/失败次数和 unavailable 拒绝次数。
代码不会无限重试 OOM 请求。真实 CUDA OOM 后的服务恢复仍需在可控 GPU
环境单独验收，不能在承载重要共存服务的 GPU 上随意制造 OOM。

## 图片输入边界

输入只接受 JPEG 和 PNG。服务在 OpenCV 完整解码前依次检查：

1. Base64 编码长度是否可能超过配置的解码字节上限。
2. Base64 是否严格合法。
3. 文件签名是否为 JPEG 或 PNG。
4. JPEG SOF 或 PNG IHDR 中的宽高和像素数是否超限。
5. OpenCV 是否能成功解码，并再次检查实际像素数。

GIF、WebP、伪造签名、损坏头部和超过限制的图片均返回
`invalid_request`，不会进入 GPU 队列。反向代理或 ASGI 层的 HTTP 请求体
总大小限制仍需单独配置。

## 健康检查

```bash
curl -s http://127.0.0.1:8000/health
```

```bash
curl -s http://127.0.0.1:8000/ready
```

## 推理请求

先生成请求 JSON：

```bash
python3 -c "import base64,json,pathlib; p=pathlib.Path('example.jpg'); print(json.dumps({'request_id':'smoke-001','prompt':'标出未佩戴安全帽的作业人员。','image_base64':base64.b64encode(p.read_bytes()).decode('ascii')},ensure_ascii=False))" \
> /tmp/lisa-request.json
```

调用接口：

```bash
curl -s \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: smoke-001" \
  --data-binary @/tmp/lisa-request.json \
  http://127.0.0.1:8000/v1/segment
```

如果配置了 `LISA_API_KEY`，增加：

```bash
-H "X-API-Key: ${LISA_API_KEY}"
```

响应中的 `masks[].data` 是 PNG Base64。

## Docker

在仓库根目录远程构建：

```bash
docker build \
  -f production/Dockerfile \
  -t lisa-safety-seg:lisa13b-clean030-v1 \
  .
```

远程运行：

```bash
docker run --rm \
  --gpus '"device=0"' \
  --shm-size 8g \
  --env-file production/.env \
  -p 8000:8000 \
  -v /models/lisa13b-clean030-v1:/models/lisa13b-clean030-v1:ro \
  -v /models/clip-vit-large-patch14:/models/clip-vit-large-patch14:ro \
  lisa-safety-seg:lisa13b-clean030-v1
```

模型目录和 CLIP vision tower 不进入镜像，也不提交 Git。

## 本地纯逻辑测试

本地安装基础 Python 依赖后可执行：

```bash
python3 -m unittest discover \
  -s production/tests \
  -v
```

本地测试不运行模型、不加载权重、不使用GPU。

## 生产限制

- 当前 `/metrics` 返回 JSON 快照，后续接入 Prometheus 时可替换为标准文本格式。
- 当前请求只接受 Base64 图片，不允许服务端访问任意 URL，从而避免 SSRF。
- 请求超时只控制HTTP等待时间，已经提交到GPU的同步推理不会被强制中断；
  专用GPU worker会等它真正结束后才处理下一个任务，避免产生隐性并发。
- 应用不记录请求体、API Key、完整图片或 Prompt；返回客户端的推理错误
  使用固定脱敏文本，不包含底层异常和服务器路径。
- 13B模型没有在本地执行；必须在远程GPU环境进行冒烟、benchmark和压测。
- 正式上线前仍必须完成 `todo.md` 中的 golden test、制品冻结、精度复评、监控、灰度和回滚。
