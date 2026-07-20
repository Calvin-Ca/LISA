# LISA 生产推理服务

该目录提供 LISA 施工安全分割模型的生产化基础实现。服务使用自定义 LISA 推理链路，不是普通文本生成接口。

## 已实现

- 模型延迟加载和进程内单例
- 图片 Base64 输入
- HTTP 请求体、Prompt 和图片大小校验
- PNG Base64 mask 输出
- 多 mask 和空 mask 协议
- 有界 GPU 任务队列和固定数量的 GPU worker
- 排队超时、推理超时和队列满保护
- HTTP 推理超时后继续占用原 GPU worker，直到底层同步推理真正结束
- CUDA OOM 专用错误、一次缓存清理和恢复失败后的 unavailable 状态
- JPEG/PNG 文件签名、编码长度和头部像素尺寸预检
- 根据 Content-Length 和实际接收字节数限制完整 HTTP 请求体
- 不记录 API Key、完整 Base64 图片、Prompt 或底层推理异常详情
- 可选 API Key
- `/health`、`/ready`、`/metrics`、`/metrics/prometheus`、`/alerts`
- 完整 HTTP 状态、滚动延迟分位数、队列等待和 GPU 推理时间
- 当前队列长度、GPU 在途任务数及其历史最大值
- 空 mask、多 mask、CUDA OOM、鉴权和输入校验失败指标
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

完整 HTTP 请求体默认限制为 30 MiB：

```text
LISA_MAX_REQUEST_BYTES=31457280
```

服务会先检查 `Content-Length`，对没有可信长度或使用 chunked 传输的请求，
再按实际接收字节累计。超过限制时返回 HTTP 413 和
`request_too_large`，不会解析 JSON、解码 Base64 或进入 GPU 队列；响应要求
客户端关闭并重新建立连接。

输入只接受 JPEG 和 PNG。服务在 OpenCV 完整解码前依次检查：

1. Base64 编码长度是否可能超过配置的解码字节上限。
2. Base64 是否严格合法。
3. 文件签名是否为 JPEG 或 PNG。
4. JPEG SOF 或 PNG IHDR 中的宽高和像素数是否超限。
5. OpenCV 是否能成功解码，并再次检查实际像素数。

GIF、WebP、伪造签名、损坏头部和超过限制的图片均返回
`invalid_request`，不会进入 GPU 队列。应用侧已经执行 30 MiB 硬限制；
正式对外暴露时仍应在 Nginx/Caddy 等反向代理配置相同或更小的请求体上限，
形成入口和应用双层保护。

## 健康检查

```bash
curl -s http://127.0.0.1:8000/health
```

```bash
curl -s http://127.0.0.1:8000/ready
```

JSON 指标：

```bash
curl -s http://127.0.0.1:8000/metrics
```

Prometheus 指标：

```bash
curl -s http://127.0.0.1:8000/metrics/prometheus
```

进程内告警状态：

```bash
curl -s http://127.0.0.1:8000/alerts
```

配置 API Key 时，上述三个监控接口使用 `X-API-Key` 或
`Authorization: Bearer`。完整 Prometheus、Alertmanager、Grafana 和 NVIDIA
DCGM Exporter 接入说明位于 `production/monitoring/README.md`。

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

镜像只复制 `production/`、`model/` 和 `utils/` 运行目录。
`production/.env`、数据集、实验输出、训练权重和 Git 元数据均不进入镜像。
镜像安装独立的 `production/requirements.txt`，不复用包含训练、评估和
Demo 依赖的根目录 `requirements.txt`。

仓库同时提供根目录 `.dockerignore` 和
`production/Dockerfile.dockerignore`。前者兼容 legacy Docker builder，
后者供支持 Dockerfile-specific ignore 的 BuildKit 使用；两者均采用目录
白名单并在白名单之后排除 `.env`，避免将大模型、数据集、实验输出或本地
运行凭据发送给 Docker daemon。

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

## 正式发布模型与镜像

当前决定：暂不实施外部制品发布。本阶段不安装或配置 rclone，不创建或接入
镜像仓库，不上传模型权重，也不执行 `production/publish_release.sh`。当前
模型和镜像继续保留在已完成验收的单台 GPU 服务器上；这不影响该服务器本地
启动服务，但不具备跨机器恢复、扩容和远端灾备能力。

这一步的目的不是提高模型精度或推理速度，而是把只存在于当前服务器的运行
环境和 27 GB 模型保存到独立远端。当服务器磁盘损坏、镜像被清理、模型目录
误删、需要迁移到新服务器或需要回滚时，可以重新取得与验收版本完全一致的
镜像和权重。

未来出现多机部署、正式灾备、服务器迁移或严格回滚要求时，再按以下流程
实施。相关通用工具保留在仓库中，不代表当前已经完成外部发布。

`production/publish_release.sh` 发布已经通过容器 GPU 验收的确切镜像，不会
重新构建一个未经相同验收的新镜像。脚本会：

1. 从容器 `summary.json` 读取已验证镜像标签、镜像 ID 和源码 Git commit。
2. 重新校验冻结模型的 `SHA256SUMS`。
3. 用模型版本和已验证 commit 生成不可变镜像标签并推送。
4. 核对推送后镜像 ID 与容器验收中的镜像 ID 完全一致。
5. 生成绑定模型校验值、镜像 digest 和精度/容器报告的
   `release-manifest.json`。
6. 使用 rclone 的 immutable/check 流程上传并复核模型制品。

发布脚本不会执行 `docker login`、创建 rclone 凭据或把凭据写入仓库。先在
远程发布服务器外部完成镜像仓库和对象存储认证，再设置两个非敏感目标：

```bash
export LISA_IMAGE_REPOSITORY="registry.example.com/safety/lisa-safety-seg"

export LISA_MODEL_REMOTE="internal-models:lisa-safety-seg/lisa13b-clean030-v1"
```

远程 Linux 发布服务器执行：

```bash
bash production/publish_release.sh
```

其中 `LISA_MODEL_REMOTE` 是已配置好的 rclone remote 目标。脚本使用
`--immutable`，目标中已有同名但内容不同的文件时会失败，不会静默覆盖已发布
制品。最终以镜像 registry digest 和 `release-manifest.json` 作为发布身份，
不再只依赖可变 tag。

未来实施顺序：

1. 准备私有 Docker 镜像仓库和独立模型存储。
2. 在发布服务器外部完成认证，不把凭据写入仓库。
3. 设置 `LISA_IMAGE_REPOSITORY` 和 `LISA_MODEL_REMOTE`。
4. 重新校验冻结模型 `SHA256SUMS` 和容器验收报告。
5. 推送已验收镜像并取得 registry digest。
6. 生成 `release-manifest.json`，上传模型制品。
7. 从远端重新下载并校验，最后演练新服务器恢复和旧版本回滚。

## 本地纯逻辑测试

本地安装基础 Python 依赖后可执行：

```bash
python3 -m unittest discover \
  -s production/tests \
  -v
```

本地测试不运行模型、不加载权重、不使用GPU。

## 生产限制

- 当前同时提供 JSON 和 Prometheus 指标，但外部 Prometheus、Alertmanager、
  Grafana 与 NVIDIA DCGM Exporter 尚未在目标服务器部署。
- 当前请求只接受 Base64 图片，不允许服务端访问任意 URL，从而避免 SSRF。
- 请求超时只控制HTTP等待时间，已经提交到GPU的同步推理不会被强制中断；
  专用GPU worker会等它真正结束后才处理下一个任务，避免产生隐性并发。
- 应用不记录请求体、API Key、完整图片或 Prompt；返回客户端的推理错误
  使用固定脱敏文本，不包含底层异常和服务器路径。
- 13B模型没有在本地执行；必须在远程GPU环境进行冒烟、benchmark和压测。
- 正式上线前仍必须完成 `todo.md` 中的 golden test、制品冻结、精度复评、监控、灰度和回滚。
