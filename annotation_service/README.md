# GroundingDINO Free Detection Service

该服务只保留自由检测模式：

```text
上传 JPEG/PNG
  -> 提交任意 grounding_prompt
  -> GroundingDINO 异步检测
  -> bbox JSON + 带框 PNG
```

Spring 不再提交 `requested_categories`，服务不再暴露隐患规则、SAM、Qwen、
Task 或 Release API。Prompt 内容不使用类别白名单；只进行非空、首尾空白和
最大 2000 字符的通用请求校验。Worker 为符合 GroundingDINO 输入约定，会在
模型 Caption 末尾补英文句点，但数据库和 API 返回用户提交的原始 Prompt。

API 进程不加载模型。GroundingDINO 由共享同一个持久化目录的独立 GPU Worker
懒加载。

## API

```text
GET  /health
GET  /ready
POST /v1/annotation/assets
GET  /v1/annotation/assets/{asset_id}
GET  /v1/annotation/assets/{asset_id}/content
POST /v1/annotation/jobs
GET  /v1/annotation/jobs/{job_id}
GET  /v1/annotation/jobs/{job_id}/detections
GET  /v1/annotation/jobs/{job_id}/assets/{asset_id}/bbox-image
```

除 `/health` 外，部署时均应使用以下任一种认证：

```http
X-API-Key: <ANNOTATION_API_KEY>
```

或：

```http
Authorization: Bearer <ANNOTATION_API_KEY>
```

完整 Spring 契约见：

- `docs_caich/annotation_api.md`
- `docs_caich/annotation_openapi.yaml`

## 持久化

启用存储后，服务初始化：

```text
annotation-data/
├── annotation.db
├── images/
├── overlays/jobs/
└── tmp/
```

数据库使用 WAL 和外键约束。schema v7 增加 `grounding_prompt` 与
`job_artifacts`。旧数据库会原地升级；旧的类别 Job 会保留用于审计，但新的
自由检测 Worker 不会领取它们。

API 与 Worker 必须使用完全相同的 `ANNOTATION_STORAGE_ROOT`。正式部署应备份
整个目录，不能只备份 SQLite 文件。

## 环境配置

以下命令均在远程 Linux 服务器执行：

```bash
cd <LISA仓库目录>
cp annotation_service/.env.example annotation_service/.env
chmod 600 annotation_service/.env
```

必须在未提交的 `.env` 中填写：

- `ANNOTATION_API_KEY`
- `ANNOTATION_STORAGE_ROOT`
- `ANNOTATION_GROUNDING_DINO_ROOT`
- `ANNOTATION_GROUNDING_DINO_CONFIG`
- `ANNOTATION_GROUNDING_DINO_CHECKPOINT`
- `ANNOTATION_GROUNDING_DINO_BERT`

`ANNOTATION_GROUNDING_DINO_ROOT` 是包含 `groundingdino` Python 包的固定源码
目录。配置、权重和离线 BERT 使用 MODEL_STORE 中同一模型制品的绝对路径。

正式密钥可在服务器生成：

```bash
openssl rand -hex 32
```

## 依赖

使用服务器现有的 PyTorch/CUDA/GroundingDINO Python 环境，不要随意覆盖其中
的 PyTorch 版本：

```bash
python -m pip install -r annotation_service/requirements.txt
python -m pip install -r annotation_service/docker/requirements-worker.txt
```

路径预检不会加载权重：

```bash
cd <LISA仓库目录>
set -a
source annotation_service/.env
set +a
python -c "from annotation_service.worker.settings import GroundingDINOWorkerSettings as S; s=S.from_env(); s.validate_model_files(); print('GroundingDINO paths OK')"
```

## 直接启动 Python

终端一启动 API：

```bash
cd <LISA仓库目录>
set -a
source annotation_service/.env
set +a
python -m uvicorn annotation_service.app:app --host 0.0.0.0 --port 8008
```

终端二启动持续检测 Worker：

```bash
cd <LISA仓库目录>
set -a
source annotation_service/.env
set +a
python -m annotation_service.worker
```

首次联调可将 Worker 命令改为：

```bash
python -m annotation_service.worker --once
```

没有可领取 Job 时，`--once` 返回退出码 3。

## curl 联调

健康检查：

```bash
curl -fsS http://127.0.0.1:8008/health
curl -fsS -H "X-API-Key: <API_KEY>" http://127.0.0.1:8008/ready
```

上传图片：

```bash
curl -fsS -X POST http://127.0.0.1:8008/v1/annotation/assets \
  -H "X-API-Key: <API_KEY>" \
  -F "file=@/path/to/test.jpg" \
  -F "group_id=manual-test"
```

创建自由检测 Job：

```bash
curl -fsS -X POST http://127.0.0.1:8008/v1/annotation/jobs \
  -H "X-API-Key: <API_KEY>" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: detection-test-001" \
  -d '{"asset_ids":["<ASSET_ID>"],"grounding_prompt":"the worker beside the blue excavator"}'
```

轮询状态并读取结果：

```bash
curl -fsS -H "X-API-Key: <API_KEY>" \
  http://127.0.0.1:8008/v1/annotation/jobs/<JOB_ID>
curl -fsS -H "X-API-Key: <API_KEY>" \
  http://127.0.0.1:8008/v1/annotation/jobs/<JOB_ID>/detections
curl -fsS -H "X-API-Key: <API_KEY>" \
  -o bbox.png \
  http://127.0.0.1:8008/v1/annotation/jobs/<JOB_ID>/assets/<ASSET_ID>/bbox-image
```

## 本地 CPU 测试

测试使用 Fake Predictor，不加载模型：

```bash
python -m unittest discover -s annotation_service/tests -v
```
