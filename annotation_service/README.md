# Annotation Service

本目录是施工安全辅助标注后端的独立服务骨架。API 契约位于：

- `docs_caich/annotation_api.md`
- `docs_caich/annotation_openapi.yaml`

## 当前阶段

当前已实现：

- FastAPI 应用工厂；
- `/health`、`/ready`；
- Bearer Token 和 `X-API-Key` 鉴权；
- request ID 生成与回传；
- 统一 JSON 错误协议；
- 完整请求体大小限制；
- 显式 CORS 白名单；
- v1 Pydantic 契约模型；
- Job、Task、Release 状态迁移约束；
- SQLite WAL 元数据仓库和显式 schema 版本；
- 图片 SHA256 去重和内容寻址存储；
- JPEG/PNG 安全校验、解码像素上限和上传大小限制；
- Asset 上传、元数据查询和原图读取接口；
- Asset 上传幂等键和重复图片关联；
- Job 创建、状态查询和原子幂等入队接口；
- SQLite 队列容量限制、Worker 原子领取和租约心跳；
- Worker 异常后的过期 Job 恢复领取；
- GroundingDINO 独立 GPU Worker、模型懒加载和类别实体 Prompt 映射；
- GroundingDINO 归一化框到原图绝对 `xyxy` 坐标转换；
- 检测结果在有效 Worker 租约下事务覆盖，支持异常恢复后幂等重跑；
- `stop_after=grounding_dino` 检测型 Job 的成功、部分失败和失败闭环；
- 基于检测框的 PPE 缺失、人机接近、洞口/临边防护和现场整理候选规则；
- 隐患候选的规则版本、置信度、检测框证据和视觉复核标记持久化；
- `stop_after=hazard_rules` Job 闭环及隐患候选查询接口；
- 隐患候选到人工复核 Task 的幂等构建接口；
- Task 列表、详情、草稿、提交、审核和 Artifact 下载接口；
- Task cursor 分页、组合筛选和 API Key 鉴权；
- Task 乐观锁、完整版本快照、Review 历史及审计查询接口；
- submit 和 accept 前的 polygon 边界、非零面积、目标 mask 非空及
  3+2+1 Prompt 自动验收；
- 千问视觉事实、3+2+1 Prompt 的严格 JSON Schema、解析器和版本化模板；
- Qwen2.5-VL OpenAI 兼容 provider、原图/mask/crop 多模态输入和两阶段生成；
- Prompt enrichment Operation 入队、查询、租约恢复和独立 Qwen Worker；
- SAM bbox Prompt 推理适配器、mask/overlay/crop/polygon 生成；
- mask candidate Operation 入队、查询、租约恢复和独立 SAM Worker；
- 自动串联 GroundingDINO、隐患规则、SAM、Qwen 和 Task 的完整 Pipeline
  Worker；
- mask、overlay、crop 的图片内容校验和原子文件写入；
- Release 创建、查询、manifest/archive 下载接口；
- 独立纯 CPU Release Worker、租约恢复和确定性 ReasonSeg ZIP 导出；
- 按 `group_id` 防泄漏划分 train/val/golden；
- Job 重启恢复查询和幂等记录；
- 环境变量配置和纯逻辑测试。

当前代码已经具备全自动标注闭环。省略 `stop_after` 的 Job 由完整 Pipeline
Worker 消费，自动生成带 SAM polygon 和 3+2+1 Prompt 的 `generated` Task；
模型结果仍必须人工 submit/review，不会自动接受。单独的 SAM、Qwen Operation
也可用于分阶段重生成候选。

本地测试不会加载模型。真实 GroundingDINO、SAM、Qwen2.5-VL 权重加载和 GPU
端到端延迟必须在远程服务器验收。

API 进程不得加载 GroundingDINO、SAM、千问或 LISA 权重。后续模型能力由独立
GPU Worker 提供。

## 持久化目录

启用 `ANNOTATION_STORAGE_ENABLED=true` 后，服务初始化：

```text
annotation-data/
├── annotation.db
├── images/
├── masks/
├── overlays/
├── crops/
├── exports/
└── tmp/
```

数据库使用 WAL、外键约束和 `busy_timeout`。数据库只保存相对路径，不保存
Base64 或对外暴露宿主机绝对路径。正式部署应将整个目录挂载到持久卷；不得
只保存 SQLite 文件而遗漏图片和 Artifact。

## Asset 接口

前端通过以下接口管理待标注原图：

- `POST /v1/annotation/assets`：上传 JPEG/PNG，使用
  `multipart/form-data` 传递 `file`、`group_id`、可选的 `source_id` 和
  `metadata_json`；
- `GET /v1/annotation/assets/{asset_id}`：读取图片元数据；
- `GET /v1/annotation/assets/{asset_id}/content`：读取原始图片字节。

上传接口支持可选的 `Idempotency-Key` 请求头。同一幂等键和相同请求会返回
首次创建的 Asset；同一幂等键对应不同请求时返回 `409
idempotency_conflict`。相同图片在不同请求中上传时保留独立 Asset 元数据，
底层图片按 SHA256 复用，并通过 `duplicate_of` 指向首个 Asset。

`group_id` 用于视频、拍摄序列或工地级的数据隔离，后续划分训练集和测试集时
不得把同组相邻帧拆到不同数据集。`metadata_json` 必须是 JSON object，不接受
数组或任意文本。

## Job 接口与 Worker 队列

前端通过以下接口提交和轮询自动标注任务：

- `POST /v1/annotation/jobs`：校验 Asset 后创建异步 Job，返回 HTTP 202；
- `GET /v1/annotation/jobs/{job_id}`：读取状态、当前阶段、进度、错误和已生成
  Task ID；
- `GET /v1/annotation/jobs/{job_id}/detections`：读取 GroundingDINO 检测框，
  可用 `asset_id` 查询参数筛选单张图片。
- `GET /v1/annotation/jobs/{job_id}/hazard-candidates`：读取规则推导的隐患
  候选，可用 `asset_id` 筛选单张图片。

创建接口支持 `Idempotency-Key`。Asset 校验、队列容量检查、Job 写入和幂等
记录位于同一事务中。`ANNOTATION_MAX_QUEUED_JOBS` 控制等待中的 Job 数量，
超过上限返回 `429 queue_full`；幂等重试不会重复占用容量。

GPU Worker 不通过公开 HTTP 接口领取任务，而是与 API 进程共享持久化目录，
使用 `AnnotationStore` 的内部队列方法：

- `claim_next_job(worker_id=..., lease_seconds=...)`：原子领取最早的排队 Job，
  并优先恢复租约已过期的运行中 Job；
- `heartbeat_job(job_id, worker_id=..., lease_seconds=...)`：续租；
- `update_job(..., worker_id=...)`：在有效租约下更新阶段、进度或终态。

公开 Job 响应不暴露 `claimed_by`、租约时间或内部 Asset 列表。GPU Worker
领取到的内部 Job payload 包含执行所需的 Asset、类别、选项和租约信息。
数据库启动时会自动迁移到 schema v6。v5 为隐患候选来源建立唯一关联，并为
Release Worker 增加领取令牌、租约、心跳和重试字段；v6 增加可恢复的模型
Operation 队列。已有 SQLite 数据会在启动时原地迁移。

### GroundingDINO 与隐患规则 Worker

当前 Worker 领取明确设置以下边界之一的分阶段 Job：

```json
{
  "options": {
    "generate_masks": false,
    "enrich_prompts": false,
    "prompt_count": 6,
    "stop_after": "grounding_dino"
  }
}
```

或将 `stop_after` 改为 `hazard_rules`。省略 `stop_after` 的完整流水线 Job
只由 `annotation_service.pipeline_worker` 领取，不会被分阶段 Worker
误领取。

检测阶段只输出基础实体框。规则阶段读取这些框并生成“隐患候选”，例如：

- 人员框的头部区域没有匹配到安全帽框；
- 人员框和施工设备框的边缘距离低于阈值；
- 洞口或平台边沿附近没有匹配到盖板、围挡或护栏框；
- 材料框与通道框相交，或画面中检测到建筑垃圾。

缺失类规则依赖“未匹配到防护用品框”这一弱负证据，可能受到 GroundingDINO
漏检影响，因此结果始终携带 `requires_visual_verification=true`，只作为
千问复核和人工审核的候选，不直接成为最终标签。`safe` 不根据框存在自动
推导；宽泛的 `unsafe` 只从已命中的具体空间/现场规则派生，不从单个人员框
臆测 PPE 缺失。

每个候选包含 `category`、具体目标框、关联 `target_detection_ids`、
`confidence`、`rule_id`、`rule_version`、`evidence` 和 `metadata`。规则阶段
分阶段 Job 完成后，`grounding_dino`、`hazard_rules` 都进入终态，SAM 及之后
阶段标记为 `skipped`。完整 Job 则继续执行 SAM、Qwen 和 Task 构建。其中
`confidence` 是由检测分数和空间关系组合出的规则排序分，
不是经过校准的真实风险概率。

Worker 只在领取到兼容 Job 后懒加载模型。FastAPI API 进程不会导入
GroundingDINO 或 PyTorch。

## 配置

复制环境变量模板：

```bash
cp annotation_service/.env.example annotation_service/.env
```

`.env` 不得提交到仓库。正式部署必须设置 `ANNOTATION_API_KEY`，并在
`ANNOTATION_CORS_ORIGINS` 中填写明确的前端来源域名。

## 纯逻辑测试

测试不加载模型、不使用 GPU：

```bash
python -m unittest discover -s annotation_service/tests -v
```

## 远程启动

以下启动命令仅用于远程 Linux 服务环境，本地不要启动服务或加载模型：

```bash
cd <LISA仓库目录>
set -a
source annotation_service/.env
set +a
python -m uvicorn annotation_service.app:app \
  --host 0.0.0.0 \
  --port 8001
```

GroundingDINO Worker 必须在远程 Linux + GPU 环境执行。先在未提交的
`annotation_service/.env` 中配置 `.env.example` 展示的模型路径。另开一个
终端，进入同一个仓库并加载同一份环境配置后运行：

```bash
cd <LISA仓库目录>
set -a
source annotation_service/.env
set +a
python -m annotation_service.worker --once
```

`--once` 最多领取一个兼容 Job，适合首次联调；移除 `--once` 后持续轮询队列。
没有兼容 Job 时 `--once` 返回退出码 3。API 和 Worker 必须使用同一个
`ANNOTATION_STORAGE_ROOT`。

首次规则联调顺序为：上传 Asset、创建带 `stop_after=hazard_rules` 的 Job、
执行一次 Worker、轮询 Job 到终态，再依次调用
`GET /v1/annotation/jobs/{job_id}/detections` 和
`GET /v1/annotation/jobs/{job_id}/hazard-candidates` 检查检测框、候选类别
及证据链。确认候选后调用
`POST /v1/annotation/jobs/{job_id}/review-tasks`，即可幂等生成待人工补齐
polygon 与 Prompt 的 Task；重复调用不会为同一候选创建重复 Task。

## Docker Compose 启动（无需反向代理）

远程 Linux 服务器可以直接通过 Docker Compose 启动 API 和纯 CPU Release
Worker。API 端口直接映射到宿主机，不依赖 Nginx：

```bash
cd <LISA仓库目录>
cp annotation_service/docker/.env.example annotation_service/docker/.env
chmod 600 annotation_service/docker/.env
docker compose \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  up -d --build
```

部署前必须在未提交的 `.env` 中填写随机 `ANNOTATION_API_KEY`、持久化存储
目录 `ANNOTATION_STORAGE_HOST_PATH` 和实际前端来源
`ANNOTATION_CORS_ORIGINS`。`ANNOTATION_CONTAINER_UID/GID` 应与持久化目录
所有者一致，避免产生 root 权限文件。还需分别配置构建时使用的
`GROUNDING_DINO_SOURCE_HOST_PATH` 和运行时只读挂载的
`GROUNDING_DINO_MODEL_HOST_PATH`；源码固定进镜像，权重、配置和离线 BERT
保留在完整的 MODEL_STORE 版本目录中。默认启动 `api` 与
`release-worker`，不会启动或加载模型。默认直接监听宿主机
`0.0.0.0:18001`：

```text
API:     http://<服务器地址>:18001
Swagger: http://<服务器地址>:18001/docs
```

状态和日志检查：

```bash
docker compose \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  ps
docker compose \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  logs --tail=100 api worker
```

API 和 Release Worker 使用同一个持久化目录，均设置
`restart: unless-stopped`。完整远程部署、升级、回滚、备份和验收命令见
`annotation_service/docker/README.md`。

前端可使用 `docs_caich/annotation_openapi.yaml` 生成契约类型，并接入当前已
完成的 Asset、Job、Task、SAM/Prompt Operation 和 Release 接口。运行时与
静态 OpenAPI 均包含 mask candidate 和 prompt enrichment 路由。

## Task 人工审核接口

即使模型 Worker 或 Qwen 模型容器尚未启动，前端也可以使用下列接口联调人工
编辑与二级审核：

```text
GET  /v1/annotation/tasks
GET  /v1/annotation/tasks/{task_id}
GET  /v1/annotation/tasks/{task_id}/versions
GET  /v1/annotation/tasks/{task_id}/reviews
PUT  /v1/annotation/tasks/{task_id}/draft
POST /v1/annotation/tasks/{task_id}/submit
POST /v1/annotation/tasks/{task_id}/review
GET  /v1/annotation/tasks/{task_id}/artifacts/{artifact_type}
```

Task 列表支持 `status`、`category`、`group_id`、`job_id`、
`annotator_id`、`reviewer_id`、`bad_case_type`、`limit` 和 `cursor`。
`cursor` 只能原样使用上一页返回的 `next_cursor`，不能由前端构造。

草稿允许 mask 或 Prompt 暂不完整。提交一级审核，以及二级审核执行
`decision=accept` 时，后端强制检查：

- 至少一个非零面积且不越过原图边界的 `target` polygon；
- Prompt 恰好为 3 条 `visual`、2 条 `risk` 和 1 条 `agent`；
- Prompt 非空且互不重复；
- `safe`/`unsafe` 的目标不能仍是宽泛的“安全区域”或“不安全目标”。

只运行 GroundingDINO/规则的分阶段模式下，可通过
`POST /v1/annotation/jobs/{job_id}/review-tasks` 将规则候选构建为人工 Task。
该接口只预填候选目标、位置锚点、风险候选和来源证据，不会把检测框伪装成
polygon，也不会生成冒充千问输出的 Prompt。前端必须补齐 polygon 和 3+2+1
Prompt 后才能提交或接受。

## Release 导出

前端创建 Release 后，由纯 CPU Worker 异步导出：

```text
POST /v1/annotation/releases
GET  /v1/annotation/releases/{release_id}
GET  /v1/annotation/releases/{release_id}/manifest
GET  /v1/annotation/releases/{release_id}/archive
```

Release 只读取 `accepted` Task，按 `group_id` 确定性分配到
train/val/golden。同名 Release 不覆盖；创建接口支持 `Idempotency-Key`。
导出前再次执行 polygon 与 3+2+1 验收，输出 jpg/json 配对、审计
`annotation_manifest.jsonl`、构建摘要、数据卡、外部哈希 manifest 和 ZIP。

不使用 Docker 时，Release Worker 在远程 Linux 的同一仓库和同一存储目录
运行：

```bash
cd <LISA仓库目录>
set -a
source annotation_service/.env
set +a
python -m annotation_service.release_worker --once
```

## SAM mask 候选

分阶段模式先为 Task 提交 bbox：

```http
POST /v1/annotation/tasks/{task_id}/mask-candidates
Content-Type: application/json

{"expected_version": 1, "box_xyxy": [120, 80, 460, 720]}
```

接口返回 Operation。SAM Worker 使用 bbox 生成二值 mask、红色叠加图、目标
crop 和 polygon，并将 artifact 原子写入共享存储。以下命令仅在远程 Linux
GPU 环境执行：

```bash
cd <LISA仓库目录>
set -a
source annotation_service/.env
set +a
python -m annotation_service.sam_worker --once
```

移除 `--once` 后持续消费 `mask_candidate` Operation。Worker 从
`ANNOTATION_SAM_CHECKPOINT` 加载 MODEL_STORE 权重；API 进程不导入 PyTorch。

## 完整 Pipeline Worker

创建完整 Job 时省略 `stop_after`，并保持：

```json
{
  "generate_masks": true,
  "enrich_prompts": true,
  "prompt_count": 6
}
```

完整 Worker 自动执行：

```text
GroundingDINO -> 隐患规则 -> Task -> SAM -> Qwen2.5-VL
```

以下命令仅在远程 Linux GPU 环境执行：

```bash
cd <LISA仓库目录>
set -a
source annotation_service/.env
set +a
python -m annotation_service.pipeline_worker --once
```

成功 Job 的 `task_ids` 指向已经包含 SAM polygon、mask artifact、视觉事实和
3+2+1 Prompt 的生成态 Task。完整 Worker 不会自动 submit 或 accept。

## Qwen2.5-VL 多 Prompt 生成

`qwen_contract.py` 固定两次 Qwen2.5-VL 调用之间的边界：

1. 视觉事实阶段只接收原图、目标框或 mask，以及作为定位线索的检测/规则
   上下文，输出 `QwenVisualFacts`；
2. Prompt 富化阶段只接收已验证的视觉事实和来源类别，输出严格的
   `QwenPromptSet`（3 visual + 2 risk + 1 agent）。

解析器接受纯 JSON 或单个 JSON code fence，拒绝额外字段、重复事实、重复
Prompt、非 3+2+1 数量和非 JSON 文本。`qwen_provider.py` 使用 vLLM 的
OpenAI 兼容 `/chat/completions` 接口，第一阶段发送原图、SAM mask/overlay
以及可选 crop；第二阶段只发送已经通过契约校验的视觉事实。没有“模型不可用
时伪造事实”的 fallback。provider 内部调用：

```text
build_visual_facts_messages -> parse_visual_facts
build_prompt_enrichment_messages -> parse_prompt_set
```

模板版本分别为 `construction-visible-facts-v1` 和
`construction-prompts-3-2-1-v1`，应写入 Task provenance。

前端或 Spring 后端先保证 Task 已有 `mask-overlay` 或 `mask` artifact，再
提交：

```http
POST /v1/annotation/tasks/{task_id}/prompt-enrichments
Content-Type: application/json

{"expected_version": 1}
```

接口返回 HTTP 202。随后轮询：

```text
GET /v1/annotation/operations/{operation_id}
```

成功时 `result` 同时包含锁定的视觉事实、恰好 3 条 visual、2 条 risk、
1 条 agent Prompt，以及 Qwen 模型和模板版本。候选结果不会自动覆盖 Task；
调用方确认后应通过草稿接口显式写回，保留人工审核边界。

模型服务就绪后，配置 `.env` 中的 `ANNOTATION_QWEN_BASE_URL` 和
`ANNOTATION_QWEN_MODEL`。当前服务器上已有的宿主机网关和 vLLM
`served-model-name` 对应配置为：

```env
ANNOTATION_QWEN_BASE_URL=http://127.0.0.1:18000/qwen25/v1
ANNOTATION_QWEN_MODEL=qwen25vl
```

如果 Qwen Worker 后续改为运行在 `model-gateway` Docker 网络内，`BASE_URL`
应改成 `http://qwen25vl:8000/v1`；当前直接运行 Python Worker 时使用上面的
宿主机回环地址。

以下命令仅在远程 Linux 服务环境执行：

```bash
cd <LISA仓库目录>
set -a
source annotation_service/.env
set +a
python -m annotation_service.qwen_worker --once
```

`--once` 最多消费一个 Prompt Operation；移除该参数后持续轮询。API 服务
可以先启动，Qwen 模型和 Worker 后启动。
