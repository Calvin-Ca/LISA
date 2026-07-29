# 施工安全标注服务 API 契约（v1）

本文档冻结第一版施工安全标注后端的接口与数据契约。目标设计覆盖图片登记、
GroundingDINO 检测、隐患规则推导、SAM mask、千问 Prompt 富化、人工审核状态
和 ReasonSeg 数据导出；前端由其他项目实现。目标设计不等于当前全部落地，
真实实现边界见下方状态矩阵。

配套的机器可读规范位于
[`annotation_openapi.yaml`](annotation_openapi.yaml)。如果本文档与 OpenAPI
规范存在差异，以 OpenAPI 规范为请求/响应结构依据，以本文档的状态约束和
业务语义为补充。

## 当前结论与实现边界（2026-07-30）

当前代码已经形成从图片到 mask、Prompt、人工审核和 Release 的完整 API 与
Worker 链路。API 进程不加载模型；GroundingDINO、SAM、Qwen 和完整 Pipeline
分别由独立 Worker 或外部 vLLM 服务执行。

| 阶段 | 当前状态 | 对外接口或执行入口 | 说明 |
|---|---|---|---|
| 服务状态与鉴权 | 已实现 | `/health`、`/ready` | API 可独立启动，不加载模型 |
| 图片登记 | 已实现 | `POST /assets` | 支持 JPEG/PNG 校验、去重和持久化 |
| GroundingDINO | 已实现 | Job API + `annotation_service.worker` | 分阶段 Worker 消费检测/规则 Job |
| 隐患规则 | 已实现 | `GET /jobs/{id}/hazard-candidates` | 输出待视觉复核的候选，不作为最终真值 |
| Task 构建 | 已实现 | `POST /jobs/{id}/review-tasks` | 初始 Task 不包含 polygon 和 Prompt |
| SAM mask | 已实现 | `POST /tasks/{id}/mask-candidates` + SAM Worker | 输出 mask、overlay、crop、polygon |
| mask artifact 写入 | 已实现 | 由 SAM Worker 内部原子写入 | Spring 不上传服务器文件路径 |
| Qwen2.5-VL Prompt | 已实现 | `POST /tasks/{id}/prompt-enrichments` + Qwen Worker | Task 必须已经有 SAM artifact |
| Prompt Operation 查询 | 已实现 | `GET /operations/{id}` | 结果是候选，不自动覆盖 Task |
| 人工草稿与审核 | 已实现 | Task draft/submit/review API | 提交和接受前强制校验 mask 与 3+2+1 Prompt |
| ReasonSeg Release | 已实现 | Release API + Release Worker | 只导出已经接受的 Task |
| 默认完整 Job 编排 | 已实现 | `python -m annotation_service.pipeline_worker` | 自动执行 DINO、规则、SAM、Qwen、Task 构建 |

实现边界：

- 完整 Job 会自动把 SAM polygon 和 Qwen 3+2+1 Prompt 写入生成态 Task，
  但不会自动提交或接受，仍需人工审核；
- 单独的 mask/Prompt Operation 只产生候选，不直接覆盖 Task，避免异步模型
  结果覆盖人工编辑；
- `annotation_openapi.yaml` 与运行时路由保持一致；
- 本地测试使用假模型，不加载权重。GroundingDINO、SAM 和 Qwen2.5-VL 的
  真实 GPU 端到端验收必须在远程服务器启动模型后执行。

### 默认完整链路

```text
上传图片
  -> 创建不带 stop_after 的完整 Job
  -> Full Pipeline Worker
  -> GroundingDINO
  -> 隐患规则
  -> 创建 Task
  -> SAM mask / overlay / crop / polygon
  -> Qwen2.5-VL 视觉事实与 3+2+1 Prompt
  -> 返回完整 generated Task
  -> submit
  -> review
  -> Release
```

完整 Job 创建示例：

```json
{
  "asset_ids": ["ast_01J..."],
  "requested_categories": ["helmet_missing"],
  "pipeline_version": "grounded-sam-qwen25vl-v1",
  "options": {
    "generate_masks": true,
    "enrich_prompts": true,
    "prompt_count": 6
  }
}
```

`stop_after` 必须省略。Spring 轮询 Job，成功后读取 `task_ids`，再进入人工
审核。

### 分阶段候选链路

```text
stop_after=hazard_rules Job
  -> review-tasks
  -> POST mask-candidates
  -> 轮询 mask Operation
  -> POST prompt-enrichments
  -> 轮询 Prompt Operation
  -> Spring 合并 shapes、facts、prompts
  -> PUT task draft
  -> submit / review / Release
```

### Spring 后端对接清单

- API Base URL 使用 `http://<annotation-host>:<port>`，业务路径以
  `/v1/annotation` 开头；不要把 Qwen 网关地址暴露给浏览器。
- `/ready` 只表示 API 和共享存储可用，不代表 GPU Worker 或 Qwen 容器已
  启动；模型可用性以 Job/Operation 状态和部署监控为准。
- 除 `/health` 外统一携带 `X-API-Key` 或 Bearer Token。密钥只保存在 Spring
  服务端配置，不返回前端。
- Asset、Job、Release 创建请求设置 `Idempotency-Key`；网络超时后使用同一
  key 和同一请求重试。
- Job、Operation、Release 均为异步资源。Spring 轮询状态，不要把 HTTP 202
  当成模型已完成。
- Task 写接口必须传 `expected_version`。收到 HTTP 409 后重新获取 Task，
  不得盲目重试旧请求。
- 图片、mask、overlay 和 Release ZIP 按二进制流转发，不转成 JSON Base64。
- Qwen Operation 的 `result` 是候选。Spring 需先获取最新 Task，将 facts 和
  prompts 合并到完整 `annotation` 后调用 draft；不能只提交 prompts 字段。
- 默认完整 Job 省略 `stop_after`，并保持 `generate_masks=true`、
  `enrich_prompts=true`；确保完整 Pipeline Worker 正在运行。
- 只联调 GroundingDINO 和隐患规则时使用 `stop_after=hazard_rules`，同时
  设置 `generate_masks=false`、`enrich_prompts=false`。

## 1. 版本与兼容规则

- API 根路径固定为 `/v1/annotation`。
- v1 内允许新增可选字段，不删除字段、不修改既有字段含义。
- 枚举值扩展属于向前兼容变更，前端必须为未知枚举值提供兜底展示。
- 破坏性变更必须发布新主版本路径，例如 `/v2/annotation`。
- 所有 JSON 使用 UTF-8；时间使用 UTC ISO 8601，例如
  `2026-07-24T10:30:00Z`。
- 所有 ID 都是不透明字符串，前端不得解析 ID 或根据其格式推断业务含义。

## 2. 通用约定

### 2.1 鉴权

业务接口使用以下任一方式鉴权：

```http
Authorization: Bearer <token>
```

或：

```http
X-API-Key: <token>
```

`/health` 不要求鉴权；`/ready` 是否公开由部署配置决定。

### 2.2 请求追踪

前端可以传入：

```http
X-Request-ID: frontend-request-id
```

后端在每个响应中返回 `X-Request-ID`。未传入时由后端生成。请求 ID 最长
128 个字符，不得包含密钥、图片内容或个人敏感信息。

### 2.3 幂等

以下创建接口接受 `Idempotency-Key`：

- `POST /v1/annotation/assets`
- `POST /v1/annotation/jobs`
- `POST /v1/annotation/releases`

同一调用方在幂等保留期内用相同 key 和相同请求体重试，应得到第一次创建的
资源；相同 key 携带不同请求体返回 HTTP 409。

### 2.4 分页

列表接口使用 cursor 分页：

```json
{
  "items": [],
  "next_cursor": "opaque-cursor-or-null"
}
```

cursor 是不透明字符串。前端不得自行生成、修改或持久解析 cursor。

### 2.5 坐标

- bbox 格式为 `[x1, y1, x2, y2]`。
- polygon 格式为 `[[x, y], ...]`。
- 坐标均为原图绝对像素坐标，原点位于左上角。
- `x` 向右递增，`y` 向下递增。
- bbox 使用半开区间语义：左上边界包含，右下边界不包含。
- 前端可以缩放图片展示，但提交时必须转换回原图坐标。
- 后端接受整数或浮点数，导出 ReasonSeg 时可按统一规则取整。

### 2.6 内容类型

- 图片上传使用 `multipart/form-data`。
- 其他写接口使用 `application/json`。
- 原图响应使用原始 `image/jpeg` 或 `image/png`。
- mask 原图统一返回 `image/png`。
- 标注导出归档使用 `application/zip`。

## 3. 核心资源

### 3.1 Asset

`asset` 表示一张原始图片。其主要字段为：

| 字段 | 含义 |
|---|---|
| `asset_id` | 后端图片 ID |
| `source_id` | 前端或原业务系统中的图片 ID |
| `group_id` | 视频、拍摄序列、工地或其他防泄漏分组 |
| `sha256` | 原始图片内容哈希 |
| `width` / `height` | 原图尺寸 |
| `content_url` | 鉴权图片读取接口 |
| `duplicate_of` | 内容重复时已有 asset ID |

`group_id` 是数据划分的强约束。相同 `group_id` 的样本不得跨
train、val、golden。

### 3.2 Annotation Job

`job` 表示一次异步自动标注过程。一个 job 可以处理多张 asset，并为每张
图片生成零个、一个或多个人工审核 task。

Job 状态：

```text
queued
running
succeeded
partial_failed
failed
cancelled
```

流水线阶段：

```text
grounding_dino
hazard_rules
sam
qwen_facts
qwen_prompts
build_review_tasks
```

阶段状态：

```text
pending
running
succeeded
failed
skipped
```

`partial_failed` 表示至少一个 asset 成功且至少一个 asset 或中间阶段失败。
错误必须出现在 `errors` 中，不能只写入日志。

Job 创建与 GPU 执行解耦。API 进程只在 SQLite 队列中原子写入 `queued`
任务，不加载任何模型权重。独立 GPU Worker 使用租约领取任务；领取后状态
变为 `running`，定期发送心跳续租。Worker 异常退出且租约过期后，其他
Worker 可以恢复领取同一 Job。每次写回状态时必须校验 `worker_id` 和有效
租约，避免旧 Worker 覆盖新 Worker 的结果。

队列等待上限由 `ANNOTATION_MAX_QUEUED_JOBS` 控制。队列已满时新请求返回
HTTP 429；相同 `Idempotency-Key` 的有效重试不重复占用队列容量。

### 3.3 Annotation Task

`task` 是前端实际编辑和审核的单位。一个 task 对应一张图片上的一个确定分割
语义。相同图片的不同类别或不同目标粒度应拆成不同 task。

Task 状态：

```text
generated
annotating
review_pending
changes_requested
needs_expert
accepted
rejected
frozen
```

允许的主要状态迁移：

```text
generated -> annotating
generated -> review_pending
annotating -> review_pending
review_pending -> accepted
review_pending -> changes_requested
review_pending -> needs_expert
review_pending -> rejected
changes_requested -> annotating
needs_expert -> accepted
needs_expert -> changes_requested
needs_expert -> rejected
accepted -> frozen
```

普通草稿保存不得直接把 task 变成 `accepted` 或 `frozen`。

### 3.4 Release

`release` 表示由已接受任务构建的不可变 ReasonSeg 数据版本。Release 状态：

```text
queued
building
succeeded
failed
```

同名 release 不得原地覆盖。需要修改数据时创建新版本名称。

## 4. 分类体系

v1 冻结以下任务类别：

```text
helmet_missing
no_helmet
no_jacket
harness_missing
equipment_proximity
opening_unprotected
guardrail_missing
poor_housekeeping
safe
unsafe
```

`safe` 和 `unsafe` 只能作为来源类别。进入人工提交和最终导出前，必须在
`target_object`、`mask_granularity` 和 Prompt 中具体化可分割对象，不得只写
“安全区域”“危险区域”或“不安全目标”。

GroundingDINO 主要检测可见基础实体，不直接承担否定或组合语义判断。隐患类别
由基础实体、空间规则和后续人工审核共同确定。

## 5. 标注数据结构

一个 task 的 `annotation` 包含五项语义不变量、polygon 和 Prompt：

```json
{
  "target_object": "靠近黄色挖掘机的一名作业人员",
  "instance_count": 1,
  "visual_anchor": [
    "位于画面中央",
    "位于黄色挖掘机左侧"
  ],
  "mask_granularity": "人员整体",
  "risk_semantics": "人机距离过近，存在碰撞风险",
  "shapes": [
    {
      "shape_id": "shp_001",
      "label": "target",
      "shape_type": "polygon",
      "points": [[121, 46], [154, 42], [309, 468]]
    }
  ],
  "prompts": [
    {
      "prompt_id": "p1",
      "type": "visual",
      "text": "分割画面中央靠近黄色挖掘机的一名作业人员。"
    }
  ]
}
```

约束：

- `target_object` 必须是具体可分割对象。
- `instance_count` 必须与 mask 和所有 Prompt 一致。
- `visual_anchor` 只能包含图中可证实的方位、颜色、结构或邻近关系。
- `mask_granularity` 说明标的是人员整体、设备整体、构件、边沿或空间。
- `risk_semantics` 可以为空；非空时必须由类别与可见事实共同支持。
- shape 的 `label` 只能为 `target` 或 `ignore`。
- 一个 polygon 至少三个点，面积必须大于零，坐标必须位于原图范围内。
- 同一 task 的多个 `target` polygon 合并为一张二值目标 mask。
- `shape_id` 在同一 task 内必须非空且唯一。

## 6. Prompt 契约

Prompt 类型：

```text
visual
risk
agent
```

草稿阶段允许 Prompt 不完整。调用 submit 时必须满足：

- 恰好 3 条 `visual`；
- 恰好 2 条 `risk`；
- 恰好 1 条 `agent`；
- 所有 Prompt 非空且不重复；
- `prompt_id` 在同一 task 内必须非空且唯一；
- 所有 Prompt 指向相同像素、相同实例和相同目标粒度；
- 风险型 Prompt 去掉风险修饰后仍包含具体可分割对象；
- 不得添加无法从图片、mask 和来源类别支持的动作、因果或违规事实。

千问生成的 Prompt 始终是候选值。重新富化不得自动覆盖人工草稿，必须由前端
显式选择并通过 draft 接口保存。

后端的模型无关契约实现位于 `annotation_service/qwen_contract.py`。视觉
事实与 Prompt 分两次调用，前一次只允许输出图中可直接确认的事实，后一次
只允许根据这些事实生成严格 3+2+1 Prompt。解析失败必须令对应阶段失败并
记录错误，不能用规则文本或默认句子冒充千问输出。

## 7. 并发编辑

每个 task 带单调递增的 `version`。修改请求必须提交
`expected_version`：

```json
{
  "expected_version": 3
}
```

如果数据库中的当前版本不是 3，后端返回：

```http
HTTP/1.1 409 Conflict
```

```json
{
  "request_id": "req_...",
  "code": "version_conflict",
  "message": "annotation task has been modified",
  "details": [
    {
      "field": "expected_version",
      "reason": "expected 3 but current version is 4"
    }
  ]
}
```

前端必须重新获取 task，不得自动用旧草稿覆盖新版本。

## 8. Bad case 类型

v1 支持以下主要状态：

```text
prompt_ok
prompt_rewritten
prompt_mask_mismatch
mask_overflow
mask_missing
instance_ambiguous
target_unrecognizable
dino_false_positive
dino_false_negative
hazard_rule_error
qwen_visual_hallucination
qwen_prompt_semantic_drift
other
```

`primary_result` 是单个主要状态；其他问题可以记录在评论或后续扩展字段中。

## 9. 接口摘要

### 9.1 服务状态

```text
GET /health
GET /ready
```

### 9.2 图片

```text
POST /v1/annotation/assets
GET  /v1/annotation/assets/{asset_id}
GET  /v1/annotation/assets/{asset_id}/content
```

上传字段：

| 字段 | 类型 | 必填 |
|---|---|---|
| `file` | JPEG/PNG binary | 是 |
| `source_id` | string | 否 |
| `group_id` | string | 是 |
| `metadata_json` | JSON object encoded as string | 否 |

### 9.3 自动标注 Job

```text
POST /v1/annotation/jobs
GET  /v1/annotation/jobs/{job_id}
GET  /v1/annotation/jobs/{job_id}/detections
GET  /v1/annotation/jobs/{job_id}/hazard-candidates
POST /v1/annotation/jobs/{job_id}/review-tasks
```

创建请求示例：

```json
{
  "asset_ids": ["ast_01J..."],
  "requested_categories": [
    "helmet_missing",
    "equipment_proximity"
  ],
  "pipeline_version": "grounded-qwen-v1",
  "options": {
    "generate_masks": false,
    "enrich_prompts": false,
    "prompt_count": 6,
    "stop_after": "hazard_rules"
  }
}
```

创建成功返回 HTTP 202。

`stop_after` 是可选调试/分阶段执行参数。当前独立 Worker 领取
`stop_after=grounding_dino` 或 `stop_after=hazard_rules` 的 Job；完整
Pipeline Worker 只领取省略 `stop_after` 的 Job。设置上述任一停止点时必须
同时令 `generate_masks=false`、`enrich_prompts=false`；省略停止点时两项
必须都为 `true`。

Job 结束后，前端通过 `detections` 接口获取检测框。可传
`asset_id` 查询参数只读取 Job 中某一张图片；不传则按 Job 内图片顺序返回
全部检测结果。每项包含 `asset_id`、检测实体、原图绝对像素
`box_xyxy`、`box_score`、`phrase_score`、模型元数据和生成时间。Job 或指定
图片不存在时返回 HTTP 404；尚未生成检测结果时 `items` 为空且 `total=0`。

`hazard-candidates` 接口返回由检测框推导的可解释候选。每项包含具体
`category`、被建议分割的 `target_entity` 和 `box_xyxy`、参与推导的
`target_detection_ids`、候选 `confidence`、`rule_id`、`rule_version`、
`evidence` 与 `metadata`。缺少安全帽、背心、安全带、护栏等规则使用
“未匹配到对应检测框”作为弱负证据，统一标记
`requires_visual_verification=true`；它们不能绕过千问视觉复核或人工审核
直接成为最终标签。`confidence` 仅是检测分数与空间关系组成的规则排序分，
不是经过校准的事故概率或标签正确率。

规则 v1 的主要映射如下：

| 类别 | 触发条件 | 候选目标 |
|---|---|---|
| `helmet_missing` / `no_helmet` | 人员头部区域未匹配安全帽框 | 人员框 |
| `no_jacket` | 人员躯干区域未匹配反光背心框 | 人员框 |
| `harness_missing` | 人员躯干区域未匹配安全带框 | 人员框 |
| `equipment_proximity` | 人员与施工设备框距离低于图像尺度阈值 | 人员框 |
| `opening_unprotected` | 洞口附近未匹配盖板、围挡或护栏 | 洞口框 |
| `guardrail_missing` | 临边/洞口附近未匹配护栏 | 临边或洞口框 |
| `poor_housekeeping` | 可见垃圾，或材料与通道相交 | 垃圾或材料框 |

`safe` 不通过框存在自动推导。`unsafe` 只包装已经命中的具体空间关系或现场
整理规则，并在 `metadata.derived_category` 中保留原始具体类别。

`review-tasks` 只允许在 `hazard_rules` 成功且 Job 已进入终态后调用。它将每个
候选幂等映射为一个人工 Task，并保留 `source_hazard`、规则版本和检测来源。
初始 Task 没有 polygon 和 Prompt，因此不能直接 submit；重复调用返回相同
Task ID，不产生重复数据。首次构建完成后，Job 的 `build_review_tasks` 阶段
记为 `succeeded`，并更新 `progress.generated_tasks`，但不改写 Job 原终态和
完成时间。

### 9.4 Task 查询和编辑

```text
GET  /v1/annotation/tasks
GET  /v1/annotation/tasks/{task_id}
GET  /v1/annotation/tasks/{task_id}/versions
GET  /v1/annotation/tasks/{task_id}/reviews
PUT  /v1/annotation/tasks/{task_id}/draft
POST /v1/annotation/tasks/{task_id}/submit
POST /v1/annotation/tasks/{task_id}/review
```

列表接口支持：

```text
status
category
group_id
job_id
annotator_id
reviewer_id
bad_case_type
limit
cursor
```

这些接口已在当前后端实现。草稿保存允许不完整标注；`submit` 和
`review(decision=accept)` 会在存储事务内重新执行第 11 节自动验收，避免
HTTP 之外的内部调用绕过非空 mask、polygon 和 3+2+1 Prompt 约束。
`cursor` 是后端生成的不透明值，前端只能原样传回。

`versions` 返回 generated、draft、submit、review、freeze 的不可变快照；
`reviews` 返回二级审核决策、主要 bad case、评论和对应 Task 版本。前端可用
这两个接口展示完整审计时间线。

### 9.5 模型候选重生成

当前可调用：

```text
POST /v1/annotation/tasks/{task_id}/mask-candidates
POST /v1/annotation/tasks/{task_id}/prompt-enrichments
GET  /v1/annotation/operations/{operation_id}
```

两个创建接口都返回 HTTP 202 和 `operation_id`，新结果作为候选保存，不直接
覆盖 `annotation`。前端通过 Operation 查询接口轮询，直到状态变为
`succeeded` 或 `failed`。

SAM 请求示例：

```http
POST /v1/annotation/tasks/tsk_01J.../mask-candidates
Content-Type: application/json
X-API-Key: <token>

{
  "expected_version": 1,
  "box_xyxy": [120, 80, 460, 720]
}
```

SAM 成功后，Operation `result` 包含：

```json
{
  "box_xyxy": [120, 80, 460, 720],
  "predicted_iou": 0.93,
  "mask_area_pixels": 158420,
  "shapes": [
    {
      "shape_id": "sam-target-1",
      "label": "target",
      "shape_type": "polygon",
      "points": [[121, 82], [458, 83], [456, 718], [123, 716]]
    }
  ],
  "artifacts": {
    "mask": "/v1/annotation/tasks/tsk_01J.../artifacts/mask",
    "mask_overlay": "/v1/annotation/tasks/tsk_01J.../artifacts/mask-overlay",
    "crop": "/v1/annotation/tasks/tsk_01J.../artifacts/crop"
  },
  "provenance": {
    "sam_version": "sam-vit-h-4b8939"
  }
}
```

Prompt 请求示例：

```http
POST /v1/annotation/tasks/tsk_01J.../prompt-enrichments
Content-Type: application/json
X-API-Key: <token>

{"expected_version": 1}
```

接受响应：

```json
{
  "operation_id": "op_01J...",
  "status": "queued",
  "created_at": "2026-07-30T02:00:00Z"
}
```

Operation 状态：

```text
queued
running
succeeded
failed
```

当前可能出现的 Operation 类型：

```text
mask_candidate
prompt_enrichment
```

Prompt Operation 成功后，`result` 直接返回视觉事实、6 条候选 Prompt 和
provenance；失败原因通过 `error` 返回。示例：

```json
{
  "operation_id": "op_01J...",
  "operation_type": "prompt_enrichment",
  "task_id": "tsk_01J...",
  "task_version": 1,
  "status": "succeeded",
  "result": {
    "facts": {
      "target_object": "画面中央的一名作业人员",
      "instance_count": 1,
      "visual_anchor": ["位于画面中央"],
      "mask_granularity": "人员整体",
      "visible_facts": ["人员头部未见安全帽"],
      "risk_semantics": "头部防护缺失"
    },
    "prompts": [
      {"prompt_id": "visual-1", "type": "visual", "text": "..."},
      {"prompt_id": "visual-2", "type": "visual", "text": "..."},
      {"prompt_id": "visual-3", "type": "visual", "text": "..."},
      {"prompt_id": "risk-1", "type": "risk", "text": "..."},
      {"prompt_id": "risk-2", "type": "risk", "text": "..."},
      {"prompt_id": "agent-1", "type": "agent", "text": "..."}
    ],
    "provenance": {
      "qwen_provider": "vllm-openai-compatible",
      "qwen_model": "qwen25vl",
      "qwen_facts_prompt_version": "construction-visible-facts-v1",
      "qwen_enrichment_prompt_version": "construction-prompts-3-2-1-v1"
    }
  },
  "error": null,
  "created_at": "2026-07-30T02:00:00Z",
  "started_at": "2026-07-30T02:00:01Z",
  "completed_at": "2026-07-30T02:00:04Z"
}
```

Operation 完成不会自动增加 Task 的 `version`。Spring 必须读取最新 Task，
将 SAM `shapes`、Qwen facts 和 prompts 合并进完整 `annotation`，再以当前
`expected_version` 调用 `PUT /tasks/{task_id}/draft`。若 Task 版本已变化，
应放弃旧候选或重新生成，不能覆盖其他用户修改。

Prompt 请求要求 Task 已经保存 `mask-overlay` 或 `mask` artifact，因此分阶段
调用必须先等待 SAM Operation 成功。独立 Qwen Worker 使用原图、mask 叠加图
和可选 crop 调用 Qwen2.5-VL，不在 API 进程加载模型。

Qwen 成功结果包含视觉事实、3 条 visual、2 条 risk、1 条 agent Prompt 和
模型 provenance。结果不会自动修改 Task；调用方确认后通过 draft 接口显式
写回。

Qwen Worker 直接运行在服务器宿主机时使用：

```env
ANNOTATION_QWEN_BASE_URL=http://127.0.0.1:18000/qwen25/v1
ANNOTATION_QWEN_MODEL=qwen25vl
```

Qwen 容器未运行时，Operation 会失败并返回稳定错误；API、人工审核和 Release
接口仍可正常运行。

### 9.6 Release

```text
POST /v1/annotation/releases
GET  /v1/annotation/releases/{release_id}
GET  /v1/annotation/releases/{release_id}/manifest
GET  /v1/annotation/releases/{release_id}/archive
```

只允许从 `accepted` task 构建 release。构建时按照 `group_id` 完成
train、val、golden 划分。创建接口返回 HTTP 202；纯 CPU Release Worker
使用租约领取任务，失败后记录可见错误，异常退出且租约过期后允许其他 Worker
恢复。`manifest` 和 `archive` 在状态为 `succeeded` 前返回 409。

## 10. 审核决策

一级提交：

```json
{
  "expected_version": 4,
  "annotator_id": "user-1001",
  "primary_result": "prompt_rewritten",
  "comment": "确认目标为靠近挖掘机的一名人员"
}
```

二级审核决策：

```text
accept
request_changes
needs_expert
reject
```

审核请求示例：

```json
{
  "expected_version": 5,
  "reviewer_id": "reviewer-01",
  "decision": "request_changes",
  "primary_result": "mask_overflow",
  "comment": "mask包含了人员右侧设备区域"
}
```

`accept` 必须通过完整自动验收。`request_changes`、`needs_expert` 和 `reject`
必须提供明确的 `primary_result`，建议提供评论。

## 11. 自动验收

submit、accept 和 release 构建前至少检查：

- 图片存在且可解析；
- polygon 合法、非零面积且未越界；
- `target` mask 非空；
- Prompt 为非空字符串且不重复；
- shape ID 与 Prompt ID 各自在样本内唯一；
- Prompt 满足 3+2+1；
- 实例数量、对象和粒度保持一致；
- 风险语义有可见事实支持；
- `safe/unsafe` 已具体化；
- accepted task 未跨 `group_id` 泄漏；
- release 中 jpg/json 成对；
- 导出 JSON 可由 `utils/data_processing.py::get_mask_from_json` 读取。

## 12. Release 输出

成功 release 至少包含：

```text
ReasonSegGroundedV1/
├── train/
├── val/
├── golden/
├── annotation_manifest.jsonl
├── build_summary.json
└── dataset_card.md
```

ReasonSeg JSON 示例：

```json
{
  "shapes": [
    {
      "label": "target",
      "points": [[121, 46], [154, 42], [309, 468]]
    }
  ],
  "text": [
    "分割画面中央靠近黄色挖掘机的一名作业人员。"
  ],
  "is_sentence": true,
  "source": {
    "sample_id": "tsk_01J...",
    "sample_key": "equipment_proximity",
    "group_id": "site01_video03"
  }
}
```

审核记录、Prompt 类型和模型调用详情保存在
`annotation_manifest.jsonl`，不依赖 ReasonSeg 训练读取器解释这些管理字段。
ZIP 内成员名、成员顺序和时间戳固定；相同 Task 快照与 split policy 产生相同
归档哈希。外部 `manifest.json` 记录每个样本和成员文件的 SHA256。

## 13. 错误协议

所有 JSON 错误使用：

```json
{
  "request_id": "req_...",
  "code": "annotation_validation_failed",
  "message": "annotation payload is invalid",
  "details": [
    {
      "field": "annotation.prompts",
      "reason": "expected 3 visual prompts, got 2"
    }
  ]
}
```

主要状态码：

| HTTP | 含义 |
|---|---|
| 200 | 查询或更新成功 |
| 201 | asset 创建成功 |
| 202 | 异步任务已受理 |
| 400 | 无法解释的业务请求 |
| 401 | 未认证 |
| 403 | 无操作权限 |
| 404 | 资源不存在 |
| 409 | 版本冲突、幂等冲突或非法状态迁移 |
| 413 | 图片或请求体过大 |
| 415 | 不支持的图片或内容类型 |
| 422 | 字段或标注内容不合法 |
| 429 | 异步/GPU 队列已满 |
| 503 | 模型或下游千问服务不可用 |

稳定错误码包括：

```text
validation_error
annotation_validation_failed
unauthorized
forbidden
not_found
version_conflict
idempotency_conflict
invalid_state_transition
unsupported_media_type
request_too_large
queue_full
model_unavailable
downstream_unavailable
internal_error
```

## 14. 模型可追溯性

每个 task 必须记录：

```text
grounding_dino_version
grounding_dino_prompt_version
grounding_dino_thresholds
hazard_rule_version
sam_version
sam_qc_version
qwen_provider
qwen_model
qwen_facts_prompt_version
qwen_enrichment_prompt_version
pipeline_version
```

模型路径、服务器路径、API Key 和权重文件不得写入 task、日志、API 响应或
仓库。

## 15. 第一版非目标

v1 首版不包含：

- WebSocket 协同编辑；
- 多人实时光标；
- 前端直传服务器文件路径；
- 训练、推理和评估启动接口；
- 自动接受模型标注；
- 根据模型 IoU 自动删除标签；
- release 原地覆盖；
- 在 API 进程中加载 GroundingDINO、SAM 或千问权重。

以上能力如有需要，在不破坏 v1 契约的前提下单独设计。
