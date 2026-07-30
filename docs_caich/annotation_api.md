# GroundingDINO 自由检测 API（Spring 后端对接）

文档版本：`1.0.0`

更新时间：`2026-07-30`

## 0. 当前联调地址

Spring 后端与自动标注服务不在同一台主机。当前测试环境使用：

```text
Base URL: http://172.19.2.2:8008
```

其中 `172.19.2.2` 是自动标注服务器地址，`8008` 是自动标注 API 端口。
Spring 自身监听的 `18080` 是 Spring 对外提供业务接口的端口，不是自动标注
服务端口。

推荐在 Spring 的外部配置中声明：

```yaml
annotation:
  base-url: ${ANNOTATION_BASE_URL:http://172.19.2.2:8008}
  api-key: ${ANNOTATION_API_KEY}
```

当前测试 API Key：

```text
<TEST_API_KEY>
```

测试交接副本可以直接填写测试密钥；正式部署时应替换密钥，并改由 Spring
部署环境注入。

不要使用 `127.0.0.1` 或 `localhost`：它们会指向 Spring 所在主机，无法访问
另一台服务器上的自动标注服务。

## 1. 服务范围

该服务只提供：

```text
图片上传 -> 任意文本 Prompt -> GroundingDINO -> bbox JSON + 带框 PNG
```

不再提供类别白名单检测、隐患规则、SAM、Qwen、人工 Task 或数据集 Release。

所有业务接口携带：

```http
X-API-Key: <ANNOTATION_API_KEY>
```

也可以使用：

```http
Authorization: Bearer <ANNOTATION_API_KEY>
```

Spring 推荐统一使用 `X-API-Key`。

## 2. 通用约定

- JSON 使用 UTF-8。
- 时间为 ISO 8601 UTC。
- 响应中的 `content_url` 是相对路径，需要与 Base URL 拼接。
- `POST /assets`、`POST /jobs` 支持 `Idempotency-Key`。
- 幂等键长度为 8～128；相同键和相同请求返回首次结果，不同请求返回 409。
- Prompt 不做类别或关键词白名单限制。
- Prompt 必须非空，去除首尾空白后最长 2000 字符。
- 一个 Job 最多包含 500 个不重复的 Asset。
- bbox 坐标为原图绝对像素 `xyxy=[x1,y1,x2,y2]`。
- 所有响应都可能携带 `X-Request-ID`；Spring 应记录它以便排查服务端日志。
- API 是异步接口。创建 Job 返回 202 只表示已入队，不表示推理已经完成。

### API 一览

| 方法 | 路径 | 鉴权 | 说明 |
|---|---|---:|---|
| GET | `/health` | 否 | 进程存活检查 |
| GET | `/ready` | 是 | API 和持久化存储就绪检查 |
| POST | `/v1/annotation/assets` | 是 | 上传 JPEG/PNG |
| GET | `/v1/annotation/assets/{asset_id}` | 是 | 查询图片元数据 |
| GET | `/v1/annotation/assets/{asset_id}/content` | 是 | 下载原图 |
| POST | `/v1/annotation/jobs` | 是 | 创建自由文本检测任务 |
| GET | `/v1/annotation/jobs/{job_id}` | 是 | 查询任务状态 |
| GET | `/v1/annotation/jobs/{job_id}/detections` | 是 | 获取 bbox JSON |
| GET | `/v1/annotation/jobs/{job_id}/assets/{asset_id}/bbox-image` | 是 | 下载带框 PNG |

统一错误：

```json
{
  "request_id": "req_xxx",
  "code": "validation_error",
  "message": "request payload is invalid",
  "details": [
    {
      "field": "body.grounding_prompt",
      "reason": "value must not be blank"
    }
  ]
}
```

常见状态码：

| 状态码 | 含义 |
|---|---|
| 200 | 查询成功 |
| 201 | Asset 创建成功 |
| 202 | Job 已入队 |
| 401 | API Key 无效 |
| 404 | 资源或结果图不存在 |
| 409 | 幂等键冲突 |
| 413 | 请求或图片过大 |
| 415 | 图片格式不支持 |
| 422 | 参数校验失败 |
| 429 | Job 队列已满 |
| 503 | 存储未就绪 |

Spring 不应只依赖 HTTP 状态码判断异步任务结果，还要读取 Job 响应中的
`status` 和 `errors`。

## 3. 健康检查

### `GET /health`

不需要认证。

```json
{"status":"ok"}
```

### `GET /ready`

需要认证。HTTP 200 示例：

```json
{
  "status": "ready",
  "dependencies": {
    "api": "ready",
    "storage": "ready"
  }
}
```

## 4. 上传图片

### `POST /v1/annotation/assets`

请求：

```http
Content-Type: multipart/form-data
X-API-Key: <API_KEY>
Idempotency-Key: upload-image-0001
```

字段：

| 字段 | 必填 | 说明 |
|---|---:|---|
| `file` | 是 | JPEG 或 PNG |
| `group_id` | 是 | 来源分组，1～256 字符 |
| `source_id` | 否 | 上游业务 ID |
| `metadata_json` | 否 | JSON object 字符串 |

响应 HTTP 201：

```json
{
  "asset_id": "ast_xxx",
  "source_id": "spring-file-1001",
  "group_id": "manual-detection",
  "width": 1920,
  "height": 1080,
  "sha256": "<64位小写SHA256>",
  "media_type": "image/jpeg",
  "content_url": "/v1/annotation/assets/ast_xxx/content",
  "duplicate_of": null,
  "metadata": {},
  "created_at": "2026-07-30T10:00:00+00:00"
}
```

查询：

```text
GET /v1/annotation/assets/{asset_id}
GET /v1/annotation/assets/{asset_id}/content
```

## 5. 创建自由检测 Job

### `POST /v1/annotation/jobs`

请求：

```http
Content-Type: application/json
X-API-Key: <API_KEY>
Idempotency-Key: detection-job-0001
```

```json
{
  "asset_ids": ["ast_xxx"],
  "grounding_prompt": "the worker beside the blue excavator",
  "pipeline_version": "groundingdino-free-form-v1"
}
```

`pipeline_version` 可省略，默认：

```text
groundingdino-free-form-v1
```

任意中文、英文、标点和自然语言描述都可以提交，例如：

```json
{
  "asset_ids": ["ast_xxx"],
  "grounding_prompt": "找出画面右侧蓝色设备旁边的人员"
}
```

服务不再接受旧字段：

```json
{
  "requested_categories": ["helmet_missing"],
  "options": {}
}
```

携带旧字段会返回 HTTP 422。

响应 HTTP 202：

```json
{
  "job_id": "job_xxx",
  "status": "queued",
  "stage": null,
  "pipeline_version": "groundingdino-free-form-v1",
  "grounding_prompt": "the worker beside the blue excavator",
  "progress": {
    "total_assets": 1,
    "completed_assets": 0
  },
  "stages": {
    "grounding_dino": {
      "status": "pending",
      "started_at": null,
      "completed_at": null,
      "message": null
    }
  },
  "errors": [],
  "created_at": "2026-07-30T10:00:00+00:00",
  "started_at": null,
  "completed_at": null
}
```

## 6. 轮询 Job

### `GET /v1/annotation/jobs/{job_id}`

建议每 1～2 秒轮询，直到：

```text
succeeded
partial_failed
failed
cancelled
```

成功时：

```json
{
  "job_id": "job_xxx",
  "status": "succeeded",
  "stage": "grounding_dino",
  "pipeline_version": "groundingdino-free-form-v1",
  "grounding_prompt": "the worker beside the blue excavator",
  "progress": {
    "total_assets": 1,
    "completed_assets": 1
  },
  "stages": {
    "grounding_dino": {
      "status": "succeeded",
      "started_at": "2026-07-30T10:00:01+00:00",
      "completed_at": "2026-07-30T10:00:02+00:00",
      "message": "1/1 assets detected; 0 failed"
    }
  },
  "errors": [],
  "created_at": "2026-07-30T10:00:00+00:00",
  "started_at": "2026-07-30T10:00:01+00:00",
  "completed_at": "2026-07-30T10:00:02+00:00"
}
```

## 7. 获取 bbox JSON

### `GET /v1/annotation/jobs/{job_id}/detections`

可选查询参数：

```text
asset_id=ast_xxx
```

响应：

```json
{
  "job_id": "job_xxx",
  "items": [
    {
      "detection_id": "det_xxx",
      "asset_id": "ast_xxx",
      "entity": "person",
      "box_xyxy": [120.5, 80.0, 460.25, 720.75],
      "box_score": 0.91,
      "phrase_score": 0.91,
      "metadata": {
        "caption": "the worker beside the blue excavator .",
        "grounding_prompt": "the worker beside the blue excavator",
        "model_version": "groundingdino-swint-ogc",
        "prompt_version": "free-form-v1",
        "box_threshold": 0.35,
        "text_threshold": 0.25,
        "ordinal": 0
      },
      "created_at": "2026-07-30T10:00:02+00:00"
    }
  ],
  "total": 1
}
```

无目标时返回 HTTP 200，`items=[]`、`total=0`。

## 8. 获取带框图片

### `GET /v1/annotation/jobs/{job_id}/assets/{asset_id}/bbox-image`

成功响应：

```http
Content-Type: image/png
ETag: "<SHA256>"
Cache-Control: private, no-cache
```

即使未检测到 bbox，成功完成的 Asset 也会生成 PNG；此时图片内容为未画框的
原图 PNG。Job 尚未完成、Asset 不属于 Job 或生成失败时返回 HTTP 404。

Spring 示例：

```java
byte[] image = webClient.get()
    .uri(baseUrl + "/v1/annotation/jobs/{jobId}/assets/{assetId}/bbox-image",
        jobId, assetId)
    .header("X-API-Key", apiKey)
    .retrieve()
    .bodyToMono(byte[].class)
    .block();
```

## 9. 推荐调用顺序

```text
1. POST /assets
2. 保存 asset_id
3. POST /jobs，传入 grounding_prompt
4. 保存 job_id
5. GET /jobs/{job_id} 轮询终态
6. GET /jobs/{job_id}/detections
7. GET /jobs/{job_id}/assets/{asset_id}/bbox-image
```

不要在 Job 未进入成功或部分成功终态前将 bbox 图片 404 当成永久失败。

## 10. Spring 接入建议

### 10.1 WebClient 配置

下面仅展示连接配置，DTO 字段应按本文各接口的 JSON 定义创建：

```java
@Bean
WebClient annotationWebClient(
        WebClient.Builder builder,
        @Value("${annotation.base-url}") String baseUrl,
        @Value("${annotation.api-key}") String apiKey) {
    return builder
        .baseUrl(baseUrl)
        .defaultHeader("X-API-Key", apiKey)
        .build();
}
```

上传图片必须使用 `multipart/form-data`，创建 Job 使用 `application/json`。
下载带框图片时可直接接收 `byte[]`，并向前端返回 `image/png`。

### 10.2 轮询策略

- 建议每 1～2 秒查询一次 Job。
- `queued`、`running`：继续轮询。
- `succeeded`：读取 detections 和带框图片。
- `partial_failed`：读取成功结果，同时记录 `errors`。
- `failed`、`cancelled`：停止轮询并返回明确业务错误。
- 建议设置总超时，不要让单个 HTTP 请求同步阻塞到模型推理结束。

### 10.3 幂等键

上传和创建 Job 建议分别生成不同的幂等键，例如：

```text
asset-<Spring业务请求ID>
job-<Spring业务请求ID>
```

重试同一次业务请求时复用原 Key 和原请求内容。改变图片、Prompt 或 Asset
列表时必须生成新 Key，否则会返回 HTTP 409。

### 10.4 CORS

Spring 到本服务属于服务端到服务端请求，不受浏览器 CORS 限制。推荐由浏览器
只访问 Spring 的 `18080`，再由 Spring 调用本服务；不要把自动标注 API Key
下发到浏览器。

## 11. 联调检查清单

交给 Spring 后端同事的必要信息：

- 本文档。
- `ANNOTATION_BASE_URL=http://172.19.2.2:8008`。
- 通过安全渠道提供的 API Key。
- 一张用于联调的 JPEG 或 PNG。

后端开始联调前可执行：

```bash
curl -fsS http://172.19.2.2:8008/health
curl -fsS \
  -H "X-API-Key: ${ANNOTATION_API_KEY}" \
  http://172.19.2.2:8008/ready
```

两项均成功后即可按第 9 节调用。当前契约只包含 GroundingDINO 自由检测、
bbox JSON 和带框 PNG，不包含 SAM mask、Qwen Prompt 生成或数据集发布接口。
