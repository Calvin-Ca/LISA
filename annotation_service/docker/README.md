# GroundingDINO 自由检测 Docker 部署

当前 Compose 只包含 API 和自由检测 GroundingDINO Worker，不包含 SAM、
Qwen、Task 或 Release 服务。用户当前采用宿主机 Python 联调时无需构建镜像；
本文件用于后续容器化。

以下命令均在远程 Linux 服务器执行。

## 配置

```bash
cd <LISA仓库目录>
cp annotation_service/docker/.env.example annotation_service/docker/.env
chmod 600 annotation_service/docker/.env
```

至少填写：

- `ANNOTATION_API_KEY`
- `ANNOTATION_CORS_ORIGINS`
- `ANNOTATION_STORAGE_HOST_PATH`
- `GROUNDING_DINO_SOURCE_HOST_PATH`
- `GROUNDING_DINO_MODEL_HOST_PATH`
- `ANNOTATION_CONTAINER_UID/GID`

`GROUNDING_DINO_MODEL_HOST_PATH` 指向：

```text
<MODEL_STORE>/groundingdino/swint-ogc/upstream-v1
```

该制品目录应包含：

```text
GroundingDINO_SwinT_OGC.py
groundingdino_swint_ogc.pth
text_encoder/bert-base-uncased/
```

真实路径、密钥和权重不得提交。

## 启动

```bash
docker compose \
  --profile models \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  config --quiet
docker compose \
  --profile models \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  up -d --build api worker
```

默认地址：

```text
API:     http://<服务器地址>:18001
Swagger: http://<服务器地址>:18001/docs
```

检查：

```bash
docker compose \
  --profile models \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  ps
docker compose \
  --profile models \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  logs --tail=100 api worker
curl -fsS -H "X-API-Key: <API_KEY>" \
  http://127.0.0.1:18001/ready
```

API 和 Worker 必须挂载同一个完整持久化目录。升级前应停止两个容器并备份整个
目录；schema v7 会自动迁移，旧代码回滚时必须同时恢复迁移前备份。
