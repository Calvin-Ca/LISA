# 标注后端 Docker 部署（无反向代理）

本方案在远程 Linux 服务器直接暴露 FastAPI 端口，不依赖 Nginx。默认只启动
不加载模型的 `api` 和 `release-worker`；GroundingDINO 位于可选 `models`
profile，模型阶段暂缓时无需启动。

## 1. 准备

以下命令均为远程执行：

```bash
cd <LISA仓库目录>
cp annotation_service/docker/.env.example annotation_service/docker/.env
chmod 600 annotation_service/docker/.env
```

编辑 `.env`，至少设置：

- `ANNOTATION_API_KEY`：长随机密钥；
- `ANNOTATION_CORS_ORIGINS`：前端真实 Origin，例如
  `http://frontend.example.internal:8080`，不要填写 `*`；
- `ANNOTATION_STORAGE_HOST_PATH`：持久化目录；
- `ANNOTATION_CONTAINER_UID/GID`：该目录的所有者 UID/GID；
- `ANNOTATION_BIND_ADDRESS`：需要跨机器调用时用 `0.0.0.0`，仅本机调用时用
  `127.0.0.1`。

模型路径按当前约定保留在同一个 `.env` 里，后续 worker 复用：

- `GROUNDING_DINO_SOURCE_HOST_PATH`：构建镜像时使用的 `GroundingDINO`
  源码目录；镜像只复制其中的 `groundingdino/` Python 包，不复制权重；
- `GROUNDING_DINO_MODEL_HOST_PATH`：完整、已校验的 GroundingDINO
  MODEL_STORE 版本目录，运行时只读挂载；
- `ANNOTATION_SAM_CHECKPOINT`：预留的 `sam_vit_h_4b8939.pth` 服务器路径；
- `ANNOTATION_QWEN_BASE_URL`：Qwen2.5-VL 的 OpenAI 兼容 `/v1` 地址；
- `ANNOTATION_QWEN_MODEL`：vLLM 的实际 `served-model-name`。

当前这版 Docker compose 在构建阶段消费 GroundingDINO 源码路径，在运行
阶段消费 GroundingDINO MODEL_STORE 路径。SAM、Qwen Prompt 和完整 Pipeline
Worker 都已有 Python 入口，但按当前“先不构建新镜像”的约定尚未加入本
Compose。当前直接在宿主机运行 Qwen Worker 时使用：

```env
ANNOTATION_QWEN_BASE_URL=http://127.0.0.1:18000/qwen25/v1
ANNOTATION_QWEN_MODEL=qwen25vl
```

若以后把 Qwen Worker 加入 `model-gateway` Docker 网络，地址改为
`http://qwen25vl:8000/v1`。

创建持久化目录并确认权限：

```bash
sudo install -d -o <uid> -g <gid> -m 0750 <ANNOTATION_STORAGE_HOST_PATH>
```

真实路径、密钥、权重和 `.env` 不得提交到仓库。

## 2. 启动非模型服务

```bash
docker compose \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  config --quiet
docker compose \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  up -d --build api release-worker
```

默认端口是宿主机 `18001`：

```text
API:     http://<服务器地址>:18001
Swagger: http://<服务器地址>:18001/docs
```

若服务器防火墙启用，只向前端所在网段开放该 TCP 端口，不要向公网无条件开放。

当前 Compose 默认仍只启动 API、Release Worker，并可通过 `models` profile
启动分阶段 GroundingDINO Worker。SAM、Qwen 和完整 Pipeline Worker 需要按
主 README 的远程 Python 命令单独运行；这是部署方式限制，不是业务代码缺口。

## 3. 验收

```bash
docker compose \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  ps
curl -fsS \
  -H "X-API-Key: <ANNOTATION_API_KEY>" \
  http://127.0.0.1:18001/ready
docker compose \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  logs --tail=100 api release-worker
```

`/ready` 应返回 `{"status":"ready",...}`。随后可创建一个无匹配 Task 的测试
Release，并确认 Release Worker 将其置为 `failed` 且 API 返回可读错误；正式
验收应使用已通过二级审核的 Task，并下载 manifest 与 ZIP 检查 jpg/json 配对。

## 4. 更新与回滚

更新前先备份整个持久化目录，而不只是 SQLite：

```bash
docker compose \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  stop api release-worker
sudo cp -a <ANNOTATION_STORAGE_HOST_PATH> <备份目录>
docker compose \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  up -d --build api release-worker
```

数据库在 API 或 Worker 初始化时自动向前迁移。回滚旧代码前必须同时恢复更新前
的完整数据目录；不要让旧程序直接打开已迁移的新数据库。

## 5. 可选模型 Worker

恢复 GroundingDINO 后，在 `.env` 填写
`GROUNDING_DINO_SOURCE_HOST_PATH` 和 `GROUNDING_DINO_MODEL_HOST_PATH`，
先构建包含固定 GroundingDINO 源码的镜像，再执行：

```bash
docker compose \
  --profile models \
  --env-file annotation_service/docker/.env \
  -f annotation_service/docker/compose.yaml \
  up -d worker
```

GroundingDINO 源码固定进镜像；权重、模型配置和离线 BERT 保留在同一个
MODEL_STORE 制品中，以只读方式挂载到 `/models/groundingdino`。权重不写入
镜像或仓库。
