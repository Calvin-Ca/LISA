# lisa13b-clean030-container-smoke-v1

## 状态

已准备，等待远程共享 GPU 执行。

## 背景

裸进程生产 API 已完成精度复评、三轮 shared-GPU 性能基线、超时串行化、
异常输入与队列保护、客户端并发 1/2/4 和连续 100 请求验证。本实验验证相同
服务封装进生产 Docker 镜像后没有引入环境、权限、挂载、GPU、健康检查或
重启回归。

目标 A100 40GB 上长期运行不可停止的 `bge-m3` vLLM pooling 服务。容器
测试保持它在线，只启动一个 LISA 容器，并在结束时确认实验前已有的
`VLLM::EngineCore` PID 没有消失。

## 模型、输入与容器配置

- LISA 版本：`lisa13b-clean030-v1`
- 冻结制品：`artifacts/lisa-safety-seg/lisa13b-clean030-v1`
- 模型目录：容器内 `/models/lisa13b-clean030-v1/merged_hf`
- CLIP：完整 Hugging Face 模型缓存只读挂载到 `/models/clip-cache`
- CLIP snapshot：运行时自动从服务器缓存定位
- 精度：bf16，未启用 8bit/4bit
- Dockerfile：`production/Dockerfile`
- 镜像标签：`lisa-safety-seg:lisa13b-clean030-v1-0092463`
- 测试容器名：`lisa-clean030-container-smoke-v1`
- 宿主机地址：`127.0.0.1:8004`
- 容器服务端口：8000
- Uvicorn worker：1
- GPU worker：1
- 最大等待队列：8
- 排队超时：30 秒
- 推理超时：120 秒
- 共享内存：8 GiB
- 容器运行用户：`lisa`，预期 UID 10001
- 模型和 CLIP 挂载：只读
- Hugging Face 和 Transformers：offline
- API Key：实验工具运行时随机生成，不写入输出或仓库
- 固定图片：
  `dataset/reason_seg/ReasonSeg/val/val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing.jpg`
- 固定 Prompt：`标出未按规定佩戴安全帽的作业人员。`

CLIP 不能只挂载 snapshot 子目录。Hugging Face snapshot 中可能存在指向
`blobs/` 的相对符号链接，因此脚本挂载完整
`models--openai--clip-vit-large-patch14` 目录，同时在容器内引用确定的
snapshot。

Docker 构建只复制生产运行所需的 `production/`、`model/` 和 `utils/`，
不把数据集、训练输出、模型制品、Git 元数据或本地协作文档复制进镜像。
服务器现有的 `production/.env` 由 Docker ignore 明确排除；随机 API Key
通过权限 0600 的临时 env 文件传入，容器启动后立即删除该文件。

## 实验流程

1. 检查 Docker、NVIDIA Runtime、端口、冻结制品和完整 CLIP 缓存。
2. 重新执行冻结模型 `SHA256SUMS` 校验。
3. 构建固定标签的生产镜像并保存完整构建日志。
4. 在镜像内以 CPU 模式运行当前全部 42 项纯逻辑测试。
5. 记录实验前共享 GPU 计算进程和显存。
6. 启动 200 ms 间隔的 GPU 显存、利用率和温度采样。
7. 使用 GPU 0 启动一个容器，等待 Docker healthcheck 和 `/ready`。
8. 检查镜像配置用户、容器实际 UID/GID，以及模型、CLIP 挂载只读属性。
9. 执行第一轮真实 GPU 冒烟，验证响应和 PNG mask。
10. 正常重启同一容器，再次等待 healthy/ready 并执行第二轮冒烟。
11. 检查两个进程周期均只加载一次模型，请求成功且 GPU 最大在途数为 1。
12. 正常停止容器，记录退出码、日志、峰值显存和停止后显存。
13. 确认原有共享 GPU PID 仍存在，删除测试容器但保留镜像和实验输出。

容器失败时工具也会尽力保存日志、停止并删除测试容器，避免随机 API Key
长期留在 Docker 元数据中。工具不会删除已有镜像、数据或模型。

## 准入条件

全部检查必须通过：

- Docker 镜像构建成功。
- 镜像内全部纯逻辑测试成功，数量不少于 42。
- 镜像配置用户为 `lisa`，运行 UID 为 10001，GID 非 root。
- 模型和 CLIP 两个挂载均为只读。
- 镜像内不存在 `production/.env`、协作文档、数据集、实验输出、训练输出或
  模型制品副本。
- 首次启动和重启后均为 Docker healthy 且 `/ready=ready`。
- 两次冒烟均返回 HTTP 200、正确模型版本和至少一个有效 PNG mask。
- 两个进程周期均只加载一次模型。
- 运行时请求、成功请求和 GPU 成功数均为 2。
- GPU 失败、排队超时、推理超时、队列拒绝、取消、unexpected error 和
  CUDA OOM 均为 0。
- `gpu_inference_in_flight_max=1`，两轮结束时均为 0。
- GPU 总峰值不超过 36,864 MiB，峰值剩余不少于 4,096 MiB。
- 容器停止后整卡显存相对实验前漂移绝对值不超过 500 MiB。
- 实验前已有共享 GPU 计算进程 PID 没有消失。
- 容器日志没有 CUDA OOM、Traceback、ERROR、API Key 或宿主机私有路径。
- 容器正常停止，退出码为 0。

## 执行命令

远程 Linux GPU 服务器执行：

```bash
bash exp/runs/lisa13b-clean030-container-smoke-v1/command.sh
```

脚本自包含模型、CLIP 发现、镜像、容器、端口、输入、队列和准入参数。不依赖
用户提前 `export` 环境变量，不读取或修改正式 `production/.env`，不停止
`bge-m3`。

首次构建需要下载基础 CUDA 镜像和 Python wheel，耗时取决于服务器网络。
后续重复构建可复用 Docker layer。

## 预期产物

```text
exp/runs/lisa13b-clean030-container-smoke-v1/outputs/
├── runtime_config.json
├── build.log
├── unit_tests.log
├── gpu_metrics.csv
├── server.log
├── smoke-request-metadata.json
├── smoke-cycle-1-response.json
├── smoke-cycle-1-mask-0.png
├── smoke-cycle-2-response.json
├── smoke-cycle-2-mask-0.png
├── cycle-1-metrics.json
├── cycle-2-metrics.json
├── container_inspect.json
├── summary.json
└── summary.md
```

结构化输出不保存图片 Base64、API Key、Docker 环境变量、宿主机挂载源路径或
服务器私有绝对路径。

## 结果

等待远程执行。

## 局限

- 使用一个固定图片和 Prompt，只验证容器链路，不替代 86 样本精度回归和
  并发压测。
- 只确认 `bge-m3` GPU 进程存活，未接入它的业务健康和延迟指标。
- 不主动制造 CUDA OOM，避免影响共享 GPU 上的常驻服务。
- 当前 Docker 依赖中仍有少量间接或未固定版本，完整依赖锁定仍是独立待办。
- 本实验验证单机 Docker，不代表 Kubernetes、反向代理或网关配置已经完成。
