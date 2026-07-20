# LISA 生产化开发 TODO

## 目标

将施工安全 ReasonSeg 微调模型封装为可版本化、可验证、可监控、可灰度和可回滚的生产推理服务。

当前推荐：

- 生产候选制品：`artifacts/lisa-safety-seg/lisa13b-clean030-v1/merged_hf`
- 生产候选来源：`runs/lisa13b-clean030-lora-v1/merged_hf`
- 影子候选来源：`runs/lisa13b-relabel303-lora-v1/merged_hf`，尚未冻结为生产制品
- 当前对比集：`ReasonSeg|val`，86 个样本
- 当前生产候选指标：gIoU `0.4494`、cIoU `0.3858`、Dice `0.5156`

Clean030 LoRA 当前总体指标优于 Relabel303 LoRA。冻结制品 SHA-256、生产精度复评、API 稳定性、并发压测和真实容器验收均已通过。独立 golden test 与外部模型/镜像发布当前暂缓；后续优先完成监控告警、灰度和回滚演练。

## P0：模型版本与制品冻结

- [x] 确认首个生产候选使用 Clean030 LoRA，不直接用 Relabel303 替换。
- [x] 为生产模型定义固定版本，例如 `lisa13b-clean030-v1`。
- [x] 将合并后的 Hugging Face 模型目录复制到独立制品目录，不直接引用可变化的训练目录。
- [x] 确认部署对象是 `merged_hf/`，不是 `ckpt_model/` 或单独的 `pytorch_model.bin`。
- [x] 确认 CLIP vision tower `clip-vit-large-patch14` 可在服务器本地读取。
- [x] 提供生成模型 SHA-256 文件清单的制品冻结工具。
- [x] 制品 manifest 自动保存 Git commit、文件大小和校验值。
- [x] 提供厂商无关的正式发布脚本：绑定已通过 GPU 验收的镜像 ID、远端镜像 digest、模型 SHA-256 和验收报告，不重新构建未验收镜像。
- [x] 提供 rclone immutable 上传与远端 checksum 复核流程，凭据和目标地址不写入仓库。
- [x] 已记录 Python、CUDA、驱动、PyTorch、Transformers、DeepSpeed、bitsandbytes、FastAPI、Uvicorn 和 OpenCV 版本。
- [ ] （暂缓，不作为当前单机部署阻塞项）将模型权重上传到对象存储、内部模型仓库或制品平台，不提交到 Git。
- [x] 定义制品目录结构：

```text
artifacts/lisa-safety-seg/
└── lisa13b-clean030-v1/
    ├── merged_hf/
    ├── manifest.json
    ├── SHA256SUMS
    └── MODEL_CARD.md
```

### 外部制品发布目的与暂缓记录

- 决策日期：2026-07-20
- 当前决定：暂不安装或配置 rclone，不接入镜像仓库，不上传模型和镜像。
- 当前运行方式：模型制品和已验收 Docker 镜像继续保留在单台 GPU 服务器。
- 对当前推理的影响：不影响本机启动、精度、延迟、显存或容器运行。
- 当前风险：服务器或磁盘故障、误删模型、清理 Docker 镜像时缺少独立恢复副本；暂不支持从新服务器自动恢复、横向扩容和严格版本回滚。
- 未来触发条件：进入正式多机生产、需要灾备、迁移服务器、接入发布平台或要求一键回滚时再实施。

未来实施步骤：

1. 准备私有 Docker 镜像仓库和独立模型存储。
2. 在仓库外配置镜像仓库认证和对象存储/rclone 凭据。
3. 设置非敏感的镜像 repository 与模型 remote 目标。
4. 校验冻结模型 `SHA256SUMS`，确认容器验收报告为 `PASS`。
5. 将已验收镜像按模型版本和 Git commit 打不可变标签并推送。
6. 记录 registry digest，生成模型、镜像和验收证据联合 release manifest。
7. 使用 immutable 模式上传约 27 GB 模型，并执行远端 checksum 对比。
8. 从远端恢复到另一目录或新服务器，重新执行 SHA-256 和容器冒烟。
9. 保存恢复、灰度和回滚所需的版本对应关系与运维文档。

远程执行，校验冻结制品：

```bash
cd artifacts/lisa-safety-seg/lisa13b-clean030-v1 \
  && sha256sum --check SHA256SUMS \
  && cd ../../..
```

## P0：模型产物检查

- [x] 检查 `merged_hf/config.json` 存在。
- [x] tokenizer 已从冻结制品成功加载，`tokenizer_config.json` 等运行所需文件可用。
- [x] 模型权重或分片已从冻结制品成功加载，并完成 86 个样本的完整 benchmark。
- [x] 模型配置的 `vision_tower`、`mm_vision_tower` 与运行时 CLIP 一致，`[SEG]` 为独立 token 且不是 unknown token。
- [x] 检查生产机器能读取模型和 CLIP vision tower。
- [x] 已在冻结制品目录执行 `sha256sum --check --quiet SHA256SUMS`，全部文件校验通过。
- [x] 服务使用 `local_files_only=True`，禁止启动时从公网自动下载权重。

远程执行，列出模型文件：

```bash
find artifacts/lisa-safety-seg/lisa13b-clean030-v1/merged_hf \
  -maxdepth 1 \
  -type f \
  -printf '%f\n' \
  | sort
```

### 冻结制品完整性校验记录

- 日期：2026-07-19
- 制品版本：`lisa13b-clean030-v1`
- 制品目录：`artifacts/lisa-safety-seg/lisa13b-clean030-v1`
- 校验文件：`SHA256SUMS`
- 校验命令：`sha256sum --check --quiet SHA256SUMS`
- 校验结果：通过，冻结制品全部文件完整

### 生产运行环境与模型配置记录

- 日期：2026-07-20
- Python：3.10
- GPU：NVIDIA A100-PCIE-40GB，40,960 MiB
- 驱动：`580.159.03`
- PyTorch：`2.1.0+cu121`
- Transformers：`4.31.0`
- DeepSpeed：`0.12.6`
- bitsandbytes：`0.41.1`
- FastAPI：`0.100.1`
- Uvicorn：`0.23.2`
- OpenCV：`4.8.0.74`
- 模型类型：`llava`
- CLIP：`openai/clip-vit-large-patch14`，snapshot `32bd64288804d66eefd0ccbe215aa642df71cc41`
- `vision_tower`、`mm_vision_tower` 与运行时 CLIP：一致
- `[SEG]` token ID：`32000`
- unknown token ID：`0`
- 结论：模型配置、CLIP 配置、预处理配置和特殊 token 检查通过

## P0：生产同构环境冒烟测试

- [x] 在远程 Linux A100 40GB GPU 机器准备并核对 CUDA 和 Python 环境。
- [x] 使用生产 FastAPI 后端验证合并模型能成功加载。
- [x] 使用固定图片和固定 Prompt 作为 smoke case。
- [x] 检查模型文本输出包含有效分割响应。
- [x] 检查输出 mask 非空、尺寸正确且像素值有效。
- [x] 已通过三轮 shared-GPU 实验记录模型ready、首次推理、预热和正式请求延迟。
- [x] 已记录共享基线、模型加载、预热、总峰值、剩余显存和预热后显存漂移。
- [x] 已在容器验收输出中保留 smoke case 的输入元数据、准入预期、响应、mask 和实际指标。

### 冒烟测试记录

- 日期：2026-07-16
- 服务器环境：Python 3.10、PyTorch `2.1.0+cu121`、Transformers `4.31.0`
- API依赖：FastAPI `0.100.1`、Uvicorn `0.23.2`、OpenCV `4.8.0`
- GPU：NVIDIA A100-PCIE-40GB，单GPU
- 模型：`lisa13b-clean030-v1`
- 模型路径：`runs/lisa13b-clean030-lora-v1/merged_hf`（首次 API 冒烟；冻结制品随后已通过完整 benchmark）
- 模型大小：约 27GB
- 推理精度：bf16
- 并发：1
- 服务端口：`127.0.0.1:8001`
- smoke Prompt：`标出未按规定佩戴安全帽的作业人员。`
- 返回文本：`Sure, [SEG] .`
- API首次实测延迟：`942.156 ms`
- 输入/输出尺寸：`512 × 512`
- mask数量：1
- mask像素范围：`0～255`
- mask前景像素：11,012
- mask前景比例：`0.042007`
- `/metrics`：模型只加载1次，请求1次且成功1次，返回mask 1张
- 人工叠加图检查：通过

远程执行，bf16 模型链路诊断（非 API 冒烟）：

```bash
CUDA_VISIBLE_DEVICES=0 \
python chat.py \
  --version ./artifacts/lisa-safety-seg/lisa13b-clean030-v1/merged_hf \
  --vision-tower ./clip-vit-large-patch14 \
  --precision bf16 \
  --vis_save_path ./vis_output/clean030-production-smoke
```

如果生产计划使用 8bit，远程执行：

```bash
CUDA_VISIBLE_DEVICES=0 \
python chat.py \
  --version ./artifacts/lisa-safety-seg/lisa13b-clean030-v1/merged_hf \
  --vision-tower ./clip-vit-large-patch14 \
  --precision fp16 \
  --load_in_8bit \
  --vis_save_path ./vis_output/clean030-production-8bit-smoke
```

## P0：生产精度复评

- [x] 首轮生产候选明确使用 bf16；量化方案如后续启用需重新评估。
- [x] 使用最终生产精度重新运行完整 benchmark。
- [x] 不使用 bf16 指标代替量化模型指标；首版生产配置即为 bf16，未启用量化。
- [x] 对比生产候选与 Base、Clean030 历史结果。
- [x] 检查 gIoU、cIoU、Dice、Precision 和 Recall。
- [x] 检查零 IoU、低 IoU、FP Area 和 FN Area，结果与历史 Clean030 一致。
- [x] 逐样本指标与历史 Clean030 完全一致，`unsafe`、`safe` 和其他类别均未出现生产制品回归。
- [x] 保存生产预检的 `summary.json`、逐样本指标和按 IoU 排序的样本报告。

### 生产精度预检记录

- 日期：2026-07-19
- 模型版本：`lisa13b-clean030-v1`
- 冻结制品：`artifacts/lisa-safety-seg/lisa13b-clean030-v1/merged_hf`
- 推理精度：bf16，未启用 8bit/4bit 量化
- 评估集：`ReasonSeg|val`
- 样本数：86
- gIoU：`0.4494459628`
- cIoU：`0.3858078868`
- Mean Dice：`0.5156411951`
- Mean Precision：`0.5331799687`
- Mean Recall：`0.5416047745`
- 平均耗时：`0.4162837093 秒/样本`
- 历史 Clean030 对照：六项精度指标一致，无精度回归
- 逐样本回归：86 个历史样本与 86 个预检样本完全对应，新增/缺失样本均为 0，IoU、Dice、Precision、Recall、FP Area、FN Area 差异均为 0
- IoU = 0：16
- IoU < 0.1：28
- IoU < 0.3：38
- IoU >= 0.5：39
- Total FP Area：1,623,031
- Total FN Area：1,677,931
- 输出目录：`exp/runs/lisa13b-clean030-lora-v1/production-preflight`
- 已生成：`summary.json`、`summary.md`、逐样本 CSV/JSONL、按 IoU 排序的 CSV/Markdown
- 数据集元数据字段：`dataset_dir` 和 `val_dataset`；其中 `val_dataset` 应为 `ReasonSeg|val`。`dataset`、`split` 不是当前汇总格式使用的字段，因此查询结果为 `None`。

远程执行，bf16 完整复评：

```bash
CUDA_VISIBLE_DEVICES=0 \
python benchmark_reason_seg.py \
  --version ./artifacts/lisa-safety-seg/lisa13b-clean030-v1/merged_hf \
  --vision-tower ./clip-vit-large-patch14 \
  --dataset_dir ./dataset \
  --val_dataset "ReasonSeg|val" \
  --vision_pretrained ./data_pipeline/sam_vit_h_4b8939.pth \
  --output_dir ./exp/runs/lisa13b-clean030-lora-v1/production-preflight \
  --precision bf16 \
  --workers 4 \
  --save_visualizations \
  --max_visualizations -1 \
  --save_masks
```

## P0：独立 golden test

- [ ] 准备 30～60 张没有进入现有 train/val 的独立图片。
- [ ] 按原始视频、拍摄序列或工地分组，避免相邻帧泄漏。
- [ ] 覆盖 10 个安全类别和主要目标尺寸。
- [ ] 人工确定目标对象、实例数量和目标粒度。
- [ ] 人工修正 mask，不直接把 SAM 伪标签当作最终真值。
- [ ] 每个目标准备标准模板型、视觉定位型和 Agent 查询型 Prompt。
- [ ] 冻结 golden test 后禁止根据模型结果修改。
- [ ] 按 `source_file_name` 聚类计算置信区间。
- [ ] 为上线提前设定 golden test 准入阈值。
- [ ] 分别评估 Clean030 和 Relabel303，决定生产模型与影子模型。

## P0：生产推理核心模块

- [x] 从 `chat.py` 提取无交互的模型加载和推理逻辑。
- [x] 服务进程启动时只加载一次 tokenizer、LISA、CLIP 和预处理器。
- [x] 将输入统一为图片加自然语言 Prompt。
- [x] 将输出统一为 mask、原图尺寸、文本响应和模型版本。
- [x] 支持 PNG Base64 标准 mask 格式。
- [x] 明确多 mask 的返回协议。
- [x] 明确空 mask 的业务含义和响应字段。
- [x] 固定并记录 mask threshold。
- [x] 增加启动时模型预热加载选项。
- [x] 增加 CUDA OOM 捕获与恢复策略：当前请求不自动重试，清理异常链、执行 GC 和 `empty_cache()`；恢复失败时保持 unavailable 并要求重启。
- [x] 增加后端推理等待超时，并明确超时不会中断已经提交的同步 GPU 任务。
- [ ] 增加可中断正在执行的GPU推理机制。
- [x] 编写纯逻辑单元测试，不在本地加载模型。

说明：LISA 使用自定义 `LISAForCausalLM.evaluate()` 和分割分支，不能直接作为普通文本生成模型部署到标准 vLLM OpenAI 接口。需要保留自定义视觉预处理、SAM输入和 mask 输出逻辑。

## P0：生产 API

- [x] 使用 FastAPI 建立服务。
- [x] 实现 `POST /v1/segment`。
- [x] 实现 `GET /health`，仅检查进程存活。
- [x] 实现 `GET /ready`，检查模型已加载并可接收请求。
- [x] 实现 `GET /metrics`，输出进程内JSON指标。
- [x] 实现 `GET /metrics/prometheus`，保持原 JSON 指标接口兼容。
- [x] 实现 `GET /alerts`，按可配置阈值返回进程内告警状态。
- [x] 限制解码后图片字节数和像素数，并拒绝 OpenCV 无法解码的内容。
- [x] 增加 JPEG/PNG 文件签名白名单，并在 OpenCV 解码前检查编码长度和头部像素尺寸。
- [x] 限制 Prompt 长度。
- [x] 支持 request ID 和基础链路追踪。
- [x] 增加可选 API Key 访问控制和进程内并发限制。
- [x] 增加结构化错误码。
- [x] 应用不记录 API Key、完整 Base64 图片和 Prompt，推理异常响应使用固定脱敏文本。
- [ ] 继续核查容器、反向代理和依赖启动日志，避免凭据或服务器私有路径进入外部日志系统。

建议请求：

```json
{
  "image_base64": "图片内容的Base64字符串",
  "prompt": "定位画面左侧未佩戴安全帽的作业人员",
  "request_id": "request-001"
}
```

建议响应：

```json
{
  "request_id": "request-001",
  "model_version": "lisa13b-clean030-v1",
  "width": 1920,
  "height": 1080,
  "has_segmentation": true,
  "mask_count": 1,
  "masks": [
    {
      "index": 0,
      "format": "png_base64",
      "data": "..."
    }
  ],
  "text": "Sure, [SEG] .",
  "latency_ms": 850
}
```

## P1：GPU并发与任务队列

- [x] 默认从单进程、单GPU、串行推理开始。
- [x] 使用有界任务队列和固定数量的 GPU worker 限制进程内 GPU 推理并发。
- [x] 修复超时后的并发槽释放问题：底层 `to_thread` 推理未结束前，原 GPU worker 不处理新任务；真实共享 GPU 双请求回归的历史最大在途数为 1。
- [x] 明确最大队列长度和排队超时，默认等待队列为 8、排队超时为 30 秒。
- [x] 准备客户端并发 1、2、4 和并发 4 连续 100 请求的自包含共享 GPU 验证脚本及准入阈值。
- [x] 完成并发 1、2、4 吞吐与显存验证：195 请求零失败，最差 P95 1,567.105 ms，最低吞吐 2.568395 req/s，峰值显存 31,770 MiB。
- [ ] 评估动态 batching 是否适用于图片尺寸和 Prompt 长度差异。
- [ ] 多GPU时采用一GPU一进程，并由网关负载均衡。
- [ ] 记录队列时间、预处理时间、GPU推理时间和后处理时间。

## P1：容器化和配置

- [x] 编写生产 Dockerfile。
- [x] 固定基础 CUDA 12.1.1 + cuDNN 8 镜像版本。
- [x] Docker 构建仅复制 `production/`、`model/` 和 `utils/` 运行目录，不复制数据集、实验输出、权重或本地协作文档。
- [x] 同时提供 legacy builder 使用的根目录 `.dockerignore` 和 BuildKit 使用的 Dockerfile-specific ignore；构建上下文采用目录白名单并排除所有 `.env`。
- [x] 准备双启动、双冒烟、非 root、只读挂载、GPU 显存和日志脱敏的自包含容器验收实验。
- [x] 完成真实 shared-GPU 容器验收：44 项测试、两轮启动与推理及全部准入项通过。
- [x] 将生产推理依赖与训练/评估依赖拆分，直接生产依赖均固定版本，不在 runtime 镜像安装 `pycocotools`、Gradio、Ray 或编译工具链。
- [ ] 生成包含全部间接依赖及哈希的完整 lock 文件。
- [ ] 在镜像构建阶段运行静态检查和CPU单元测试。
- [x] 权重通过只读卷挂载，不提交到镜像源码层。
- [x] 使用可迁移的模型路径配置，不写服务器私有绝对路径。
- [x] 使用环境变量注入 GPU、模型版本和阈值。
- [x] 增加非 root 用户。
- [x] 增加容器健康检查。
- [x] 容器验收已使用 `--gpus device=0` 和 `--shm-size 8g` 验证 GPU 与共享内存运行参数。
- [x] Dockerfile 支持 OCI 源码 commit 和 LISA 模型版本标签；后续容器验收构建会写入真实值。
- [x] 当前纯逻辑测试累计 59 项：包含制品发布、HTTP 指标、滚动分位数、告警阈值、Prometheus 输出、mask 计数和监控验收工具测试。

### 容器验收记录

- 日期：2026-07-20
- 实验：`lisa13b-clean030-container-smoke-v1`
- Git commit：`0de93e76d7c29da8b70eb326a64816f4526ef990`
- 镜像：`lisa-safety-seg:lisa13b-clean030-v1-0092463`
- 构建时间：17.179 秒
- 容器内测试：44 项全部通过
- 启动与重启：两轮均 healthy、ready
- 推理：两轮均 HTTP 200，各返回 1 个有效 mask
- 客户端延迟：1,022.529 ms、635.262 ms
- GPU 基线 / 峰值 / 停止后：2,337 / 30,214 / 2,337 MiB
- 峰值剩余显存：10,746 MiB
- 停止后显存漂移：0 MiB
- 运行用户：`lisa`，UID/GID 10001/10001
- 模型和 CLIP：只读挂载
- 敏感文件、私有路径和错误日志检查：全部通过
- 共享 GPU 原有计算进程：保持存活
- 准入结果：`PASS`

建议挂载：

```text
/models/lisa13b-clean030-v1/merged_hf
/models/clip-vit-large-patch14
```

## P1：模型量化与容量优化

当前首发候选保持 bf16。量化用于显存或并发容量优化，不作为首版上线的默认前提；只有 bf16 峰值显存、目标并发或目标 GPU 规格不满足要求时才启动。

### 量化决策与执行顺序

1. 先记录 bf16 基线：模型加载显存、单请求峰值显存、冷启动时间、预热后 P50/P95/P99、吞吐量和连续运行稳定性。
2. 根据容量目标决定是否量化；如果 A100 40GB 上 bf16 单并发有足够余量且满足延迟目标，继续使用 bf16。
3. 需要节省显存时先评估 8bit；只有 8bit 仍无法满足容量目标时才评估 4bit。
4. 量化实验必须使用独立模型版本、配置、输出目录和报告，不覆盖 `lisa13b-clean030-v1`。
5. 先完成 API 冒烟和 86 样本 `ReasonSeg|val` 回归，再进入 golden test、性能压测和稳定性测试。
6. 所有准入条件通过后才能将量化版本作为新生产候选；否则继续使用 bf16，并保留一键回滚配置。

### 实现与版本管理

- [ ] 确认量化目标是语言模型权重，CLIP 和 LISA/SAM 分割分支保持原精度，并核对实际被量化和跳过的模块清单。
- [x] 明确当前后端采用 bitsandbytes 加载时量化，不将其误记为已经导出的独立 INT8/INT4 权重文件。
- [x] 已固定并记录 bitsandbytes、Transformers、PyTorch、CUDA 和驱动版本。
- [ ] 定义 8bit 版本，例如 `lisa13b-clean030-int8-v1`，使用独立环境配置和运行 manifest。
- [ ] 仅在 8bit 不满足容量目标时定义 4bit 版本，例如 `lisa13b-clean030-int4-v1`。
- [ ] 为量化版本保存源 bf16 制品 SHA-256、Git commit、量化参数、跳过模块和运行依赖。
- [x] 保留 `lisa13b-clean030-v1` bf16 配置作为回滚目标，不覆盖或删除原制品。
- [x] shared-GPU基线峰值31,740 MiB、剩余9,220 MiB，当前单并发目标继续使用bf16，暂不启动8bit。

### 精度验收

- [ ] 实验前确认量化准入阈值；建议初始阈值为 gIoU 和 Dice 相对 bf16 绝对下降均不超过 `0.02`、零 IoU 样本增加不超过 2 个。
- [ ] 使用与 bf16 完全一致的 `ReasonSeg|val`、Prompt、mask threshold 和样本顺序运行完整 benchmark。
- [ ] 对比 gIoU、cIoU、Dice、Precision、Recall、FP/FN Area 和零/低 IoU 分布。
- [ ] 逐样本比较 bf16 与量化结果，检查 `unsafe`、`safe`、`equipment_proximity` 等关键类别回归。
- [ ] 在冻结的独立 golden test 上比较 bf16、8bit；如启用4bit，再加入4bit。
- [ ] 量化配置或依赖发生任何变化时重新完成精度验收，不复用旧结果。

### 性能与稳定性验收

- [ ] 记录 bf16、8bit 和可选4bit的模型加载显存、单请求峰值显存及实际节省比例。
- [ ] 对比冷启动、首次请求、预热后 P50/P95/P99 和吞吐量；建议量化版本 P95 延迟退化不超过20%。
- [ ] 在并发 1、2、4 下测试显存、吞吐、排队时间和 CUDA OOM。
- [ ] 连续运行至少100个代表性请求，检查显存增长、空 mask、多 mask、超时和服务恢复。
- [ ] 确认量化确实满足既定显存或并发目标；如果只有显存下降但精度、延迟或稳定性不达标，不升级生产版本。

### 计划实验目录

```text
exp/runs/lisa13b-clean030-int8-v1/
├── EXPERIMENT.md
├── command.sh
├── reasonseg-val-outputs/
├── golden-outputs/
└── performance-outputs/

exp/runs/lisa13b-clean030-int4-v1/
└── 仅在8bit无法满足容量目标时创建
```

启动任何量化实验前，先确认实验背景、阈值和配置，再创建对应 `EXPERIMENT.md` 与自包含 `command.sh`；确认前不在服务器执行。

## P1：性能压测

- [x] 准备 bf16 单GPU、单worker、并发1的自包含性能基线脚本和准入阈值。
- [x] 针对不可停止的 `bge-m3` 新增 shared-GPU 三轮共存基线，记录已有进程、相对显存增量、总峰值和最差延迟。
- [x] 三轮模型ready时间为14.29～14.54秒。
- [x] 记录首次请求、5次预热和每轮30次正式请求延迟。
- [x] 正式请求最差P50为394.976 ms、最差P95为400.362 ms、最差P99为401.107 ms。
- [x] 最低串行吞吐为2.525612 req/s。
- [x] 共享基线2,337 MiB，总峰值31,740 MiB，峰值剩余9,220 MiB。
- [x] 三轮共408次请求零失败，最大预热后显存漂移为0 MiB。
- [x] 完成 `lisa13b-clean030-api-concurrency-v1`：客户端并发 1、2、4 和并发 4 连续 100 请求全部通过。
- [x] 测试并发 1、2、4：P95 分别为 383.547、775.983、1,557.151 ms，吞吐分别为 2.622934、2.608311、2.593920 req/s。
- [x] 真实 API 已验证超大头部图片、损坏 JPEG/PNG、非法 Base64、GIF 和 WebP 均在进入 GPU 前被拒绝。
- [ ] 超长、空和纯空格 Prompt 已通过真实 API 验证；仍需补充控制字符等异常字符集。
- [ ] 测试空 mask、多 mask 和极小目标。
- [ ] 测试 CUDA OOM 后服务恢复。

注意：benchmark 中约 `0.42` 秒/样本不等于生产端到端延迟。生产延迟还包括图片传输、解码、排队、预处理、结果编码和网络返回。

当前目标GPU长期运行 `bge-m3` vLLM pooling服务，不能停止。首轮性能数据必须标记为shared-GPU，只用于当前共存部署判断；未来有空闲GPU时仍需补充独占GPU纯性能基线。

### shared-GPU 性能基线记录

- 日期：2026-07-20
- 模型：`lisa13b-clean030-v1` bf16
- GPU模式：与 `bge-m3` 的 `VLLM::EngineCore` 共存
- 轮次：3
- 总请求：408
- 失败请求：0
- 最大ready时间：14,535.709 ms
- 最差正式请求P95：400.362 ms
- 最低吞吐：2.525612 req/s
- GPU总峰值：31,740 MiB
- 相对共享基线峰值增量：29,403 MiB
- 峰值剩余显存：9,220 MiB
- 最大预热后显存漂移：0 MiB
- 准入结果：PASS，3/3轮通过
- 量化决策：当前A100 40GB单并发继续使用bf16，不启动8bit；目标GPU或并发需求变化时重新评估
- 局限：只确认 `bge-m3` GPU进程存活，尚未接入其业务健康和延迟指标

### shared-GPU 并发与稳定性记录

- 日期：2026-07-20
- 实验：`lisa13b-clean030-api-concurrency-v1`
- 服务配置：bf16、单 Uvicorn worker、单 GPU worker、等待队列 8
- 客户端并发：1、2、4
- 总请求：195
- 失败、排队超时、队列拒绝、推理超时和 CUDA OOM：均为 0
- 并发 1 P95 / 吞吐：383.547 ms / 2.622934 req/s
- 并发 2 P95 / 吞吐：775.983 ms / 2.608311 req/s
- 并发 4 P95 / 吞吐：1,557.151 ms / 2.593920 req/s
- 并发 4 连续 100 请求 P95 / 吞吐：1,567.105 ms / 2.568395 req/s
- GPU 历史最大在途数：1
- GPU 总峰值 / 峰值剩余：31,770 MiB / 9,190 MiB
- 预热后显存漂移：0 MiB
- 最终服务状态：ready
- 准入结果：PASS，29/29 项通过
- 结论：当前配置可安全吸收客户端并发 4，但单 GPU worker 吞吐上限约 2.6 req/s；更高吞吐应采用多 GPU 横向扩展

## P1：监控与告警

- [x] 提供进入模型运行时的请求数、成功数、推理失败数、推理超时数和 mask 数指标。
- [x] 将鉴权失败、请求校验失败、Prompt 校验失败和图片解码失败纳入完整 HTTP 请求/失败指标。
- [x] 提供 `/v1/segment` HTTP 延迟、队列等待和 GPU 推理时间的滚动 P50、P95、P99。
- [x] 监控队列长度、容量、利用率和排队时间。
- [ ] 监控 GPU 利用率、显存和温度。
- [x] 进程内指标记录 CUDA OOM、恢复成功、恢复失败和 unavailable 拒绝次数。
- [ ] 将 CUDA OOM、恢复失败和进程/模型重启次数接入外部监控告警。
- [x] 记录空 mask 响应、多 mask 响应和总 mask 数；比率由 Prometheus 计算。
- [ ] 监控输入图片尺寸和 Prompt 长度分布。
- [x] 在响应、响应头和指标中记录模型版本。
- [x] 建立 4xx/5xx、P95、队列利用率、CUDA OOM、unexpected error、模型 ready、GPU 显存和温度的初始告警阈值与 Prometheus 规则。
- [ ] 建立人工抽检和线上 bad case 回流机制。

当前应用内监控实现完成；外部 Prometheus、Alertmanager、Grafana 与 NVIDIA
DCGM Exporter 尚未部署，因此“健康检查、监控和告警已接入”仍保持未完成。

- [x] 准备 `lisa13b-clean030-monitoring-v1` 自包含容器验收实验，覆盖
  JSON/Prometheus 一致性、Bearer 鉴权、4xx 告警触发与恢复、GPU 串行化、
  显存和日志脱敏。
- [x] 完成远程 shared-GPU 监控验收：59 项容器测试、22 项准入全部通过，
  4xx 告警正确触发与恢复，JSON/Prometheus 指标一致，峰值显存 31,752 MiB，
  停止后显存漂移 0 MiB。

## P1：灰度发布与回滚

- [ ] 保留当前线上模型及完整启动配置。
- [ ] 为 Clean030 和 Relabel303 使用不同模型版本标识。
- [ ] 先在预发布环境验证。
- [ ] 开启影子流量，不将影子结果返回用户。
- [ ] 对影子结果进行人工抽检和离线比较。
- [ ] 通过准入后进行 5% 灰度。
- [ ] 再扩大到 20% 灰度。
- [ ] 指标稳定后全量发布。
- [ ] 准备一键切换到上一版本的回滚配置。
- [ ] 明确自动回滚条件。

推荐自动回滚条件包括：

- 请求成功率明显下降。
- P95/P99 超过约定阈值。
- CUDA OOM 持续出现。
- 空 mask 率异常升高。
- golden regression 或人工抽检明显退化。

## P2：安全与合规

- [ ] 明确生产图片的保存周期。
- [x] 默认不持久化原始图片。
- [ ] 如需保存 bad case，进行访问控制和脱敏。
- [x] 不接受外部图片 URL，避免 SSRF。
- [x] 对 Base64 解码后的字节数和 OpenCV 解码后的像素数设置上限。
- [ ] 在反向代理或 ASGI 层限制 HTTP 请求体大小，避免超大 Base64 在业务校验前占用内存。
- [x] 增加 JPEG/PNG 签名和头部尺寸预检，在 OpenCV 完整解码前拒绝像素数超限图片。
- [x] 使用 OpenCV 解码，并拒绝无法作为图片解码的内容。
- [x] 对服务接口增加可选 API Key 身份认证。
- [ ] 增加持久化审计日志。
- [ ] 定期扫描镜像和依赖漏洞。
- [ ] 明确模型输出只作为辅助判断，不能替代人工安全验收。

## 上线验收清单

- [x] 生产模型版本 `lisa13b-clean030-v1` 已冻结为独立制品。
- [x] 冻结制品的 `SHA256SUMS` 已实际执行并全部校验通过。
- [x] 冻结制品 bf16 benchmark 与历史 Clean030 指标一致，无精度回归。
- [ ] 独立 golden test 达标。
- [x] API单请求冒烟功能测试通过。
- [x] 并发和稳定性压测通过。
- [x] 生产 Docker 镜像双启动、真实 GPU 冒烟、权限、挂载、显存和日志准入通过。
- [ ] 健康检查、监控和告警已接入。
- [x] 模型版本能够在响应和响应头中追踪。
- [ ] 灰度方案已确认。
- [ ] 回滚流程已演练。
- [ ] 运维文档和故障处理手册已完成。

## 当前下一步

- [x] 确认生产首发模型为 Clean030 LoRA。
- [x] 确认首轮环境为 A100 40GB、单GPU、并发1。
- [x] 确认首轮生产候选使用 bf16。
- [x] 冻结模型制品并生成 manifest、`SHA256SUMS` 和模型卡。
- [x] 使用冻结制品完成 `ReasonSeg|val` 生产精度预检。
- [x] 在冻结制品目录执行 `sha256sum --check --quiet SHA256SUMS`，全部文件校验通过。
- [x] 核对零 IoU、低 IoU、FP/FN Area 和关键类别回归，逐样本结果与历史 Clean030 一致。
- [ ] （暂缓）准备独立 golden test 图片和人工标签。
- [x] 完成无交互推理核心和 FastAPI 服务基础实现。
- [x] 使用有界队列和专用 GPU worker 修复推理超时后并发槽提前释放问题，并通过纯逻辑并发回归测试。
- [x] 准备 bf16 冷启动、延迟、显存和100次连续请求基线实验。
- [x] 准备保持 `bge-m3` 在线的三轮 shared-GPU 共存性能实验。
- [x] 完成三轮 shared-GPU 共存实验，408次请求零失败，性能和显存准入通过。
- [x] 完成 `lisa13b-clean030-timeout-guard-v1` 真实 GPU 回归：2 个超时响应、2 个后台推理成功、历史最大 GPU 在途数 1、无 CUDA OOM。
- [x] 完成 OOM 恢复状态机、JPEG/PNG 预检和推理错误脱敏实现；加入 robustness 验收逻辑后共通过 33 项本地纯逻辑测试。
- [x] 完成 `lisa13b-clean030-api-robustness-v1`：15/15 用例及 20/20 准入项通过，异常后服务 ready，敏感哨兵无泄漏。
- [x] 准备 `lisa13b-clean030-api-concurrency-v1` 自包含脚本、逐阶段指标快照和并发准入逻辑。
- [x] 准备 `lisa13b-clean030-container-smoke-v1` 容器构建、双启动冒烟和准入脚本；修复生产镜像误装 `pycocotools`、299.6 GB 构建上下文及本地 `.env` 进入镜像的问题，生产纯逻辑测试累计 44 项。
- [x] 完成 `lisa13b-clean030-container-smoke-v1` 真实容器验收：构建 17.179 秒，两轮 healthy/ready 和 GPU 推理通过，峰值 30,214 MiB，停止后显存漂移 0 MiB，最终 `PASS`。
- [x] 完成正式发布工具：复用并绑定已验收镜像，生成模型/镜像/验收证据联合 release manifest，并提供不可变上传复核。
- [x] 已记录外部模型/镜像发布的目的、风险、触发条件和未来执行步骤；当前决定暂不实际发布。
- [x] 完成应用内 HTTP、延迟、队列、推理、mask 和错误指标，提供 Prometheus 输出、进程内 alerts 与外部告警规则模板。
- [x] 准备 `lisa13b-clean030-monitoring-v1` 真实容器监控验收脚本。
- [x] 完成 `lisa13b-clean030-monitoring-v1`：59 项测试和 22 项准入全部通过，告警触发/恢复、指标一致性、共享 GPU 与日志脱敏均通过。
- [x] 完成 API 多次请求、核心异常输入、并发和显存稳定性压测；控制字符、空/多 mask 与真实 OOM 仍保留为独立待办。
- [x] 根据bf16实测显存决定当前不启动8bit；4bit仅在未来8bit仍不满足容量目标时评估。
- [x] 完成容器实测。
- [x] 在远程环境完成新增监控接口验收。
- [ ] 接入外部 Prometheus/DCGM/Alertmanager。
- [ ] 完成灰度和回滚演练。
