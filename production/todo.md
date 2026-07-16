# LISA 生产化开发 TODO

## 目标

将施工安全 ReasonSeg 微调模型封装为可版本化、可验证、可监控、可灰度和可回滚的生产推理服务。

当前推荐：

- 生产候选：`runs/lisa13b-clean030-lora-v1/merged_hf`
- 影子候选：`runs/lisa13b-relabel303-lora-v1/merged_hf`
- 当前对比集：`ReasonSeg|val`，86 个样本
- 当前生产候选指标：gIoU `0.4494`、cIoU `0.3858`、Dice `0.5156`

Clean030 LoRA 当前总体指标优于 Relabel303 LoRA。正式上线前仍需通过独立 golden test、生产精度复评和生产同构环境压测。

## P0：模型版本与制品冻结

- [ ] 确认首个生产候选使用 Clean030 LoRA，不直接用 Relabel303 替换。
- [x] 为生产模型定义固定版本，例如 `lisa13b-clean030-v1`。
- [ ] 将合并后的 Hugging Face 模型目录复制到独立制品目录，不直接引用可变化的训练目录。
- [ ] 确认部署对象是 `merged_hf/`，不是 `ckpt_model/` 或单独的 `pytorch_model.bin`。
- [ ] 确认 CLIP vision tower `clip-vit-large-patch14` 可作为独立制品部署。
- [x] 提供生成模型 SHA-256 文件清单的制品冻结工具。
- [x] 制品 manifest 自动保存 Git commit、文件大小和校验值。
- [ ] 记录 Python、CUDA、PyTorch、Transformers、DeepSpeed 和 bitsandbytes 版本。
- [ ] 将模型权重上传到对象存储、内部模型仓库或制品平台，不提交到 Git。
- [x] 定义制品目录结构：

```text
artifacts/lisa-safety-seg/
└── lisa13b-clean030-v1/
    ├── merged_hf/
    ├── manifest.json
    ├── SHA256SUMS
    └── MODEL_CARD.md
```

远程执行，生成校验文件：

```bash
find runs/lisa13b-clean030-lora-v1/merged_hf \
  -type f \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > runs/lisa13b-clean030-lora-v1/merged_hf.sha256
```

## P0：模型产物检查

- [ ] 检查 `merged_hf/config.json` 存在。
- [ ] 检查 tokenizer 配置和 tokenizer 模型存在。
- [ ] 检查模型权重或分片权重完整。
- [ ] 检查模型配置中的 vision tower 和特殊 token。
- [ ] 检查生产机器能读取模型和 CLIP vision tower。
- [ ] 校验传输后的 SHA-256 与训练服务器一致。
- [ ] 禁止服务启动时从公网自动下载权重。

远程执行，列出模型文件：

```bash
find runs/lisa13b-clean030-lora-v1/merged_hf \
  -maxdepth 1 \
  -type f \
  -printf '%f\n' \
  | sort
```

## P0：生产同构环境冒烟测试

- [ ] 在预发布 Linux GPU 机器准备与生产一致的 CUDA 和 Python 环境。
- [ ] 使用 `chat.py` 验证合并模型能成功加载。
- [ ] 使用固定图片和固定 Prompt 作为 smoke case。
- [ ] 检查模型文本输出包含有效分割响应。
- [ ] 检查输出 mask 非空、尺寸正确且无 NaN。
- [ ] 记录模型加载时间、首次推理时间和预热后推理时间。
- [ ] 记录模型加载显存和推理峰值显存。
- [ ] 保留 smoke case 的输入、期望输出和实际产物。

远程执行，bf16 冒烟测试：

```bash
CUDA_VISIBLE_DEVICES=0 \
python chat.py \
  --version ./runs/lisa13b-clean030-lora-v1/merged_hf \
  --vision-tower ./clip-vit-large-patch14 \
  --precision bf16 \
  --vis_save_path ./vis_output/clean030-production-smoke
```

如果生产计划使用 8bit，远程执行：

```bash
CUDA_VISIBLE_DEVICES=0 \
python chat.py \
  --version ./runs/lisa13b-clean030-lora-v1/merged_hf \
  --vision-tower ./clip-vit-large-patch14 \
  --precision fp16 \
  --load_in_8bit \
  --vis_save_path ./vis_output/clean030-production-8bit-smoke
```

## P0：生产精度复评

- [ ] 明确生产使用 bf16、fp16、8bit 或 4bit。
- [ ] 使用最终生产精度重新运行完整 benchmark。
- [ ] 不使用 bf16 指标代替量化模型指标。
- [ ] 对比生产候选与 Base、Clean030 历史结果。
- [ ] 检查 gIoU、cIoU、Dice、Precision 和 Recall。
- [ ] 检查零 IoU、低 IoU、FP Area 和 FN Area。
- [ ] 检查 `unsafe`、`safe` 和其他关键类别回归。
- [ ] 保存生产预检的 `summary.json`、逐样本指标、mask 和可视化。

远程执行，bf16 完整复评：

```bash
CUDA_VISIBLE_DEVICES=0 \
python benchmark_reason_seg.py \
  --version ./runs/lisa13b-clean030-lora-v1/merged_hf \
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
- [ ] 增加 CUDA OOM 捕获与恢复策略。
- [x] 增加HTTP推理等待超时。
- [ ] 增加可中断正在执行的GPU推理机制。
- [x] 编写纯逻辑单元测试，不在本地加载模型。

说明：LISA 使用自定义 `LISAForCausalLM.evaluate()` 和分割分支，不能直接作为普通文本生成模型部署到标准 vLLM OpenAI 接口。需要保留自定义视觉预处理、SAM输入和 mask 输出逻辑。

## P0：生产 API

- [x] 使用 FastAPI 建立服务。
- [x] 实现 `POST /v1/segment`。
- [x] 实现 `GET /health`，仅检查进程存活。
- [x] 实现 `GET /ready`，检查模型已加载并可接收请求。
- [x] 实现 `GET /metrics`，输出进程内JSON指标。
- [x] 限制图片大小、分辨率和文件格式。
- [x] 限制 Prompt 长度。
- [x] 支持 request ID 和基础链路追踪。
- [x] 增加可选 API Key 访问控制和进程内并发限制。
- [x] 增加结构化错误码。
- [ ] 对请求日志和图片信息进行脱敏。
- [ ] 禁止在日志中记录凭据、完整私有图片或服务器私有路径。

建议请求：

```json
{
  "image": "base64或受控对象存储地址",
  "prompt": "定位画面左侧未佩戴安全帽的作业人员",
  "request_id": "request-001"
}
```

建议响应：

```json
{
  "request_id": "request-001",
  "model_version": "lisa13b-clean030-v1",
  "mask_format": "png_base64",
  "mask": "...",
  "width": 1920,
  "height": 1080,
  "has_segmentation": true,
  "latency_ms": 850
}
```

## P1：GPU并发与任务队列

- [x] 默认从单进程、单GPU、串行推理开始。
- [x] 增加有界进程内并发控制，避免请求同时占用GPU。
- [ ] 明确最大队列长度和排队超时。
- [ ] 测试并发 1、2、4 时的吞吐与显存。
- [ ] 评估动态 batching 是否适用于图片尺寸和 Prompt 长度差异。
- [ ] 多GPU时采用一GPU一进程，并由网关负载均衡。
- [ ] 记录队列时间、预处理时间、GPU推理时间和后处理时间。

## P1：容器化和配置

- [x] 编写生产 Dockerfile。
- [x] 固定基础 CUDA 12.1.1 + cuDNN 8 镜像版本。
- [ ] 冻结 Python 依赖版本。
- [ ] 在镜像构建阶段运行静态检查和CPU单元测试。
- [x] 权重通过只读卷挂载，不提交到镜像源码层。
- [x] 使用可迁移的模型路径配置，不写服务器私有绝对路径。
- [x] 使用环境变量注入 GPU、模型版本和阈值。
- [x] 增加非 root 用户。
- [x] 增加容器健康检查。
- [ ] 设置共享内存和GPU运行参数。

建议挂载：

```text
/models/lisa13b-clean030-v1/merged_hf
/models/clip-vit-large-patch14
```

## P1：性能压测

- [ ] 记录模型冷启动时间。
- [ ] 记录首次请求和预热后请求延迟。
- [ ] 统计端到端 P50、P95、P99。
- [ ] 统计请求吞吐量。
- [ ] 统计模型加载显存和峰值显存。
- [ ] 测试连续运行稳定性和显存泄漏。
- [ ] 测试并发 1、2、4。
- [ ] 测试超大图片、损坏图片和不支持的格式。
- [ ] 测试超长 Prompt、空 Prompt 和异常字符。
- [ ] 测试空 mask、多 mask 和极小目标。
- [ ] 测试 CUDA OOM 后服务恢复。
- [ ] 测试依赖的对象存储或图片下载超时。

注意：benchmark 中约 `0.42` 秒/样本不等于生产端到端延迟。生产延迟还包括图片传输、解码、排队、预处理、结果编码和网络返回。

## P1：监控与告警

- [x] 提供请求数、成功数、失败数、超时数和mask数的进程内指标。
- [ ] 监控 P50、P95、P99 延迟。
- [ ] 监控队列长度和排队时间。
- [ ] 监控 GPU 利用率、显存和温度。
- [ ] 监控 CUDA OOM 和模型重载次数。
- [ ] 监控空 mask 率和多 mask 率。
- [ ] 监控输入图片尺寸和 Prompt 长度分布。
- [x] 在响应、响应头和指标中记录模型版本。
- [ ] 建立关键指标告警阈值。
- [ ] 建立人工抽检和线上 bad case 回流机制。

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
- [x] 限制文件大小和解码像素数，防止资源耗尽攻击。
- [x] 使用 OpenCV 解码并校验真实图片格式。
- [x] 对服务接口增加可选 API Key 身份认证。
- [ ] 增加持久化审计日志。
- [ ] 定期扫描镜像和依赖漏洞。
- [ ] 明确模型输出只作为辅助判断，不能替代人工安全验收。

## 上线验收清单

- [ ] 生产模型版本已冻结且 SHA-256 校验通过。
- [ ] 生产精度 benchmark 达标。
- [ ] 独立 golden test 达标。
- [ ] API功能测试通过。
- [ ] 并发和稳定性压测通过。
- [ ] 健康检查、监控和告警已接入。
- [x] 模型版本能够在响应和响应头中追踪。
- [ ] 灰度方案已确认。
- [ ] 回滚流程已演练。
- [ ] 运维文档和故障处理手册已完成。

## 当前下一步

- [ ] 确认生产首发模型为 Clean030 LoRA。
- [ ] 确认生产GPU型号、显存和期望并发。
- [ ] 确认生产使用 bf16、8bit 还是4bit。
- [ ] 准备独立 golden test 图片和人工标签。
- [ ] 冻结模型制品并生成 manifest。
- [x] 完成无交互推理核心和 FastAPI 服务基础实现。
