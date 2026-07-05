# CLAUDE.md

本文件为在此仓库工作的 Claude 提供项目上下文与协作约定。

## ⚠️ 执行模型:本地只编程,训练/运行在远程服务器

- **本地(这台 Windows 机器)只负责:编写/修改代码、设计、看文档。**
- **训练、GPU 推理、SAM / GroundingDINO / LISA 的实际运行,全部在远程服务器上进行。**
- 因此在本地:
  - **不要**尝试跑训练(`train_ds.py`)、加载大模型权重、下载数据集等重活;
  - 可以跑**纯逻辑校验**(如 `build_dataset.py --dry-run`、规则单测、`grounded_ingest.py --frames-only`),这些不需要 GPU/权重;
  - 给出的运行命令默认是"给远程服务器执行的",本地只做编辑与 dry-run 验证。
- 交付方式:本地改好代码 → 推送/同步到远程 → 在远程跑。写代码时要保证**在 Linux + GPU 环境可运行**(注意跨平台路径、CUDA 依赖)。

## 项目背景

- 基座是开源项目 **LISA / LISA++**(reasoning segmentation,推理分割多模态模型)。
- 目标:**面试 Agent 应用开发岗的简历项目**——在建筑施工场景微调 LISA,并把它包成一个 Agent 里的"视觉定位工具"。
- 核心论点(贯穿所有设计):**LISA 只是被 LLM 规划器编排的一个工具,不是 Agent 本身;Agent 的价值在编排,不在单模型。** 只有 LISA 需要微调,其余(RAG/VLM/DB/工单)都是现成拼装。
- 主打场景 A:**施工安全隐患巡检 + 整改闭环 Agent**(未戴安全帽/未穿反光衣/临边无防护/裸露电线)。
- 面试准备文档见 `MS.md`(含场景选型、STAR 故事线、Agent 架构图、关键设计点答辩);架构图 HTML 版见 `MS.html`。

## 关键目录(本项目新增,非 LISA 原生)

```
MS.md                        面试准备主文档(先读它了解全貌)
MS.html                      场景 A 架构图(Mermaid,浏览器打开)
data_pipeline/               数据合成 pipeline(检测框数据 → LISA 训练数据)
  ├── grounded_ingest.py     无框图/视频前端:抽帧+开放词表检测+规则推导隐患 → 出 .txt 框
  ├── config.py              隐患分类体系、类别→隐患映射、质检阈值、路径
  ├── box_to_mask.py         复用仓库 SAM,框→像素掩码
  ├── instruction_bank.py    推理指令模板库 + LLM 改写钩子
  ├── build_dataset.py       主编排(Stage 0-6):出掩码→生成指令→组装 LISA json→划分
  ├── quality_check.py       叠加掩码可视化,人工抽检
  ├── deploy_to_dataset.py   按 split.json 把 out/ 分发到 dataset/ 训练目录
  └── ingest/ raw/ out/ vis/ 工作目录(内容 .gitignore,保留 .gitkeep 骨架)
dataset/                     LISA 训练目录(= train_ds.py --dataset_dir)
  └── reason_seg/ReasonSeg/{train,val,explanatory}/
```

## 数据流

```
无框图/视频 → ingest/ ─(grounded_ingest.py)→ raw/ (带框 .txt)
已有检测框数据 ───────────────────────────→ raw/
raw/ ─(build_dataset.py)→ out/(LISA 格式 jpg+json + split.json)
out/ ─(deploy_to_dataset.py)→ dataset/reason_seg/ReasonSeg/{train,val}/
dataset/ ─(train_ds.py, 远程)→ LoRA 微调
```

## LISA ReasonSeg 数据格式(产物必须对齐)

- 每样本 = 同名成对 `<name>.jpg` + `<name>.json`,放在 split 目录下。
- 加载逻辑(`utils/reason_seg_dataset.py`):`glob("*.jpg")` 后把后缀换成 `.json`。
- json 规格(见 `utils/data_processing.py::get_mask_from_json`):
  ```json
  {
    "shapes": [{"label": "target", "points": [[x, y], ...]}],
    "text": ["圈出图中没有佩戴安全帽的工人。"],
    "is_sentence": true
  }
  ```
  - `label` 含 `ignore` → 值=255(评测忽略);否则值=1(目标)。

## 易踩的坑

- 图片**必须 `.jpg` 后缀**(glob 写死 `*.jpg`),`.png/.jpeg` 会被静默漏掉 → 先统一转 jpg。
- 文件名里别再出现 `.jpg` 字样(`replace(".jpg",".json")` 是无脑替换)。
- 训练加 `--explanatory -1` 跳过 explanatory 支路(其 basename 提取在 Windows 路径下有 bug,但远程 Linux 训练不受影响)。
- **验证集必须真实/人工核验**,不能用合成脏数据,否则 mIoU 不可信。
- 外部数据集下载注意重名陷阱:施工 "SODA" ≠ HuggingFace `allenai/soda`(后者是对话数据集)。
- 数据集导出优先选 **COCO**(绝对坐标 + 类别名),避免 YOLO(归一化坐标 + 数字 id)。

## 数据量参考

- 在 LISA 预训练权重上做 LoRA,数据需求小(原论文 ReasonSeg 微调仅 ~239 张)。
- 快速验证/简历:~300–800 合成样本 + 100–200 干净 val;稳妥:1.5k–3k。按隐患类型均衡,多样性 > 数量。

## 训练命令(远程执行,仅 reason_seg)

```bash
python train_ds.py \
  --dataset_dir ./dataset \
  --dataset "reason_seg" \
  --reason_seg_data "ReasonSeg|train" \
  --explanatory -1 \
  --exp_name lisa-construction
```

## 待办 / 未完成

- `grounded_ingest.py::GroundingDetector` 的 Model 初始化 + detect 输出解析(接 GroundingDINO,远程装权重后补全)。
- `import_coco.py`(Roboflow COCO 导出 → raw/ .txt 格式)尚未创建。
- `edge_no_guardrail`(临边无防护)几何规则难覆盖,待人工/专用模型补充。
- Agent 集成层(LISA tool 封装 + ReAct 规划器 + RAG/VLM/工单)尚未开始。
