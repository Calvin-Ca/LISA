# 数据合成 Pipeline(场景 A · 施工安全隐患巡检)

**一句话**:把现成的"检测框"数据集,零人工标注地转成 LISA 需要的"推理指令 → 像素掩码"数据。

## 为什么需要它(面试亮点)

LISA 训练需要 `(图像, 推理指令, 分割掩码)` 三元组,而这种数据**没有现成的**、纯人工标注**极贵**。
本 pipeline 用三步把便宜的检测数据"升级"成 LISA 数据:

1. **框 → 掩码**:复用仓库自带 SAM,用 bbox 作 box prompt 生成像素级掩码(免标注)。
2. **类别 → 推理指令**:用模板库 + 可选 LLM 改写,把类别名写成组合语义指令("圈出没戴安全帽的工人")。
3. **掩码 → 多边形 → LISA json**:输出与 `utils/data_processing.py::get_mask_from_json` 完全对齐的 LabelMe 格式。

## 目录

| 文件 | Stage | 作用 |
|------|-------|------|
| `grounded_ingest.py` | 0- | **无框数据前端**:原始图/视频 →(抽帧)→ 开放词表检测 → 规则推导隐患 → 出 `.txt` 框 |
| `config.py` | 0 | 路径、隐患分类体系、类别→隐患映射、质检阈值 |
| `box_to_mask.py` | 1 | SAM 框→掩码 + 掩码/框 IoU 质检 |
| `instruction_bank.py` | 3 | 推理指令模板库 + LLM 改写钩子 |
| `build_dataset.py` | 0-6 | 主编排:读原料→出掩码→生成指令→组装 json→划分 |
| `quality_check.py` | 6 | 叠加掩码可视化,供人工抽检 |

## 两种输入路径

- **已有检测框数据**:直接放进 `raw/`,跑 `build_dataset.py`。
- **无框的原始照片/视频**:放进 `ingest/`,先跑 `grounded_ingest.py` 自动出框到 `raw/`,再跑 `build_dataset.py`。
  - 原理:开放词表检测器只出"基础实体"(person/helmet/vest/wire),隐患("未戴安全帽")由**几何规则**从实体组合推导——组合语义交给规则,不硬塞给检测器。
  - 视频先抽帧去重;`--frames-only` 只需 opencv 即可跑(不加载检测器)。
  - 进阶:视频可用 SAM 2 做掩码传播,一帧标注传播到多帧,样本量翻倍且近乎免费。

## 产物规格(与 LISA 对齐)

```
out/<name>.jpg
out/<name>.json = {
  "shapes": [{"label": "target", "points": [[x,y], ...]}, ...],
  "text":   ["圈出图中没有佩戴安全帽的工人。"],
  "is_sentence": true
}
out/split.json = {"train": [...], "val": [...]}
```

其中 `label` 用 `"target"`(值=1);如需标"忽略区域",用含 `ignore` 的 label(值=255)。

## 使用

```bash
# 1. 准备原料:把图片 + 同名 .txt 检测标注放进 raw/
#    每行: class_name x1 y1 x2 y2   (绝对像素坐标)

# 2. 先干跑,校验类别映射与数据流(不加载 SAM,无需 GPU)
python build_dataset.py --dry-run

# 3. 下载 SAM 权重到 config.SAM_CHECKPOINT,正式合成
python build_dataset.py

# 4. 抽检
python quality_check.py

# 5. 把 out/ 里的 jpg+json 放到
#    <base_image_dir>/reason_seg/ReasonSeg/train/ 下,即可用于 LISA 训练
```

## 待适配点(按你的真实数据集改)

- `load_samples()`:改成你的检测标注格式(YOLO txt / COCO json / VOC xml)。
- `HAZARD_TAXONOMY.source_classes`:填你数据集里真实的类别名。
- 组合语义隐患(如"临边无防护")现成数据少,可先用规则/人工少量补充,再进本 pipeline。
- `instruction_bank.llm_paraphrase()`:接入 LLM 客户端提升指令多样性。
