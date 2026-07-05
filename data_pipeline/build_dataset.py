"""
主编排脚本:把"检测框数据集"合成为"LISA ReasonSeg 数据集"。

流程(与 MS.md 里的 pipeline 图一一对应):
  Stage 0  加载原料      —— 读图 + 检测标注(bbox + class)
  Stage 1  框 → 掩码     —— SAM(box_to_mask.py)
  Stage 2  类别 → 隐患   —— HAZARD_TAXONOMY 映射
  Stage 3  生成指令      —— instruction_bank.py
  Stage 4  掩码 → 多边形 —— cv2.findContours + approxPolyDP
  Stage 5  组装 LISA json（LabelMe 多边形格式,含 text / is_sentence）
  Stage 6  质检 + 划分   —— 交给 quality_check.py / split

产物规格(与 utils/data_processing.py::get_mask_from_json 完全对齐):
  <name>.jpg
  <name>.json = {
      "shapes": [{"label": "target", "points": [[x,y],...]}, ...],
      "text":   ["<推理指令>"],
      "is_sentence": true
  }
"""

import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from config import (HAZARD_TAXONOMY, OUT_DIR, QC, RAW_DIR, SAM_CHECKPOINT,
                    SAM_MODEL_TYPE, VAL_RATIO, RANDOM_SEED)
from instruction_bank import sample_instruction

# 反向索引:原始检测类别 -> 隐患 key
CLASS_TO_HAZARD = {
    cls.lower(): hz
    for hz, cfg in HAZARD_TAXONOMY.items()
    for cls in cfg["source_classes"]
}


def mask_to_polygons(mask, epsilon_ratio, min_points):
    """0/1 掩码 -> LabelMe 多边形点列表 [[x,y],...]。"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polys = []
    for c in contours:
        eps = epsilon_ratio * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) >= min_points:
            polys.append(approx.reshape(-1, 2).tolist())
    return polys


def load_samples(raw_dir: Path):
    """
    Stage 0:约定原料格式(可按你的实际数据集改写这个函数)。
    这里假设每张图配一个同名 .txt 检测标注,每行:
        class_name x1 y1 x2 y2   (绝对像素坐标)
    产出: [{"image": Path, "boxes": [(cls, [x1,y1,x2,y2]), ...]}, ...]
    """
    samples = []
    for img_path in sorted(raw_dir.glob("*.jpg")):
        ann_path = img_path.with_suffix(".txt")
        if not ann_path.exists():
            continue
        boxes = []
        for line in ann_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, *xyxy = parts
            boxes.append((cls.lower(), [float(v) for v in xyxy]))
        if boxes:
            samples.append({"image": img_path, "boxes": boxes})
    return samples


def build(dry_run=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    samples = load_samples(RAW_DIR)
    print(f"[Stage 0] 载入 {len(samples)} 张原料图")

    # SAM 只在真正需要出掩码时才加载(dry_run 跳过,便于无 GPU 检查逻辑)
    b2m = None
    if not dry_run:
        from box_to_mask import BoxToMask, box_mask_iou
        b2m = BoxToMask(SAM_CHECKPOINT, SAM_MODEL_TYPE)

    stats = Counter()
    produced = []

    for s in samples:
        img = cv2.imread(str(s["image"]))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        img_area = H * W

        # Stage 2:按隐患类型把框分组(同一隐患的多个实例合成一张 json)
        hazard_boxes = {}
        for cls, box in s["boxes"]:
            hz = CLASS_TO_HAZARD.get(cls)
            if hz:
                hazard_boxes.setdefault(hz, []).append(box)

        if not hazard_boxes:
            stats["skip_no_hazard"] += 1
            continue

        if not dry_run:
            b2m.set_image(img_rgb)

        for hz, boxes in hazard_boxes.items():
            boxes_arr = np.array(boxes, dtype=np.float32)

            # Stage 1:框 -> 掩码
            if dry_run:
                stats[f"dryrun_{hz}"] += len(boxes)
                continue
            masks = b2m.boxes_to_masks(boxes_arr)

            # Stage 5:组装多边形 shapes(逐实例质检后合并)
            shapes = []
            for box, m in zip(boxes, masks):
                area_ratio = m.sum() / img_area
                if not (QC["min_mask_area_ratio"] <= area_ratio <= QC["max_mask_area_ratio"]):
                    stats["drop_area"] += 1
                    continue
                if box_mask_iou(box, m) < QC["min_box_mask_iou"]:
                    stats["drop_iou"] += 1
                    continue
                for poly in mask_to_polygons(m, QC["poly_epsilon_ratio"], QC["poly_min_points"]):
                    shapes.append({"label": "target", "points": poly})

            if not shapes:
                stats["drop_empty"] += 1
                continue

            # Stage 3:生成指令
            is_sentence = HAZARD_TAXONOMY[hz]["is_sentence"]
            instruction = sample_instruction(hz, is_sentence=is_sentence, rng=rng)

            # 写出 LISA 格式样本
            name = f"{s['image'].stem}__{hz}"
            cv2.imwrite(str(OUT_DIR / f"{name}.jpg"), img)
            anno = {"shapes": shapes, "text": [instruction], "is_sentence": is_sentence}
            (OUT_DIR / f"{name}.json").write_text(
                json.dumps(anno, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            produced.append(name)
            stats[f"ok_{hz}"] += 1

    # Stage 6:train/val 划分清单
    rng.shuffle(produced)
    n_val = int(len(produced) * VAL_RATIO)
    manifest = {"val": produced[:n_val], "train": produced[n_val:]}
    (OUT_DIR / "split.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[完成] 统计:", dict(stats))
    print(f"[完成] 产出 {len(produced)} 条样本 -> {OUT_DIR}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="不加载 SAM、不出掩码,只校验数据流与类别映射")
    args = ap.parse_args()
    build(dry_run=args.dry_run)
