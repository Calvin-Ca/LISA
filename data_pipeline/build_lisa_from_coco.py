"""
Convert phase-1 COCO bbox data into LISA ReasonSeg samples.

Default input:
  data/phase1_feasibility/
    train/_annotations.coco.json
    val/_annotations.coco.json

Default output:
  dataset/reason_seg/ReasonSeg/
    train/<name>.jpg + <name>.json
    val/<name>.jpg + <name>.json

Phase-1 category policy:
  keep all annotated COCO categories. Each original category is converted into
  a normalized sample key, and no category is filtered by default.

Each output json follows the LabelMe-like format expected by
utils/data_processing.py::get_mask_from_json.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from config import QC, SAM_CHECKPOINT, SAM_MODEL_TYPE
from box_to_mask import BoxToMask, box_mask_iou


INSTRUCTION_BANK = {
    "helmet_missing": [
        "圈出图中没有佩戴安全帽的工人。",
        "标出未按规定佩戴安全帽的作业人员。",
        "现场哪些人员存在未戴安全帽的安全隐患?请分割出来。",
        "把没有做好头部防护、未戴安全帽的人分割出来。",
    ],
    "no_helmet": [
        "圈出图中没有佩戴安全帽的工人。",
        "标出未按规定佩戴安全帽的作业人员。",
        "现场哪些人员存在未戴安全帽的安全隐患?请分割出来。",
        "把没有做好头部防护、未戴安全帽的人分割出来。",
    ],
    "no_jacket": [
        "圈出没有穿反光衣或安全背心的作业人员。",
        "标出未按要求穿戴反光背心的工人。",
        "现场哪些人员没有穿安全背心?请分割出来。",
        "把缺少反光衣防护的作业人员分割出来。",
    ],
    "harness_missing": [
        "圈出高处作业但没有系安全带的人员。",
        "标出未按要求佩戴安全带的作业人员。",
        "现场哪些工人存在未系安全带的高处作业风险?请分割出来。",
        "把缺少安全带防护的作业人员分割出来。",
    ],
    "guardrail_missing": [
        "标出缺少防护栏杆的区域。",
        "指出没有设置防护栏杆的临边区域。",
        "图中哪些位置存在防护栏杆缺失隐患?请分割出来。",
        "把缺少栏杆防护、存在坠落风险的部位分割出来。",
    ],
    "opening_unprotected": [
        "圈出没有防护的洞口或临边区域。",
        "标出存在洞口未防护隐患的位置。",
        "图中哪些洞口或临边没有做防护?请分割出来。",
        "把缺少防护、可能导致坠落的开口区域分割出来。",
    ],
    "equipment_proximity": [
        "标出人员或设备距离过近、存在碰撞风险的区域。",
        "圈出施工现场设备邻近作业人员的危险区域。",
        "图中哪些位置存在设备靠近人员的安全隐患?请分割出来。",
    ],
    "poor_housekeeping": [
        "标出现场材料堆放混乱或文明施工不到位的区域。",
        "圈出施工现场杂乱、可能影响通行安全的位置。",
        "图中哪些区域存在场地整理不到位的安全隐患?请分割出来。",
    ],
    "safe": [
        "圈出图中被标注为安全状态的作业人员或区域。",
        "标出现场处于安全状态的目标。",
        "请分割图中符合安全要求的目标区域。",
    ],
    "unsafe": [
        "圈出图中被标注为不安全状态的作业人员或区域。",
        "标出现场存在不安全状态的目标。",
        "请分割图中存在安全风险的目标区域。",
    ],
}


def normalize_category_name(category_name: str) -> str:
    """Convert a source COCO category name into a stable sample key."""
    name = category_name.strip().lower().replace(" ", "_").replace("-", "_")
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unknown"


def sample_instruction(category_name: str, sample_key: str, rng: random.Random) -> str:
    if sample_key in INSTRUCTION_BANK:
        return rng.choice(INSTRUCTION_BANK[sample_key])
    readable_name = category_name.replace("_", " ").replace("-", " ")
    return rng.choice(
        [
            f"请分割图中的 {readable_name} 目标。",
            f"标出图中属于 {readable_name} 的区域。",
            f"圈出现场中与 {readable_name} 相关的目标。",
        ]
    )


def find_image_path(split_dir: Path, file_name: str) -> Path | None:
    candidates = [
        split_dir / file_name,
        split_dir / "images" / file_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(split_dir.rglob(Path(file_name).name))
    return matches[0] if matches else None


def clip_xywh_to_xyxy(bbox: list[float], width: int, height: int) -> list[float] | None:
    x, y, w, h = bbox
    x1 = max(0.0, min(float(width), float(x)))
    y1 = max(0.0, min(float(height), float(y)))
    x2 = max(0.0, min(float(width), float(x + w)))
    y2 = max(0.0, min(float(height), float(y + h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def mask_to_polygons(mask: np.ndarray, epsilon_ratio: float, min_points: int):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        eps = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)
        if len(approx) >= min_points:
            polygons.append(approx.reshape(-1, 2).tolist())
    return polygons


def load_coco(split_dir: Path):
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")

    data = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in data.get("images", [])}
    categories = {category["id"]: category for category in data.get("categories", [])}
    anns_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("image_id") in images and ann.get("category_id") in categories:
            anns_by_image[ann["image_id"]].append(ann)
    return images, categories, anns_by_image


def clear_existing_output(split_out: Path) -> None:
    split_out.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.jpg", "*.json"):
        for path in split_out.glob(pattern):
            path.unlink()


def build_split(
    split: str,
    input_root: Path,
    output_root: Path,
    b2m: BoxToMask | None,
    dry_run: bool,
    overwrite: bool,
    seed: int,
) -> tuple[list[dict], Counter]:
    split_dir = input_root / split
    split_out = output_root / split
    if overwrite and not dry_run:
        clear_existing_output(split_out)
    else:
        split_out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed + (0 if split == "train" else 1000))
    images, categories, anns_by_image = load_coco(split_dir)

    stats = Counter()
    manifest_rows = []

    for image_id, annotations in sorted(anns_by_image.items()):
        image_info = images[image_id]
        image_path = find_image_path(split_dir, image_info["file_name"])
        if image_path is None:
            stats["skip_missing_image"] += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            stats["skip_unreadable_image"] += 1
            continue
        height, width = image.shape[:2]

        sample_boxes: dict[str, list[list[float]]] = defaultdict(list)
        source_categories: dict[str, str] = {}
        for ann in annotations:
            category_name = categories[ann["category_id"]]["name"]
            sample_key = normalize_category_name(category_name)

            box = clip_xywh_to_xyxy(ann["bbox"], width, height)
            if box is None:
                stats["drop_invalid_box"] += 1
                continue
            sample_boxes[sample_key].append(box)
            source_categories[sample_key] = category_name

        if not sample_boxes:
            stats["skip_no_valid_category"] += 1
            continue

        if dry_run:
            for sample_key, boxes in sample_boxes.items():
                stats[f"dryrun_{sample_key}"] += len(boxes)
            continue

        assert b2m is not None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_area = height * width
        b2m.set_image(image_rgb)

        for sample_key, boxes in sample_boxes.items():
            masks = b2m.boxes_to_masks(np.array(boxes, dtype=np.float32))

            shapes = []
            for box, mask in zip(boxes, masks):
                area_ratio = float(mask.sum()) / float(image_area)
                if not (QC["min_mask_area_ratio"] <= area_ratio <= QC["max_mask_area_ratio"]):
                    stats["drop_area"] += 1
                    continue
                if box_mask_iou(box, mask) < QC["min_box_mask_iou"]:
                    stats["drop_iou"] += 1
                    continue
                for polygon in mask_to_polygons(
                    mask, QC["poly_epsilon_ratio"], QC["poly_min_points"]
                ):
                    shapes.append({"label": "target", "points": polygon})

            if not shapes:
                stats["drop_empty_shapes"] += 1
                continue

            category_name = source_categories[sample_key]
            instruction = sample_instruction(category_name, sample_key, rng)
            stem = Path(image_info["file_name"]).stem
            out_name = f"{split}__{stem}__{sample_key}"
            out_image = split_out / f"{out_name}.jpg"
            out_json = split_out / f"{out_name}.json"

            shutil.copy2(image_path, out_image)
            anno = {
                "shapes": shapes,
                "text": [instruction],
                "is_sentence": True,
                "source": {
                    "split": split,
                    "image_id": image_id,
                    "file_name": image_info["file_name"],
                    "sample_key": sample_key,
                    "source_category": category_name,
                },
            }
            out_json.write_text(
                json.dumps(anno, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            stats[f"ok_{sample_key}"] += 1
            stats["ok_samples"] += 1
            manifest_rows.append(
                {
                    "split": split,
                    "sample": out_name,
                    "source_image_id": image_id,
                    "source_file_name": image_info["file_name"],
                    "sample_key": sample_key,
                    "source_category": category_name,
                    "instruction": instruction,
                    "shapes": len(shapes),
                    "manual_check_required": split == "val",
                }
            )

    return manifest_rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert phase-1 COCO bbox subset to LISA ReasonSeg format."
    )
    parser.add_argument("--input-root", default="data/phase1_feasibility", type=Path)
    parser.add_argument(
        "--output-root",
        default="dataset/reason_seg/ReasonSeg",
        type=Path,
    )
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--sam-checkpoint", default=SAM_CHECKPOINT, type=Path)
    parser.add_argument("--sam-model-type", default=SAM_MODEL_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]

    b2m = None
    if not args.dry_run:
        b2m = BoxToMask(args.sam_checkpoint, args.sam_model_type)

    all_rows = []
    summary = {}
    for split in splits:
        rows, stats = build_split(
            split=split,
            input_root=args.input_root,
            output_root=args.output_root,
            b2m=b2m,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            seed=args.seed,
        )
        all_rows.extend(rows)
        summary[split] = dict(stats)
        print(f"[{split}] {dict(stats)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary["category_policy"] = "keep_all"
    (args.output_root / "phase1_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if all_rows:
        manifest_path = args.output_root / "phase1_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[done] manifest: {manifest_path}")

    print(f"[done] output: {args.output_root}")
    print(f"[done] summary: {args.output_root / 'phase1_build_summary.json'}")


if __name__ == "__main__":
    main()
