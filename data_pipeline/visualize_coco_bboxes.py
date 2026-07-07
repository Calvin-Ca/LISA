"""
Visualize COCO bounding boxes for dataset inspection.

Examples:
  # Visualize phase-1 train/val annotations.
  python3 data_pipeline/visualize_coco_bboxes.py \
    --data-root data/phase1_feasibility \
    --output-root data/phase1_feasibility/vis_bboxes

  # Visualize raw datasets.
  python3 data_pipeline/visualize_coco_bboxes.py \
    --data-root data/raw \
    --output-root data/raw_bbox_vis \
    --limit 100
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2


PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
]


def safe_name(name: str) -> str:
    name = name.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "unknown"


def annotation_key(data_root: Path, ann_path: Path) -> str:
    rel_dir = ann_path.parent.relative_to(data_root)
    return "__".join(safe_name(part) for part in rel_dir.parts) or "root"


def find_image_path(annotation_dir: Path, file_name: str) -> Path | None:
    candidates = [
        annotation_dir / file_name,
        annotation_dir / "images" / file_name,
        annotation_dir / "train" / file_name,
        annotation_dir / "valid" / file_name,
        annotation_dir / "val" / file_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(annotation_dir.rglob(Path(file_name).name))
    return matches[0] if matches else None


def color_for_category(category_id: int) -> tuple[int, int, int]:
    return PALETTE[category_id % len(PALETTE)]


def draw_label(
    image,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y1 = max(0, y - th - baseline - 4)
    y2 = max(th + baseline + 4, y)
    cv2.rectangle(image, (x, y1), (x + tw + 6, y2), color, -1)
    cv2.putText(
        image,
        text,
        (x + 3, y2 - baseline - 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def visualize_annotation(
    ann_path: Path,
    data_root: Path,
    output_root: Path,
    limit: int | None,
    seed: int,
    overwrite: bool,
) -> Counter:
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {img["id"]: img for img in data.get("images", [])}
    categories = {cat["id"]: cat for cat in data.get("categories", [])}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("image_id") in images and ann.get("category_id") in categories:
            anns_by_image[ann["image_id"]].append(ann)

    image_ids = sorted(anns_by_image)
    if limit is not None and len(image_ids) > limit:
        rng = random.Random(seed)
        image_ids = sorted(rng.sample(image_ids, limit))

    key = annotation_key(data_root, ann_path)
    split_out = output_root / key
    split_out.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    for image_id in image_ids:
        image_info = images[image_id]
        src = find_image_path(ann_path.parent, image_info["file_name"])
        if src is None:
            stats["missing_images"] += 1
            continue

        dst = split_out / Path(image_info["file_name"]).name
        if dst.exists() and not overwrite:
            stats["skipped_existing"] += 1
            continue

        image = cv2.imread(str(src))
        if image is None:
            stats["unreadable_images"] += 1
            continue

        height, width = image.shape[:2]
        for ann in anns_by_image[image_id]:
            x, y, w, h = ann["bbox"]
            x1 = max(0, min(width - 1, int(round(x))))
            y1 = max(0, min(height - 1, int(round(y))))
            x2 = max(0, min(width - 1, int(round(x + w))))
            y2 = max(0, min(height - 1, int(round(y + h))))
            category = categories[ann["category_id"]]
            color = color_for_category(category["id"])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{category['name']} #{ann.get('id', '')}".rstrip()
            draw_label(image, label, x1, y1, color)

        cv2.imwrite(str(dst), image)
        stats["visualized_images"] += 1
        stats["visualized_boxes"] += len(anns_by_image[image_id])

    stats["images_with_annotations"] = len(anns_by_image)
    stats["selected_images"] = len(image_ids)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize COCO bbox annotations.")
    parser.add_argument("--data-root", default="data/phase1_feasibility", type=Path)
    parser.add_argument("--output-root", default=None, type=Path)
    parser.add_argument(
        "--ann-glob",
        default="**/_annotations.coco.json",
        help="Glob pattern under data-root for annotation files.",
    )
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_root = args.output_root or args.data_root / "vis_bboxes"
    ann_paths = sorted(args.data_root.glob(args.ann_glob))
    ann_paths = [p for p in ann_paths if args.output_root not in p.parents]
    if not ann_paths:
        raise FileNotFoundError(f"No COCO annotations found under {args.data_root}")

    summary = {}
    for ann_path in ann_paths:
        key = annotation_key(args.data_root, ann_path)
        stats = visualize_annotation(
            ann_path=ann_path,
            data_root=args.data_root,
            output_root=args.output_root,
            limit=args.limit,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        summary[key] = dict(stats)
        print(f"[{key}] {dict(stats)}")

    summary_path = args.output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] output: {args.output_root}")
    print(f"[done] summary: {summary_path}")


if __name__ == "__main__":
    main()
