"""
Organize COCO image files by category.

Input layout:
  data/
    001/train/_annotations.coco.json
    001/train/<images...>
    002/train/_annotations.coco.json
    ...

Output layout:
  data/data_by_category/
    001/
      train/
        no-helmet/
          001__train__image_a.jpg -> ../../../001/train/image_a.jpg
    002/
      train/
        helmet_missing/
          002__train__image_b.jpg -> ../../../002/train/image_b.jpg

Images with multiple categories are linked/copied into multiple category folders.
If image files are not present yet, the script still writes a manifest that can be
used to inspect category membership.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def safe_name(name: str) -> str:
    """Make a category name safe as a directory name."""
    name = name.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "unknown"


def dataset_key(data_root: Path, ann_path: Path) -> str:
    """Use the annotation folder path relative to data_root as dataset key."""
    rel_dir = ann_path.parent.relative_to(data_root)
    return "__".join(safe_name(part) for part in rel_dir.parts)


def output_dataset_dir(data_root: Path, output_root: Path, ann_path: Path) -> Path:
    """Mirror the annotation folder path under output_root."""
    rel_dir = ann_path.parent.relative_to(data_root)
    return output_root.joinpath(*rel_dir.parts)


def find_image_path(dataset_dir: Path, file_name: str) -> Path | None:
    """Resolve a COCO file_name relative to the dataset directory."""
    candidates = [
        dataset_dir / file_name,
        dataset_dir / "images" / file_name,
        dataset_dir / "train" / file_name,
        dataset_dir / "valid" / file_name,
        dataset_dir / "val" / file_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = list(dataset_dir.rglob(Path(file_name).name))
    return matches[0] if matches else None


def place_file(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        rel_src = os.path.relpath(src.resolve(), dst.parent.resolve())
        dst.symlink_to(rel_src)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def organize_dataset(
    ann_path: Path,
    data_root: Path,
    output_root: Path,
    mode: str,
    skip_empty_categories: bool,
    overwrite: bool,
) -> tuple[list[dict], Counter]:
    dataset_dir = ann_path.parent
    dataset_name = dataset_key(data_root, ann_path)
    dataset_output_dir = output_dataset_dir(data_root, output_root, ann_path)
    data = json.loads(ann_path.read_text(encoding="utf-8"))

    images = {img["id"]: img for img in data.get("images", [])}
    categories = {cat["id"]: cat for cat in data.get("categories", [])}

    image_to_categories: dict[int, set[int]] = defaultdict(set)
    category_counts = Counter()
    for ann in data.get("annotations", []):
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")
        if image_id not in images or category_id not in categories:
            continue
        image_to_categories[image_id].add(category_id)
        category_counts[category_id] += 1

    rows: list[dict] = []
    missing = 0
    linked = 0

    for image_id, category_ids in sorted(image_to_categories.items()):
        image = images[image_id]
        file_name = image["file_name"]
        src = find_image_path(dataset_dir, file_name)

        for category_id in sorted(category_ids):
            if skip_empty_categories and category_counts[category_id] == 0:
                continue
            category = categories[category_id]
            category_name = category["name"]
            category_dir = safe_name(category_name)
            dst_name = f"{dataset_name}__{Path(file_name).name}"
            dst = dataset_output_dir / category_dir / dst_name

            status = "missing"
            if src is not None:
                place_file(src, dst, mode, overwrite)
                status = mode
                linked += 1
            else:
                missing += 1

            rows.append(
                {
                    "dataset": dataset_name,
                    "image_id": image_id,
                    "file_name": file_name,
                    "category_id": category_id,
                    "category_name": category_name,
                    "annotation_count_for_category": category_counts[category_id],
                    "source_path": str(src) if src else "",
                    "output_path": str(dst),
                    "status": status,
                }
            )

    stats = Counter(
        {
            "images": len(images),
            "annotations": len(data.get("annotations", [])),
            "categories": len(categories),
            "rows": len(rows),
            "placed_files": linked,
            "missing_files": missing,
        }
    )
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize COCO images into category directories."
    )
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument("--output-root", default=None, type=Path)
    parser.add_argument(
        "--ann-glob",
        default="**/_annotations.coco.json",
        help="Glob pattern under data-root for COCO annotation files.",
    )
    parser.add_argument(
        "--mode",
        default="symlink",
        choices=["symlink", "copy", "hardlink"],
        help="How to place images in category directories.",
    )
    parser.add_argument(
        "--include-empty-categories",
        action="store_true",
        help="Keep categories with zero annotations in manifests.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_root = args.output_root or args.data_root / "data_by_category"
    ann_paths = sorted(args.data_root.glob(args.ann_glob))
    ann_paths = [p for p in ann_paths if args.output_root not in p.parents]
    if not ann_paths:
        raise FileNotFoundError(f"No COCO annotation files found under {args.data_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_stats = {}
    for ann_path in ann_paths:
        rows, stats = organize_dataset(
            ann_path=ann_path,
            data_root=args.data_root,
            output_root=args.output_root,
            mode=args.mode,
            skip_empty_categories=not args.include_empty_categories,
            overwrite=args.overwrite,
        )
        all_rows.extend(rows)
        key = dataset_key(args.data_root, ann_path)
        all_stats[key] = dict(stats)
        print(f"[{key}] {dict(stats)}")

    manifest_path = args.output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = args.output_root / "summary.json"
    summary_path.write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[done] manifest: {manifest_path}")
    print(f"[done] summary: {summary_path}")


if __name__ == "__main__":
    main()
