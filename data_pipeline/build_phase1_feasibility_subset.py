"""
Build a phase-1 feasibility image subset from selected COCO datasets.

Default goal:
  - Source datasets: data/002 and data/004
  - train: up to 600 images, balanced by category
  - val: up to 80 images, balanced by category and intended for manual checking

Output layout:
  data/phase1_feasibility/
    train/
      _annotations.coco.json
      <linked-or-copied images>
    val/
      _annotations.coco.json
      <linked-or-copied images>
    manifest.csv
    summary.json

The script preserves COCO annotations for the selected images and rewrites
image file names to the placed file names. If source image files are not present
yet, it still emits the subset annotations and manifest with status=missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_EXCLUDE_CATEGORIES = {
    "objects",
    "Safe-Unsafe-No_Helmet-No_Jacket",
}


def safe_name(name: str) -> str:
    name = name.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "unknown"


def find_annotation(data_root: Path, dataset_id: str) -> Path:
    dataset_dir = data_root / dataset_id
    candidates = [
        dataset_dir / "train" / "_annotations.coco.json",
        dataset_dir / "_annotations.coco.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(dataset_dir.glob("**/_annotations.coco.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No COCO annotation found for dataset {dataset_id}")


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


def load_dataset(data_root: Path, dataset_id: str, exclude_categories: set[str]) -> dict:
    ann_path = find_annotation(data_root, dataset_id)
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    categories = {cat["id"]: cat for cat in data.get("categories", [])}
    images = {img["id"]: img for img in data.get("images", [])}

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    category_to_images: dict[int, set[int]] = defaultdict(set)
    for ann in data.get("annotations", []):
        category = categories.get(ann.get("category_id"))
        image = images.get(ann.get("image_id"))
        if not category or not image:
            continue
        if category["name"] in exclude_categories:
            continue
        anns_by_image[image["id"]].append(ann)
        category_to_images[category["id"]].add(image["id"])

    usable_images = set(anns_by_image)
    return {
        "dataset_id": dataset_id,
        "ann_path": ann_path,
        "annotation_dir": ann_path.parent,
        "categories": categories,
        "images": images,
        "annotations_by_image": anns_by_image,
        "category_to_images": category_to_images,
        "usable_images": usable_images,
    }


def build_global_items(datasets: list[dict]) -> tuple[dict, dict]:
    items = {}
    category_to_items: dict[str, set[tuple[str, int]]] = defaultdict(set)

    for dataset in datasets:
        dataset_id = dataset["dataset_id"]
        for image_id in dataset["usable_images"]:
            anns = dataset["annotations_by_image"][image_id]
            category_names = sorted(
                {
                    dataset["categories"][ann["category_id"]]["name"]
                    for ann in anns
                }
            )
            key = (dataset_id, image_id)
            items[key] = {
                "dataset": dataset,
                "dataset_id": dataset_id,
                "image_id": image_id,
                "category_names": category_names,
            }
            for name in category_names:
                category_to_items[name].add(key)

    return items, category_to_items


def balanced_sample(
    category_to_items: dict[str, set[tuple[str, int]]],
    target: int,
    rng: random.Random,
    forbidden: set[tuple[str, int]] | None = None,
) -> list[tuple[str, int]]:
    forbidden = forbidden or set()
    pools = {
        category: [item for item in sorted(items) if item not in forbidden]
        for category, items in category_to_items.items()
    }
    for pool in pools.values():
        rng.shuffle(pool)

    selected: list[tuple[str, int]] = []
    selected_set: set[tuple[str, int]] = set()
    categories = sorted(pools)

    while len(selected) < target:
        added = False
        rng.shuffle(categories)
        for category in categories:
            pool = pools[category]
            while pool and (pool[-1] in selected_set or pool[-1] in forbidden):
                pool.pop()
            if not pool:
                continue
            item = pool.pop()
            selected.append(item)
            selected_set.add(item)
            added = True
            if len(selected) >= target:
                break
        if not added:
            break

    return selected


def merged_categories(datasets: list[dict]) -> tuple[list[dict], dict]:
    names = sorted(
        {
            cat["name"]
            for dataset in datasets
            for cat in dataset["categories"].values()
            if cat["name"] not in DEFAULT_EXCLUDE_CATEGORIES
        }
    )
    name_to_id = {name: idx + 1 for idx, name in enumerate(names)}
    categories = [
        {"id": cid, "name": name, "supercategory": "phase1_feasibility"}
        for name, cid in name_to_id.items()
    ]
    return categories, name_to_id


def write_split(
    split: str,
    selected: list[tuple[str, int]],
    items: dict,
    output_root: Path,
    mode: str,
    overwrite: bool,
    categories: list[dict],
    category_name_to_id: dict[str, int],
) -> tuple[list[dict], Counter]:
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    out_images = []
    out_annotations = []
    manifest_rows = []
    stats = Counter()
    new_image_id = 1
    new_ann_id = 1

    for key in selected:
        item = items[key]
        dataset = item["dataset"]
        image = dataset["images"][item["image_id"]]
        old_file_name = image["file_name"]
        placed_name = f"{item['dataset_id']}__{Path(old_file_name).name}"
        src = find_image_path(dataset["annotation_dir"], old_file_name)
        dst = split_dir / placed_name

        status = "missing"
        if src is not None:
            place_file(src, dst, mode, overwrite)
            status = mode
            stats["placed_files"] += 1
        else:
            stats["missing_files"] += 1

        out_image = {
            **{k: v for k, v in image.items() if k != "id"},
            "id": new_image_id,
            "file_name": placed_name,
            "source_dataset": item["dataset_id"],
            "source_image_id": item["image_id"],
            "manual_check_required": split == "val",
        }
        out_images.append(out_image)

        for ann in dataset["annotations_by_image"][item["image_id"]]:
            category_name = dataset["categories"][ann["category_id"]]["name"]
            if category_name not in category_name_to_id:
                continue
            new_ann = {
                **{
                    k: v
                    for k, v in ann.items()
                    if k not in {"id", "image_id", "category_id"}
                },
                "id": new_ann_id,
                "image_id": new_image_id,
                "category_id": category_name_to_id[category_name],
                "source_dataset": item["dataset_id"],
                "source_annotation_id": ann.get("id"),
                "source_category_name": category_name,
            }
            out_annotations.append(new_ann)
            new_ann_id += 1

        manifest_rows.append(
            {
                "split": split,
                "dataset": item["dataset_id"],
                "source_image_id": item["image_id"],
                "source_file_name": old_file_name,
                "placed_file_name": placed_name,
                "categories": "|".join(item["category_names"]),
                "source_path": str(src) if src else "",
                "output_path": str(dst),
                "status": status,
                "manual_check_required": split == "val",
            }
        )

        stats["images"] += 1
        new_image_id += 1

    coco = {
        "info": {
            "description": "Phase-1 feasibility subset from datasets 002 and 004",
            "split": split,
        },
        "licenses": [],
        "categories": categories,
        "images": out_images,
        "annotations": out_annotations,
    }
    (split_dir / "_annotations.coco.json").write_text(
        json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stats["annotations"] = len(out_annotations)
    return manifest_rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build phase-1 feasibility subset from COCO datasets 002 and 004."
    )
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument("--output-root", default=None, type=Path)
    parser.add_argument("--datasets", default="002,004")
    parser.add_argument("--train-size", default=600, type=int)
    parser.add_argument("--val-size", default=80, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--mode",
        default="symlink",
        choices=["symlink", "copy", "hardlink"],
        help="How to place images in the subset folders.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root or args.data_root / "phase1_feasibility"
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_ids = [part.strip() for part in args.datasets.split(",") if part.strip()]
    datasets = [
        load_dataset(args.data_root, dataset_id, DEFAULT_EXCLUDE_CATEGORIES)
        for dataset_id in dataset_ids
    ]
    items, category_to_items = build_global_items(datasets)

    rng = random.Random(args.seed)
    val_items = balanced_sample(category_to_items, args.val_size, rng)
    train_items = balanced_sample(
        category_to_items, args.train_size, rng, forbidden=set(val_items)
    )

    categories, category_name_to_id = merged_categories(datasets)
    all_rows = []
    summary = {}
    for split, selected in [("train", train_items), ("val", val_items)]:
        rows, stats = write_split(
            split=split,
            selected=selected,
            items=items,
            output_root=output_root,
            mode=args.mode,
            overwrite=args.overwrite,
            categories=categories,
            category_name_to_id=category_name_to_id,
        )
        all_rows.extend(rows)
        summary[split] = dict(stats)
        print(f"[{split}] {dict(stats)}")

    category_summary = {}
    for split, selected in [("train", train_items), ("val", val_items)]:
        image_counter = Counter()
        for key in selected:
            for category_name in items[key]["category_names"]:
                image_counter[category_name] += 1
        category_summary[split] = dict(sorted(image_counter.items()))

    with (output_root / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary["category_image_counts"] = category_summary
    summary["source_datasets"] = dataset_ids
    summary["excluded_categories"] = sorted(DEFAULT_EXCLUDE_CATEGORIES)
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[done] output: {output_root}")
    print(f"[done] manifest: {output_root / 'manifest.csv'}")
    print(f"[done] summary: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
