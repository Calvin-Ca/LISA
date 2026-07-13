"""
Build a clean ReasonSeg subset from benchmark per-sample metrics.

This script filters existing LISA/ReasonSeg samples by their benchmark IoU and
copies the original jpg/json pairs into a new dataset directory. It does not
load LISA, SAM, CLIP, or any model weights.

Default output:
  dataset/reason_seg/ReasonSegClean030/
    train/*.jpg + *.json
    val/*.jpg + *.json
    clean_subset_manifest.json
    clean_subset_summary.json

Example:
  python data_pipeline/build_clean_subset_from_benchmark.py --dry-run
  python data_pipeline/build_clean_subset_from_benchmark.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SPLITS = {
    "train": "exp/runs/lisa13b-local-train/outputs/per_sample_metrics.jsonl",
    "val": "exp/runs/lisa13b-local-val/outputs/per_sample_metrics.jsonl",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_path(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return root / path


def load_rows(metrics_path: Path, split: str, threshold: float, root: Path) -> list[dict]:
    rows = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            iou = float(row.get("iou", 0.0))
            if iou < threshold:
                continue

            image_path = resolve_path(row["image"], root)
            json_path = resolve_path(row["dataset_json_path"], root)
            rows.append(
                {
                    "split": split,
                    "line_no": line_no,
                    "iou": iou,
                    "dice": float(row.get("dice", 0.0)),
                    "precision": float(row.get("precision", 0.0)),
                    "recall": float(row.get("recall", 0.0)),
                    "source_category": row.get("source_category", ""),
                    "sample_key": row.get("sample_key", ""),
                    "source_file_name": row.get("source_file_name", ""),
                    "source_image_id": row.get("source_image_id", ""),
                    "prompt": row.get("prompt", ""),
                    "source_image": str(image_path),
                    "source_json": str(json_path),
                    "output_image": "",
                    "output_json": "",
                }
            )
    return rows


def clear_split_dirs(output_root: Path, splits: list[str]) -> None:
    for split in splits:
        split_dir = output_root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def copy_rows(rows: list[dict], output_root: Path, dry_run: bool, strict: bool) -> Counter:
    stats = Counter()
    seen_outputs = set()

    for row in rows:
        split_dir = output_root / row["split"]
        out_image = split_dir / Path(row["source_image"]).name
        out_json = split_dir / Path(row["source_json"]).name
        row["output_image"] = str(out_image)
        row["output_json"] = str(out_json)

        pair_key = (str(out_image), str(out_json))
        if pair_key in seen_outputs:
            stats["duplicate_output_pair"] += 1
            continue
        seen_outputs.add(pair_key)

        source_image = Path(row["source_image"])
        source_json = Path(row["source_json"])
        missing = [str(path) for path in (source_image, source_json) if not path.exists()]
        if missing:
            stats["missing_pair"] += 1
            if strict:
                raise FileNotFoundError("Missing source files: " + ", ".join(missing))
            continue

        stats[f"selected_{row['split']}"] += 1
        stats[f"selected_{row['source_category'] or 'unknown'}"] += 1
        if not dry_run:
            split_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, out_image)
            shutil.copy2(source_json, out_json)
            stats[f"copied_{row['split']}"] += 1

    return stats


def summarize(rows: list[dict], stats: Counter, threshold: float, output_root: Path) -> dict:
    by_split = defaultdict(list)
    by_category = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
        by_category[row["source_category"] or "unknown"].append(row)

    def split_summary(items: list[dict]) -> dict:
        if not items:
            return {"count": 0}
        ious = [row["iou"] for row in items]
        return {
            "count": len(items),
            "mean_iou": sum(ious) / len(ious),
            "min_iou": min(ious),
            "max_iou": max(ious),
        }

    return {
        "threshold": threshold,
        "output_root": str(output_root),
        "splits": {split: split_summary(items) for split, items in sorted(by_split.items())},
        "categories": {
            category: split_summary(items) for category, items in sorted(by_category.items())
        },
        "stats": dict(stats),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter benchmarked ReasonSeg samples into a clean subset."
    )
    parser.add_argument("--threshold", default=0.30, type=float, help="Minimum IoU to keep.")
    parser.add_argument(
        "--output-root",
        default="dataset/reason_seg/ReasonSegClean030",
        type=Path,
        help="Output dataset directory under dataset/reason_seg/.",
    )
    parser.add_argument(
        "--train-metrics",
        default=DEFAULT_SPLITS["train"],
        type=Path,
        help="Train split per_sample_metrics.jsonl.",
    )
    parser.add_argument(
        "--val-metrics",
        default=DEFAULT_SPLITS["val"],
        type=Path,
        help="Val split per_sample_metrics.jsonl.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output split directories before copying.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print counts.")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Skip rows whose source jpg/json files are missing instead of failing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    output_root = resolve_path(str(args.output_root), root)
    splits = {
        "train": resolve_path(str(args.train_metrics), root),
        "val": resolve_path(str(args.val_metrics), root),
    }

    all_rows = []
    for split, metrics_path in splits.items():
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file for {split}: {metrics_path}")
        rows = load_rows(metrics_path, split, args.threshold, root)
        all_rows.extend(rows)
        print(f"[select] {split}: {len(rows)} rows with IoU >= {args.threshold:.2f}")

    if not args.dry_run:
        if args.overwrite:
            clear_split_dirs(output_root, list(splits.keys()))
        else:
            for split in splits:
                (output_root / split).mkdir(parents=True, exist_ok=True)

    stats = copy_rows(
        all_rows,
        output_root=output_root,
        dry_run=args.dry_run,
        strict=not args.no_strict,
    )
    summary = summarize(all_rows, stats, args.threshold, output_root)

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    manifest_path = output_root / "clean_subset_manifest.json"
    summary_path = output_root / "clean_subset_summary.json"
    manifest_path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] output: {output_root}")
    print(f"[done] manifest: {manifest_path}")
    print(f"[done] summary: {summary_path}")


if __name__ == "__main__":
    main()
