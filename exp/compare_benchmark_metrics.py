"""
Compare two LISA benchmark per-sample metric files.

The script matches samples by image file name and reports:
  - top improved samples
  - top regressed samples
  - samples still below an IoU threshold after fine-tuning
  - per-category aggregate deltas

Example:
  python exp/compare_benchmark_metrics.py
  python exp/compare_benchmark_metrics.py --top-k 30 --bad-threshold 0.2
  python exp/compare_benchmark_metrics.py --output exp/comparisons/clean030_full_val.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


DEFAULT_BASE = "exp/runs/lisa13b-local-val/outputs/per_sample_metrics.jsonl"
DEFAULT_FINETUNED = (
    "exp/runs/lisa13b-clean030-lora-v1/full-eval-outputs/per_sample_metrics.jsonl"
)


def load_jsonl(path: Path, key: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if key == "image_name":
                sample_key = Path(row["image"]).name
            elif key == "json_name":
                sample_key = Path(row.get("dataset_json_path", "")).name
            else:
                sample_key = str(row[key])
            if not sample_key:
                raise ValueError(f"Empty match key at {path}:{line_no}")
            rows[sample_key] = row
    return rows


def build_comparison(base_rows: dict[str, dict], tuned_rows: dict[str, dict]) -> list[dict]:
    rows = []
    for sample_key in sorted(set(base_rows) & set(tuned_rows)):
        base = base_rows[sample_key]
        tuned = tuned_rows[sample_key]
        rows.append(
            {
                "sample_key": sample_key,
                "image": Path(tuned["image"]).name,
                "label": tuned.get("source_category") or tuned.get("sample_key") or "",
                "prompt": tuned.get("prompt", ""),
                "base_iou": float(base["iou"]),
                "tuned_iou": float(tuned["iou"]),
                "delta_iou": float(tuned["iou"]) - float(base["iou"]),
                "base_dice": float(base["dice"]),
                "tuned_dice": float(tuned["dice"]),
                "delta_dice": float(tuned["dice"]) - float(base["dice"]),
                "base_precision": float(base["precision"]),
                "tuned_precision": float(tuned["precision"]),
                "delta_precision": float(tuned["precision"]) - float(base["precision"]),
                "base_recall": float(base["recall"]),
                "tuned_recall": float(tuned["recall"]),
                "delta_recall": float(tuned["recall"]) - float(base["recall"]),
                "target_area": int(tuned.get("target_area", 0)),
                "base_pred_area": int(base.get("pred_area", 0)),
                "tuned_pred_area": int(tuned.get("pred_area", 0)),
            }
        )
    return rows


def fmt_float(value: float, signed: bool = False) -> str:
    if signed:
        return f"{value:+.4f}"
    return f"{value:.4f}"


def format_sample_table(rows: list[dict], title: str) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Rank | Delta IoU | Base IoU | Tuned IoU | Label | Image | Prompt |",
        "| ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for idx, row in enumerate(rows, start=1):
        prompt = str(row["prompt"]).replace("|", "\\|")
        image = str(row["image"]).replace("|", "\\|")
        label = str(row["label"]).replace("|", "\\|")
        lines.append(
            "| {idx} | {delta} | {base} | {tuned} | {label} | `{image}` | {prompt} |".format(
                idx=idx,
                delta=fmt_float(row["delta_iou"], signed=True),
                base=fmt_float(row["base_iou"]),
                tuned=fmt_float(row["tuned_iou"]),
                label=label,
                image=image,
                prompt=prompt,
            )
        )
    lines.append("")
    return lines


def category_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["label"] or "unknown"].append(row)

    summary = []
    for label, items in grouped.items():
        delta_iou = [row["delta_iou"] for row in items]
        summary.append(
            {
                "label": label,
                "count": len(items),
                "base_iou": mean(row["base_iou"] for row in items),
                "tuned_iou": mean(row["tuned_iou"] for row in items),
                "delta_iou": mean(delta_iou),
                "median_delta_iou": median(delta_iou),
                "improved": sum(row["delta_iou"] > 0 for row in items),
                "regressed": sum(row["delta_iou"] < 0 for row in items),
                "still_bad_010": sum(row["tuned_iou"] < 0.10 for row in items),
            }
        )
    return sorted(summary, key=lambda row: row["delta_iou"], reverse=True)


def format_category_table(rows: list[dict]) -> list[str]:
    lines = [
        "## Category Summary",
        "",
        "| Label | Count | Base IoU | Tuned IoU | Delta IoU | Median Delta | Improved | Regressed | Tuned IoU < 0.10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        label = str(row["label"]).replace("|", "\\|")
        lines.append(
            "| {label} | {count} | {base} | {tuned} | {delta} | {median_delta} | {improved} | {regressed} | {bad} |".format(
                label=label,
                count=row["count"],
                base=fmt_float(row["base_iou"]),
                tuned=fmt_float(row["tuned_iou"]),
                delta=fmt_float(row["delta_iou"], signed=True),
                median_delta=fmt_float(row["median_delta_iou"], signed=True),
                improved=row["improved"],
                regressed=row["regressed"],
                bad=row["still_bad_010"],
            )
        )
    lines.append("")
    return lines


def make_report(
    rows: list[dict],
    base_path: Path,
    tuned_path: Path,
    top_k: int,
    bad_threshold: float,
) -> str:
    improved = sorted(rows, key=lambda row: row["delta_iou"], reverse=True)
    regressed = sorted(rows, key=lambda row: row["delta_iou"])
    still_bad = sorted(
        [row for row in rows if row["tuned_iou"] < bad_threshold],
        key=lambda row: (row["tuned_iou"], row["delta_iou"]),
    )

    lines = [
        "# Benchmark Comparison",
        "",
        f"- Base: `{base_path}`",
        f"- Tuned: `{tuned_path}`",
        f"- Matched samples: `{len(rows)}`",
        f"- Improved: `{sum(row['delta_iou'] > 0 for row in rows)}`",
        f"- Regressed: `{sum(row['delta_iou'] < 0 for row in rows)}`",
        f"- Unchanged: `{sum(row['delta_iou'] == 0 for row in rows)}`",
        f"- Tuned IoU < {bad_threshold:.2f}: `{len(still_bad)}`",
        f"- Mean base IoU: `{mean(row['base_iou'] for row in rows):.4f}`",
        f"- Mean tuned IoU: `{mean(row['tuned_iou'] for row in rows):.4f}`",
        f"- Mean delta IoU: `{mean(row['delta_iou'] for row in rows):+.4f}`",
        "",
    ]
    lines.extend(format_category_table(category_summary(rows)))
    lines.extend(format_sample_table(improved[:top_k], f"Top {top_k} Improved"))
    lines.extend(format_sample_table(regressed[:top_k], f"Top {top_k} Regressed"))
    lines.extend(
        format_sample_table(
            still_bad[: max(top_k, 40)],
            f"Still Bad After Tuning (IoU < {bad_threshold:.2f})",
        )
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two benchmark metric jsonl files.")
    parser.add_argument("--base", default=DEFAULT_BASE, type=Path)
    parser.add_argument("--tuned", default=DEFAULT_FINETUNED, type=Path)
    parser.add_argument(
        "--match-key",
        default="image_name",
        choices=["image_name", "json_name", "image", "dataset_json_path"],
        help="Field used to match samples between two runs.",
    )
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--bad-threshold", default=0.10, type=float)
    parser.add_argument("--output", type=Path, help="Optional markdown report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base.exists():
        raise FileNotFoundError(f"Missing base metrics: {args.base}")
    if not args.tuned.exists():
        raise FileNotFoundError(f"Missing tuned metrics: {args.tuned}")

    base_rows = load_jsonl(args.base, args.match_key)
    tuned_rows = load_jsonl(args.tuned, args.match_key)
    rows = build_comparison(base_rows, tuned_rows)
    if not rows:
        raise ValueError("No matched samples found.")

    report = make_report(
        rows,
        base_path=args.base,
        tuned_path=args.tuned,
        top_k=args.top_k,
        bad_threshold=args.bad_threshold,
    )
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"\n[done] wrote report: {args.output}")


if __name__ == "__main__":
    main()
