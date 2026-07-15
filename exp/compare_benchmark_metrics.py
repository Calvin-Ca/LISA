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
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Optional


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
                "base_visualization": resolve_visualization_markdown_path(base),
                "tuned_visualization": resolve_visualization_markdown_path(tuned),
            }
        )
    return rows


def resolve_visualization_markdown_path(row: dict) -> str:
    path = row.get("visualization_markdown_path") or row.get(
        "visualization_label_path"
    )
    if not path and row.get("visualization_path"):
        path = str(Path(row["visualization_path"]).with_suffix(".md"))
    if path and Path(path).suffix == ".txt":
        path = str(Path(path).with_suffix(".md"))
    return str(path or "")


def markdown_link(target: str, report_path: Optional[Path], label: str) -> str:
    if not target:
        return ""
    link = target
    if report_path:
        link = os.path.relpath(target, start=report_path.parent)
    return f"[{label}]({Path(link).as_posix()})"


def fmt_float(value: float, signed: bool = False) -> str:
    if signed:
        return f"{value:+.4f}"
    return f"{value:.4f}"


def format_sample_table(
    rows: list[dict],
    title: str,
    report_path: Optional[Path] = None,
    heading: str = "##",
) -> list[str]:
    lines = [
        f"{heading} {title}",
        "",
        "| Rank | Change | Delta IoU | Base IoU | Tuned IoU | Delta Dice | Label | Image | Base View | Tuned View | Prompt |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(rows, start=1):
        prompt = str(row["prompt"]).replace("|", "\\|")
        image = str(row["image"]).replace("|", "\\|")
        label = str(row["label"]).replace("|", "\\|")
        if row["delta_iou"] > 0:
            change = "Improved"
        elif row["delta_iou"] < 0:
            change = "Regressed"
        else:
            change = "Unchanged"
        lines.append(
            "| {idx} | {change} | {delta} | {base} | {tuned} | {delta_dice} | "
            "{label} | `{image}` | {base_view} | {tuned_view} | {prompt} |".format(
                idx=idx,
                change=change,
                delta=fmt_float(row["delta_iou"], signed=True),
                base=fmt_float(row["base_iou"]),
                tuned=fmt_float(row["tuned_iou"]),
                delta_dice=fmt_float(row["delta_dice"], signed=True),
                label=label,
                image=image,
                base_view=markdown_link(
                    row["base_visualization"], report_path, "base"
                ),
                tuned_view=markdown_link(
                    row["tuned_visualization"], report_path, "tuned"
                ),
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
    report_path: Optional[Path] = None,
) -> str:
    improved = sorted(
        rows,
        key=lambda row: (-row["delta_iou"], -row["delta_dice"], row["image"]),
    )
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
    lines.extend(
        format_sample_table(
            improved,
            "All Samples Sorted by IoU Change (Best to Worst)",
            report_path,
        )
    )
    lines.extend(
        format_sample_table(
            improved[:top_k], f"Top {top_k} Improved", report_path
        )
    )
    lines.extend(
        format_sample_table(
            regressed[:top_k], f"Top {top_k} Regressed", report_path
        )
    )
    lines.extend(
        format_sample_table(
            still_bad[: max(top_k, 40)],
            f"Still Bad After Tuning (IoU < {bad_threshold:.2f})",
            report_path,
        )
    )
    return "\n".join(lines)


def make_embedded_section(
    rows: list[dict],
    base_path: Path,
    tuned_path: Path,
    title: str,
    markdown_path: Path,
) -> str:
    sorted_rows = sorted(
        rows,
        key=lambda row: (-row["delta_iou"], -row["delta_dice"], row["image"]),
    )
    lines = [
        f"## {title}",
        "",
        f"- Base: `{base_path}`",
        f"- Tuned: `{tuned_path}`",
        f"- Matched samples: `{len(rows)}`",
        f"- Improved: `{sum(row['delta_iou'] > 0 for row in rows)}`",
        f"- Regressed: `{sum(row['delta_iou'] < 0 for row in rows)}`",
        f"- Unchanged: `{sum(row['delta_iou'] == 0 for row in rows)}`",
        f"- Mean delta IoU: `{mean(row['delta_iou'] for row in rows):+.4f}`",
        "",
    ]
    lines.extend(
        format_sample_table(
            sorted_rows,
            "All Samples Sorted by IoU Change (Best to Worst)",
            markdown_path,
            heading="###",
        )
    )
    return "\n".join(lines).rstrip()


def update_managed_section(
    path: Path, section_id: str, section_content: str
) -> None:
    start_marker = f"<!-- benchmark-comparison:{section_id}:start -->"
    end_marker = f"<!-- benchmark-comparison:{section_id}:end -->"
    block = f"{start_marker}\n{section_content}\n{end_marker}"
    content = path.read_text(encoding="utf-8") if path.exists() else ""

    if start_marker in content or end_marker in content:
        if content.count(start_marker) != 1 or content.count(end_marker) != 1:
            raise ValueError(f"Invalid managed section markers in {path}: {section_id}")
        before, remainder = content.split(start_marker, 1)
        _, after = remainder.split(end_marker, 1)
        content = f"{before.rstrip()}\n\n{block}{after}"
    else:
        content = f"{content.rstrip()}\n\n{block}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


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
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path, help="Optional markdown report path.")
    destination.add_argument(
        "--append-to",
        type=Path,
        help="Append or replace a managed comparison section in a Markdown file.",
    )
    parser.add_argument("--section-id", help="Stable id used by --append-to markers.")
    parser.add_argument("--section-title", help="Heading used by --append-to.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the markdown report to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base.exists():
        raise FileNotFoundError(f"Missing base metrics: {args.base}")
    if not args.tuned.exists():
        raise FileNotFoundError(f"Missing tuned metrics: {args.tuned}")
    if args.append_to and (not args.section_id or not args.section_title):
        raise ValueError("--append-to requires --section-id and --section-title")

    base_rows = load_jsonl(args.base, args.match_key)
    tuned_rows = load_jsonl(args.tuned, args.match_key)
    rows = build_comparison(base_rows, tuned_rows)
    if not rows:
        raise ValueError("No matched samples found.")

    report_path = args.append_to or args.output
    report = make_report(
        rows,
        base_path=args.base,
        tuned_path=args.tuned,
        top_k=args.top_k,
        bad_threshold=args.bad_threshold,
        report_path=report_path,
    )
    if not args.quiet:
        print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"\n[done] wrote report: {args.output}")
    elif args.append_to:
        section = make_embedded_section(
            rows,
            base_path=args.base,
            tuned_path=args.tuned,
            title=args.section_title,
            markdown_path=args.append_to,
        )
        update_managed_section(args.append_to, args.section_id, section)
        print(f"\n[done] updated section in: {args.append_to}")


if __name__ == "__main__":
    main()
