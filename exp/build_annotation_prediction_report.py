"""
Build a Markdown report that compares source annotations and model predictions.

For each matched benchmark sample, the report shows:
  - source COCO annotations for all categories
  - source COCO annotation for the target category
  - generated LISA polygon annotation
  - baseline benchmark prediction mask overlay
  - fine-tuned benchmark prediction mask overlay

Default example:
  python exp/build_annotation_prediction_report.py

Useful subsets:
  python exp/build_annotation_prediction_report.py --only bad --bad-threshold 0.10
  python exp/build_annotation_prediction_report.py --only improved --limit 30
  python exp/build_annotation_prediction_report.py --sort-by delta_iou --limit 40
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError as exc:
    Image = None
    ImageDraw = None
    ImageFont = None
    PIL_IMPORT_ERROR = exc


DEFAULT_BASE_METRICS = Path("exp/runs/lisa13b-local-val/outputs/per_sample_metrics.jsonl")
DEFAULT_TUNED_METRICS = Path(
    "exp/runs/lisa13b-clean030-lora-v1/full-eval-outputs/per_sample_metrics.jsonl"
)
DEFAULT_COCO_ROOT = Path("data")
DEFAULT_OUTPUT_DIR = Path("exp/comparisons/annotation_prediction_report")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
COCO_COLORS = [
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
]


def load_font(size: int) -> ImageFont.ImageFont:
    ensure_pillow()
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def ensure_pillow() -> None:
    if Image is None or ImageDraw is None or ImageFont is None:
        raise SystemExit(
            "Missing dependency: Pillow. Install it in the remote environment with "
            "`python -m pip install Pillow`."
        ) from PIL_IMPORT_ERROR


def normalize_category_name(name: str) -> str:
    name = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def strip_dataset_prefix(file_name: str) -> str:
    return re.sub(r"^\d{3}__", "", Path(file_name).name)


def sample_source_stem(image_name: str, category: str) -> str:
    stem = Path(image_name).stem
    if "__" in stem:
        split_prefix, rest = stem.split("__", 1)
        if split_prefix in {"train", "val"}:
            stem = rest
    suffix = f"__{normalize_category_name(category)}"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def rel_link(path: Path, md_path: Path) -> str:
    return Path(os.path.relpath(path, start=md_path.parent)).as_posix()


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            image = row.get("image")
            if not image:
                raise ValueError(f"Missing image field at {path}:{line_no}")
            rows[Path(image).name] = row
    return rows


def read_lisa_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="cp1252"))


def resolve_existing_path(path_value: str | None, root: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(root / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_image_for_json(json_path: Path) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = json_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def draw_corner_tag(
    image: Image.Image,
    title: str,
    subtitle: str = "",
    color: tuple[int, int, int] = (32, 32, 32),
) -> None:
    draw = ImageDraw.Draw(image)
    title_font = load_font(20)
    subtitle_font = load_font(15)
    pad = 8
    lines = [title] + ([subtitle] if subtitle else [])
    widths = [
        int(draw.textlength(line, font=title_font if idx == 0 else subtitle_font))
        for idx, line in enumerate(lines)
    ]
    height = 28 + (21 if subtitle else 0)
    rect = (8, 8, 8 + max(widths, default=80) + pad * 2, 8 + height + pad)
    draw.rounded_rectangle(rect, radius=7, fill=color)
    y = 14
    draw.text((16, y), title, fill=(255, 255, 255), font=title_font)
    if subtitle:
        y += 25
        draw.text((16, y), subtitle, fill=(230, 230, 230), font=subtitle_font)


def placeholder_panel(
    size: tuple[int, int],
    title: str,
    message: str,
    color: tuple[int, int, int] = (72, 72, 72),
) -> Image.Image:
    image = Image.new("RGB", size, (238, 238, 238))
    draw = ImageDraw.Draw(image)
    text_font = load_font(16)
    draw_corner_tag(image, title, color=color)
    max_width = image.width - 40
    y = 90
    for line in wrap_text(message, draw, text_font, max_width):
        draw.text((20, y), line, fill=(48, 48, 48), font=text_font)
        y += 22
    return image


def wrap_text(
    text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int
) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in str(text):
        if char == "\n":
            lines.append(current)
            current = ""
            continue
        candidate = current + char
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def draw_coco_segmentation(
    draw: ImageDraw.ImageDraw,
    segmentation: Any,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
) -> int:
    count = 0
    if not isinstance(segmentation, list):
        return count
    for segment in segmentation:
        if not isinstance(segment, list) or len(segment) < 6:
            continue
        points = [
            (float(segment[idx]), float(segment[idx + 1]))
            for idx in range(0, len(segment) - 1, 2)
        ]
        draw.polygon(points, fill=fill)
        draw.line(points + [points[0]], fill=outline, width=3)
        count += 1
    return count


class CocoIndex:
    def __init__(self, coco_root: Path):
        self.coco_root = coco_root
        self.datasets: list[dict[str, Any]] = []
        self.by_file: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        self.by_prefixed_file: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = (
            defaultdict(list)
        )
        self.by_split_image_id: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = (
            defaultdict(list)
        )
        self._load()

    def _load(self) -> None:
        for ann_path in sorted(self.coco_root.rglob("_annotations.coco.json")):
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            categories = {cat["id"]: cat for cat in data.get("categories", [])}
            anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for ann in data.get("annotations", []):
                if ann.get("category_id") in categories:
                    anns_by_image[int(ann["image_id"])].append(ann)

            dataset = {
                "ann_path": ann_path,
                "tag": ann_path.parent.name,
                "categories": categories,
                "anns_by_image": anns_by_image,
            }
            self.datasets.append(dataset)
            for image in data.get("images", []):
                file_name = Path(image["file_name"]).name
                item = (dataset, image)
                self.by_file[file_name].append(item)
                self.by_file[strip_dataset_prefix(file_name)].append(item)
                self.by_prefixed_file[f"{dataset['tag']}__{file_name}"].append(item)
                self.by_split_image_id[(dataset["tag"], int(image["id"]))].append(item)

    def find(self, lisa_anno: dict[str, Any], image_name: str, category: str):
        source = lisa_anno.get("source") or {}
        source_file = Path(str(source.get("file_name", ""))).name
        source_split = str(source.get("split", "")).strip()
        source_image_id = source.get("image_id")
        source_stem = sample_source_stem(image_name, category)

        keys = []
        if source_file:
            keys.extend([source_file, strip_dataset_prefix(source_file)])
        if source_stem:
            keys.extend([f"{source_stem}.jpg", strip_dataset_prefix(f"{source_stem}.jpg")])

        for key in keys:
            matches = self.by_prefixed_file.get(key) or self.by_file.get(key)
            if matches:
                return self._prefer_match(matches, source_file, source_split)

        if source_split and source_image_id is not None:
            try:
                matches = self.by_split_image_id.get((source_split, int(source_image_id)))
            except (TypeError, ValueError):
                matches = None
            if matches:
                return self._prefer_match(matches, source_file, source_split)

        return None

    @staticmethod
    def _prefer_match(
        matches: list[tuple[dict[str, Any], dict[str, Any]]],
        source_file: str,
        source_split: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        source_prefix = ""
        prefix_match = re.match(r"^(\d{3})__", source_file)
        if prefix_match:
            source_prefix = prefix_match.group(1)
        for dataset, image in matches:
            if source_prefix and dataset["tag"] == source_prefix:
                return dataset, image
        for dataset, image in matches:
            if source_split and dataset["tag"] == source_split:
                return dataset, image
        return matches[0]


def draw_coco_annotations(
    base_image: Image.Image,
    anns: list[dict[str, Any]],
    categories: dict[int, dict[str, Any]],
    title: str,
    subtitle: str,
) -> Image.Image:
    panel = base_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    label_draw = ImageDraw.Draw(panel)
    font = load_font(17)
    for ann in anns:
        color = COCO_COLORS[int(ann["category_id"]) % len(COCO_COLORS)]
        fill = (*color, 78)
        outline = (*color, 255)
        draw_coco_segmentation(draw, ann.get("segmentation"), fill=fill, outline=outline)
        x, y, w, h = ann["bbox"]
        box = (round(x), round(y), round(x + w), round(y + h))
        draw.rectangle(box, outline=outline, width=4)
        label = str(categories[ann["category_id"]]["name"])
        bbox = label_draw.textbbox((box[0], max(0, box[1] - 24)), label, font=font)
        label_draw.rectangle(
            (bbox[0] - 5, bbox[1] - 3, bbox[2] + 5, bbox[3] + 3), fill=color
        )
        label_draw.text((bbox[0], bbox[1]), label, fill=(255, 255, 255), font=font)

    panel = Image.alpha_composite(panel, overlay).convert("RGB")
    draw_corner_tag(panel, title, subtitle, color=(93, 68, 31))
    return panel


def draw_coco_panels(
    base_image: Image.Image,
    coco_index: CocoIndex,
    lisa_anno: dict[str, Any],
    row: dict[str, Any],
) -> tuple[Image.Image, Image.Image, int, int]:
    category = row.get("source_category") or row.get("sample_key") or ""
    sample_key = normalize_category_name(str(row.get("sample_key") or category))
    source_category = str((lisa_anno.get("source") or {}).get("source_category") or category)
    source_category_norm = normalize_category_name(source_category)
    found = coco_index.find(lisa_anno, Path(row["image"]).name, str(category))

    if not found:
        message = "No matching COCO image was found. Check --coco-root and source file names."
        return (
            placeholder_panel(base_image.size, "COCO source", message, color=(93, 68, 31)),
            placeholder_panel(base_image.size, "COCO target", message, color=(93, 68, 31)),
            0,
            0,
        )

    dataset, image_info = found
    anns = dataset["anns_by_image"].get(int(image_info["id"]), [])
    categories = dataset["categories"]
    target_anns = []
    for ann in anns:
        cat_name = str(categories[ann["category_id"]]["name"])
        cat_norm = normalize_category_name(cat_name)
        if cat_norm in {sample_key, source_category_norm}:
            target_anns.append(ann)

    source_panel = draw_coco_annotations(
        base_image,
        anns,
        categories,
        "COCO source annotations",
        f"{dataset['tag']} / all boxes: {len(anns)}",
    )
    target_panel = draw_coco_annotations(
        base_image,
        target_anns,
        categories,
        "COCO target annotations",
        f"{dataset['tag']} / target boxes: {len(target_anns)}",
    )
    return source_panel, target_panel, len(anns), len(target_anns)


def draw_lisa_panel(base_image: Image.Image, lisa_anno: dict[str, Any]) -> tuple[Image.Image, int]:
    panel = base_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    shape_count = 0
    for shape in lisa_anno.get("shapes", []):
        points = shape.get("points") or []
        if len(points) < 3:
            continue
        polygon = [(float(x), float(y)) for x, y in points]
        label = str(shape.get("label", "target")).lower()
        if "ignore" in label:
            fill = (255, 55, 55, 95)
            outline = (220, 0, 0, 255)
        else:
            fill = (0, 210, 90, 105)
            outline = (0, 145, 58, 255)
        draw.polygon(polygon, fill=fill)
        draw.line(polygon + [polygon[0]], fill=outline, width=3)
        shape_count += 1
    panel = Image.alpha_composite(panel, overlay).convert("RGB")
    draw_corner_tag(panel, "LISA annotation", f"polygons: {shape_count}", color=(28, 92, 49))
    return panel, shape_count


def draw_prediction_panel(
    base_image: Image.Image,
    row: dict[str, Any],
    title: str,
    repo_root: Path,
    color: tuple[int, int, int],
) -> Image.Image:
    mask_path = resolve_existing_path(row.get("pred_mask_path"), repo_root)
    if not mask_path:
        return placeholder_panel(
            base_image.size,
            title,
            f"Missing prediction mask: {row.get('pred_mask_path', '')}",
            color=color,
        )

    mask = Image.open(mask_path).convert("L")
    if mask.size != base_image.size:
        mask = mask.resize(base_image.size, resample=Image.Resampling.NEAREST)
    mask = mask.point(lambda value: 150 if value > 0 else 0)
    overlay = Image.new("RGBA", base_image.size, (*color, 0))
    overlay.putalpha(mask)
    panel = Image.alpha_composite(base_image.convert("RGBA"), overlay).convert("RGB")
    subtitle = (
        f"IoU {float(row.get('iou', 0.0)):.4f} / "
        f"P {float(row.get('precision', 0.0)):.3f} / "
        f"R {float(row.get('recall', 0.0)):.3f}"
    )
    draw_corner_tag(panel, title, subtitle, color=color)
    return panel


def resize_panel(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) <= max_side:
        return image
    scale = max_side / float(max(image.size))
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(new_size, resample=Image.Resampling.LANCZOS)


def save_panel(image: Image.Image, path: Path, max_side: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = resize_panel(image, max_side=max_side)
    image.save(path, quality=92)


def build_rows(base_rows: dict[str, dict[str, Any]], tuned_rows: dict[str, dict[str, Any]]):
    rows = []
    for image_name in sorted(set(base_rows) & set(tuned_rows)):
        base = base_rows[image_name]
        tuned = tuned_rows[image_name]
        rows.append(
            {
                "image_name": image_name,
                "base": base,
                "tuned": tuned,
                "category": tuned.get("source_category") or tuned.get("sample_key") or "",
                "prompt": tuned.get("prompt", ""),
                "base_iou": float(base.get("iou", 0.0)),
                "tuned_iou": float(tuned.get("iou", 0.0)),
                "delta_iou": float(tuned.get("iou", 0.0)) - float(base.get("iou", 0.0)),
            }
        )
    return rows


def filter_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.only == "bad":
        rows = [row for row in rows if row["tuned_iou"] < args.bad_threshold]
    elif args.only == "improved":
        rows = [row for row in rows if row["delta_iou"] > 0]
    elif args.only == "regressed":
        rows = [row for row in rows if row["delta_iou"] < 0]

    if args.category:
        categories = {normalize_category_name(item) for item in args.category.split(",")}
        rows = [row for row in rows if normalize_category_name(row["category"]) in categories]

    if args.sort_by == "tuned_iou":
        rows = sorted(rows, key=lambda row: (row["tuned_iou"], row["category"], row["image_name"]))
    elif args.sort_by == "base_iou":
        rows = sorted(rows, key=lambda row: (row["base_iou"], row["category"], row["image_name"]))
    elif args.sort_by == "delta_iou":
        rows = sorted(rows, key=lambda row: row["delta_iou"], reverse=True)
    elif args.sort_by == "regression":
        rows = sorted(rows, key=lambda row: row["delta_iou"])
    elif args.sort_by == "category":
        rows = sorted(rows, key=lambda row: (row["category"], row["tuned_iou"], row["image_name"]))
    else:
        rows = sorted(rows, key=lambda row: int(row["tuned"].get("sample_index", 0)))

    if args.limit:
        rows = rows[: args.limit]
    return rows


def make_safe_stem(index: int, image_name: str) -> str:
    stem = Path(image_name).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return f"{index:04d}_{stem}"


def tuned_visualization_markdown_path(row: dict[str, Any]) -> Path:
    path = row.get("visualization_markdown_path") or row.get(
        "visualization_label_path"
    )
    if not path and row.get("visualization_path"):
        path = str(Path(row["visualization_path"]).with_suffix(".md"))
    if not path:
        raise ValueError(f"Missing tuned visualization Markdown path: {row.get('image')}")
    path = Path(path)
    if path.suffix == ".txt":
        path = path.with_suffix(".md")
    return path


def write_tuned_comparison_page(
    md_path: Path,
    row: dict[str, Any],
    base: dict[str, Any],
    tuned: dict[str, Any],
    paths: dict[str, Path],
    source_count: int,
    target_count: int,
    lisa_count: int,
) -> None:
    delta = float(tuned.get("iou", 0.0)) - float(base.get("iou", 0.0))
    lines = [
        "# Base / Tuned Sample Comparison",
        "",
        f"- Sample: `{row['image_name']}`",
        f"- Category: `{row['category']}`",
        f"- Prompt: {row['prompt']}",
        (
            f"- Base IoU: `{row['base_iou']:.4f}` | "
            f"Tuned IoU: `{row['tuned_iou']:.4f}` | Delta: `{delta:+.4f}`"
        ),
        (
            f"- COCO source boxes: `{source_count}` | COCO target boxes: `{target_count}` | "
            f"LISA polygons: `{lisa_count}`"
        ),
        "",
        "| COCO source annotations | COCO target annotations | LISA annotations |",
        "| --- | --- | --- |",
        (
            f"| ![]({rel_link(paths['coco_source'], md_path)}) "
            f"| ![]({rel_link(paths['coco_target'], md_path)}) "
            f"| ![]({rel_link(paths['lisa'], md_path)}) |"
        ),
        "",
        "| Base benchmark prediction | Tuned prediction |",
        "| --- | --- |",
        (
            f"| ![]({rel_link(paths['base'], md_path)}) "
            f"| ![]({rel_link(paths['tuned'], md_path)}) |"
        ),
        "",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def generate_report(args: argparse.Namespace) -> Path:
    repo_root = Path.cwd()
    output_dir = args.output_dir
    asset_dir = output_dir / "assets"
    md_path = output_dir / "report.md"
    if not args.update_tuned_pages:
        output_dir.mkdir(parents=True, exist_ok=True)
        asset_dir.mkdir(parents=True, exist_ok=True)

    base_rows = load_jsonl(args.base_metrics)
    tuned_rows = load_jsonl(args.tuned_metrics)
    rows = filter_rows(build_rows(base_rows, tuned_rows), args)
    if not rows:
        raise SystemExit("No matched rows after filtering.")

    coco_index = CocoIndex(args.coco_root)
    missing_assets = 0
    lines = [
        "# Annotation And Prediction Comparison",
        "",
        f"- Base metrics: `{args.base_metrics}`",
        f"- Tuned metrics: `{args.tuned_metrics}`",
        f"- COCO root: `{args.coco_root}`",
        f"- Samples in report: `{len(rows)}`",
        f"- Filter: `{args.only}`",
        "",
    ]

    for rank, row in enumerate(rows, start=1):
        tuned = row["tuned"]
        base = row["base"]
        image_path = resolve_existing_path(tuned.get("image"), repo_root)
        json_path = resolve_existing_path(tuned.get("dataset_json_path"), repo_root)
        if not json_path:
            guessed_json = Path(str(tuned.get("image", ""))).with_suffix(".json")
            json_path = resolve_existing_path(str(guessed_json), repo_root)

        if image_path:
            base_image = Image.open(image_path).convert("RGB")
        else:
            base_image = placeholder_panel(
                (512, 512),
                "Missing image",
                str(tuned.get("image", "")),
                color=(95, 40, 40),
            )
            missing_assets += 1

        if json_path:
            lisa_anno = read_lisa_json(json_path)
            coco_source_panel, coco_target_panel, source_count, target_count = draw_coco_panels(
                base_image, coco_index, lisa_anno, tuned
            )
            lisa_panel, lisa_count = draw_lisa_panel(base_image, lisa_anno)
        else:
            lisa_anno = {}
            source_count = 0
            target_count = 0
            lisa_count = 0
            coco_source_panel = placeholder_panel(
                base_image.size,
                "COCO source annotations",
                "Missing LISA JSON, cannot locate source COCO annotation.",
                color=(93, 68, 31),
            )
            coco_target_panel = placeholder_panel(
                base_image.size,
                "COCO target annotations",
                "Missing LISA JSON, cannot locate target COCO annotation.",
                color=(93, 68, 31),
            )
            lisa_panel = placeholder_panel(
                base_image.size,
                "LISA annotation",
                str(tuned.get("dataset_json_path", "")),
                color=(28, 92, 49),
            )
            missing_assets += 1

        base_pred_panel = draw_prediction_panel(
            base_image, base, "Base prediction", repo_root, color=(36, 96, 180)
        )
        tuned_pred_panel = draw_prediction_panel(
            base_image, tuned, "LoRA prediction", repo_root, color=(180, 70, 34)
        )

        sample_index = int(tuned.get("sample_index", rank - 1))
        safe_stem = make_safe_stem(sample_index, row["image_name"])
        if args.update_tuned_pages:
            sample_md_path = tuned_visualization_markdown_path(tuned)
            sample_asset_dir = sample_md_path.parent / "comparison_assets"
        else:
            sample_md_path = md_path
            sample_asset_dir = asset_dir
        paths = {
            "coco_source": sample_asset_dir / f"{safe_stem}_coco_source.jpg",
            "coco_target": sample_asset_dir / f"{safe_stem}_coco_target.jpg",
            "lisa": sample_asset_dir / f"{safe_stem}_lisa.jpg",
            "base": sample_asset_dir / f"{safe_stem}_base_pred.jpg",
            "tuned": sample_asset_dir / f"{safe_stem}_tuned_pred.jpg",
        }
        save_panel(coco_source_panel, paths["coco_source"], args.max_side)
        save_panel(coco_target_panel, paths["coco_target"], args.max_side)
        save_panel(lisa_panel, paths["lisa"], args.max_side)
        save_panel(base_pred_panel, paths["base"], args.max_side)
        save_panel(tuned_pred_panel, paths["tuned"], args.max_side)

        if args.update_tuned_pages:
            write_tuned_comparison_page(
                sample_md_path,
                row,
                base,
                tuned,
                paths,
                source_count,
                target_count,
                lisa_count,
            )
            continue

        delta = row["delta_iou"]
        lines.extend(
            [
                f"## {rank:03d}. `{row['image_name']}`",
                "",
                (
                    f"- Category: `{row['category']}` | "
                    f"Base IoU: `{row['base_iou']:.4f}` | "
                    f"LoRA IoU: `{row['tuned_iou']:.4f}` | "
                    f"Delta: `{delta:+.4f}`"
                ),
                f"- Prompt: {row['prompt']}",
                (
                    f"- COCO source boxes: `{source_count}` | "
                    f"COCO target boxes: `{target_count}` | LISA polygons: `{lisa_count}`"
                ),
                "",
                "| COCO source annotations | COCO target annotations | LISA annotations | Base benchmark prediction | Tuned prediction |",
                "| --- | --- | --- | --- | --- |",
                (
                    f"| ![]({rel_link(paths['coco_source'], md_path)}) "
                    f"| ![]({rel_link(paths['coco_target'], md_path)}) "
                    f"| ![]({rel_link(paths['lisa'], md_path)}) "
                    f"| ![]({rel_link(paths['base'], md_path)}) "
                    f"| ![]({rel_link(paths['tuned'], md_path)}) |"
                ),
                "",
            ]
        )

    if args.update_tuned_pages:
        print(f"[done] updated tuned visualization pages: {len(rows)}")
        return args.tuned_metrics

    if missing_assets:
        lines.extend(
            [
                "## Notes",
                "",
                (
                    f"- Missing image/json assets were encountered for `{missing_assets}` "
                    "sample(s). The report includes placeholder panels for them."
                ),
                "",
            ]
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] report: {md_path}")
    print(f"[done] assets: {asset_dir}")
    return md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Markdown report comparing COCO/LISA annotations and predictions."
    )
    parser.add_argument("--base-metrics", default=DEFAULT_BASE_METRICS, type=Path)
    parser.add_argument("--tuned-metrics", default=DEFAULT_TUNED_METRICS, type=Path)
    parser.add_argument(
        "--coco-root",
        default=DEFAULT_COCO_ROOT,
        type=Path,
        help="Root searched recursively for _annotations.coco.json files.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument(
        "--update-tuned-pages",
        action="store_true",
        help=(
            "Replace each tuned visualization Markdown page with a five-panel "
            "COCO/LISA/base/tuned comparison. Generated JPG assets are stored "
            "beside the tuned pages under comparison_assets/."
        ),
    )
    parser.add_argument(
        "--only",
        choices=["all", "bad", "improved", "regressed"],
        default="all",
        help="Subset to include.",
    )
    parser.add_argument("--bad-threshold", default=0.10, type=float)
    parser.add_argument(
        "--sort-by",
        choices=["sample_index", "tuned_iou", "base_iou", "delta_iou", "regression", "category"],
        default="sample_index",
    )
    parser.add_argument(
        "--category",
        help="Optional comma-separated category filter, e.g. unsafe,no_helmet.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-side",
        default=480,
        type=int,
        help="Maximum side length for each embedded panel image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_pillow()
    if not args.base_metrics.exists():
        raise FileNotFoundError(f"Missing base metrics: {args.base_metrics}")
    if not args.tuned_metrics.exists():
        raise FileNotFoundError(f"Missing tuned metrics: {args.tuned_metrics}")
    if not args.coco_root.exists():
        raise FileNotFoundError(f"Missing COCO root: {args.coco_root}")
    if args.max_side < 128:
        raise ValueError("--max-side must be at least 128")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if not math.isfinite(args.bad_threshold):
        raise ValueError("--bad-threshold must be finite")
    generate_report(args)


if __name__ == "__main__":
    main()
