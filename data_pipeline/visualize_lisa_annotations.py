"""
Visualize LISA/ReasonSeg JSON annotations on their paired images.

Default output:
  data/phase1_feasibility/lisa_visualizations/

Examples:
  # Visualize one image/json pair
  python data_pipeline/visualize_lisa_annotations.py \
    --image path/to/sample.jpg \
    --json path/to/sample.json

  # Visualize all jpg/json pairs under a split directory
  python data_pipeline/visualize_lisa_annotations.py \
    --input-dir dataset/reason_seg/ReasonSeg/train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_OUTPUT_DIR = Path("data/phase1_feasibility/lisa_visualizations")
DEFAULT_COCO_ROOT = Path("data/phase1_feasibility")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
COCO_PALETTE = [
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


def normalize_category_name(name: str) -> str:
    return (
        str(name).strip().lower().replace(" ", "_").replace("-", "_")
    )


def read_annotation(json_path: Path) -> dict:
    return json.loads(json_path.read_text(encoding="utf-8"))


def read_instruction(json_path: Path) -> str:
    anno = read_annotation(json_path)
    text = anno.get("text") or []
    return str(text[0]) if text else ""


def find_image_for_json(json_path: Path) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = json_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def collect_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for json_path in sorted(input_dir.rglob("*.json")):
        if json_path.name in {"split.json", "phase1_build_summary.json"}:
            continue
        image_path = find_image_for_json(json_path)
        if image_path is not None:
            pairs.append((image_path, json_path))
    return pairs


def load_font(size: int) -> ImageFont.ImageFont:
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
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def wrap_by_pixel_width(
    text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int
) -> list[str]:
    lines = []
    current = ""
    for char in text:
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


def normalize_points(points: list[list[float]]) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in points]


def color_for_category(category_id: int) -> tuple[int, int, int]:
    return COCO_PALETTE[category_id % len(COCO_PALETTE)]


def draw_corner_tag(
    image: Image.Image,
    text: str,
    *,
    x: int = 10,
    y: int = 10,
    bg_color: tuple[int, int, int] = (24, 24, 24),
) -> None:
    draw = ImageDraw.Draw(image)
    font = load_font(18)
    bbox = draw.textbbox((x, y), text, font=font)
    pad_x = 8
    pad_y = 6
    rect = (
        bbox[0] - pad_x,
        bbox[1] - pad_y,
        bbox[2] + pad_x,
        bbox[3] + pad_y,
    )
    draw.rounded_rectangle(rect, radius=8, fill=bg_color)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def load_coco_index(coco_root: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for split in ("train", "val"):
        ann_path = coco_root / split / "_annotations.coco.json"
        if not ann_path.exists():
            continue
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        images = {img["id"]: img for img in data.get("images", [])}
        categories = {cat["id"]: cat for cat in data.get("categories", [])}
        anns_by_image: dict[int, list[dict]] = {}
        for ann in data.get("annotations", []):
            image_id = ann.get("image_id")
            category_id = ann.get("category_id")
            if image_id not in images or category_id not in categories:
                continue
            anns_by_image.setdefault(image_id, []).append(ann)
        index[split] = {
            "images": images,
            "categories": categories,
            "anns_by_image": anns_by_image,
            "ann_path": ann_path,
        }
    return index


def draw_coco_boxes_panel(
    image: Image.Image,
    anns: list[dict],
    categories: dict[int, dict],
    title: str,
) -> tuple[Image.Image, int]:
    annotated = image.copy()
    draw_corner_tag(annotated, title, bg_color=(40, 64, 96))
    draw = ImageDraw.Draw(annotated)
    font = load_font(16)
    for ann in anns:
        x, y, w, h = ann["bbox"]
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))
        color = color_for_category(int(ann["category_id"]))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = str(categories[ann["category_id"]]["name"])
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        rect = (
            text_bbox[0] - 5,
            max(0, text_bbox[1] - 4),
            text_bbox[2] + 5,
            text_bbox[3] + 4,
        )
        draw.rectangle(rect, fill=color)
        draw.text((x1, rect[1] + 2), label, fill=(255, 255, 255), font=font)
    return annotated, len(anns)


def draw_coco_bboxes(
    image: Image.Image,
    anno: dict,
    coco_index: dict[str, dict],
) -> tuple[Image.Image, Image.Image, int, int]:
    source = anno.get("source") or {}
    split = str(source.get("split", "")).strip()
    source_image_id = source.get("image_id")
    source_category = str(source.get("source_category", "")).strip()
    sample_key = str(source.get("sample_key", "")).strip()
    coco_split = coco_index.get(split)

    if coco_split is None or source_image_id is None:
        all_panel, all_count = draw_coco_boxes_panel(
            image, [], {}, "COCO all categories"
        )
        target_panel, target_count = draw_coco_boxes_panel(
            image, [], {}, "COCO target category"
        )
        return all_panel, target_panel, all_count, target_count

    try:
        image_id = int(source_image_id)
    except (TypeError, ValueError):
        all_panel, all_count = draw_coco_boxes_panel(
            image, [], {}, "COCO all categories"
        )
        target_panel, target_count = draw_coco_boxes_panel(
            image, [], {}, "COCO target category"
        )
        return all_panel, target_panel, all_count, target_count

    anns = coco_split["anns_by_image"].get(image_id, [])
    categories = coco_split["categories"]
    filtered_anns = []
    for ann in anns:
        category_name = str(categories[ann["category_id"]]["name"]).strip()
        if source_category and category_name == source_category:
            filtered_anns.append(ann)
            continue
        if sample_key and normalize_category_name(category_name) == sample_key:
            filtered_anns.append(ann)

    all_panel, all_count = draw_coco_boxes_panel(
        image, anns, categories, "COCO all categories"
    )
    target_panel, target_count = draw_coco_boxes_panel(
        image, filtered_anns, categories, "COCO target category"
    )
    return all_panel, target_panel, all_count, target_count


def build_meta_lines(anno: dict, shape_count: int) -> list[tuple[str, tuple[int, int, int]]]:
    source = anno.get("source") or {}
    source_category = str(source.get("source_category", "")).strip()
    sample_key = str(source.get("sample_key", "")).strip()
    source_file_name = str(source.get("file_name", "")).strip()
    source_image_id = str(source.get("image_id", "")).strip()

    meta_lines = [
        ("target/ignore polygons: {}".format(shape_count), (180, 220, 255)),
    ]
    if source_category:
        meta_lines.append((f"COCO source label: {source_category}", (255, 225, 160)))
    if sample_key and sample_key != source_category:
        meta_lines.append((f"LISA sample_key: {sample_key}", (255, 205, 145)))
    if source_image_id:
        meta_lines.append((f"COCO image id: {source_image_id}", (190, 190, 190)))
    if source_file_name:
        meta_lines.append((f"COCO file: {source_file_name}", (190, 190, 190)))
    return meta_lines


def draw_shapes(image: Image.Image, json_path: Path) -> tuple[Image.Image, dict, str, int]:
    anno = read_annotation(json_path)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    shape_count = 0

    for shape in anno.get("shapes", []):
        points = normalize_points(shape.get("points", []))
        if len(points) < 3:
            continue

        label = str(shape.get("label", "target")).lower()
        if "ignore" in label:
            fill = (255, 60, 60, 115)
            outline = (220, 0, 0, 255)
        else:
            fill = (0, 220, 80, 115)
            outline = (0, 150, 40, 255)

        draw.polygon(points, fill=fill)
        draw.line(points + [points[0]], fill=outline, width=2)
        shape_count += 1

    instruction = str((anno.get("text") or [""])[0])
    annotated = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw_corner_tag(annotated, "LISA annotations", bg_color=(20, 92, 44))
    return (annotated, anno, instruction, shape_count)


def draw_text_panel(
    canvas: Image.Image,
    anno: dict,
    instruction: str,
    json_name: str,
    shape_count: int,
    coco_all_box_count: int,
    coco_target_box_count: int,
    panel_height: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(18)
    text_font = load_font(17)
    draw.rectangle((0, 0, canvas.width, panel_height), fill=(24, 24, 24))
    draw.text((12, 10), json_name, fill=(235, 235, 235), font=title_font)
    meta_lines = build_meta_lines(anno, shape_count)
    meta_lines.insert(1, (f"COCO all-category boxes: {coco_all_box_count}", (170, 205, 255)))
    meta_lines.insert(2, (f"COCO target-category boxes: {coco_target_box_count}", (170, 205, 255)))
    y = 34
    line_gap = 21
    for meta_text, color in meta_lines:
        wrapped = wrap_by_pixel_width(meta_text, draw, text_font, canvas.width - 24)
        for line in wrapped:
            draw.text((12, y), line, fill=color, font=text_font)
            y += line_gap
    prompt_lines = wrap_by_pixel_width(
        f"LISA prompt: {instruction}", draw, text_font, canvas.width - 24
    )
    for line in prompt_lines:
        draw.text((12, y), line, fill=(210, 240, 210), font=text_font)
        y += line_gap


def visualize_pair(
    image_path: Path,
    json_path: Path,
    output_dir: Path,
    coco_index: dict[str, dict],
) -> Path:
    image = Image.open(image_path).convert("RGB")
    original = image.copy()
    draw_corner_tag(original, "Original image", bg_color=(56, 56, 56))
    annotated, anno, instruction, shape_count = draw_shapes(image, json_path)
    (
        coco_all_annotated,
        coco_target_annotated,
        coco_all_box_count,
        coco_target_box_count,
    ) = draw_coco_bboxes(image, anno, coco_index)

    panel_height = 200
    canvas = Image.new(
        "RGB", (image.width * 4, image.height + panel_height), (255, 255, 255)
    )
    canvas.paste(original, (0, panel_height))
    canvas.paste(coco_all_annotated, (image.width, panel_height))
    canvas.paste(coco_target_annotated, (image.width * 2, panel_height))
    canvas.paste(annotated, (image.width * 3, panel_height))
    draw_text_panel(
        canvas,
        anno,
        instruction,
        json_path.name,
        shape_count,
        coco_all_box_count,
        coco_target_box_count,
        panel_height,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{json_path.stem}_lisa_vis.jpg"
    canvas.save(output_path, quality=95)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize LISA JSON polygon annotations on paired images."
    )
    parser.add_argument("--image", type=Path, help="Path to one image file.")
    parser.add_argument("--json", type=Path, help="Path to one LISA JSON file.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing paired image/json files. Used when --image/--json are omitted.",
    )
    parser.add_argument(
        "--coco-root",
        default=DEFAULT_COCO_ROOT,
        type=Path,
        help="COCO subset root containing train/val/_annotations.coco.json.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--limit", default=None, type=int)
    args = parser.parse_args()

    if args.image or args.json:
        if not args.image or not args.json:
            raise SystemExit("--image and --json must be provided together.")
        pairs = [(args.image, args.json)]
    else:
        if args.input_dir is None:
            raise SystemExit("Provide either --image/--json or --input-dir.")
        pairs = collect_pairs(args.input_dir)

    if args.limit is not None:
        pairs = pairs[: args.limit]

    if not pairs:
        raise SystemExit("No paired image/json files found.")

    coco_index = load_coco_index(args.coco_root)
    manifest_rows = []
    for image_path, json_path in pairs:
        output_path = visualize_pair(image_path, json_path, args.output_dir, coco_index)
        manifest_rows.append(f"{image_path}\t{json_path}\t{output_path}")
        print(f"[ok] {output_path}")

    manifest_path = args.output_dir / "manifest.tsv"
    manifest_path.write_text(
        "image\tjson\tvisualization\n" + "\n".join(manifest_rows) + "\n",
        encoding="utf-8",
    )
    print(f"[done] {len(pairs)} visualization(s) -> {args.output_dir}")
    print(f"[done] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
