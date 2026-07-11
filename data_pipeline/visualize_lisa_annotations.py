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
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


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
    return (
        Image.alpha_composite(image.convert("RGBA"), overlay),
        anno,
        instruction,
        shape_count,
    )


def draw_text_panel(
    canvas: Image.Image,
    anno: dict,
    instruction: str,
    json_name: str,
    shape_count: int,
    panel_height: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(18)
    text_font = load_font(17)
    draw.rectangle((0, 0, canvas.width, panel_height), fill=(24, 24, 24))
    draw.text((12, 10), json_name, fill=(235, 235, 235), font=title_font)
    meta_lines = build_meta_lines(anno, shape_count)
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
) -> Path:
    image = Image.open(image_path).convert("RGB")
    annotated, anno, instruction, shape_count = draw_shapes(image, json_path)

    panel_height = 200
    canvas = Image.new(
        "RGB", (image.width * 2, image.height + panel_height), (255, 255, 255)
    )
    canvas.paste(image, (0, panel_height))
    canvas.paste(annotated.convert("RGB"), (image.width, panel_height))
    draw_text_panel(canvas, anno, instruction, json_path.name, shape_count, panel_height)

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

    manifest_rows = []
    for image_path, json_path in pairs:
        output_path = visualize_pair(image_path, json_path, args.output_dir)
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
