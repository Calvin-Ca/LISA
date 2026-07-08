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
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_OUTPUT_DIR = Path("data/phase1_feasibility/lisa_visualizations")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_instruction(json_path: Path) -> str:
    anno = json.loads(json_path.read_text(encoding="utf-8"))
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
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def normalize_points(points: list[list[float]]) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in points]


def draw_shapes(image: Image.Image, json_path: Path) -> tuple[Image.Image, str, int]:
    anno = json.loads(json_path.read_text(encoding="utf-8"))
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
    return Image.alpha_composite(image.convert("RGBA"), overlay), instruction, shape_count


def draw_text_panel(canvas: Image.Image, instruction: str, json_name: str, shape_count: int) -> None:
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(18)
    text_font = load_font(16)
    draw.rectangle((0, 0, canvas.width, 84), fill=(24, 24, 24))
    draw.text((12, 10), json_name, fill=(235, 235, 235), font=title_font)
    meta = f"target/ignore polygons: {shape_count}"
    draw.text((12, 34), meta, fill=(180, 220, 255), font=text_font)
    for idx, line in enumerate(textwrap.wrap(instruction, width=52)[:2]):
        draw.text((12, 56 + idx * 20), line, fill=(210, 240, 210), font=text_font)


def visualize_pair(image_path: Path, json_path: Path, output_dir: Path) -> Path:
    image = Image.open(image_path).convert("RGB")
    annotated, instruction, shape_count = draw_shapes(image, json_path)

    canvas = Image.new("RGB", (image.width * 2, image.height + 84), (255, 255, 255))
    canvas.paste(image, (0, 84))
    canvas.paste(annotated.convert("RGB"), (image.width, 84))
    draw_text_panel(canvas, instruction, json_path.name, shape_count)

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
