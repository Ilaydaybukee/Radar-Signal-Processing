"""Create contact sheets for sea_candidate or ship_candidate patch folders."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def find_images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def load_font(size: int = 12) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def create_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    thumb_size: int = 128,
    columns: int = 5,
    label_height: int = 34,
    limit: int | None = None,
) -> None:
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise ValueError("No images found for contact sheet.")

    font = load_font()
    rows = math.ceil(len(image_paths) / columns)
    cell_w = thumb_size
    cell_h = thumb_size + label_height
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, path in enumerate(image_paths):
        col = idx % columns
        row = idx // columns
        x = col * cell_w
        y = row * cell_h

        with Image.open(path) as image:
            image = image.convert("L")
            image.thumbnail((thumb_size, thumb_size))
            tile = Image.new("RGB", (thumb_size, thumb_size), "black")
            paste_x = (thumb_size - image.width) // 2
            paste_y = (thumb_size - image.height) // 2
            tile.paste(image.convert("RGB"), (paste_x, paste_y))

        sheet.paste(tile, (x, y))
        label = path.name
        if len(label) > 24:
            label = label[:21] + "..."
        draw.text((x + 3, y + thumb_size + 3), label, fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_folder", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--thumb-size", type=int, default=128)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.candidate_folder / f"{args.candidate_folder.name}_review_grid.jpg"
    images = find_images(args.candidate_folder)
    create_contact_sheet(images, output, args.thumb_size, args.columns, limit=args.limit)
    print(f"Saved review grid: {output}")


if __name__ == "__main__":
    main()
