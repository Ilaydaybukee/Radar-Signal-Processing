"""Build a small balanced SAR frequency-band demo dataset.

The output dataset is written outside the repository by default:
C:\\SAR_Datasets\\multifrequency_band_demo

This script copies normalized 256x256 grayscale PNGs. It does not move,
delete, compress, or modify source imagery.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path(r"C:\SAR_Datasets\multifrequency_band_demo")
DEFAULT_C_BAND_ROOT = REPO_ROOT / "data" / "processed"
DEFAULT_S_BAND_ROOT = Path(r"C:\SAR_Datasets\novasar_sband\multiclass_clean_6class\balanced_158")
DEFAULT_L_BAND_ROOTS = [
    Path(r"C:\SAR_Datasets\lband_alos\lband_ship_sea_pilot"),
    Path(r"C:\SAR_Datasets\lband_alos\final_datasets"),
    Path(r"C:\SAR_Datasets\lband_alos\candidate_patches"),
]

BANDS = ["C_band", "S_band", "L_band"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_paths(values: list[str] | None, defaults: list[Path]) -> list[Path]:
    if not values:
        return defaults
    paths: list[Path] = []
    for value in values:
        for item in value.split(";"):
            item = item.strip()
            if item:
                paths.append(Path(item))
    return paths


def find_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def collect_sources(c_root: Path, s_root: Path, l_roots: list[Path]) -> dict[str, list[Path]]:
    sources = {
        "C_band": find_images(c_root),
        "S_band": find_images(s_root),
        "L_band": [],
    }
    for root in l_roots:
        sources["L_band"].extend(find_images(root))
    sources["L_band"] = sorted(set(sources["L_band"]))
    return sources


def reset_output(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def make_png(image_path: Path, output_path: Path, image_size: int, augmentation: str = "none") -> None:
    with Image.open(image_path) as image:
        image = image.convert("L")
        image = ImageOps.autocontrast(image)
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        if augmentation == "hflip":
            image = ImageOps.mirror(image)
        elif augmentation == "vflip":
            image = ImageOps.flip(image)
        elif augmentation == "rot90":
            image = image.rotate(90, expand=False)
        elif augmentation == "rot180":
            image = image.rotate(180, expand=False)
        elif augmentation == "rot270":
            image = image.rotate(270, expand=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)


def split_names(count: int, train_fraction: float, val_fraction: float) -> list[str]:
    train_count = max(1, round(count * train_fraction))
    val_count = max(1, round(count * val_fraction)) if count >= 3 else 0
    if train_count + val_count >= count:
        train_count = max(1, count - 1)
        val_count = 0 if count < 3 else 1
    test_count = count - train_count - val_count
    return ["train"] * train_count + ["val"] * val_count + ["test"] * test_count


def build_dataset(
    sources: dict[str, list[Path]],
    output_root: Path,
    image_size: int,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    augment_lband: bool,
) -> pd.DataFrame:
    missing = [band for band, paths in sources.items() if not paths]
    if missing:
        raise RuntimeError(f"No source images found for: {', '.join(missing)}")

    rng = random.Random(seed)
    per_band_count = min(len(paths) for paths in sources.values())
    if per_band_count <= 0:
        raise RuntimeError("At least one band has zero images.")

    records: list[dict[str, object]] = []
    reset_output(output_root)

    for band in BANDS:
        selected = list(sources[band])
        rng.shuffle(selected)
        selected = selected[:per_band_count]
        splits = split_names(len(selected), train_fraction, val_fraction)
        rng.shuffle(splits)

        for idx, (source_path, split) in enumerate(zip(selected, splits)):
            filename = f"{band}_{idx:04d}.png"
            raw_path = output_root / "raw" / band / filename
            split_path = output_root / split / band / filename
            make_png(source_path, raw_path, image_size)
            split_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_path, split_path)
            records.append(
                {
                    "band": band,
                    "split": split,
                    "source_path": str(source_path),
                    "raw_path": str(raw_path),
                    "split_path": str(split_path),
                    "augmentation": "none",
                }
            )

            if augment_lband and band == "L_band":
                for augmentation in ["hflip", "vflip", "rot90", "rot180", "rot270"]:
                    aug_filename = f"{band}_{idx:04d}_{augmentation}.png"
                    aug_raw_path = output_root / "raw_augmented" / band / aug_filename
                    make_png(source_path, aug_raw_path, image_size, augmentation=augmentation)

    manifest = pd.DataFrame(records)
    manifest.to_csv(output_root / "manifest.csv", index=False)
    write_summary(output_root, manifest, sources, per_band_count, image_size, augment_lband)
    return manifest


def write_summary(
    output_root: Path,
    manifest: pd.DataFrame,
    sources: dict[str, list[Path]],
    per_band_count: int,
    image_size: int,
    augment_lband: bool,
) -> None:
    summary = {
        "output_root": str(output_root),
        "image_size": image_size,
        "classes": BANDS,
        "available_source_counts": {band: len(paths) for band, paths in sources.items()},
        "selected_per_band": per_band_count,
        "split_counts": {
            f"{split}/{band}": int(count)
            for (split, band), count in manifest.groupby(["split", "band"]).size().items()
        },
        "lband_augmented_preview_written": augment_lband,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "Multi-frequency SAR band demo dataset",
        f"Output: {output_root}",
        f"Image size: {image_size}x{image_size} grayscale PNG",
        f"Selected images per band: {per_band_count}",
        "",
        "Available source counts:",
    ]
    lines.extend(f"- {band}: {len(paths)}" for band, paths in sources.items())
    lines.extend(["", "Split counts:", manifest.groupby(["split", "band"]).size().to_string()])
    (output_root / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--c-band-root", type=Path, default=DEFAULT_C_BAND_ROOT)
    parser.add_argument("--s-band-root", type=Path, default=DEFAULT_S_BAND_ROOT)
    parser.add_argument("--l-band-root", action="append", help="L-band source root. Repeat or separate with semicolons.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--augment-lband", action="store_true", help="Also save simple L-band augmentation previews.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    l_roots = parse_paths(args.l_band_root, DEFAULT_L_BAND_ROOTS)
    sources = collect_sources(args.c_band_root, args.s_band_root, l_roots)

    for band in BANDS:
        print(f"{band}: found {len(sources[band])} source images")

    manifest = build_dataset(
        sources=sources,
        output_root=args.output_root,
        image_size=args.image_size,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        augment_lband=args.augment_lband,
    )
    print(f"Built demo dataset: {args.output_root}")
    print(f"Manifest rows: {len(manifest)}")


if __name__ == "__main__":
    main()
