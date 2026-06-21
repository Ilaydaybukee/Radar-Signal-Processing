"""Prepare raw SAR ship/sea images for binary classification.

This script reads images from data/raw, creates a reproducible train/val/test
split, converts each image to grayscale, resizes it, saves it as PNG, and writes
a metadata.csv file for the processed dataset.
"""

import csv
from pathlib import Path
import random

# Default project settings used when config/dataset_config.yaml is missing or
# does not define a value.
CONFIG_PATH = Path("config/dataset_config.yaml")
DEFAULT_RAW_DATA_DIR = Path("data/raw")
DEFAULT_PROCESSED_DATA_DIR = Path("data/processed")
DEFAULT_CLASSES = ["ship", "sea"]
DEFAULT_PATCH_SIZE = 256
DEFAULT_SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}
DEFAULT_RANDOM_SEED = 42

# The script prepares these image file types.
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# The processed dataset always uses these split folder names.
SPLITS = ("train", "val", "test")


def read_dataset_config(config_path: Path) -> dict:
    """Read simple dataset settings from config/dataset_config.yaml.

    This beginner-friendly parser handles the small YAML structure used by this
    project without adding another package dependency.
    """
    config = {
        "raw_data_dir": DEFAULT_RAW_DATA_DIR,
        "processed_data_dir": DEFAULT_PROCESSED_DATA_DIR,
        "classes": DEFAULT_CLASSES.copy(),
        "patch_size": DEFAULT_PATCH_SIZE,
        "split_ratios": DEFAULT_SPLIT_RATIOS.copy(),
        "random_seed": DEFAULT_RANDOM_SEED,
    }

    if not config_path.exists():
        return config

    section = None
    classes = []

    for line in config_path.read_text().splitlines():
        stripped_line = line.strip()

        # Ignore blank lines and comments.
        if not stripped_line or stripped_line.startswith("#"):
            continue

        if not line.startswith(" "):
            section = None

            if stripped_line == "classes:":
                section = "classes"
                continue

            if stripped_line == "paths:":
                section = "paths"
                continue

            if stripped_line == "splits:":
                section = "splits"
                continue

            if stripped_line in {"image:", "patch:"}:
                section = "image"
                continue

            if stripped_line.startswith("random_seed:"):
                config["random_seed"] = int(stripped_line.split(":", 1)[1].strip())

            if stripped_line.startswith("patch_size:"):
                config["patch_size"] = int(stripped_line.split(":", 1)[1].strip())

            continue

        if section == "classes" and stripped_line.startswith("-"):
            classes.append(stripped_line.lstrip("- ").strip())

        if section == "paths" and ":" in stripped_line:
            name, value = stripped_line.split(":", 1)
            if name.strip() == "raw_data":
                config["raw_data_dir"] = Path(value.strip())
            if name.strip() == "processed_data":
                config["processed_data_dir"] = Path(value.strip())

        if section == "splits" and ":" in stripped_line:
            name, value = stripped_line.split(":", 1)
            name = name.strip()
            if name in config["split_ratios"]:
                config["split_ratios"][name] = float(value.strip())

        if section == "image" and ":" in stripped_line:
            name, value = stripped_line.split(":", 1)
            if name.strip() == "patch_size":
                config["patch_size"] = int(value.strip())

    if classes:
        config["classes"] = classes

    return config


def normalize_split_ratios(split_ratios: dict[str, float]) -> dict[str, float]:
    """Make split ratios add up to 1.0."""
    total = sum(split_ratios.values())

    if total <= 0:
        return DEFAULT_SPLIT_RATIOS.copy()

    return {split: ratio / total for split, ratio in split_ratios.items()}


def find_raw_images(class_folder: Path) -> list[Path]:
    """Find supported image files directly inside one raw class folder."""
    if not class_folder.exists():
        return []

    images = []

    for item in class_folder.iterdir():
        if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            images.append(item)

    # Sort first so the same seed gives the same result on every run.
    return sorted(images)


def clear_previous_processed_images(processed_data_dir: Path, classes: list[str]) -> None:
    """Clear old processed image files while keeping .gitkeep files."""
    for split in SPLITS:
        for class_name in classes:
            output_folder = processed_data_dir / split / class_name
            output_folder.mkdir(parents=True, exist_ok=True)

            for item in output_folder.iterdir():
                if item.name == ".gitkeep":
                    continue

                if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    item.unlink()


def split_images(images: list[Path], split_ratios: dict[str, float]) -> dict[str, list[Path]]:
    """Split one class of images into train, validation, and test lists."""
    image_count = len(images)
    train_count = int(image_count * split_ratios["train"])
    val_count = int(image_count * split_ratios["val"])
    test_start = train_count + val_count

    # Slicing keeps each original image in exactly one split.
    return {
        "train": images[:train_count],
        "val": images[train_count:test_start],
        "test": images[test_start:],
    }


def prepare_one_image(source_path: Path, saved_path: Path, patch_size: int) -> tuple[int, int]:
    """Open one image with Pillow, convert it to grayscale, resize, and save PNG."""
    from PIL import Image

    with Image.open(source_path) as image:
        prepared_image = image.convert("L").resize((patch_size, patch_size))
        prepared_image.save(saved_path, format="PNG")

    return patch_size, patch_size


def prepare_split_images(
    class_name: str,
    split_to_images: dict[str, list[Path]],
    processed_data_dir: Path,
    patch_size: int,
) -> tuple[dict[str, int], list[dict[str, str | int]]]:
    """Prepare and save all split images for one class."""
    saved_counts = {}
    metadata_rows = []

    for split, images in split_to_images.items():
        output_folder = processed_data_dir / split / class_name
        output_folder.mkdir(parents=True, exist_ok=True)

        for image_number, source_path in enumerate(images, start=1):
            saved_name = f"{image_number:05d}_{source_path.stem}.png"
            saved_path = output_folder / saved_name
            width, height = prepare_one_image(source_path, saved_path, patch_size)

            metadata_rows.append(
                {
                    "original_path": str(source_path),
                    "saved_path": str(saved_path),
                    "class_name": class_name,
                    "split": split,
                    "width": width,
                    "height": height,
                }
            )

        saved_counts[split] = len(images)

    return saved_counts, metadata_rows


def write_metadata(metadata_path: Path, metadata_rows: list[dict[str, str | int]]) -> None:
    """Write metadata.csv for the processed dataset."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["original_path", "saved_path", "class_name", "split", "width", "height"]

    with metadata_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)


def main() -> None:
    """Prepare the full processed dataset."""
    config = read_dataset_config(CONFIG_PATH)
    raw_data_dir = config["raw_data_dir"]
    processed_data_dir = config["processed_data_dir"]
    classes = config["classes"]
    patch_size = config["patch_size"]
    split_ratios = normalize_split_ratios(config["split_ratios"])
    random_generator = random.Random(config["random_seed"])

    clear_previous_processed_images(processed_data_dir, classes)

    raw_counts = {}
    saved_summary = {}
    all_metadata_rows = []

    for class_name in classes:
        raw_class_folder = raw_data_dir / class_name
        images = find_raw_images(raw_class_folder)
        raw_counts[class_name] = len(images)

        # Shuffle a copy so the original list stays unchanged.
        shuffled_images = images.copy()
        random_generator.shuffle(shuffled_images)

        split_to_images = split_images(shuffled_images, split_ratios)
        saved_counts, metadata_rows = prepare_split_images(
            class_name,
            split_to_images,
            processed_data_dir,
            patch_size,
        )
        saved_summary[class_name] = saved_counts
        all_metadata_rows.extend(metadata_rows)

    metadata_path = processed_data_dir / "metadata.csv"
    write_metadata(metadata_path, all_metadata_rows)

    print("SAR Ship-Sea Dataset Preparation")
    print("================================")

    for class_name in classes:
        print(f"Raw {class_name} images found: {raw_counts[class_name]}")

    print()

    for class_name in classes:
        print(f"{class_name.title()} processed images saved:")
        for split in SPLITS:
            print(f"  {split}: {saved_summary[class_name][split]}")

        if raw_counts[class_name] == 0:
            print(f"  WARNING: The {class_name} class has zero images.")

    print()
    print(f"metadata.csv path: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
