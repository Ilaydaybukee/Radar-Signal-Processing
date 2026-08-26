"""Split raw SAR ship/sea images into train, validation, and test folders.

This script copies images from data/raw into data/processed. The raw images are
left untouched so the split can be recreated later.
"""

from pathlib import Path
import random
import shutil

# Project paths used by the dataset pipeline.
CONFIG_PATH = Path("config/dataset_config.yaml")
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# The dataset has two classes for binary classification.
CLASSES = ("ship", "sea")

# The output dataset is split into these three folders.
SPLITS = ("train", "val", "test")

# These defaults are used when config/dataset_config.yaml is missing or does
# not contain split ratios.
DEFAULT_SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

# A fixed seed makes the shuffled split reproducible.
DEFAULT_RANDOM_SEED = 42

# Image file extensions supported by this script.
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def read_split_ratios(config_path: Path) -> dict[str, float]:
    """Read split ratios from the YAML config file when it is available.

    The config file for this beginner project is simple, so this function reads
    only the values under the top-level "splits:" section without requiring any
    extra third-party YAML package.
    """
    split_ratios = DEFAULT_SPLIT_RATIOS.copy()

    if not config_path.exists():
        return split_ratios

    in_splits_section = False

    for line in config_path.read_text().splitlines():
        stripped_line = line.strip()

        # Skip blank lines and comments.
        if not stripped_line or stripped_line.startswith("#"):
            continue

        if stripped_line == "splits:":
            in_splits_section = True
            continue

        # Stop reading split values when the next top-level section begins.
        if in_splits_section and not line.startswith(" "):
            break

        if in_splits_section and ":" in stripped_line:
            name, value = stripped_line.split(":", 1)
            name = name.strip()
            value = value.strip()

            if name in split_ratios:
                split_ratios[name] = float(value)

    return split_ratios


def normalize_split_ratios(split_ratios: dict[str, float]) -> dict[str, float]:
    """Make sure split ratios add up to 1.0."""
    total = sum(split_ratios.values())

    if total <= 0:
        return DEFAULT_SPLIT_RATIOS.copy()

    return {name: ratio / total for name, ratio in split_ratios.items()}


def find_images(class_folder: Path) -> list[Path]:
    """Find supported image files directly inside one raw class folder."""
    if not class_folder.exists():
        return []

    images = []

    for item in class_folder.iterdir():
        if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            images.append(item)

    # Sorting before shuffling keeps the result stable across operating systems.
    return sorted(images)


def clear_previous_processed_images() -> None:
    """Remove old processed image files while keeping .gitkeep files."""
    for split in SPLITS:
        for class_name in CLASSES:
            output_folder = PROCESSED_DATA_DIR / split / class_name
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

    # The test split receives the remaining images so every image is used once.
    test_count_start = train_count + val_count

    return {
        "train": images[:train_count],
        "val": images[train_count:test_count_start],
        "test": images[test_count_start:],
    }


def copy_split_images(class_name: str, split_to_images: dict[str, list[Path]]) -> dict[str, int]:
    """Copy split images into the processed dataset folders."""
    copied_counts = {}

    for split, images in split_to_images.items():
        output_folder = PROCESSED_DATA_DIR / split / class_name
        output_folder.mkdir(parents=True, exist_ok=True)

        for image_path in images:
            destination = output_folder / image_path.name
            shutil.copy2(image_path, destination)

        copied_counts[split] = len(images)

    return copied_counts


def main() -> None:
    """Create the processed train/val/test dataset split."""
    split_ratios = normalize_split_ratios(read_split_ratios(CONFIG_PATH))
    random_generator = random.Random(DEFAULT_RANDOM_SEED)

    # Clear old processed image files before writing the new split.
    clear_previous_processed_images()

    raw_counts = {}
    copied_summary = {}

    for class_name in CLASSES:
        raw_class_folder = RAW_DATA_DIR / class_name
        images = find_images(raw_class_folder)
        raw_counts[class_name] = len(images)

        # Shuffle a copy of the list so each image can appear in only one split.
        shuffled_images = images.copy()
        random_generator.shuffle(shuffled_images)

        split_to_images = split_images(shuffled_images, split_ratios)
        copied_summary[class_name] = copy_split_images(class_name, split_to_images)

    print("SAR Ship-Sea Dataset Split")
    print("==========================")
    print(f"Raw ship images found: {raw_counts['ship']}")
    print(f"Raw sea images found:  {raw_counts['sea']}")
    print()

    for class_name in CLASSES:
        print(f"{class_name.title()} images copied:")
        for split in SPLITS:
            print(f"  {split}: {copied_summary[class_name][split]}")

        if raw_counts[class_name] == 0:
            print(f"  WARNING: The {class_name} class has zero images.")

    print()
    print(f"Processed dataset path: {PROCESSED_DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
