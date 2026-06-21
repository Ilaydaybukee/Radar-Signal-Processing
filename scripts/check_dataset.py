"""Check the raw SAR ship/sea image dataset.

This script gives a simple summary of the binary classification dataset used by
this project. It checks the expected raw class folders and counts common image
file types in each folder.
"""

from pathlib import Path

# The dataset root is where the raw class folders should live.
DATASET_ROOT = Path("data/raw")

# These are the two class folders expected for binary ship/sea classification.
CLASS_FOLDERS = {
    "ship": DATASET_ROOT / "ship",
    "sea": DATASET_ROOT / "sea",
}

# Image file extensions supported by this checker.
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def count_images(folder: Path) -> int:
    """Count supported image files directly inside a folder."""
    # If the folder is missing, it cannot contain any images.
    if not folder.exists():
        return 0

    image_count = 0

    # Look at each item in the folder and count files with supported extensions.
    for item in folder.iterdir():
        if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            image_count += 1

    return image_count


def main() -> None:
    """Print a clear summary of the raw dataset."""
    # Resolve the path so the summary shows exactly where the script looked.
    dataset_root = DATASET_ROOT.resolve()

    # Count images for each class. Missing folders count as zero images.
    ship_count = count_images(CLASS_FOLDERS["ship"])
    sea_count = count_images(CLASS_FOLDERS["sea"])
    total_count = ship_count + sea_count

    print("SAR Ship-Sea Dataset Check")
    print("==========================")
    print(f"Dataset root: {dataset_root}")
    print(f"Ship images:  {ship_count}")
    print(f"Sea images:   {sea_count}")
    print(f"Total images: {total_count}")

    # Print warnings after the summary so beginners can quickly see any issues.
    for class_name, folder in CLASS_FOLDERS.items():
        if not folder.exists():
            print(f"WARNING: Missing {class_name} class folder: {folder}")

    if ship_count == 0:
        print("WARNING: The ship class has zero images.")

    if sea_count == 0:
        print("WARNING: The sea class has zero images.")

    if ship_count == 0 and sea_count == 0:
        print("WARNING: Both classes are empty.")


if __name__ == "__main__":
    main()
