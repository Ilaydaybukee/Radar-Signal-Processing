"""Split SAR images into train, validation, and test folders.

The full splitting logic will be added later. For now, this file documents
where the binary ship/sea dataset split step belongs.
"""

from pathlib import Path

PROCESSED_DATA_DIR = Path("data/processed")
SPLITS = ("train", "val", "test")
CLASSES = ("ship", "sea")


def main() -> None:
    """Print the expected processed dataset folders."""
    print("Dataset splitting placeholder")
    for split in SPLITS:
        for class_name in CLASSES:
            print(f"Expected folder: {PROCESSED_DATA_DIR / split / class_name}")


if __name__ == "__main__":
    main()
