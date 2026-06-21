"""Prepare raw SAR ship/sea images for later processing.

This beginner-friendly placeholder will eventually validate raw files,
standardize image formats, and copy clean inputs into a staging area.
"""

from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
CLASSES = ("ship", "sea")


def main() -> None:
    """Print the expected raw dataset folders."""
    print("Dataset preparation placeholder")
    for class_name in CLASSES:
        print(f"Expected folder: {RAW_DATA_DIR / class_name}")


if __name__ == "__main__":
    main()
