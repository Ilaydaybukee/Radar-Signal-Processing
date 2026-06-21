"""Check the expected SAR ship/sea dataset folders.

This script currently performs a simple folder-existence check. More detailed
image validation can be added as the dataset grows.
"""

from pathlib import Path

REQUIRED_FOLDERS = [
    Path("data/raw/ship"),
    Path("data/raw/sea"),
    Path("data/processed/train/ship"),
    Path("data/processed/train/sea"),
    Path("data/processed/val/ship"),
    Path("data/processed/val/sea"),
    Path("data/processed/test/ship"),
    Path("data/processed/test/sea"),
]


def main() -> None:
    """Report whether the expected dataset folders exist."""
    missing = [folder for folder in REQUIRED_FOLDERS if not folder.exists()]
    if missing:
        print("Missing dataset folders:")
        for folder in missing:
            print(f"- {folder}")
        return

    print("All expected dataset folders exist.")


if __name__ == "__main__":
    main()
