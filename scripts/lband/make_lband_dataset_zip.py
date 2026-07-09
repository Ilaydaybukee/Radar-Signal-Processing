"""Create a Google Drive upload zip for a prepared L-band dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_DATA_ROOT = Path(r"C:\SAR_Datasets\lband_alos")
DEFAULT_FINAL_ROOT = DEFAULT_DATA_ROOT / "final_datasets"
DEFAULT_UPLOAD_ROOT = DEFAULT_DATA_ROOT / "drive_upload"


INCLUDED_DIRS = {"raw", "train", "val", "test", "notes"}
INCLUDED_FILES = {"manifest.csv", "summary.txt"}
SAMPLE_GRID_PATTERNS = ("*sample_grid*.jpg", "*review_grid*.jpg", "*contact_sheet*.jpg")


def should_include(path: Path, dataset_dir: Path) -> bool:
    relative = path.relative_to(dataset_dir)
    if relative.parts and relative.parts[0] in INCLUDED_DIRS:
        return True
    if path.name in INCLUDED_FILES:
        return True
    return any(path.match(pattern) for pattern in SAMPLE_GRID_PATTERNS)


def make_zip(dataset_dir: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as zip_file:
        for path in sorted(dataset_dir.rglob("*")):
            if path.is_file() and should_include(path, dataset_dir):
                zip_file.write(path, arcname=path.relative_to(dataset_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--final-root", type=Path, default=DEFAULT_FINAL_ROOT)
    parser.add_argument("--upload-root", type=Path, default=DEFAULT_UPLOAD_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.final_root / args.dataset_name
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset folder does not exist: {dataset_dir}")
    output_zip = args.upload_root / f"{args.dataset_name}.zip"
    make_zip(dataset_dir, output_zip)
    print(f"Saved Drive upload package: {output_zip}")


if __name__ == "__main__":
    main()
