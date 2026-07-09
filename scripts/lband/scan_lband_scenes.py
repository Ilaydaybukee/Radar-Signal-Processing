"""Scan local ALOS PALSAR L-band scenes and write a scene index CSV.

This script only reads file metadata. It does not move or modify raw scenes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DATA_ROOT = Path(r"C:\SAR_Datasets\lband_alos")
DEFAULT_RAW_SCENES = DEFAULT_DATA_ROOT / "raw_scenes"
DEFAULT_INDEX_CSV = DEFAULT_DATA_ROOT / "notes" / "lband_scene_index.csv"


def scene_id_from_hh_path(hh_path: Path) -> str:
    """Return the scene id from a file named like HH-<scene_id>.tif."""
    name = hh_path.stem
    if name.upper().startswith("HH-"):
        return name[3:]
    return name


def find_first(scene_dir: Path, pattern: str) -> str:
    matches = sorted(scene_dir.glob(pattern))
    return str(matches[0]) if matches else ""


def build_scene_record(hh_path: Path) -> dict[str, object]:
    scene_dir = hh_path.parent
    scene_id = scene_id_from_hh_path(hh_path)

    polarizations = {}
    for pol in ("HH", "HV", "INC", "MASK"):
        found = sorted(scene_dir.glob(f"{pol}-*.tif"))
        polarizations[pol.lower()] = str(found[0]) if found else ""

    jpg_preview = find_first(scene_dir, "*.jpg") or find_first(scene_dir, "*.jpeg")
    summary_txt = str(scene_dir / "summary.txt") if (scene_dir / "summary.txt").exists() else ""

    return {
        "scene_id": scene_id,
        "scene_dir": str(scene_dir),
        "hh_path": str(hh_path),
        "file_size_mb": round(hh_path.stat().st_size / (1024 * 1024), 2),
        "hh_available": bool(polarizations["hh"]),
        "hv_available": bool(polarizations["hv"]),
        "inc_available": bool(polarizations["inc"]),
        "mask_available": bool(polarizations["mask"]),
        "hh_file": polarizations["hh"],
        "hv_file": polarizations["hv"],
        "inc_file": polarizations["inc"],
        "mask_file": polarizations["mask"],
        "jpg_preview": jpg_preview,
        "summary_txt": summary_txt,
    }


def scan_scenes(raw_scenes_dir: Path) -> pd.DataFrame:
    hh_files = sorted(raw_scenes_dir.rglob("HH-*.tif"))
    records = [build_scene_record(path) for path in hh_files]
    return pd.DataFrame(records)


def print_scene_index(index: pd.DataFrame) -> None:
    if index.empty:
        print("No HH-*.tif scenes found.")
        return

    for row in index.to_dict("records"):
        available = [
            pol.upper()
            for pol in ("hh", "hv", "inc", "mask")
            if row.get(f"{pol}_available")
        ]
        print(f"Scene: {row['scene_id']}")
        print(f"  HH path: {row['hh_path']}")
        print(f"  Size: {row['file_size_mb']} MB")
        print(f"  Available files: {', '.join(available)}")
        print(f"  JPG preview: {row['jpg_preview'] or 'missing'}")
        print(f"  summary.txt: {row['summary_txt'] or 'missing'}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-scenes-dir", type=Path, default=DEFAULT_RAW_SCENES)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_INDEX_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = scan_scenes(args.raw_scenes_dir)
    print_scene_index(index)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.output_csv, index=False)
    print(f"Saved scene index: {args.output_csv}")


if __name__ == "__main__":
    main()
