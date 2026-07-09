"""Extract sea and ship candidate patches from indexed ALOS PALSAR HH scenes.

This can process large GeoTIFF scenes. Run it intentionally with scene limits,
stride, and max-patch settings that are appropriate for the machine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from create_review_grids import create_contact_sheet  # noqa: E402


Image.MAX_IMAGE_PIXELS = None

DEFAULT_DATA_ROOT = Path(r"C:\SAR_Datasets\lband_alos")
DEFAULT_INDEX_CSV = DEFAULT_DATA_ROOT / "notes" / "lband_scene_index.csv"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "candidate_patches"
PATCH_SIZE = 256


def normalize_to_uint8(array: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    array = array.astype(np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape, dtype=np.uint8)

    low, high = np.percentile(array[finite], [low_percentile, high_percentile])
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)

    clipped = np.clip(array, low, high)
    scaled = (clipped - low) / (high - low)
    return (scaled * 255).astype(np.uint8)


def patch_metrics(patch: np.ndarray) -> dict[str, float]:
    mean = float(patch.mean())
    std = float(patch.std())
    dark_ratio = float((patch < 25).mean())
    bright_ratio_200 = float((patch > 200).mean())
    bright_ratio_230 = float((patch > 230).mean())
    max_value = float(patch.max())
    ship_score = max_value + 160.0 * bright_ratio_230 + 40.0 * bright_ratio_200 - mean
    sea_score = 100.0 - std - abs(mean - 55.0) - 40.0 * bright_ratio_200
    return {
        "mean": mean,
        "std": std,
        "dark_ratio": dark_ratio,
        "bright_ratio_200": bright_ratio_200,
        "bright_ratio_230": bright_ratio_230,
        "max": max_value,
        "ship_score": float(ship_score),
        "sea_score": float(sea_score),
    }


def is_sea_candidate(metrics: dict[str, float], max_std: float, max_bright_ratio: float) -> bool:
    return (
        20.0 <= metrics["mean"] <= 110.0
        and metrics["std"] <= max_std
        and metrics["bright_ratio_200"] <= max_bright_ratio
        and metrics["dark_ratio"] <= 0.65
    )


def is_ship_candidate(metrics: dict[str, float], min_bright_ratio: float, min_std: float) -> bool:
    return (
        metrics["bright_ratio_230"] >= min_bright_ratio
        and metrics["std"] >= min_std
        and metrics["max"] >= 230.0
    )


def save_patch(patch: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(patch, mode="L").save(output_path)


def iter_patch_locations(width: int, height: int, patch_size: int, stride: int):
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            yield x, y


def extract_scene(
    scene_row: pd.Series,
    output_root: Path,
    patch_size: int,
    stride: int,
    max_sea: int,
    max_ship: int,
    max_sea_std: float,
    max_sea_bright_ratio: float,
    min_ship_bright_ratio: float,
    min_ship_std: float,
    grid_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scene_id = str(scene_row["scene_id"])
    hh_path = Path(scene_row["hh_path"])
    scene_output = output_root / scene_id
    sea_dir = scene_output / "sea_candidate"
    ship_dir = scene_output / "ship_candidate"

    print(f"Loading {hh_path}")
    with Image.open(hh_path) as image:
        image_array = np.array(image)
    normalized = normalize_to_uint8(image_array)

    height, width = normalized.shape[:2]
    sea_records: list[dict[str, object]] = []
    ship_records: list[dict[str, object]] = []

    for x, y in iter_patch_locations(width, height, patch_size, stride):
        patch = normalized[y : y + patch_size, x : x + patch_size]
        metrics = patch_metrics(patch)

        if len(sea_records) < max_sea and is_sea_candidate(metrics, max_sea_std, max_sea_bright_ratio):
            filename = f"{scene_id}_HH_sea_x{x}_y{y}.png"
            save_patch(patch, sea_dir / filename)
            sea_records.append(make_record(scene_id, x, y, metrics, "sea_candidate", sea_dir / filename))

        if len(ship_records) < max_ship and is_ship_candidate(metrics, min_ship_bright_ratio, min_ship_std):
            filename = f"{scene_id}_HH_ship_x{x}_y{y}.png"
            save_patch(patch, ship_dir / filename)
            ship_records.append(make_record(scene_id, x, y, metrics, "ship_candidate", ship_dir / filename))

        if len(sea_records) >= max_sea and len(ship_records) >= max_ship:
            break

    sea_manifest = pd.DataFrame(sea_records)
    ship_manifest = pd.DataFrame(ship_records)
    if not sea_manifest.empty:
        sea_manifest.to_csv(sea_dir / "manifest.csv", index=False)
        create_contact_sheet(sorted(sea_dir.glob("*.png")), sea_dir / "sea_candidate_sample_grid.jpg", limit=grid_limit)
    if not ship_manifest.empty:
        ship_manifest.to_csv(ship_dir / "manifest.csv", index=False)
        create_contact_sheet(sorted(ship_dir.glob("*.png")), ship_dir / "ship_candidate_sample_grid.jpg", limit=grid_limit)

    return sea_manifest, ship_manifest


def make_record(
    scene_id: str,
    x: int,
    y: int,
    metrics: dict[str, float],
    candidate_type: str,
    filename: Path,
) -> dict[str, object]:
    score = metrics["sea_score"] if candidate_type == "sea_candidate" else metrics["ship_score"]
    return {
        "source_scene": scene_id,
        "x": x,
        "y": y,
        "mean": round(metrics["mean"], 4),
        "std": round(metrics["std"], 4),
        "dark_ratio": round(metrics["dark_ratio"], 6),
        "bright_ratio_200": round(metrics["bright_ratio_200"], 6),
        "bright_ratio_230": round(metrics["bright_ratio_230"], 6),
        "score": round(score, 4),
        "band": "L",
        "sensor": "ALOS PALSAR",
        "polarization": "HH",
        "candidate_type": candidate_type,
        "filename": str(filename),
    }


def load_selected_scenes(index_csv: Path, scene_ids: list[str] | None, limit: int | None) -> pd.DataFrame:
    index = pd.read_csv(index_csv)
    if scene_ids:
        index = index[index["scene_id"].astype(str).isin(scene_ids)]
    if limit is not None:
        index = index.head(limit)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-index", type=Path, default=DEFAULT_INDEX_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scene-id", action="append", help="Scene id to process. Repeat for multiple scenes.")
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-sea", type=int, default=200)
    parser.add_argument("--max-ship", type=int, default=200)
    parser.add_argument("--max-sea-std", type=float, default=24.0)
    parser.add_argument("--max-sea-bright-ratio", type=float, default=0.002)
    parser.add_argument("--min-ship-bright-ratio", type=float, default=0.0008)
    parser.add_argument("--min-ship-std", type=float, default=28.0)
    parser.add_argument("--grid-limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = load_selected_scenes(args.scene_index, args.scene_id, args.limit_scenes)
    if scenes.empty:
        raise SystemExit("No scenes selected. Check the scene index or --scene-id values.")

    all_sea = []
    all_ship = []
    for _, scene_row in scenes.iterrows():
        sea_manifest, ship_manifest = extract_scene(
            scene_row,
            args.output_root,
            args.patch_size,
            args.stride,
            args.max_sea,
            args.max_ship,
            args.max_sea_std,
            args.max_sea_bright_ratio,
            args.min_ship_bright_ratio,
            args.min_ship_std,
            args.grid_limit,
        )
        all_sea.append(sea_manifest)
        all_ship.append(ship_manifest)

    args.output_root.mkdir(parents=True, exist_ok=True)
    pd.concat(all_sea, ignore_index=True).to_csv(args.output_root / "sea_candidate_manifest_all.csv", index=False)
    pd.concat(all_ship, ignore_index=True).to_csv(args.output_root / "ship_candidate_manifest_all.csv", index=False)
    print(f"Saved candidates under: {args.output_root}")


if __name__ == "__main__":
    main()
