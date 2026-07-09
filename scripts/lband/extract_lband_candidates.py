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


def parse_mask_values(value_text: str | None) -> set[float] | None:
    if not value_text:
        return None
    values = set()
    for value in value_text.split(","):
        value = value.strip()
        if value:
            values.add(float(value))
    return values or None


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


def load_scene_mask(scene_row: pd.Series, hh_shape: tuple[int, int]) -> np.ndarray | None:
    mask_path_text = str(scene_row.get("mask_file", "") or "")
    mask_path = Path(mask_path_text) if mask_path_text else None
    if mask_path is None or not mask_path.exists():
        scene_dir = Path(str(scene_row.get("scene_dir", Path(scene_row["hh_path"]).parent)))
        matches = sorted(scene_dir.glob("MASK-*.tif"))
        mask_path = matches[0] if matches else None
    if mask_path is None or not mask_path.exists():
        return None

    print(f"Loading mask {mask_path}")
    with Image.open(mask_path) as image:
        if image.size != (hh_shape[1], hh_shape[0]):
            image = image.resize((hh_shape[1], hh_shape[0]), Image.Resampling.NEAREST)
        return np.array(image)


def infer_water_mask(
    mask_array: np.ndarray,
    normalized_hh: np.ndarray,
    explicit_water_values: set[float] | None,
) -> np.ndarray:
    if explicit_water_values is not None:
        return np.isin(mask_array, list(explicit_water_values))

    values, counts = np.unique(mask_array, return_counts=True)
    total_pixels = mask_array.size
    value_scores = []
    for value, count in zip(values, counts):
        coverage = count / total_pixels
        if coverage < 0.005:
            continue
        pixels = normalized_hh[mask_array == value]
        if pixels.size == 0:
            continue
        mean = float(pixels.mean())
        std = float(pixels.std())
        bright_ratio = float((pixels > 200).mean())
        dark_ratio = float((pixels < 25).mean())
        score = mean + 1.5 * std + 500.0 * bright_ratio + 20.0 * dark_ratio
        value_scores.append((score, value, coverage, mean, std, bright_ratio))

    if not value_scores:
        return np.ones(mask_array.shape, dtype=bool)

    value_scores.sort(key=lambda item: item[0])
    water_value = value_scores[0][1]
    print(
        "Inferred MASK water value "
        f"{water_value!r} from HH smoothness/brightness statistics. "
        "Use --mask-water-values if this convention is wrong."
    )
    return mask_array == water_value


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
        "very_bright_ratio_245": float((patch > 245).mean()),
        "max": max_value,
        "ship_score": float(ship_score),
        "sea_score": float(sea_score),
    }


def background_metrics(patch: np.ndarray, target_threshold: int = 220) -> dict[str, float]:
    background = patch[patch < target_threshold]
    if background.size < patch.size * 0.80:
        background = patch
    return {
        "background_mean": float(background.mean()),
        "background_std": float(background.std()),
        "background_bright_ratio_180": float((background > 180).mean()),
    }


def mask_metrics(water_patch: np.ndarray | None) -> dict[str, float]:
    if water_patch is None:
        return {
            "water_ratio": 1.0,
            "land_ratio": 0.0,
            "mask_used": False,
        }
    water_ratio = float(water_patch.mean())
    return {
        "water_ratio": water_ratio,
        "land_ratio": 1.0 - water_ratio,
        "mask_used": True,
    }


def local_texture_ratio(patch: np.ndarray, threshold: float) -> float:
    patch_float = patch.astype(np.float32)
    horizontal = np.abs(np.diff(patch_float, axis=1))
    vertical = np.abs(np.diff(patch_float, axis=0))
    return float(((horizontal[:-1, :] + vertical[:, :-1]) * 0.5 > threshold).mean())


def is_sea_candidate(
    metrics: dict[str, float],
    water_metrics: dict[str, float],
    max_std: float,
    max_bright_ratio: float,
    min_water_ratio: float,
    max_land_ratio: float,
    require_mask_water: bool,
) -> bool:
    if require_mask_water and not water_metrics["mask_used"]:
        return False
    return (
        20.0 <= metrics["mean"] <= 110.0
        and metrics["std"] <= max_std
        and metrics["bright_ratio_200"] <= max_bright_ratio
        and metrics["dark_ratio"] <= 0.65
        and water_metrics["water_ratio"] >= min_water_ratio
        and water_metrics["land_ratio"] <= max_land_ratio
    )


def is_ship_candidate(
    metrics: dict[str, float],
    background: dict[str, float],
    water_metrics: dict[str, float],
    texture_ratio: float,
    min_bright_ratio: float,
    min_std: float,
    min_water_ratio: float,
    max_land_ratio: float,
    max_ship_std: float,
    max_background_std: float,
    max_texture_ratio: float,
    max_very_bright_ratio: float,
    require_mask_water: bool,
) -> bool:
    if require_mask_water and not water_metrics["mask_used"]:
        return False
    return (
        metrics["bright_ratio_230"] >= min_bright_ratio
        and metrics["std"] >= min_std
        and metrics["std"] <= max_ship_std
        and metrics["max"] >= 230.0
        and metrics["very_bright_ratio_245"] <= max_very_bright_ratio
        and background["background_std"] <= max_background_std
        and background["background_bright_ratio_180"] <= 0.025
        and water_metrics["water_ratio"] >= min_water_ratio
        and water_metrics["land_ratio"] <= max_land_ratio
        and texture_ratio <= max_texture_ratio
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
    max_ship_std: float,
    max_ship_background_std: float,
    max_ship_texture_ratio: float,
    max_ship_very_bright_ratio: float,
    require_mask_water: bool,
    ignore_mask: bool,
    min_water_ratio: float,
    max_land_ratio: float,
    mask_water_values: set[float] | None,
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
    water_mask = None
    if require_mask_water and not ignore_mask:
        scene_mask = load_scene_mask(scene_row, normalized.shape[:2])
        if scene_mask is not None:
            water_mask = infer_water_mask(scene_mask, normalized, mask_water_values)

    height, width = normalized.shape[:2]
    sea_records: list[dict[str, object]] = []
    ship_records: list[dict[str, object]] = []

    for x, y in iter_patch_locations(width, height, patch_size, stride):
        patch = normalized[y : y + patch_size, x : x + patch_size]
        metrics = patch_metrics(patch)
        water_patch = None if water_mask is None else water_mask[y : y + patch_size, x : x + patch_size]
        water = mask_metrics(water_patch)

        if len(sea_records) < max_sea and is_sea_candidate(
            metrics,
            water,
            max_sea_std,
            max_sea_bright_ratio,
            min_water_ratio,
            max_land_ratio,
            require_mask_water,
        ):
            filename = f"{scene_id}_HH_sea_x{x}_y{y}.png"
            save_patch(patch, sea_dir / filename)
            sea_records.append(make_record(scene_id, x, y, metrics, water, "sea_candidate", sea_dir / filename))

        background = background_metrics(patch)
        texture_ratio = local_texture_ratio(patch, threshold=35.0)
        if len(ship_records) < max_ship and is_ship_candidate(
            metrics,
            background,
            water,
            texture_ratio,
            min_ship_bright_ratio,
            min_ship_std,
            min_water_ratio,
            max_land_ratio,
            max_ship_std,
            max_ship_background_std,
            max_ship_texture_ratio,
            max_ship_very_bright_ratio,
            require_mask_water,
        ):
            filename = f"{scene_id}_HH_ship_x{x}_y{y}.png"
            save_patch(patch, ship_dir / filename)
            record = make_record(scene_id, x, y, metrics, water, "ship_candidate", ship_dir / filename)
            record.update(
                {
                    "background_std": round(background["background_std"], 4),
                    "background_bright_ratio_180": round(background["background_bright_ratio_180"], 6),
                    "texture_ratio": round(texture_ratio, 6),
                }
            )
            ship_records.append(record)

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
    water_metrics: dict[str, float],
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
        "very_bright_ratio_245": round(metrics["very_bright_ratio_245"], 6),
        "water_ratio": round(water_metrics["water_ratio"], 6),
        "land_ratio": round(water_metrics["land_ratio"], 6),
        "mask_used": water_metrics["mask_used"],
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
    parser.add_argument("--max-ship-std", type=float, default=75.0)
    parser.add_argument("--max-ship-background-std", type=float, default=42.0)
    parser.add_argument("--max-ship-texture-ratio", type=float, default=0.18)
    parser.add_argument("--max-ship-very-bright-ratio", type=float, default=0.01)
    parser.add_argument("--require-mask-water", action="store_true")
    parser.add_argument("--ignore-mask", action="store_true", help="Force HH-only extraction and do not load MASK files.")
    parser.add_argument("--min-water-ratio", type=float, default=0.92)
    parser.add_argument("--max-land-ratio", type=float, default=0.08)
    parser.add_argument(
        "--mask-water-values",
        default=None,
        help="Comma-separated MASK pixel values that mean water. If omitted, water is inferred from HH statistics.",
    )
    parser.add_argument("--grid-limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.require_mask_water and args.ignore_mask:
        raise SystemExit("Use either --require-mask-water or --ignore-mask, not both.")

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
            args.max_ship_std,
            args.max_ship_background_std,
            args.max_ship_texture_ratio,
            args.max_ship_very_bright_ratio,
            args.require_mask_water,
            args.ignore_mask,
            args.min_water_ratio,
            args.max_land_ratio,
            parse_mask_values(args.mask_water_values),
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
