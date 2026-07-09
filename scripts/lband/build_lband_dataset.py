"""Build a balanced ship-sea dataset from accepted review CSV files."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import pandas as pd


DEFAULT_DATA_ROOT = Path(r"C:\SAR_Datasets\lband_alos")
DEFAULT_REVIEWS_ROOT = DEFAULT_DATA_ROOT / "reviews"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "final_datasets"


def read_reviews(review_root: Path) -> pd.DataFrame:
    review_files = sorted(review_root.glob("*.csv"))
    frames = [pd.read_csv(path) for path in review_files]
    if not frames:
        return pd.DataFrame()
    reviews = pd.concat(frames, ignore_index=True)
    return reviews[reviews["decision"].isin(["ship", "sea"])].copy()


def balanced_reviews(reviews: pd.DataFrame, seed: int) -> pd.DataFrame:
    ship = reviews[reviews["decision"] == "ship"].drop_duplicates("filename")
    sea = reviews[reviews["decision"] == "sea"].drop_duplicates("filename")
    count = min(len(ship), len(sea))
    if count == 0:
        raise ValueError("Need at least one accepted ship and one accepted sea patch.")
    ship = ship.sample(n=count, random_state=seed)
    sea = sea.sample(n=count, random_state=seed)
    return pd.concat([ship, sea], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def assign_splits(frame: pd.DataFrame, seed: int, val_fraction: float, test_fraction: float) -> pd.DataFrame:
    rng = random.Random(seed)
    frame = frame.copy()
    splits = []
    for label, group in frame.groupby("decision"):
        indexes = list(group.index)
        rng.shuffle(indexes)
        n = len(indexes)
        n_test = max(1, round(n * test_fraction)) if n >= 3 else 0
        n_val = max(1, round(n * val_fraction)) if n >= 3 else 0
        split_map = {}
        for idx in indexes[:n_test]:
            split_map[idx] = "test"
        for idx in indexes[n_test : n_test + n_val]:
            split_map[idx] = "val"
        for idx in indexes[n_test + n_val :]:
            split_map[idx] = "train"
        splits.extend(split_map.items())
    frame["split"] = pd.Series(dict(splits))
    return frame


def copy_dataset_files(dataset: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    records = []
    for _, row in dataset.iterrows():
        label = row["decision"]
        src = Path(row["filename"])
        dst_name = src.name
        raw_dst = output_dir / "raw" / label / dst_name
        split_dst = output_dir / row["split"] / label / dst_name
        raw_dst.parent.mkdir(parents=True, exist_ok=True)
        split_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, raw_dst)
        shutil.copy2(src, split_dst)
        record = row.to_dict()
        record["raw_path"] = str(raw_dst)
        record["split_path"] = str(split_dst)
        records.append(record)
    return pd.DataFrame(records)


def write_summary(dataset: pd.DataFrame, output_dir: Path, dataset_name: str) -> None:
    lines = [
        f"Dataset: {dataset_name}",
        "Sensor: ALOS PALSAR",
        "Band: L",
        "Polarization: HH",
        "",
        "Counts by label:",
        dataset["decision"].value_counts().to_string(),
        "",
        "Counts by split and label:",
        dataset.groupby(["split", "decision"]).size().to_string(),
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews-root", type=Path, default=DEFAULT_REVIEWS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.dataset_name
    reviews = read_reviews(args.reviews_root)
    if reviews.empty:
        raise SystemExit("No accepted review rows found.")

    dataset = balanced_reviews(reviews, args.seed)
    dataset = assign_splits(dataset, args.seed, args.val_fraction, args.test_fraction)
    manifest = copy_dataset_files(dataset, output_dir)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    write_summary(manifest, output_dir, args.dataset_name)
    print(f"Built dataset: {output_dir}")


if __name__ == "__main__":
    main()
