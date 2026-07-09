"""Streamlit review app for L-band ALOS candidate patches."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


DEFAULT_DATA_ROOT = Path(r"C:\SAR_Datasets\lband_alos")
DEFAULT_CANDIDATE_ROOT = DEFAULT_DATA_ROOT / "candidate_patches"
DEFAULT_REVIEW_ROOT = DEFAULT_DATA_ROOT / "reviews"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def find_candidate_folders(root: Path) -> list[Path]:
    if not root.exists():
        return []
    folders = []
    for path in root.rglob("*"):
        if path.is_dir() and path.name in {"sea_candidate", "ship_candidate"}:
            folders.append(path)
    return sorted(folders)


def load_manifest(folder: Path) -> pd.DataFrame:
    manifest_path = folder / "manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        images = sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        manifest = pd.DataFrame({"filename": [str(path) for path in images]})
    manifest["filename"] = manifest["filename"].astype(str)
    manifest["file_exists"] = manifest["filename"].map(lambda value: Path(value).exists())
    return manifest[manifest["file_exists"]].reset_index(drop=True)


def review_csv_path(review_root: Path, candidate_folder: Path) -> Path:
    scene_id = candidate_folder.parent.name
    return review_root / f"{scene_id}_{candidate_folder.name}_review.csv"


def append_decision(output_csv: Path, row: pd.Series, decision: str) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    record = row.to_dict()
    record["decision"] = decision
    record["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
    record["review_csv"] = str(output_csv)
    frame = pd.DataFrame([record])
    frame.to_csv(output_csv, mode="a", index=False, header=not output_csv.exists())


def main() -> None:
    st.set_page_config(page_title="L-band Candidate Review", layout="wide")
    st.title("L-band ALOS Candidate Review")

    candidate_root = Path(st.sidebar.text_input("Candidate root", str(DEFAULT_CANDIDATE_ROOT)))
    review_root = Path(st.sidebar.text_input("Review output root", str(DEFAULT_REVIEW_ROOT)))
    folders = find_candidate_folders(candidate_root)

    if not folders:
        st.info("No sea_candidate or ship_candidate folders were found.")
        return

    folder_labels = [str(folder) for folder in folders]
    selected_label = st.sidebar.selectbox("Candidate folder", folder_labels)
    selected_folder = Path(selected_label)
    manifest = load_manifest(selected_folder)

    if manifest.empty:
        st.warning("The selected folder has no readable images.")
        return

    index = st.sidebar.number_input("Image index", min_value=0, max_value=len(manifest) - 1, value=0, step=1)
    row = manifest.iloc[int(index)]
    image_path = Path(row["filename"])
    output_csv = review_csv_path(review_root, selected_folder)

    left, right = st.columns([2, 1])
    with left:
        st.subheader(image_path.name)
        st.image(Image.open(image_path), clamp=True, use_container_width=False)

    with right:
        st.subheader("Metadata")
        st.dataframe(pd.DataFrame([row.drop(labels=["file_exists"], errors="ignore")]).T, use_container_width=True)
        st.caption(f"Review decisions: {output_csv}")

        col1, col2 = st.columns(2)
        if col1.button("Accept as ship", use_container_width=True):
            append_decision(output_csv, row, "ship")
            st.success("Saved: ship")
        if col2.button("Accept as sea", use_container_width=True):
            append_decision(output_csv, row, "sea")
            st.success("Saved: sea")

        col3, col4 = st.columns(2)
        if col3.button("Reject", use_container_width=True):
            append_decision(output_csv, row, "reject")
            st.success("Saved: reject")
        if col4.button("Unsure", use_container_width=True):
            append_decision(output_csv, row, "unsure")
            st.success("Saved: unsure")


if __name__ == "__main__":
    main()
