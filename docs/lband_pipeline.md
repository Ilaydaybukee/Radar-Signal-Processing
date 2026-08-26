# L-band ALOS PALSAR ship-sea dataset pipeline

This repository stores reproducible code and documentation for preparing an
L-band SAR ship-sea classification dataset. Raw scenes, candidate patches,
final dataset images, and zip packages stay outside GitHub under:

```text
C:\SAR_Datasets\lband_alos
```

## Why imagery is not committed

Raw SAR scenes and generated dataset images can be large, licensed, and hard to
audit in Git history. Keeping them outside the repository prevents accidental
redistribution, keeps clones lightweight, and makes the dataset pipeline
reproducible from documented local inputs instead of opaque committed binaries.

Google Drive is used for sharing packaged dataset releases. The repository keeps
the scripts that build those releases, while the generated zip files are written
to `C:\SAR_Datasets\lband_alos\drive_upload`.

## Local folder layout

```text
C:\SAR_Datasets\lband_alos
  raw_scenes\
  notes\
  candidate_patches\
  reviews\
  final_datasets\
  drive_upload\
```

Each raw scene folder may contain `HH-*.tif`, `HV-*.tif`, `INC-*.tif`,
`MASK-*.tif`, a JPG preview, and `summary.txt`.

## Workflow

1. Scan scenes:

   ```powershell
   python scripts/lband/scan_lband_scenes.py
   ```

   This writes `C:\SAR_Datasets\lband_alos\notes\lband_scene_index.csv`.

2. Extract candidates after choosing scene ids and extraction settings:

   ```powershell
   python scripts/lband/extract_lband_candidates.py --scene-id ALPSRP240081010-H2.2_UA --stride 256 --max-sea 200 --max-ship 200
   ```

   Candidate PNGs and manifests are saved under
   `C:\SAR_Datasets\lband_alos\candidate_patches\<scene_id>`.

3. Create extra review grids when needed:

   ```powershell
   python scripts/lband/create_review_grids.py C:\SAR_Datasets\lband_alos\candidate_patches\<scene_id>\ship_candidate
   ```

4. Review candidates:

   ```powershell
   streamlit run app/lband_review_app.py
   ```

   Review decisions are saved as CSV files under
   `C:\SAR_Datasets\lband_alos\reviews`.

5. Build a balanced final dataset:

   ```powershell
   python scripts/lband/build_lband_dataset.py --dataset-name ALOS_LBand_ShipSea_Pilot_14
   ```

   The output goes to
   `C:\SAR_Datasets\lband_alos\final_datasets\<dataset_name>`.

6. Create a Google Drive upload package:

   ```powershell
   python scripts/lband/make_lband_dataset_zip.py --dataset-name ALOS_LBand_ShipSea_Pilot_14
   ```

   The zip is saved under `C:\SAR_Datasets\lband_alos\drive_upload`.

## Current pilot result

`ALOS_LBand_ShipSea_Pilot_14` contains 14 reviewed patches:

- 7 ship patches
- 7 sea patches
- Source scene: `ALPSRP240081010-H2.2_UA`
- Band/polarization: L-band HH

This pilot is intentionally small. It validates the review and packaging flow
before larger extraction runs are performed.
