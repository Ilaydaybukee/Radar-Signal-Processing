# Global SAR Ship-Sea Classification

## Project Goal

Global SAR Ship-Sea Classification is a student engineering research project for building a clean, topic-based binary image classification workflow for Synthetic Aperture Radar (SAR) imagery.

The goal is to classify SAR image patches into one of two classes:

1. **ship** — SAR image patches that contain visible ship targets.
2. **sea** — SAR image patches that contain open-sea background without ship targets.

This repository is intentionally organized around a single research topic: global ship-versus-sea SAR classification. It does not include the previous restoration, denoising, corrupted/clean image-pair, or region-based dataset workflow.

## Dataset Folder Structure

The dataset is organized into raw and processed folders. Data folders are kept empty in version control except for `.gitkeep` files.

```text
data/
├── raw/
│   ├── ship/
│   │   └── .gitkeep
│   └── sea/
│       └── .gitkeep
└── processed/
    ├── train/
    │   ├── ship/
    │   │   └── .gitkeep
    │   └── sea/
    │       └── .gitkeep
    ├── val/
    │   ├── ship/
    │   │   └── .gitkeep
    │   └── sea/
    │       └── .gitkeep
    └── test/
        ├── ship/
        │   └── .gitkeep
        └── sea/
            └── .gitkeep
```

Place original SAR images in `data/raw/ship/` and `data/raw/sea/`. The prepared train, validation, and test splits should be written to `data/processed/`.

## Planned Pipeline

The planned engineering workflow is:

1. Collect or curate global SAR image patches for ship and sea classes.
2. Place raw images into the class-specific raw data folders.
3. Validate the dataset structure and image readability.
4. Split the dataset into train, validation, and test subsets.
5. Build a PyTorch dataset loader for binary classification.
6. Train a baseline convolutional neural network or transfer-learning model.
7. Evaluate classification metrics on the test set.
8. Use a future Streamlit interface for simple image upload and prediction demos.

## Repository Structure

```text
config/                 Configuration files
scripts/                Dataset preparation and checking scripts
src/                    Beginner-friendly Python modules for model work
app/                    Future Streamlit interface
notebooks/              Exploratory notebooks
data/                   Local dataset folders ignored except .gitkeep files
```

## Student Engineering Research Purpose

This project is designed as a readable research and engineering foundation for students studying SAR image processing, remote sensing, and deep learning classification. The current code files are placeholders that describe the intended responsibilities of each module without implementing a full deep learning system yet.

## Future Streamlit Interface

A future Streamlit application will provide a simple interface where a user can upload a SAR image patch and receive a predicted class: `ship` or `sea`. The initial `app/streamlit_app.py` file is a placeholder for that interface.
