# Global SAR Ship-Sea Classification

This repository contains a prototype deep learning pipeline for **Synthetic Aperture Radar (SAR) ship-sea classification**.

The main goal of the project is to classify grayscale SAR image patches into two classes:

* `ship`
* `sea`

The project includes dataset preparation, train/validation/test splitting, a PyTorch dataset loader, a simple CNN model, GPU-supported training, and test evaluation.

---

## Project Overview

Synthetic Aperture Radar images are widely used in remote sensing because SAR sensors can operate independently of daylight and are less affected by cloud coverage compared to optical sensors.

In this project, SAR image patches are used for a binary classification task:

| Class  | Description                             |
| ------ | --------------------------------------- |
| `ship` | SAR patches containing ship targets     |
| `sea`  | SAR patches containing open sea surface |

The current implementation is a first prototype and uses a lightweight convolutional neural network for ship-sea classification.

---

## Dataset

The dataset used in the current prototype is balanced:

| Class | Number of Images |
| ----- | ---------------: |
| Ship  |               50 |
| Sea   |               50 |
| Total |              100 |

The dataset is split as:

| Split      | Ship | Sea | Total |
| ---------- | ---: | --: | ----: |
| Train      |   35 |  35 |    70 |
| Validation |    7 |   7 |    14 |
| Test       |    8 |   8 |    16 |

### Data Sources

* Ship images were collected from SAR ship image datasets such as OpenSARShip-style SAR ship patches.
* Sea images were generated as Sentinel-1 SAR open sea patches using Google Earth Engine.

> Note: Since the ship and sea samples may come from different data sources, the current results should be interpreted as prototype-level results.

---

## Project Structure

```text
Radar-Signal-Processing/
│
├── app/
│   └── Streamlit or demo application files
│
├── config/
│   └── dataset_config.yaml
│
├── data/
│   ├── raw/
│   │   ├── ship/
│   │   └── sea/
│   │
│   └── processed/
│       ├── train/
│       ├── val/
│       ├── test/
│       └── metadata.csv
│
├── models/
│   └── trained model files
│
├── notebooks/
│   └── experiment notebooks
│
├── scripts/
│   ├── check_dataset.py
│   ├── split_dataset.py
│   └── prepare_dataset.py
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

---

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU-supported PyTorch installation, use the official PyTorch installation command according to your CUDA version.

Example:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## Dataset Preparation

Raw images should be placed under:

```text
data/raw/ship/
data/raw/sea/
```

Check the number of raw images:

```powershell
python scripts/check_dataset.py
```

Prepare the dataset:

```powershell
python scripts/prepare_dataset.py
```

This script:

* reads raw SAR images,
* converts them to grayscale,
* resizes them to 256×256,
* splits them into train/validation/test folders,
* saves processed images as PNG files,
* creates a metadata CSV file.

---

## Training

Train the CNN model:

```powershell
python src/train.py
```

The training script uses:

* PyTorch
* CrossEntropyLoss
* Adam optimizer
* CUDA GPU if available

The trained model is saved to:

```text
models/simple_sar_classifier.pth
```

---

## Evaluation

Evaluate the trained model on the test set:

```powershell
python src/evaluate.py
```

Example test result from the current prototype:

```text
SAR Ship-Sea Test Evaluation
============================
Using device: cuda
Test samples: 16
Test accuracy: 1.0000

Per-class results:
  sea: 8/8 correct | accuracy: 1.0000
  ship: 8/8 correct | accuracy: 1.0000
```

---

## Model Architecture

The current model is a simple CNN classifier:

```text
Input: 1 × 256 × 256 grayscale SAR image

Conv2D: 1 → 16
ReLU
MaxPool2D

Conv2D: 16 → 32
ReLU
MaxPool2D

Conv2D: 32 → 64
ReLU
MaxPool2D

Flatten
Linear: 65536 → 128
ReLU
Dropout
Linear: 128 → 2
```

The output classes are:

```text
0 -> sea
1 -> ship
```

---

## Current Results

The first prototype achieved:

| Metric              | Result |
| ------------------- | -----: |
| Training Accuracy   |   100% |
| Validation Accuracy | 92.86% |
| Test Accuracy       |   100% |

These results show that the pipeline works successfully on the current small dataset.

---

### Result Figures

#### Training Curves

The training and validation loss/accuracy curves are shown below:

![Training Curves](outputs/training_curves.png)

#### Confusion Matrix

The confusion matrix on the test set is shown below:

![Confusion Matrix](outputs/confusion_matrix.png)


## Limitations

The current version is an early prototype. The main limitations are:

* The dataset size is small.
* The test set contains only 16 images.
* Ship and sea images may come from different data sources.
* The model may learn source-related differences in addition to ship-sea features.
* More diverse SAR images from different regions, seasons, incidence angles, and sea states are needed.

Therefore, the current accuracy should not be interpreted as final real-world performance.

---

## Future Work

Planned improvements include:

* increasing the dataset size,
* adding more diverse Sentinel-1 SAR patches,
* using external SAR ship datasets,
* testing stronger CNN architectures,
* adding data augmentation,
* evaluating with confusion matrix and precision/recall/F1-score,
* developing a Streamlit demo interface,
* comparing SAR preprocessing methods.

---

## Project Status

Current completed stages:

* Dataset collection
* Dataset preparation
* Train/validation/test split
* PyTorch dataset loader
* CNN model implementation
* GPU-supported training
* Test evaluation
* GitHub integration

This repository currently represents a working prototype for SAR-based ship-sea classification.
