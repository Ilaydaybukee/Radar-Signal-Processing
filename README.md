# SAR Image Processing Project

This repository presents a SAR image processing pipeline for ship/sea analysis and multi-frequency SAR experimentation.

The project started as a C-band SAR ship/sea classification prototype and has now been expanded into a **multi-frequency SAR image processing pipeline** using C-band, S-band, and L-band SAR imagery.

---

## Current Project Status

The latest version focuses on a **multi-frequency SAR band classification demo**.

The goal of this demo is not final ship/sea classification.  
Instead, it validates that SAR image patches from different radar frequency bands can be collected into one pipeline and classified by a lightweight CNN.

Supported SAR bands in the current demo:

| Class | SAR Band | Sensor / Dataset | Demo Role |
|---|---|---|---|
| `C_band` | C-band | Sentinel-1 | Ship/sea prototype source |
| `S_band` | S-band | NovaSAR / NASTaR | Ship-type SAR source |
| `L_band` | L-band | ALOS PALSAR | L-band SAR pilot/candidate source |

---

## Latest Demo: Multi-Frequency SAR Band Classification

The current meeting demo classifies SAR image patches into three frequency-band classes:

- `C_band`
- `S_band`
- `L_band`

This experiment shows that different SAR frequency bands can be processed within a single pipeline and separated using a lightweight CNN model.

---

## Demo Summary

This demo focuses on **multi-frequency SAR band classification** rather than final ship/sea classification.

### Supported SAR Bands

| Class | Sensor / Dataset |
|---|---|
| `C_band` | Sentinel-1 |
| `S_band` | NovaSAR / NASTaR |
| `L_band` | ALOS PALSAR |

### Dataset Size

| Item | Value |
|---|---:|
| Number of bands | 3 |
| Images per band | 593 |
| Total images | 1779 |
| Image size | 256 × 256 |
| Image format | Grayscale PNG |

### Goal

The goal of this experiment is to show that SAR image patches from different radar frequency bands can be collected into a single pipeline and classified with a lightweight CNN.

### Generated Result Figures

We generated **3 main result figures** for the meeting demo:

1. **Training Curves**
2. **Confusion Matrix**
3. **Sample Grid**

---

## Result Figures

### 1. Training Curves

Shows training and validation loss/accuracy across epochs.

![Training Curves](docs/assets/multifrequency_demo/training_curves.png)

### 2. Confusion Matrix

Shows the band-level classification performance on the test set.

![Confusion Matrix](docs/assets/multifrequency_demo/confusion_matrix.png)

### 3. Sample Grid

Shows example SAR patches from C-band, S-band, and L-band.

![Sample Grid](docs/assets/multifrequency_demo/sample_grid.png)

---

## Demo Result

The trained lightweight CNN achieved the following result on the test set:

| Metric | Value |
|---|---:|
| Correct predictions | 256 / 267 |
| Approximate test accuracy | 95.88% |

---

## Key Takeaway

This demo shows that:

- multi-band SAR patches can be prepared in a common pipeline,
- a lightweight CNN can learn band-level distinctions,
- and the repository is ready for larger multi-frequency SAR experiments.

Full demo documentation:

[Multi-Frequency SAR Demo](docs/multifrequency_demo.md)
