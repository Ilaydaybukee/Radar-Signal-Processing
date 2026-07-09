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

```text
C_band
S_band
L_band