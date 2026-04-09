# 🌫️ AISE Pollution — PM2.5 Spatio-Temporal Forecasting

A deep-learning pipeline for **hourly PM2.5 concentration forecasting** over a 140 × 124 spatial grid, built for the **[AISEHACK Theme 2 — Pollution Forecasting](https://www.kaggle.com/competitions/aisehack-theme-2/overview)** Kaggle competition.

> **Core idea:** Given 10 hours of meteorological, emission, and derived atmospheric features, predict the next 16 hours of PM2.5 concentrations at every grid cell.

---

## Table of Contents

- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Features & Preprocessing](#features--preprocessing)
- [Pipeline Stages](#pipeline-stages)
  - [1. Dataset Preparation](#1-dataset-preparation-prepare_datasetpy)
  - [2. Model Training](#2-model-training-trainpy)
  - [3. Inference](#3-inference-inferpy)
- [Configuration](#configuration)
- [Using with Kaggle](#using-with-kaggle)
- [Requirements](#requirements)
- [Kaggle Data Locations](#kaggle-data-locations)
- [Kaggle Resource Tips](#kaggle-resource-tips)

---

## Architecture

The model is a **Wavelet Neural Operator (WNO)** that replaces classic Fourier spectral convolutions with **2D Haar Discrete Wavelet Transforms**:

```
Input → Conv1×1 Encoder → [WNO Block ×4] → GELU → Conv1×1 Decoder → + Residual → Output
```

Each **WNO Block** performs:

1. **GroupNorm** on the input
2. **Haar DWT** — decomposes into LL / LH / HL / HH sub-bands at half resolution
3. **Spectral Mixer** — group-wise 3×3 conv across the 4 sub-bands
4. **Inverse Haar DWT** — reconstructs back to full resolution
5. **Spatial Mixer** — depthwise 5×5 conv for long-range spatial context
6. **Pointwise Conv** + **GELU** activation + **Residual connection**

The decoder head outputs a **delta** (change) which is added to the last known PM2.5 state via a residual connection — the network learns the *change* from the current state rather than absolute values.

### Training Enhancements

| Technique | Detail |
|---|---|
| **Multi-Seed Ensemble** | 3 models trained with seeds `[0, 42, 2026]`, averaged at inference |
| **SWA** | Stochastic Weight Averaging kicks in at ~57.6% of training |
| **Cosine Annealing LR** | Decays from `7.82e-4` to `1e-6` with SWA override |
| **Spatial Gradient Loss** | L1 penalty on spatial gradients to preserve sharp concentration fronts |
| **Horizon Weighting** | Later forecast steps (hours 11–26) receive up to 1.5× loss weight |
| **Input Noise** | Gaussian noise (`σ = 0.01`) injected during training (topography channel excluded) |
| **Gradient Clipping** | Max norm = 1.0 |

---

## Repository Structure

```
AISE-Pollution/
├── configs/                         # YAML configuration files
│   ├── prepare_dataset.yaml         #   Dataset preparation settings
│   ├── prepare_rapid.yaml           #   Rapid/debug dataset prep variant
│   ├── train.yaml                   #   Training hyperparameters
│   ├── train_rapid.yaml             #   Rapid/debug training variant
│   └── infer.yaml                   #   Inference settings
├── models/
│   └── baseline_model.py            # WNO model (FNO2D class w/ Haar wavelet blocks)
├── scripts/
│   ├── prepare_dataset.py           # Stage 1 — data preprocessing & normalization
│   ├── train.py                     # Stage 2 — multi-seed ensemble training
│   └── infer.py                     # Stage 3 — ensemble inference & submission
├── src/
│   └── utils/
│       ├── adam.py                   # Custom Adam optimizer implementation
│       ├── config.py                # YAML config loader with dot-attribute access
│       ├── metrics.py               # RMSE, MFB, SMAPE evaluation metrics
│       └── utilities3.py            # Lp loss utilities
├── notebooks/                       # Jupyter notebooks (for EDA / experimentation)
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

---

## Features & Preprocessing

### Input Variables (17 channels)

| Category | Variables |
|---|---|
| **Meteorological (7)** | `cpm25`, `t2`, `u10`, `v10`, `swdown`, `pblh`, `rain` |
| **Emission (7)** | `PM25`, `NH3`, `SO2`, `NOx`, `NMVOC_e`, `NMVOC_finn`, `bio` |
| **Derived (3)** | `wind_speed`, `vent_coef`, `rain_mask` |

A **topography proxy** channel (z-score normalized median surface pressure) is also appended as a static spatial feature.

### Normalization Strategy

| Variable Type | Method |
|---|---|
| Meteorological + Derived | **Robust grid-wise** — per-pixel `(x − median) / IQR` with IQR floored at 5.0 |
| Emission | **Global min–max** after `log1p(x × 1e11)` scaling |
| Rain, PBLH | `log1p` transform before robust normalization |

Statistics are computed across all training months and saved to `grid_robust_stats.npy`.

---

## Pipeline Stages

Run the three stages **sequentially** — each stage's outputs feed the next:

```bash
python scripts/prepare_dataset.py → python scripts/train.py → python scripts/infer.py
```

### 1. Dataset Preparation (`prepare_dataset.py`)

- Loads raw `.npy` arrays per month from the competition data
- Computes derived features (`wind_speed = √(u² + v²)`, `vent_coef = log1p(ws × pblh)`, `rain_mask`)
- Computes grid-wise robust statistics and saves them
- Generates a topography proxy from median surface pressure
- Normalizes all features and concatenates into a single `(T, H, W, F)` tensor per month
- Builds a sliding-window index (`horizon = 26`, `stride = 4`) for efficient sampling
- Saves `train_data.npy` and `train_indices.npy`

### 2. Model Training (`train.py`)

- Loads prepared data **fully into RAM** via `FastInMemoryDataset`
- Trains **3 independent models** (seeds `[0, 42, 2026]`) sequentially, clearing GPU memory between runs
- Uses composite loss: **horizon-weighted MSE** + **0.3228 × spatial gradient L1** (computed in physical units)
- Applies **SWA** in the final ~42% of epochs
- Saves one checkpoint per seed: `fno_baseline_seed{0,42,2026}.pt`

### 3. Inference (`infer.py`)

- Loads each of the 3 seed checkpoints sequentially (memory-safe)
- Applies identical normalization and derived-feature computation to test data
- Accumulates predictions from all 3 models and computes the ensemble average
- De-normalizes predictions back to physical PM2.5 units
- Clips negative values to 0 (physical bound)
- Saves final `preds.npy` with shape `(N_test, H, W, 16)`

---

## Configuration

All hyperparameters are controlled via YAML files in `configs/`. Key parameters:

| Config | Parameter | Default |
|---|---|---|
| `prepare_dataset.yaml` | `months` | `APRIL_16, JULY_16, OCT_16, DEC_16` |
| | `horizon` / `stride` | `26` / `4` |
| | `val_frac` | `0.2` |
| `train.yaml` | `width` | `192` |
| | `batch_size` | `16` |
| | `epochs` | `80` |
| | `lr` / `weight_decay` | `7.82e-4` / `7.6e-5` |
| `infer.yaml` | `ntest` | `996` |

> **Rapid configs** (`prepare_rapid.yaml`, `train_rapid.yaml`) are provided for quick debugging runs with reduced settings.

---

## Using with Kaggle

This repository is designed to be used **as a Kaggle dataset**.

### Setup

1. Upload this repository as a **Kaggle Dataset**
2. Open the **[baseline Kaggle notebook](https://www.kaggle.com/code/siddharthandileep/baseline-run-aisehack-test/)** → Copy and Edit
3. Add:
   - This repository (as a dataset)
   - The official competition dataset
4. Set accelerator to **GPU (P100)**
5. Click **Save & Run All**

The notebook will execute dataset preparation, training, and inference end-to-end, producing `preds.npy` for submission.

---

## Requirements

```
torch>=2.0
triton
numpy
scipy
pandas
h5py
netCDF4
xarray
matplotlib
seaborn
scikit-learn
joblib
tqdm
PyYAML
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Kaggle Data Locations

| Artifact | Path |
|---|---|
| Raw training data | `/kaggle/input/competitions/aisehack-theme-2/raw/<MONTH>/<feature>.npy` |
| Test inputs | `/kaggle/input/competitions/aisehack-theme-2/test_in/<feature>.npy` |
| Grid-wise statistics | `/kaggle/working/grid_robust_stats.npy` |
| Topography proxy | `/kaggle/working/topo_proxy.npy` |
| Prepared datasets | `/kaggle/temp/data/train/` |
| Model checkpoints | `/kaggle/working/fno_baseline_seed{0,42,2026}.pt` |
| Final predictions | `/kaggle/working/preds.npy` |

---

## Kaggle Resource Tips

- **GPU P100** — peak CPU RAM limit of **29 GB**
- `/kaggle/working/` — persistent (up to **20 GB**); save final predictions here
- `/kaggle/temp/` — non-persistent; good for large intermediate data
- `/kaggle/input/` — **read-only**
- Be mindful of session time limits and weekly GPU quotas; the full pipeline takes **~9.8 hours**
- Final test predictions **must** be saved as `/kaggle/working/preds.npy`
