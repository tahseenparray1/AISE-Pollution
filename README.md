# 🌫️ PM2.5 Concentration Forecasting

## 📖 Overview

India's rapid industrialisation and urbanisation has led to severe air quality degradation. PM2.5 (particulate matter < 2.5 micrometres) is the dominant pollutant across most regions. These fine particles enter the lungs during breathing and cause serious health issues — the Global Burden of Disease 2019 estimated ~1.67 million deaths attributable to air pollution in India.

Traditional air quality models solve complex differential equations and need large computational resources. Machine Learning models offer a faster alternative by learning directly from data and providing quicker forecasts, enabling timely public health interventions.

This project builds a deep learning pipeline for **hourly PM2.5 concentration forecasting** over a 140 × 124 spatial grid covering India. Given 10 hours of meteorological, emission, and atmospheric features, the model predicts the next 16 hours of PM2.5 concentrations at every grid cell.

The dataset used is from WRF-Chem (a numerical weather-chemistry model) for the year 2016, provided as part of the [AISEHACK Theme 2 Kaggle competition](https://www.kaggle.com/competitions/aisehack-theme-2/overview).

---

## 🧠 Model Architecture

The model is a **Wavelet Neural Operator (WNO)** that replaces classic Fourier spectral convolutions with **2D Haar Discrete Wavelet Transforms**:

```
Input → Conv1×1 Encoder → [WNO Block ×4] → GELU → Conv1×1 Decoder → + Residual → Output
```

Each **WNO Block** performs:
1. 🔹 GroupNorm on the input
2. 🔹 Haar DWT — decomposes into LL / LH / HL / HH sub-bands
3. 🔹 Spectral Mixer — group-wise 3×3 conv across the 4 sub-bands
4. 🔹 Inverse Haar DWT — reconstructs to full resolution
5. 🔹 Spatial Mixer — depthwise 5×5 conv for long-range spatial context
6. 🔹 Pointwise Conv + GELU activation + Residual connection

The decoder outputs a **delta** (change) added to the last known PM2.5 state — the network learns the *change* from the current state rather than absolute values.

Training uses a **3-model ensemble** (seeds `[0, 42, 2026]`) with SWA, cosine annealing LR, spatial gradient loss, and horizon weighting.

---

## 📁 Repository Structure

```
AISE-Pollution/
├── 📂 configs/                       # YAML configuration files
│   ├── prepare_dataset.yaml         #   Dataset preparation settings
│   ├── train.yaml                   #   Training hyperparameters
│   └── infer.yaml                   #   Inference settings
├── 📂 models/
│   └── baseline_model.py            # WNO model definition
├── 📂 scripts/
│   ├── prepare_dataset.py           # Stage 1 — data preprocessing & normalization
│   ├── train.py                     # Stage 2 — multi-seed ensemble training
│   ├── infer.py                     # Stage 3 — ensemble inference & submission
│   ├── data_exploration.py          # EDA utilities
│   ├── results_analysis.py          # Result evaluation
│   └── interpret_results.py         # Result interpretation & visualisation
├── 📂 src/
│   └── utils/                       # Helper modules (optimizer, config loader, metrics)
├── 📂 notebooks/                     # Jupyter notebooks for experimentation
├── 📄 requirements.txt               # Python dependencies
└── 📄 README.md
```

---

## 🔧 Scripts Overview

The project has 6 scripts inside `scripts/`. The first three form the **main pipeline** (run in order), and the remaining three are for analysis.

### ⚙️ Main Pipeline

| Script | What it does |
|---|---|
| `prepare_dataset.py` | 📥 Loads raw `.npy` data files for each month, computes derived weather features (wind speed, ventilation coefficient, rain mask), normalises all variables, and saves processed training data. |
| `train.py` | 🏋️ Trains 3 copies of the WNO model with different random seeds. Uses a combined loss (MSE + spatial gradient penalty) and applies Stochastic Weight Averaging (SWA). Saves one checkpoint per seed. |
| `infer.py` | 🔮 Loads all 3 trained checkpoints, runs each on test data, averages predictions (ensemble), converts back to physical PM2.5 units, clips negatives to zero, and saves `preds.npy`. |

### 📊 Analysis & Visualisation

| Script | What it does |
|---|---|
| `data_exploration.py` | 🗺️ Generates exploratory plots — seasonal PM2.5 maps, emission source hotspot maps, and PM2.5 vs boundary layer height time-series at the most polluted grid cell. |
| `results_analysis.py` | 📈 Compares predictions against ground truth — RMSE per forecast hour, spatial error map, and predicted vs actual PM2.5 curves. Prints overall test RMSE. |
| `interpret_results.py` | 🔍 Detailed analysis of `preds.npy` — summary statistics, spatial heatmaps, distribution histograms, hotspot/coldspot maps, anomaly maps, and hour-to-hour change rates. |

---

## 📦 Package Requirements

All required packages are listed in `requirements.txt`:

| Package | Purpose |
|---|---|
| `torch>=2.0` | 🔥 Deep learning framework (training, inference, GPU) |
| `triton` | ⚡ Torch compiler backend |
| `numpy` | 🔢 Array operations |
| `scipy` | 🧪 Scientific computing |
| `pandas` | 🗃️ Data handling |
| `h5py` | 💾 HDF5 file I/O |
| `netCDF4` | 🌐 Reading WRF-Chem data |
| `xarray` | 📐 Labelled multi-dimensional arrays |
| `matplotlib` | 📉 Plotting |
| `seaborn` | 🎨 Statistical visualisation |
| `scikit-learn` | 🤖 ML utilities and metrics |
| `joblib` | ⚙️ Parallel utilities |
| `tqdm` | ⏳ Progress bars |
| `PyYAML` | 📝 YAML config loading |

Install all at once with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Run Instructions (Kaggle Notebook)

Open the Kaggle Notebook 👉 [**Group ID 46 — PRML Project**](https://www.kaggle.com/code/tahseen123/group-id-46-prml-project)

and run the code cells in order.

---

## 📌 Notes

- ⏱️ The full pipeline takes approximately **3-4 hours** on a Kaggle P100 GPU.
- 💾 `/kaggle/working/` is persistent storage (up to 20 GB) — final outputs are saved here.
- 🗑️ `/kaggle/temp/` is non-persistent — used for large intermediate data.
