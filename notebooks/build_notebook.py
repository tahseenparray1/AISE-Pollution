"""Builds the pm25_pipeline.ipynb notebook programmatically."""
import json, os

def cell(source, cell_type="code"):
    if cell_type == "markdown":
        return {"cell_type":"markdown","metadata":{},"source": source.split("\n") if isinstance(source, str) else source}
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source": source.split("\n") if isinstance(source, str) else source}

def md(s): return cell(s, "markdown")
def code(s): return cell(s, "code")

cells = []

# ============================================================
# SECTION 1: SETUP
# ============================================================
cells.append(md("# 🇮🇳 India in the Haze: PM2.5 Concentration Forecasting\n\n---\n\n**End-to-end pipeline**: Deep EDA → Data Preprocessing → Model Training → Inference\n\nThis notebook uses the **WNO (Wavelet Neural Operator)** model from `models/` and orchestrates\ndata preprocessing, training, and inference via `scripts/`."))

cells.append(md("## 1. Setup & Imports"))

cells.append(code("""import sys, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings('ignore')

# --- Project Root Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
print(f"Project root: {PROJECT_ROOT}")

# Plotting defaults
plt.rcParams.update({
    'figure.dpi': 120, 'savefig.dpi': 150,
    'font.size': 11, 'axes.titlesize': 13,
    'axes.labelsize': 11, 'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22', 'text.color': '#c9d1d9',
    'axes.labelcolor': '#c9d1d9', 'xtick.color': '#8b949e',
    'ytick.color': '#8b949e', 'axes.edgecolor': '#30363d',
    'grid.color': '#21262d', 'grid.alpha': 0.6,
})

RAW_PATH = 'raw'
MONTHS = ['APRIL_16', 'JULY_16', 'OCT_16', 'DEC_16']
SEASON_LABELS = {'APRIL_16': 'Summer (Apr)', 'JULY_16': 'Monsoon (Jul)',
                 'OCT_16': 'Post-Monsoon (Oct)', 'DEC_16': 'Winter (Dec)'}
SEASON_COLORS = {'APRIL_16': '#ff6b6b', 'JULY_16': '#51cf66',
                 'OCT_16': '#ffd43b', 'DEC_16': '#74c0fc'}

MET_VARS = ['cpm25','q2','t2','u10','v10','swdown','pblh','rain']
EMI_VARS = ['PM25','NH3','SO2','NOx','NMVOC_e','NMVOC_finn','bio']
ALL_FEATURES = MET_VARS + EMI_VARS

print("Setup complete ✅")"""))

# ============================================================
# SECTION 2: DEEP EDA
# ============================================================
cells.append(md("---\n## 2. Deep Exploratory Data Analysis\n\nWe explore the raw WRF-Chem simulation data across four seasonal months of 2016."))

# 2.1 Data Overview
cells.append(md("### 2.1 Data Overview — Shapes, Sizes & Missing Values"))
cells.append(code("""# Load lat/lon grid
lat_lon = np.load(os.path.join(RAW_PATH, 'lat_long.npy'))
print(f"Lat/Lon grid shape: {lat_lon.shape}")
lat = lat_lon[0]  # (140, 124)
lon = lat_lon[1]  # (140, 124)

# Scan all months and features
overview_rows = []
for month in MONTHS:
    for feat in ALL_FEATURES + ['time']:
        fpath = os.path.join(RAW_PATH, month, f'{feat}.npy')
        if os.path.exists(fpath):
            arr = np.load(fpath, mmap_mode='r')
            size_mb = os.path.getsize(fpath) / (1024**2)
            overview_rows.append({
                'Month': SEASON_LABELS[month], 'Feature': feat,
                'Shape': str(arr.shape), 'Dtype': str(arr.dtype),
                'Size (MB)': f'{size_mb:.1f}'
            })

df_overview = pd.DataFrame(overview_rows)
print(f"\\nTotal files scanned: {len(df_overview)}")
display(df_overview.pivot_table(index='Feature', columns='Month', values='Shape', aggfunc='first'))"""))

# 2.2 Spatial Grid
cells.append(md("### 2.2 Spatial Grid — India Domain (140 × 124)"))
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Latitude map
im0 = axes[0].pcolormesh(lon, lat, lat, cmap='inferno', shading='auto')
axes[0].set_title('Latitude Grid', fontweight='bold')
axes[0].set_xlabel('Longitude (°E)'); axes[0].set_ylabel('Latitude (°N)')
plt.colorbar(im0, ax=axes[0], label='Latitude (°N)')

# Longitude map
im1 = axes[1].pcolormesh(lon, lat, lon, cmap='viridis', shading='auto')
axes[1].set_title('Longitude Grid', fontweight='bold')
axes[1].set_xlabel('Longitude (°E)'); axes[1].set_ylabel('Latitude (°N)')
plt.colorbar(im1, ax=axes[1], label='Longitude (°E)')

fig.suptitle('India Domain — 25 km × 25 km Spatial Grid', fontsize=15, fontweight='bold', color='#58a6ff')
plt.tight_layout()
plt.show()"""))

# 2.3 PM2.5 Seasonal Heatmaps
cells.append(md("### 2.3 PM2.5 Concentration — Seasonal Spatial Comparison"))
cells.append(code("""fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.ravel()

vmin_global, vmax_global = 0, 0
means = {}
for month in MONTHS:
    arr = np.load(os.path.join(RAW_PATH, month, 'cpm25.npy'), mmap_mode='r')
    m = np.mean(arr, axis=0)
    means[month] = m
    vmax_global = max(vmax_global, np.percentile(m, 98))

for i, month in enumerate(MONTHS):
    im = axes[i].pcolormesh(lon, lat, means[month], cmap='hot_r',
                            shading='auto', vmin=0, vmax=vmax_global)
    axes[i].set_title(SEASON_LABELS[month], fontweight='bold', fontsize=13,
                      color=SEASON_COLORS[month])
    axes[i].set_xlabel('Lon (°E)'); axes[i].set_ylabel('Lat (°N)')
    plt.colorbar(im, ax=axes[i], label='PM2.5 (µg/m³)', shrink=0.85)

fig.suptitle('Mean Ground-Level PM2.5 Concentration by Season (2016)',
             fontsize=16, fontweight='bold', color='#58a6ff', y=1.02)
plt.tight_layout()
plt.show()

# Print domain-average stats
print("\\n📊 Domain-Average PM2.5 Statistics:")
for month in MONTHS:
    m = means[month]
    print(f"  {SEASON_LABELS[month]:>25s} | Mean: {m.mean():.2f} | Max: {m.max():.2f} | Std: {m.std():.2f}")"""))

# 2.4 Temporal Evolution
cells.append(md("### 2.4 PM2.5 Temporal Evolution & Diurnal Cycles"))
cells.append(code("""fig, axes = plt.subplots(2, 2, figsize=(15, 8))
axes = axes.ravel()

for i, month in enumerate(MONTHS):
    arr = np.load(os.path.join(RAW_PATH, month, 'cpm25.npy'), mmap_mode='r')
    domain_avg = np.nanmean(arr, axis=(1, 2))
    hours = np.arange(len(domain_avg))

    axes[i].plot(hours, domain_avg, color=SEASON_COLORS[month], linewidth=0.8, alpha=0.9)
    axes[i].fill_between(hours, domain_avg, alpha=0.15, color=SEASON_COLORS[month])
    axes[i].set_title(SEASON_LABELS[month], fontweight='bold', color=SEASON_COLORS[month])
    axes[i].set_xlabel('Hour Index'); axes[i].set_ylabel('PM2.5 (µg/m³)')
    axes[i].grid(True, alpha=0.3)

fig.suptitle('Domain-Averaged PM2.5 Time Series per Season',
             fontsize=15, fontweight='bold', color='#58a6ff')
plt.tight_layout()
plt.show()

# Diurnal cycle
fig, ax = plt.subplots(figsize=(10, 5))
for month in MONTHS:
    arr = np.load(os.path.join(RAW_PATH, month, 'cpm25.npy'), mmap_mode='r')
    domain_avg = np.nanmean(arr, axis=(1, 2))
    n_hours = len(domain_avg)
    n_days = n_hours // 24
    reshaped = domain_avg[:n_days*24].reshape(n_days, 24)
    diurnal = reshaped.mean(axis=0)
    ax.plot(range(24), diurnal, marker='o', markersize=4, linewidth=2,
            label=SEASON_LABELS[month], color=SEASON_COLORS[month])

ax.set_xlabel('Hour of Day (UTC)'); ax.set_ylabel('PM2.5 (µg/m³)')
ax.set_title('Diurnal Cycle of PM2.5 by Season', fontweight='bold', color='#58a6ff')
ax.set_xticks(range(0, 24, 3)); ax.legend(framealpha=0.3); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()"""))

# 2.5 Meteorological Variables
cells.append(md("### 2.5 Meteorological Variables — Distributions & Spatial Patterns"))
cells.append(code("""met_display = ['t2', 'pblh', 'rain', 'swdown', 'q2', 'psfc']
met_labels = {'t2': 'Temperature 2m (K)', 'pblh': 'PBL Height (m)',
              'rain': 'Rainfall (mm)', 'swdown': 'SW Radiation (W/m²)',
              'q2': 'Humidity (kg/kg)', 'psfc': 'Surface Pressure (Pa)'}
met_cmaps = {'t2': 'RdYlBu_r', 'pblh': 'YlOrRd', 'rain': 'Blues',
             'swdown': 'YlOrBr', 'q2': 'BuGn', 'psfc': 'terrain'}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

# Use DEC_16 for demonstration (most polluted season)
demo_month = 'DEC_16'
for i, feat in enumerate(met_display):
    arr = np.load(os.path.join(RAW_PATH, demo_month, f'{feat}.npy'), mmap_mode='r')
    spatial_mean = np.mean(arr, axis=0)
    im = axes[i].pcolormesh(lon, lat, spatial_mean, cmap=met_cmaps[feat], shading='auto')
    axes[i].set_title(met_labels[feat], fontweight='bold')
    plt.colorbar(im, ax=axes[i], shrink=0.85)

fig.suptitle(f'Meteorological Variables — Time-Averaged Spatial Maps ({SEASON_LABELS[demo_month]})',
             fontsize=15, fontweight='bold', color='#58a6ff', y=1.02)
plt.tight_layout(); plt.show()"""))

# 2.6 Emission Variables
cells.append(md("### 2.6 Emission Sources — Spatial Maps"))
cells.append(code("""emi_labels = {'PM25': 'Primary PM2.5', 'NH3': 'Ammonia (NH₃)',
              'SO2': 'Sulphur Dioxide (SO₂)', 'NOx': 'Nitrogen Oxides (NOx)',
              'NMVOC_e': 'NMVOCs (Anthropogenic)', 'NMVOC_finn': 'NMVOCs (Biomass)',
              'bio': 'Biogenic Isoprene'}

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.ravel()

demo_month = 'DEC_16'
for i, feat in enumerate(EMI_VARS):
    arr = np.load(os.path.join(RAW_PATH, demo_month, f'{feat}.npy'), mmap_mode='r')
    spatial_mean = np.mean(arr, axis=0)
    im = axes[i].pcolormesh(lon, lat, np.log1p(spatial_mean), cmap='magma', shading='auto')
    axes[i].set_title(emi_labels[feat], fontweight='bold', fontsize=10)
    plt.colorbar(im, ax=axes[i], shrink=0.85, label='log(1+x)')

axes[-1].axis('off')
fig.suptitle(f'Emission Source Spatial Maps — log-scaled ({SEASON_LABELS[demo_month]})',
             fontsize=15, fontweight='bold', color='#58a6ff', y=1.02)
plt.tight_layout(); plt.show()"""))

# 2.7 Wind Fields
cells.append(md("### 2.7 Wind Vector Fields — Transport Patterns"))
cells.append(code("""fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.ravel()
skip = 5  # subsample for quiver readability

for i, month in enumerate(MONTHS):
    u = np.load(os.path.join(RAW_PATH, month, 'u10.npy'), mmap_mode='r')
    v = np.load(os.path.join(RAW_PATH, month, 'v10.npy'), mmap_mode='r')
    u_mean = np.mean(u, axis=0); v_mean = np.mean(v, axis=0)
    ws = np.sqrt(u_mean**2 + v_mean**2)

    im = axes[i].pcolormesh(lon, lat, ws, cmap='coolwarm', shading='auto', vmin=0, vmax=6)
    axes[i].quiver(lon[::skip, ::skip], lat[::skip, ::skip],
                   u_mean[::skip, ::skip], v_mean[::skip, ::skip],
                   color='white', alpha=0.7, scale=80, width=0.003)
    axes[i].set_title(SEASON_LABELS[month], fontweight='bold', color=SEASON_COLORS[month])
    plt.colorbar(im, ax=axes[i], label='Wind Speed (m/s)', shrink=0.85)

fig.suptitle('Mean Wind Fields by Season (u10, v10)',
             fontsize=15, fontweight='bold', color='#58a6ff', y=1.02)
plt.tight_layout(); plt.show()"""))

# 2.8 Correlation Analysis
cells.append(md("### 2.8 Feature–PM2.5 Correlation Analysis"))
cells.append(code("""# Compute domain-averaged time series per feature, per month, then correlate
corr_features = ['q2','t2','swdown','pblh','rain','PM25','NH3','SO2','NOx']

corr_data = {f: [] for f in corr_features}
corr_data['cpm25'] = []

for month in MONTHS:
    pm = np.load(os.path.join(RAW_PATH, month, 'cpm25.npy'), mmap_mode='r')
    corr_data['cpm25'].append(np.nanmean(pm, axis=(1,2)))
    for feat in corr_features:
        arr = np.load(os.path.join(RAW_PATH, month, f'{feat}.npy'), mmap_mode='r')
        # Trim to match cpm25 length if needed
        n = len(corr_data['cpm25'][-1])
        corr_data[feat].append(np.nanmean(arr[:n], axis=(1,2)))

# Concatenate all months
all_keys = ['cpm25'] + corr_features
concat = {k: np.concatenate(corr_data[k]) for k in all_keys}
df_corr = pd.DataFrame(concat)

fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df_corr.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Pearson Correlation'},
            annot_kws={'size': 9, 'color': 'white'})
ax.set_title('Feature Correlation Matrix (Domain-Averaged Time Series)',
             fontweight='bold', color='#58a6ff', fontsize=13)
plt.tight_layout(); plt.show()"""))

# 2.9 Summary Statistics
cells.append(md("### 2.9 Summary Statistics Table"))
cells.append(code("""stats_rows = []
for month in MONTHS:
    for feat in ALL_FEATURES:
        fpath = os.path.join(RAW_PATH, month, f'{feat}.npy')
        if os.path.exists(fpath):
            arr = np.load(fpath, mmap_mode='r')
            flat = arr.ravel()
            # Sample for speed on large arrays
            if len(flat) > 1_000_000:
                idx = np.random.choice(len(flat), 1_000_000, replace=False)
                flat = flat[idx]
            stats_rows.append({
                'Season': SEASON_LABELS[month], 'Feature': feat,
                'Min': f'{np.nanmin(flat):.4g}', 'Max': f'{np.nanmax(flat):.4g}',
                'Mean': f'{np.nanmean(flat):.4g}', 'Std': f'{np.nanstd(flat):.4g}',
                'Median': f'{np.nanmedian(flat):.4g}'
            })

df_stats = pd.DataFrame(stats_rows)
display(df_stats.style.set_caption("Feature Statistics by Season").set_table_styles(
    [{'selector': 'caption', 'props': [('font-size', '14px'), ('font-weight', 'bold')]}]))"""))

# ============================================================
# SECTION 3: DATA PREPROCESSING
# ============================================================
cells.append(md("---\n## 3. Data Preprocessing\n\nRun `scripts/prepare_dataset.py` to compute grid-wise robust statistics,\napply normalization, generate derived features, and construct train/val splits."))

cells.append(code("""print("=" * 60)
print("RUNNING DATA PREPROCESSING PIPELINE")
print("=" * 60)

%run scripts/prepare_dataset.py"""))

cells.append(code("""# Verify outputs
import glob
for split in ['train', 'val']:
    base = f'data/{split}'
    if os.path.isdir(base):
        for f in sorted(glob.glob(os.path.join(base, '*.npy'))):
            arr = np.load(f, mmap_mode='r')
            print(f"  {os.path.basename(f):>20s} -> shape: {arr.shape}, dtype: {arr.dtype}")
    else:
        print(f"  ⚠️  Directory {base} not found — preprocessing may have failed.")"""))

# ============================================================
# SECTION 4: TRAINING
# ============================================================
cells.append(md("---\n## 4. Model Training\n\nTrain the **WNO (Wavelet Neural Operator)** model defined in `models/baseline_model.py`\nusing the training script `scripts/train.py`."))

cells.append(code("""print("=" * 60)
print("RUNNING MODEL TRAINING")
print("=" * 60)

%run scripts/train.py"""))

cells.append(md("### 4.1 Training Curves"))
cells.append(code("""# Plot training curves from log.json
log_path = 'log.json'
if os.path.exists(log_path):
    import json
    with open(log_path) as f:
        log = json.load(f)

    epochs = [entry['epoch'] for entry in log]
    train_rmse = [entry['train_rmse'] for entry in log]
    val_rmse = [entry['val_rmse'] for entry in log]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_rmse, label='Train RMSE', color='#58a6ff', linewidth=2)
    ax.plot(epochs, val_rmse, label='Val RMSE', color='#f97583', linewidth=2)
    ax.fill_between(epochs, train_rmse, alpha=0.1, color='#58a6ff')
    ax.fill_between(epochs, val_rmse, alpha=0.1, color='#f97583')
    ax.set_xlabel('Epoch'); ax.set_ylabel('RMSE (µg/m³)')
    ax.set_title('Training & Validation RMSE', fontweight='bold', color='#58a6ff')
    ax.legend(framealpha=0.3); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    print(f"\\nBest Val RMSE: {min(val_rmse):.4f} at epoch {epochs[val_rmse.index(min(val_rmse))]}")
else:
    print("⚠️ log.json not found — training may not have completed.")"""))

# ============================================================
# SECTION 5: INFERENCE
# ============================================================
cells.append(md("---\n## 5. Inference on Test Data\n\nRun `scripts/infer.py` to generate predictions on the unseen 2017 test months."))

cells.append(code("""print("=" * 60)
print("RUNNING INFERENCE")
print("=" * 60)

%run scripts/infer.py"""))

cells.append(code("""# Verify predictions
preds_path = 'preds.npy'
if os.path.exists(preds_path):
    preds = np.load(preds_path)
    print(f"✅ Predictions shape: {preds.shape}")
    print(f"   Expected:          (996, 140, 124, 16)")
    print(f"   Match: {preds.shape == (996, 140, 124, 16)}")
    print(f"   Value range: [{preds.min():.2f}, {preds.max():.2f}]")
    print(f"   Mean: {preds.mean():.2f}, Std: {preds.std():.2f}")
else:
    print("⚠️ preds.npy not found — inference may not have completed.")"""))

# ============================================================
# SECTION 6: POST-INFERENCE ANALYSIS
# ============================================================
cells.append(md("---\n## 6. Post-Inference Analysis & Visualization"))

cells.append(code("""if os.path.exists('preds.npy'):
    preds = np.load('preds.npy')
    lat_lon = np.load(os.path.join(RAW_PATH, 'lat_long.npy'))
    lat, lon = lat_lon[0], lat_lon[1]

    # Show predictions for sample indices at different forecast horizons
    sample_ids = [0, 200, 500, 800]
    horizons = [0, 5, 10, 15]  # hours 1, 6, 11, 16

    fig, axes = plt.subplots(len(sample_ids), len(horizons), figsize=(20, 16))
    for r, sid in enumerate(sample_ids):
        for c, h in enumerate(horizons):
            im = axes[r, c].pcolormesh(lon, lat, preds[sid, :, :, h],
                                        cmap='hot_r', shading='auto', vmin=0)
            if r == 0:
                axes[r, c].set_title(f'T+{h+1}h', fontweight='bold', fontsize=12)
            if c == 0:
                axes[r, c].set_ylabel(f'Sample {sid}', fontweight='bold')
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    fig.suptitle('Predicted PM2.5 Fields — Selected Samples × Forecast Horizons',
                 fontsize=16, fontweight='bold', color='#58a6ff', y=1.01)
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='PM2.5 (µg/m³)')
    plt.tight_layout(rect=[0, 0, 0.92, 0.98])
    plt.show()

    # Forecast horizon error growth (domain-averaged std as proxy)
    fig, ax = plt.subplots(figsize=(8, 4))
    domain_means = preds.mean(axis=(1, 2))  # (996, 16)
    mean_per_h = domain_means.mean(axis=0)
    std_per_h = domain_means.std(axis=0)
    hours = np.arange(1, 17)
    ax.plot(hours, mean_per_h, 'o-', color='#58a6ff', linewidth=2, label='Mean Prediction')
    ax.fill_between(hours, mean_per_h - std_per_h, mean_per_h + std_per_h,
                    alpha=0.2, color='#58a6ff')
    ax.set_xlabel('Forecast Horizon (hours)'); ax.set_ylabel('PM2.5 (µg/m³)')
    ax.set_title('Prediction Statistics vs Forecast Horizon', fontweight='bold', color='#58a6ff')
    ax.legend(framealpha=0.3); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
else:
    print("⚠️ No predictions available to visualize.")"""))

cells.append(md("---\n## ✅ Pipeline Complete\n\nThe full end-to-end pipeline has been executed:\n1. **Deep EDA** — Explored PM2.5 concentrations, meteorology, emissions, wind fields, and correlations\n2. **Preprocessing** — Computed robust grid-wise stats, applied normalization, built train/val splits\n3. **Training** — Trained the WNO model with Huber + spatial gradient loss and SWA\n4. **Inference** — Generated `preds.npy` with shape `(996, 140, 124, 16)`\n5. **Analysis** — Visualized predictions across samples and forecast horizons"))

# ============================================================
# BUILD THE NOTEBOOK
# ============================================================
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

# Fix cell sources to be lists of lines with \n
for c in nb["cells"]:
    if isinstance(c["source"], str):
        lines = c["source"].split("\n")
        c["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

out_path = os.path.join(os.path.dirname(__file__), "pm25_pipeline.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook created: {out_path}")
