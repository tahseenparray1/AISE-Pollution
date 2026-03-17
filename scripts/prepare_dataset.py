"""
prepare_dataset.py  (V2 — fixed)

Changes vs original V2:
  FIX (Bug #4): Restore * 1e11 scaling before log1p for emission variables
                (PM25, NH3, SO2, NOx, NMVOC_e).  Without this, their raw
                values (~1e-12) make log1p ≈ 0, eliminating all signal.
                bio and NMVOC_finn are left on their own scale as before.
  FIX (Bottleneck): load_raw_or_derived for 'vent_coef' was re-loading
                    u10, v10, and pblh from disk even though they are loaded
                    again later in the main feature loop.  Fixed by accepting
                    an optional pre-loaded cache dict.
"""

import os
import numpy as np
from tqdm import tqdm
from src.utils.config import load_config

cfg     = load_config("configs/prepare_dataset.yaml")
RAW_PATH = cfg.paths.raw_path

# Emission variables that need the 1e11 pre-scaling before log1p
# (FIX Bug #4: these were missing the scale in V2)
EMI_SCALE_VARS = {"PM25", "NH3", "SO2", "NOx", "NMVOC_e"}
# Emission variables that only need log1p (larger natural scale)
EMI_LOGONLY_VARS = {"bio", "NMVOC_finn"}

derived_features = ['wind_speed', 'vent_coef', 'rain_mask']
all_features     = (cfg.features.met_variables_raw
                    + cfg.features.emission_variables_raw
                    + derived_features)


# ---------------------------------------------------------------------------
# Raw / derived feature loader
# FIX (Bottleneck): accept an optional month_cache so vent_coef doesn't
# reload u10/v10/pblh from disk when they've already been read this month.
# ---------------------------------------------------------------------------
def load_raw_or_derived(feat, month, month_cache=None):
    def _load(name):
        if month_cache is not None and name in month_cache:
            return month_cache[name]
        return np.load(os.path.join(RAW_PATH, month, f"{name}.npy")).astype(np.float32)

    if feat == 'wind_speed':
        u = _load('u10')
        v = _load('v10')
        return np.sqrt(u ** 2 + v ** 2)

    elif feat == 'vent_coef':
        u    = _load('u10')
        v    = _load('v10')
        ws   = np.sqrt(u ** 2 + v ** 2)
        pblh = _load('pblh')
        return np.log1p(ws * pblh)          # uses RAW pblh — matches infer.py

    elif feat == 'rain_mask':
        rain = _load('rain')
        return (rain > 0).astype(np.float32)

    else:
        arr = np.load(os.path.join(RAW_PATH, month, f"{feat}.npy")).astype(np.float32)

        if feat in EMI_SCALE_VARS:
            # FIX Bug #4: scale up tiny emission values before log1p
            arr = np.log1p(arr * 1e11)
        elif feat in EMI_LOGONLY_VARS:
            arr = np.log1p(arr)
        elif feat in ('rain', 'pblh'):
            arr = np.log1p(arr)

        return arr


# ---------------------------------------------------------------------------
# Grid-wise robust statistics (computed on training subset only)
# ---------------------------------------------------------------------------
def compute_gridwise_robust_stats(features, months):
    print("Calculating Grid-Wise Robust Statistics (Train Only)...")
    stats       = {}
    cycle_size  = (12 * 24) + (3 * 24)
    train_len   = 12 * 24

    for feat in tqdm(features, desc="Scanning features"):
        feat_data_train = []
        for month in months:
            # FIX (Bottleneck): build a per-month cache for shared raw arrays
            # so vent_coef doesn't re-read u10/v10/pblh from disk.
            month_cache = {}
            for base in ('u10', 'v10', 'pblh', 'rain'):
                p = os.path.join(RAW_PATH, month, f"{base}.npy")
                if os.path.exists(p):
                    month_cache[base] = np.load(p).astype(np.float32)

            arr = load_raw_or_derived(feat, month, month_cache=month_cache)
            total_hours = arr.shape[0]
            mask = np.zeros(total_hours, dtype=bool)
            for start_idx in range(0, total_hours, cycle_size):
                end_train = min(start_idx + train_len, total_hours)
                mask[start_idx:end_train] = True
            feat_data_train.append(arr[mask])

        feat_data_train = np.concatenate(feat_data_train, axis=0)

        median    = np.median(feat_data_train, axis=0)
        q75, q25  = np.percentile(feat_data_train, [75, 25], axis=0)
        min_iqr   = 1.0 if feat == 'rain_mask' else 5.0
        iqr       = np.clip(q75 - q25, a_min=min_iqr, a_max=None)
        stats[feat] = {
            'median': median.astype(np.float32),
            'iqr':    iqr.astype(np.float32),
        }

    np.save(cfg.paths.stats_path, stats)
    return stats


global_stats = compute_gridwise_robust_stats(all_features, cfg.data.months)

# ---------------------------------------------------------------------------
# Topography proxy from PSFC (train split only)
# ---------------------------------------------------------------------------
print("Generating Topography Map (Train Only)...")
psfc_train = []
cycle_size  = (12 * 24) + (3 * 24)
train_len   = 12 * 24

for m in cfg.data.months:
    arr         = np.load(os.path.join(RAW_PATH, m, "psfc.npy")).astype(np.float32)
    total_hours = arr.shape[0]
    mask        = np.zeros(total_hours, dtype=bool)
    for start_idx in range(0, total_hours, cycle_size):
        mask[start_idx : min(start_idx + train_len, total_hours)] = True
    psfc_train.append(arr[mask])

all_psfc_train = np.concatenate(psfc_train, axis=0)
psfc_median    = np.median(all_psfc_train, axis=0)
topo_proxy     = (psfc_median - np.mean(psfc_median)) / (np.std(psfc_median) + 1e-5)
np.save(os.path.join(os.path.dirname(cfg.paths.stats_path), "topo_proxy.npy"), topo_proxy)


# ---------------------------------------------------------------------------
# Month processing: normalize and split by cycle
# ---------------------------------------------------------------------------
def process_month(month_name):
    # FIX (Bottleneck): build cache once per month for shared raw arrays
    month_cache = {}
    for base in ('u10', 'v10', 'pblh', 'rain'):
        p = os.path.join(RAW_PATH, month_name, f"{base}.npy")
        if os.path.exists(p):
            month_cache[base] = np.load(p).astype(np.float32)

    month_data = []
    for feat in all_features:
        arr = load_raw_or_derived(feat, month_name, month_cache=month_cache)
        arr = (arr - global_stats[feat]["median"]) / global_stats[feat]["iqr"]
        month_data.append(arr)

    combined    = np.stack(month_data, axis=-1)
    total_hours = combined.shape[0]

    train_blocks, val_blocks = [], []
    for start_idx in range(0, total_hours, cycle_size):
        end_train = min(start_idx + (12 * 24), total_hours)
        end_val   = min(start_idx + cycle_size, total_hours)
        train_blocks.append(combined[start_idx:end_train])
        if end_val > end_train:
            val_blocks.append(combined[end_train:end_val])

    return train_blocks, val_blocks


def build_dataset_and_indices(blocks, window_size, stride):
    concatenated, valid_starts, current_offset = [], [], 0
    for block in blocks:
        T = block.shape[0]
        if T >= window_size:
            for i in range(0, T - window_size + 1, stride):
                valid_starts.append(current_offset + i)
        concatenated.append(block)
        current_offset += T
    return np.concatenate(concatenated, axis=0), np.array(valid_starts, dtype=np.int32)


os.makedirs(cfg.paths.train_savepath, exist_ok=True)
os.makedirs(cfg.paths.val_savepath,   exist_ok=True)

all_train_blocks, all_val_blocks = [], []
print("Processing Data...")
for month in tqdm(cfg.data.months):
    t_blocks, v_blocks = process_month(month)
    all_train_blocks.extend(t_blocks)
    all_val_blocks.extend(v_blocks)

final_train, train_indices = build_dataset_and_indices(
    all_train_blocks, cfg.data.horizon, cfg.data.stride)
final_val, val_indices = build_dataset_and_indices(
    all_val_blocks, cfg.data.horizon, cfg.data.stride)

np.save(os.path.join(cfg.paths.train_savepath, "train_data.npy"),    final_train)
np.save(os.path.join(cfg.paths.train_savepath, "train_indices.npy"), train_indices)
np.save(os.path.join(cfg.paths.val_savepath,   "val_data.npy"),      final_val)
np.save(os.path.join(cfg.paths.val_savepath,   "val_indices.npy"),   val_indices)
print(f"Success! Train shape: {final_train.shape} | Val shape: {final_val.shape}")