import os
import numpy as np
from tqdm import tqdm
from src.utils.config import load_config
import pandas as pd

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
cfg = load_config("configs/prepare_dataset.yaml")
RAW_PATH = cfg.paths.raw_path

# We define the derived features we want to calculate
derived_features = ['wind_speed', 'vent_coef','rain_mask']
skewed_features = ['vent_coef', 'rain', 'bio', 'NMVOC_finn']
all_features = cfg.features.met_variables_raw + cfg.features.emission_variables_raw + derived_features

# ==========================================
# 2. PHYSICS HELPER & STATS CALCULATOR
# ==========================================
def load_raw_or_derived(feat, month):
    """Loads raw .npy files, computes physics, and applies log-transforms to skewed data."""
    
    # 1. Physics Features
    if feat == 'wind_speed':
        u = np.load(os.path.join(RAW_PATH, month, "u10.npy")).astype(np.float32)
        v = np.load(os.path.join(RAW_PATH, month, "v10.npy")).astype(np.float32)
        return np.sqrt(u**2 + v**2)
        
    elif feat == 'vent_coef':
        u = np.load(os.path.join(RAW_PATH, month, "u10.npy")).astype(np.float32)
        v = np.load(os.path.join(RAW_PATH, month, "v10.npy")).astype(np.float32)
        ws = np.sqrt(u**2 + v**2)
        pblh = np.load(os.path.join(RAW_PATH, month, "pblh.npy")).astype(np.float32)
        vc = ws * pblh
        # Apply Log Transform immediately to compress the massive variance
        return np.log1p(vc)
        
    elif feat == 'rain_mask':
        rain = np.load(os.path.join(RAW_PATH, month, "rain.npy")).astype(np.float32)
        # Binary mask: 1.0 if raining, 0.0 if not
        return (rain > 0).astype(np.float32)
        
    # 2. Raw Features
    else:
        arr = np.load(os.path.join(RAW_PATH, month, f"{feat}.npy")).astype(np.float32)
        
        # 3. Log-Transform Skewed Raw Features
        skewed_features = ['rain', 'bio', 'NMVOC_finn']
        if feat in skewed_features:
            arr = np.log1p(arr)
            
        return arr

def compute_gridwise_robust_stats(features, months):
    print("Step 1: Calculating Grid-Wise Robust Statistics (including derived features)...")
    stats = {}
    
    for feat in tqdm(features, desc="Scanning features"):
        feat_data = []
        for month in months:
            data = load_raw_or_derived(feat, month)
            feat_data.append(data)
            
        feat_data = np.concatenate(feat_data, axis=0)
        
        median = np.median(feat_data, axis=0)
        q75, q25 = np.percentile(feat_data, [75, 25], axis=0)
        iqr = q75 - q25
        iqr = np.clip(iqr, a_min=5.0, a_max=None)
        
        stats[feat] = {
            'median': median.astype(np.float32),
            'iqr': iqr.astype(np.float32)
        }
        
    np.save('/kaggle/working/grid_robust_stats.npy', stats)
    return stats

global_stats = compute_gridwise_robust_stats(all_features, cfg.data.months)

# Calculate Static Topography Proxy (Spatial normalization of the temporal median of surface pressure)
psfc_median = global_stats['psfc']['median']
topo_proxy = (psfc_median - np.mean(psfc_median)) / (np.std(psfc_median) + 1e-5)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def process_month(month_name):
    month_data = []

    # 1. Load and Standardize Physical & Derived Features
    for feat in all_features:
        arr = load_raw_or_derived(feat, month_name)
        median = global_stats[feat]["median"]
        iqr = global_stats[feat]["iqr"]
        arr = (arr - median) / iqr
        month_data.append(arr)

    # 2. Add Static Topography Proxy
    # Get total hours from the first feature array since we no longer load time.npy
    total_hours = month_data[0].shape[0] 
    
    # Broadcast the 2D topo map across the time dimension
    topo_time = np.broadcast_to(topo_proxy[None, :, :], (total_hours, 140, 124))
    month_data.append(topo_time)

    combined = np.stack(month_data, axis=-1)

    # --- Interleaved Block Splitting ---
    train_blocks, val_blocks = [], []
    train_chunk_size = 12 * 24
    val_chunk_size = 3 * 24
    cycle_size = train_chunk_size + val_chunk_size
    total_hours = combined.shape[0]

    for start_idx in range(0, total_hours, cycle_size):
        end_train = min(start_idx + train_chunk_size, total_hours)
        end_val = min(start_idx + cycle_size, total_hours)

        train_blocks.append(combined[start_idx:end_train])

        if end_val > end_train:
            val_blocks.append(combined[end_train:end_val])

    # Return the lists of blocks, do NOT concatenate them yet!
    return train_blocks, val_blocks


def build_dataset_and_indices(blocks, window_size, stride):
    """Safely concatenates blocks while tracking valid start indices to prevent leakage."""
    concatenated = []
    valid_starts = []
    current_offset = 0

    for block in blocks:
        T = block.shape[0]
        # Only create windows if the block is larger than our required sequence length
        if T >= window_size:
            # Generate valid start indices STRICTLY within this block
            for i in range(0, T - window_size + 1, stride):
                valid_starts.append(current_offset + i)
        
        concatenated.append(block)
        current_offset += T

    return np.concatenate(concatenated, axis=0), np.array(valid_starts, dtype=np.int32)


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
os.makedirs(cfg.paths.train_savepath, exist_ok=True)
os.makedirs(cfg.paths.val_savepath, exist_ok=True)

all_train_blocks, all_val_blocks = [], []

print("\nStep 2: Processing and Splitting Data...")
for month in tqdm(cfg.data.months):
    t_blocks, v_blocks = process_month(month)
    all_train_blocks.extend(t_blocks)
    all_val_blocks.extend(v_blocks)

print("\nStep 3: Calculating Safe Boundaries and Merging...")
# Build final arrays and their safe index maps
final_train, train_indices = build_dataset_and_indices(all_train_blocks, cfg.data.horizon, cfg.data.stride)
final_val, val_indices = build_dataset_and_indices(all_val_blocks, cfg.data.horizon, cfg.data.stride)

print("Saving to disk...")
np.save(os.path.join(cfg.paths.train_savepath, "train_data.npy"), final_train)
np.save(os.path.join(cfg.paths.train_savepath, "train_indices.npy"), train_indices) # <-- NEW
np.save(os.path.join(cfg.paths.val_savepath, "val_data.npy"), final_val)
np.save(os.path.join(cfg.paths.val_savepath, "val_indices.npy"), val_indices) # <-- NEW

print(f"\nSuccess! Shapes saved:")
print(f"Train array: {final_train.shape} | Valid Windows: {len(train_indices)}")
print(f"Val array:   {final_val.shape} | Valid Windows: {len(val_indices)}")