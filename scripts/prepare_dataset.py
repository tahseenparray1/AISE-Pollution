import os
import numpy as np
from tqdm import tqdm
from src.utils.config import load_config

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
cfg = load_config("configs/prepare_dataset.yaml")
RAW_PATH = cfg.paths.raw_path
all_features = cfg.features.met_variables_raw + cfg.features.emission_variables_raw

# Global Stats Scanner (Fixes the min-max problem)
def compute_actual_stats(features, months):
    print("Step 1: Calculating Global Statistics...")
    stats = {}
    for feat in tqdm(features, desc="Scanning features"):
        feat_min, feat_max = float('inf'), float('-inf')
        for month in months:
            path = os.path.join(RAW_PATH, month, f"{feat}.npy")
            if os.path.exists(path):
                data = np.load(path, mmap_mode='r')
                feat_min = min(feat_min, np.min(data))
                feat_max = max(feat_max, np.max(data))
        stats[feat] = {'min': feat_min, 'max': feat_max, 'range': (feat_max - feat_min) + 1e-9}
    np.save('/kaggle/working/actual_min_max_stats.npy', stats)
    return stats

global_stats = compute_actual_stats(all_features, cfg.data.months)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def process_month(month_name):
    """ Loads, scales, and stacks ONE month. No windowing! """
    month_data = []
    
    for feat in all_features:
        path = os.path.join(RAW_PATH, month_name, f"{feat}.npy")
        arr = np.load(path).astype(np.float32)
        
        # Scaling
        arr = (arr - global_stats[feat]['min']) / global_stats[feat]['range']
        if feat in ["u10", "v10"]: arr = 2.0 * arr - 1.0
        if feat in cfg.features.emission_variables_raw: arr = np.clip(arr, 0, 1)
        
        month_data.append(arr)
        
    # Stack into 4D array: (Hours, 140, 124, 16)
    combined = np.stack(month_data, axis=-1)
    
    # Chronological Split (No leakage)
    split_idx = int(combined.shape[0] * (1 - cfg.data.val_frac))
    train_raw = combined[:split_idx]
    val_raw = combined[split_idx:]
    
    return train_raw, val_raw

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
os.makedirs(cfg.paths.train_savepath, exist_ok=True)
os.makedirs(cfg.paths.val_savepath, exist_ok=True)

all_train, all_val = [], []

print("\nStep 2: Processing and Stacking Data...")
for month in tqdm(cfg.data.months):
    t_m, v_m = process_month(month)
    all_train.append(t_m)
    all_val.append(v_m)

print("\nStep 3: Merging Final Sequences...")
# Combine the 4D blocks. 
# Shape will be: (Total_Train_Hours, 140, 124, 16)
final_train = np.concatenate(all_train, axis=0)
final_val = np.concatenate(all_val, axis=0)

# Save to disk
print("Saving to disk...")
np.save(os.path.join(cfg.paths.train_savepath, "train_data.npy"), final_train)
np.save(os.path.join(cfg.paths.val_savepath, "val_data.npy"), final_val)

print(f"\nSuccess! Shapes saved:")
print(f"Train array: {final_train.shape}")
print(f"Val array:   {final_val.shape}")