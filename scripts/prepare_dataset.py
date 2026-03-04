import os
import numpy as np
import gc  # Garbage Collector for RAM management
from tqdm import tqdm
from scipy import io
from src.utils.config import load_config

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
cfg = load_config("configs/prepare_dataset.yaml")
RAW_PATH = cfg.paths.raw_path
all_features = cfg.features.met_variables_raw + cfg.features.emission_variables_raw

# We calculate stats locally to solve the "Broken .mat file" issue
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
    return stats

global_stats = compute_actual_stats(all_features, cfg.data.months)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_sliding_window_view(data, window_size, stride):
    """ Creates virtual windows to save RAM """
    n_windows = (data.shape[0] - window_size) // stride + 1
    s0, s1, s2, s3 = data.strides
    new_shape = (n_windows, window_size, data.shape[1], data.shape[2], data.shape[3])
    new_strides = (stride * s0, s0, s1, s2, s3)
    return np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=new_strides)

def process_month(month_name):
    """ Loads, scales, and windows one month at a time """
    month_data = []
    for feat in all_features:
        path = os.path.join(RAW_PATH, month_name, f"{feat}.npy")
        arr = np.load(path).astype(np.float32)
        
        # Scaling
        arr = (arr - global_stats[feat]['min']) / global_stats[feat]['range']
        if feat in ["u10", "v10"]: arr = 2.0 * arr - 1.0
        if feat in cfg.features.emission_variables_raw: arr = np.clip(arr, 0, 1)
        
        month_data.append(arr)
        
    combined = np.stack(month_data, axis=-1)
    
    # Split BEFORE windowing to prevent leakage
    split_idx = int(combined.shape[0] * (1 - cfg.data.val_frac))
    train_raw = combined[:split_idx]
    val_raw = combined[split_idx:]
    
    # Create windows and SOLIDIFY into a new array
    train_windows = get_sliding_window_view(train_raw, cfg.data.horizon, cfg.data.stride).copy()
    val_windows = get_sliding_window_view(val_raw, cfg.data.horizon, cfg.data.stride).copy()
    
    return train_windows, val_windows

# ==========================================
# 3. MAIN EXECUTION (The Kaggle-Proof Fix)
# ==========================================

# 1. Create output directories
os.makedirs(cfg.paths.train_savepath, exist_ok=True)
os.makedirs(cfg.paths.val_savepath, exist_ok=True)

# 2. Process and save month-by-month to DISK (to keep RAM free)
print("\nStep 2: Windowing and Saving to temporary storage...")
temp_files_train = []
temp_files_val = []

for month in tqdm(cfg.data.months, desc="Processing months"):
    t_m, v_m = process_month(month)
    
    # Define temporary file names
    t_path = f"temp_train_{month}.npy"
    v_path = f"temp_val_{month}.npy"
    
    # Save this month to the SSD immediately
    np.save(t_path, t_m)
    np.save(v_path, v_m)
    
    temp_files_train.append(t_path)
    temp_files_val.append(v_path)
    
    # CRITICAL: Clear these huge arrays from RAM now that they are on disk
    del t_m, v_m
    gc.collect() 

# 3. Combine everything for the final training set
print("\nStep 3: Merging and Shuffling Final Dataset...")

def merge_and_finalize(file_list, save_path, shuffle=False):
    # Load all the month-sized chunks back from the SSD
    all_chunks = [np.load(f) for f in file_list]
    final_data = np.concatenate(all_chunks, axis=0)
    
    if shuffle:
        np.random.seed(cfg.data.seed)
        final_data = final_data[np.random.permutation(len(final_data))]
    
    # Save the giant unified file
    np.save(save_path, final_data)
    
    # Delete temporary files to clean up disk space
    for f in file_list: os.remove(f)
    
    del final_data, all_chunks
    gc.collect()

# Finalize Train and Val
merge_and_finalize(temp_files_train, os.path.join(cfg.paths.train_savepath, "train_data.npy"), shuffle=True)
merge_and_finalize(temp_files_val, os.path.join(cfg.paths.val_savepath, "val_data.npy"), shuffle=False)

print("\nSuccess: Dataset saved without crashing!")