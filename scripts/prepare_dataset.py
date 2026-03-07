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

# ==========================================
# 2. GRID-WISE ROBUST STATS CALCULATOR
# ==========================================
def compute_gridwise_robust_stats(features, months):
    print("Step 1: Calculating Grid-Wise Robust Statistics (Median & IQR)...")
    stats = {}
    
    for feat in tqdm(features, desc="Scanning features"):
        feat_data = []
        
        # Load all available temporal data for this feature across the 4 months
        for month in months:
            path = os.path.join(RAW_PATH, month, f"{feat}.npy")
            if os.path.exists(path):
                # Shape: (Hours, 140, 124)
                data = np.load(path).astype(np.float32) 
                feat_data.append(data)
                
        if not feat_data:
            print(f"Warning: No data found for feature {feat}")
            continue
            
        # Concatenate along the time axis -> (Total_Hours, 140, 124)
        feat_data = np.concatenate(feat_data, axis=0)
        
        # Calculate Grid-wise Median and IQR along the time axis (axis=0)
        # Resulting shapes will be exactly (140, 124)
        median = np.median(feat_data, axis=0)
        q75, q25 = np.percentile(feat_data, [75, 25], axis=0)
        iqr = q75 - q25
        
        # FIX: Prevent exploding values for sparse features (like rain). 
        # If the IQR is near zero, set it to 1.0 to avoid scaling up rare events to infinity.
        iqr = np.where(iqr < 1e-3, 1.0, iqr)
        
        stats[feat] = {
            'median': median.astype(np.float32),
            'iqr': iqr.astype(np.float32)
        }
        
    # Save the new 2D spatial stats dictionary
    np.save('/content/grid_robust_stats.npy', stats)
    return stats

# Generate our spatial stats
global_stats = compute_gridwise_robust_stats(all_features, cfg.data.months)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def process_month(month_name):
    """ Loads, scales, and stacks ONE month using grid-wise IQR. """
    month_data = []
    
    for feat in all_features:
        path = os.path.join(RAW_PATH, month_name, f"{feat}.npy")
        arr = np.load(path).astype(np.float32) # Shape: (Hours, 140, 124)
        
        # Retrieve the (140, 124) spatial stats
        median = global_stats[feat]['median']
        iqr = global_stats[feat]['iqr']
        
        # Grid-wise Scaling (numpy automatically broadcasts the 2D stats across the temporal dimension)
        arr = (arr - median) / iqr
        
        month_data.append(arr)
        
    # Stack into 4D array: (Hours, 140, 124, 16_features)
    combined = np.stack(month_data, axis=-1)
    
    # Chronological Split (No leakage)
    split_idx = int(combined.shape[0] * (1 - cfg.data.val_frac))
    train_raw = combined[:split_idx]
    val_raw = combined[split_idx:]
    
    return train_raw, val_raw

# ==========================================
# 4. MAIN EXECUTION
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
# Combine the 4D blocks -> Shape: (Total_Train_Hours, 140, 124, 16)
final_train = np.concatenate(all_train, axis=0)
final_val = np.concatenate(all_val, axis=0)

# Save to disk
print("Saving to disk...")
np.save(os.path.join(cfg.paths.train_savepath, "train_data.npy"), final_train)
np.save(os.path.join(cfg.paths.val_savepath, "val_data.npy"), final_val)

print(f"\nSuccess! Shapes saved:")
print(f"Train array: {final_train.shape}")
print(f"Val array:   {final_val.shape}")