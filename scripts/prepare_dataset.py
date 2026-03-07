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
        median = np.median(feat_data, axis=0)
        q75, q25 = np.percentile(feat_data, [75, 25], axis=0)
        iqr = q75 - q25
        
        # FIX: Prevent exploding values for sparse features (like rain). 
        iqr = np.where(iqr < 1e-3, 1.0, iqr)
        
        stats[feat] = {
            'median': median.astype(np.float32),
            'iqr': iqr.astype(np.float32)
        }
        
    # Save the new 2D spatial stats dictionary
    np.save('/kaggle/working/grid_robust_stats.npy', stats)
    return stats

# Generate our spatial stats
global_stats = compute_gridwise_robust_stats(all_features, cfg.data.months)

# ==========================================
# 3. HELPER FUNCTIONS (THE CHRONOLOGICAL FIX)
# ==========================================
def process_month(month_name):
    """ Loads, scales, and stacks ONE month using interleaved block splitting. """
    month_data = []
    
    for feat in all_features:
        path = os.path.join(RAW_PATH, month_name, f"{feat}.npy")
        arr = np.load(path).astype(np.float32) # Shape: (Hours, 140, 124)
        
        # Retrieve the (140, 124) spatial stats
        median = global_stats[feat]['median']
        iqr = global_stats[feat]['iqr']
        
        # Grid-wise Scaling
        arr = (arr - median) / iqr
        month_data.append(arr)
        
    # Stack into 4D array: (Hours, 140, 124, 16_features)
    combined = np.stack(month_data, axis=-1)
    
    # --- NEW: Interleaved Block Splitting ---
    # We split the month into repeating 15-day cycles:
    # 12 days for training (80%), 3 days for validation (20%)
    train_blocks = []
    val_blocks = []
    
    train_chunk_size = 12 * 24  # 12 days in hours
    val_chunk_size = 3 * 24     # 3 days in hours
    cycle_size = train_chunk_size + val_chunk_size
    
    total_hours = combined.shape[0]
    
    for start_idx in range(0, total_hours, cycle_size):
        end_train = min(start_idx + train_chunk_size, total_hours)
        end_val = min(start_idx + cycle_size, total_hours)
        
        # Append the 12-day train chunk
        train_blocks.append(combined[start_idx:end_train])
        
        # Append the 3-day val chunk (if there's data left in the month)
        if end_val > end_train:
            val_blocks.append(combined[end_train:end_val])
    
    # Concatenate the blocks back together
    train_raw = np.concatenate(train_blocks, axis=0)
    val_raw = np.concatenate(val_blocks, axis=0) if val_blocks else np.empty((0, *combined.shape[1:]))
    
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