import os
import numpy as np
from tqdm import tqdm
from scipy import io
from src.utils.config import load_config

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# load_config converts the YAML file into a Python object (cfg)
# so we can access paths like cfg.paths.raw_path instead of using strings.
cfg = load_config("configs/prepare_dataset.yaml")
RAW_PATH = cfg.paths.raw_path

# Create a master list of all columns we want to include in our 3D "image"
all_features = cfg.features.met_variables_raw + cfg.features.emission_variables_raw

# ==========================================
# 2. GLOBAL STATS SCANNER
# ==========================================

def compute_actual_stats(features, months):
    """
    Calculates the global Minimum and Maximum for every feature across 
    all training months. We need these to scale data between 0 and 1.
    """
    print("Step 1/3: Calculating Global Statistics from Training Months...")
    stats = {}
    
    for feat in tqdm(features, desc="Scanning features"):
        # We start with infinity so any real number will be smaller/larger
        feat_min = float('inf')
        feat_max = float('-inf')
        
        for month in months:
            path = os.path.join(RAW_PATH, month, f"{feat}.npy")
            if os.path.exists(path):
                # 'mmap_mode' is a Kaggle pro-tip: It points to the file on disk
                # without actually loading it into RAM. np.min then reads it 
                # in small chunks. This prevents the notebook from crashing.
                data = np.load(path, mmap_mode='r')
                feat_min = min(feat_min, np.min(data)) #if feat
                feat_max = max(feat_max, np.max(data))
        
        # 'range' is the denominator for our normalization formula.
        # We add 1e-9 (a tiny number) to prevent "Division by Zero" errors.
        stats[feat] = {
            'min': feat_min, 
            'max': feat_max, 
            'range': (feat_max - feat_min) + 1e-9
        }
    
    # Save these stats to a file. We will need them again during 2017 inference!
    np.save('actual_min_max_stats.npy', stats)
    return stats

# Run the scanner to build our "ruler" (global_stats)
global_stats = compute_actual_stats(all_features, cfg.data.months)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def get_sliding_window_view(data, window_size, stride):
    """
    Creates overlapping windows from a 4D array without duplicating data in RAM.
    
    Parameters:
        data: Your array of shape (Total_Time, Lat, Lon, Features)
        window_size: How long each 'snippet' is (e.g., 26 hours)
        stride: How many steps to skip between windows (e.g., 1 hour)
    """
    
    # 1. Calculate the number of windows that can actually fit.
    # If you have 100 hours and need 26-hour windows, you can't start a 
    # window at hour 90 because it would run out of data.
    n_windows = (data.shape[0] - window_size) // stride + 1
    
    # 2. Understand 'Strides'
    # In computer memory, an array is just one long line of numbers. 
    # 'Strides' are the "skip rules" that tell NumPy:
    # - How many bytes to jump to get to the next HOUR (s0)
    # - How many bytes to jump to get to the next LATITUDE (s1)
    # - How many bytes to jump to get to the next LONGITUDE (s2)
    # - How many bytes to jump to get to the next FEATURE (s3)
    s0, s1, s2, s3 = data.strides
    
    # 3. Define the New Shape
    # We want to transform (Time, Lat, Lon, Feat) 
    # INTO (Samples, Window_Time, Lat, Lon, Feat)
    new_shape = (n_windows, window_size, data.shape[1], data.shape[2], data.shape[3])
    
    # 4. Define the New Skip Rules (The Magic)
    # - To get to the next SAMPLE: Jump 'stride' amount of hours (stride * s0)
    # - To get to the next HOUR within a window: Jump 1 hour (s0)
    # - The spatial and feature jumps (s1, s2, s3) stay exactly the same.
    new_strides = (stride * s0, s0, s1, s2, s3)
    
    # 5. Create the View
    # as_strided creates a "ghost" array. It looks like a 5D array, 
    # but it is actually just pointing back to the original memory locations.
    return np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=new_strides)

def process_month(month_name):
    """
    The 'Kitchen' function: takes a month, cleans it, splits it, and windows it.
    """
    month_data = []
    
    for feat in all_features:
        path = os.path.join(RAW_PATH, month_name, f"{feat}.npy")
        # .astype(np.float32) ensures we use 32-bit floats (standard for AI)
        arr = np.load(path).astype(np.float32)
        
        # --- Normalization (Scaling) ---
        f_min = global_stats[feat]['min']
        f_den = global_stats[feat]['range']
        # Apply the formula: (x - min) / (max - min)
        arr = (arr - f_min) / f_den
        
        # Specific scaling for wind vectors to be between [-1, 1]
        if feat in ["u10", "v10"]:
            arr = 2.0 * arr - 1.0
        
        # Emissions should never be negative or over 1.0
        if feat in cfg.features.emission_variables_raw:
            arr = np.clip(arr, 0, 1)
            
        month_data.append(arr)
        
    # np.stack(..., axis=-1) glues our 16 features together like layers of an image.
    # New shape: (Time, Lat, Lon, 16 features)
    combined = np.stack(month_data, axis=-1)
    
    # --- CHRONOLOGICAL SPLIT (Prevent Data Leakage) ---
    # We take the first 80% of the month for Training, and the last 20% for Validation.
    split_idx = int(combined.shape[0] * (1 - cfg.data.val_frac))
    train_raw = combined[:split_idx]
    val_raw = combined[split_idx:]
    
    # --- WINDOWING ---
    # We turn the long month into small 26-hour overlapping sequences.
    # .copy() is used here to "solidify" the data in memory after windowing.
    train_windows = get_sliding_window_view(train_raw, cfg.data.horizon, cfg.data.stride).copy()
    val_windows = get_sliding_window_view(val_raw, cfg.data.horizon, cfg.data.stride).copy()
    
    return train_windows, val_windows

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

# Create directories for the output if they don't exist
os.makedirs(cfg.paths.train_savepath, exist_ok=True)
os.makedirs(cfg.paths.val_savepath, exist_ok=True)

all_train = []
all_val = []

print("\nStep 2/3: Windowing and Splitting Data...")
for month in tqdm(cfg.data.months, desc="Processing months"):
    t_m, v_m = process_month(month)
    all_train.append(t_m)
    all_val.append(v_m)

# np.concatenate combines April, July, Oct, and Dec into one giant array.
final_train = np.concatenate(all_train, axis=0)
final_val = np.concatenate(all_val, axis=0)

# We shuffle the TRAINING data so the model doesn't learn based on the 
# order of the months. We do NOT shuffle the validation data (it stays sequential).
np.random.seed(cfg.data.seed)
final_train = final_train[np.random.permutation(len(final_train))]

print(f"\nStep 3/3: Saving Datasets...")
print(f"Final Train Shape: {final_train.shape}") # (Total Samples, 26(hours), 140, 124, 16(features))
print(f"Final Val Shape:   {final_val.shape}")

# Save the unified files. These are much faster to read than 16 separate files.
np.save(os.path.join(cfg.paths.train_savepath, "train_data.npy"), final_train)
np.save(os.path.join(cfg.paths.val_savepath, "val_data.npy"), final_val)

print("\nSuccess: Dataset fixed and saved.")