import os
import numpy as np
from tqdm import tqdm
from src.utils.config import load_config

cfg = load_config("configs/prepare_dataset.yaml")
RAW_PATH = cfg.paths.raw_path



derived_features = ['wind_speed', 'vent_coef', 'rain_mask']
all_features = cfg.features.met_variables_raw + cfg.features.emission_variables_raw + derived_features

def load_raw_or_derived(feat, month):
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
        return np.log1p(vc)
        
    elif feat == 'rain_mask':
        rain = np.load(os.path.join(RAW_PATH, month, "rain.npy")).astype(np.float32)
        return (rain > 0).astype(np.float32)
        
    else:
        arr = np.load(os.path.join(RAW_PATH, month, f"{feat}.npy")).astype(np.float32)
        emi_vars = ["PM25", "NH3", "SO2", "NOx", "NMVOC_e", "NMVOC_finn", "bio"]
        if feat in emi_vars:
            arr = arr * 1e11
            arr = np.log1p(arr)
            return arr
            
        if feat in ['rain', 'pblh']: 
            arr = np.log1p(arr)
        return arr

def compute_gridwise_robust_stats(features, months):
    print("Calculating Grid-Wise Statistics...")
    stats = {}
    for feat in tqdm(features, desc="Scanning features"):
        feat_data = [load_raw_or_derived(feat, month) for month in months]
        feat_data = np.concatenate(feat_data, axis=0)
        
        if feat in cfg.features.emission_variables_raw:
            f_min, f_max = np.min(feat_data), np.max(feat_data)
            if f_max == f_min: f_max = f_min + 1e-5
            stats[feat] = {'min': float(f_min), 'max': float(f_max), 'type': 'minmax'}
        else:
            median = np.median(feat_data, axis=0)
            q75, q25 = np.percentile(feat_data, [75, 25], axis=0)
            iqr = np.clip(q75 - q25, a_min=5.0, a_max=None)
            stats[feat] = {'median': median.astype(np.float32), 'iqr': iqr.astype(np.float32), 'type': 'robust'}
            
    np.save(cfg.paths.stats_path, stats)
    return stats

global_stats = compute_gridwise_robust_stats(all_features, cfg.data.months)

print("Generating Topography Map...")
all_psfc = np.concatenate([np.load(os.path.join(RAW_PATH, m, "psfc.npy")).astype(np.float32) for m in cfg.data.months], axis=0)
psfc_median = np.median(all_psfc, axis=0)
topo_proxy = (psfc_median - np.mean(psfc_median)) / (np.std(psfc_median) + 1e-5)

def process_month(month_name):
    month_data = []
    for feat in all_features:
        arr = load_raw_or_derived(feat, month_name)
        if global_stats[feat].get('type') == 'minmax':
            f_min = global_stats[feat]['min']
            f_max = global_stats[feat]['max']
            arr = (arr - f_min) / (f_max - f_min)
        else:
            arr = (arr - global_stats[feat]["median"]) / global_stats[feat]["iqr"]
            
        month_data.append(arr)

    total_hours = month_data[0].shape[0] 
    topo_time = np.broadcast_to(topo_proxy[None, :, :], (total_hours, 140, 124))
    month_data.append(topo_time)

    combined = np.stack(month_data, axis=-1)
    return [combined]

def build_dataset_and_indices(blocks, window_size, stride):
    if len(blocks) == 0:
        return np.array([]), np.array([], dtype=np.int32)
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
os.makedirs(cfg.paths.val_savepath, exist_ok=True)
os.makedirs(cfg.paths.test_savepath, exist_ok=True)

train_blocks, val_blocks, test_blocks = [], [], []

val_frac = cfg.data.get('val_frac', 0.2)
test_frac = cfg.data.get('test_frac', 0.1)
train_frac = 1.0 - val_frac - test_frac

print("Processing Data and splitting chronologically (Train/Val/Test)...")
for month in tqdm(cfg.data.months):
    blocks = process_month(month)
    combined = blocks[0]
    total_hours = combined.shape[0]
    
    train_end = int(train_frac * total_hours)
    val_end = int((train_frac + val_frac) * total_hours)
    
    train_blocks.append(combined[:train_end])
    val_blocks.append(combined[train_end:val_end])
    test_blocks.append(combined[val_end:])

final_train, train_indices = build_dataset_and_indices(train_blocks, cfg.data.horizon, cfg.data.stride)
final_val, val_indices = build_dataset_and_indices(val_blocks, cfg.data.horizon, cfg.data.stride)
final_test, test_indices = build_dataset_and_indices(test_blocks, cfg.data.horizon, cfg.data.stride)

np.save(os.path.join(cfg.paths.train_savepath, "train_data.npy"), final_train)
np.save(os.path.join(cfg.paths.train_savepath, "train_indices.npy"), train_indices)

np.save(os.path.join(cfg.paths.val_savepath, "val_data.npy"), final_val)
np.save(os.path.join(cfg.paths.val_savepath, "val_indices.npy"), val_indices)

np.save(os.path.join(cfg.paths.test_savepath, "test_data.npy"), final_test)
np.save(os.path.join(cfg.paths.test_savepath, "test_indices.npy"), test_indices)

print(f"Success! Train shape: {final_train.shape}, Val shape: {final_val.shape}, Test shape: {final_test.shape}")