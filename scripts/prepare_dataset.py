import os
import numpy as np
from tqdm import tqdm
from src.utils.config import load_config

cfg = load_config("configs/prepare_rapid.yaml")
RAW_PATH = cfg.paths.raw_path

derived_features = ['wind_speed', 'vent_coef', 'rain_mask', 'wind_convergence']
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
        
    elif feat == 'wind_convergence':
        # Kinematic Convergence = -(du/dx + dv/dy)
        # We need to compute spatial gradients for each timestep.
        # Data shape: (T, 140, 124). Spatial dims are axis 1 (y/lat) and axis 2 (x/lon).
        # We assume uniform grid spacing, so we just use the raw differences.
        
        u = np.load(os.path.join(RAW_PATH, month, "u10.npy")).astype(np.float32)
        v = np.load(os.path.join(RAW_PATH, month, "v10.npy")).astype(np.float32)
        
        # np.gradient computes central differences: returns a list [dt, dy, dx]
        # v10 is meridional (y-direction wind) -> we want dv/dy (axis 1)
        # u10 is zonal (x-direction wind) -> we want du/dx (axis 2)
        
        # We don't care about the time gradient (axis 0), so we only ask for axes 1 and 2
        # However, it's easier to just compute it across all axes and take what we need
        du_dx = np.gradient(u, axis=2)
        dv_dy = np.gradient(v, axis=1)
        
        convergence = -(du_dx + dv_dy)
        return convergence
        
    else:
        arr = np.load(os.path.join(RAW_PATH, month, f"{feat}.npy")).astype(np.float32)
        
        # --- THE FIX: pblh added to log transform to compress diurnal variance ---
        skewed_features = ['rain', 'bio', 'NMVOC_finn', 'pblh'] 
        if feat in skewed_features:
            arr = np.log1p(arr)
            
        return arr

def compute_gridwise_robust_stats(features, months):
    print("Calculating Grid-Wise Robust Statistics...")
    stats = {}
    for feat in tqdm(features, desc="Scanning features"):
        feat_data = [load_raw_or_derived(feat, month) for month in months]
        feat_data = np.concatenate(feat_data, axis=0)
        
        median = np.median(feat_data, axis=0)
        q75, q25 = np.percentile(feat_data, [75, 25], axis=0)
        iqr = np.clip(q75 - q25, a_min=5.0, a_max=None)
        
        stats[feat] = {'median': median.astype(np.float32), 'iqr': iqr.astype(np.float32)}
        
    np.save(cfg.paths.stats_path, stats)
    return stats

global_stats = compute_gridwise_robust_stats(all_features, cfg.data.months)

# Generate Topography from PSFC (Loaded explicitly just for this)
print("Generating Topography Map...")
all_psfc = np.concatenate([np.load(os.path.join(RAW_PATH, m, "psfc.npy")).astype(np.float32) for m in cfg.data.months], axis=0)
psfc_median = np.median(all_psfc, axis=0)
topo_proxy = (psfc_median - np.mean(psfc_median)) / (np.std(psfc_median) + 1e-5)

def process_month(month_name):
    month_data = []
    for feat in all_features:
        arr = load_raw_or_derived(feat, month_name)
        arr = (arr - global_stats[feat]["median"]) / global_stats[feat]["iqr"]
        month_data.append(arr)

    total_hours = month_data[0].shape[0] 
    topo_time = np.broadcast_to(topo_proxy[None, :, :], (total_hours, 140, 124))
    month_data.append(topo_time)

    combined = np.stack(month_data, axis=-1)

    train_blocks, val_blocks = [], []
    cycle_size = (12 * 24) + (3 * 24)
    
    for start_idx in range(0, total_hours, cycle_size):
        end_train = min(start_idx + (12 * 24), total_hours)
        end_val = min(start_idx + cycle_size, total_hours)
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
os.makedirs(cfg.paths.val_savepath, exist_ok=True)

all_train_blocks, all_val_blocks = [], []
print("Processing Data...")
for month in tqdm(cfg.data.months):
    t_blocks, v_blocks = process_month(month)
    all_train_blocks.extend(t_blocks)
    all_val_blocks.extend(v_blocks)

final_train, train_indices = build_dataset_and_indices(all_train_blocks, cfg.data.horizon, cfg.data.stride)
final_val, val_indices = build_dataset_and_indices(all_val_blocks, cfg.data.horizon, cfg.data.stride)

np.save(os.path.join(cfg.paths.train_savepath, "train_data.npy"), final_train)
np.save(os.path.join(cfg.paths.train_savepath, "train_indices.npy"), train_indices)
np.save(os.path.join(cfg.paths.val_savepath, "val_data.npy"), final_val)
np.save(os.path.join(cfg.paths.val_savepath, "val_indices.npy"), val_indices)
print(f"Success! Train shape: {final_train.shape} | Val shape: {final_val.shape}")