import os
import torch
import numpy as np
from tqdm import tqdm
import warnings

from models.baseline_model import FNO2D
from src.utils.config import load_config

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
cfg_infer = load_config("configs/infer.yaml")
cfg_train = load_config("configs/train.yaml")

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Grid-Wise Robust Stats 
print("Loading grid-wise normalization stats...")
stats_path = cfg_infer.paths.stats_path
stats = np.load(stats_path, allow_pickle=True).item()

pm_median = stats['cpm25']['median'].reshape(1, cfg_infer.data.S1, cfg_infer.data.S2, 1)
pm_iqr = stats['cpm25']['iqr'].reshape(1, cfg_infer.data.S1, cfg_infer.data.S2, 1)

def denorm(x):
    """Reverse robust normalization then reverse log1p to get raw µg/m³."""
    log_val = (x * pm_iqr) + pm_median
    return np.expm1(log_val)

# Compute Static Topography Proxy
psfc_test = np.load(os.path.join(cfg_infer.paths.input_loc, "psfc.npy"))
psfc_median = np.median(psfc_test, axis=1)
topo_proxy = (psfc_median - np.mean(psfc_median)) / (np.std(psfc_median) + 1e-5)

# ==========================================
# 2. DATA LOADER (Time-as-Channel)
# ==========================================
class TestDataLoader(torch.utils.data.Dataset):
    """
    Outputs x as (C=19, T=26, H, W) with DYNAMIC emissions.
    Model internally slices to T=10 and flattens time into channels.
    """
    def __init__(self, cfg_infer, cfg_train, stats_dict, topo_proxy):
        self.time_in = cfg_train.data.time_input    # 10
        self.total_time = cfg_train.data.total_time  # 26
        self.S1 = cfg_train.data.S1
        self.S2 = cfg_train.data.S2
        
        self.met_variables = cfg_train.features.met_variables
        self.emi_variables = cfg_train.features.emission_variables
        self.derived_variables = cfg_train.features.derived_variables
        
        self.stats = stats_dict
        self.topo_proxy = topo_proxy

        # Load raw files lazily
        self.arrs = {}
        for feat in self.met_variables + self.emi_variables:
            path = os.path.join(cfg_infer.paths.input_loc, f"{feat}.npy")
            self.arrs[feat] = np.load(path, mmap_mode="r")

        self.N = self.arrs['cpm25'].shape[0]
        
        self.temporal_list = [f for f in self.met_variables if f != 'cpm25'] + self.derived_variables

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # 1. Extract raw sequences (all 26 hours for everything except PM2.5)
        seq_raw = {}
        for feat in self.met_variables + self.emi_variables:
            if feat == "cpm25":
                seq_raw[feat] = np.array(self.arrs[feat][idx, :self.time_in], dtype=np.float32)
            else:
                seq_raw[feat] = np.array(self.arrs[feat][idx, :self.total_time], dtype=np.float32)
                
        # 2. Compute derived features
        ws = np.sqrt(seq_raw['u10']**2 + seq_raw['v10']**2)
        vc = np.log1p(ws * seq_raw['pblh'])
        rm = (seq_raw['rain'] > 0).astype(np.float32)
        seq_raw['wind_speed'] = ws
        seq_raw['vent_coef'] = vc
        seq_raw['rain_mask'] = rm
        
        # 3. Apply log transforms
        skewed_features = ['cpm25', 'rain', 'bio', 'NMVOC_finn', 'pblh']
        for feat in skewed_features:
            seq_raw[feat] = np.log1p(seq_raw[feat])

        # 4. Normalize all features
        normalized = {}
        for feat in self.temporal_list + self.emi_variables + ['cpm25']:
            arr = seq_raw[feat]
            normalized[feat] = (arr - self.stats[feat]['median']) / self.stats[feat]['iqr']
        
        # 5. Build (C=19, T=26, H, W) tensor
        
        # PM2.5: (1, 26, H, W) — only first 10 hours have real data,
        # rest filled with normalized zero (model ignores them via :10 slice)
        pm25_norm = normalized['cpm25']  # (10, H, W)
        zero_val = (0.0 - self.stats['cpm25']['median']) / self.stats['cpm25']['iqr']
        pm25_full = np.broadcast_to(zero_val[np.newaxis, :, :], (self.total_time, self.S1, self.S2)).copy()
        pm25_full[:self.time_in] = pm25_norm
        pm25_ch = torch.from_numpy(pm25_full).unsqueeze(0)  # (1, 26, H, W)
        
        # Temporal weather: (10, 26, H, W)
        temporal_arrays = []
        for feat in self.temporal_list:
            temporal_arrays.append(normalized[feat])
        temporal_ch = torch.from_numpy(np.stack(temporal_arrays, axis=0))  # (10, 26, H, W)
        
        # DYNAMIC emissions: (7, 26, H, W) — preserves diurnal cycle
        emission_arrays = []
        for feat in self.emi_variables:
            emission_arrays.append(normalized[feat])
        emission_ch = torch.from_numpy(np.stack(emission_arrays, axis=0))  # (7, 26, H, W)
        
        # Topography: (1, 26, H, W)
        topo = torch.from_numpy(self.topo_proxy[idx]).unsqueeze(0).unsqueeze(0)
        topo_ch = topo.expand(1, self.total_time, -1, -1)  # (1, 26, H, W)
        
        # Combine: (19, 26, H, W)
        x = torch.cat((pm25_ch, temporal_ch, emission_ch, topo_ch), dim=0)
        
        return x


test_dataset = TestDataLoader(cfg_infer, cfg_train, stats, topo_proxy)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# ==========================================
# 3. MODEL INITIALIZATION (Time-as-Channel U-Net)
# ==========================================
temporal_list = [f for f in cfg_train.features.met_variables if f != 'cpm25'] + cfg_train.features.derived_variables
in_channels = 1 + len(temporal_list) + len(cfg_train.features.emission_variables) + 1

print(f"Building Time-as-Channel 2D Model with {in_channels} features × {cfg_train.data.time_input} hours...")
model = FNO2D(
    in_channels=in_channels,
    time_out=cfg_train.data.time_out,
    width=cfg_train.model.width,
    modes=cfg_train.model.modes,
    time_input=cfg_train.data.time_input,
    time_steps=cfg_train.data.total_time
).to(device)

checkpoint_path = cfg_train.paths.model_save_path.replace(".pt", "_best.pt")
print(f"Loading best weights from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Handle SWA module wrapper if present
clean_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items() if k != 'n_averaged'}
model.load_state_dict(clean_state_dict)
model.eval()

# ==========================================
# 4. INFERENCE LOOP
# ==========================================
prediction = np.zeros((len(test_dataset), cfg_train.data.S1, cfg_train.data.S2, cfg_train.data.time_out), dtype=np.float32)

print("Starting inference...")
current_idx = 0

with torch.no_grad():
    for x in tqdm(test_loader):
        x = x.to(device, non_blocking=True)
        bs = x.size(0)
        
        out = model(x)  # (B, H, W, 16)
        out_real = denorm(out.cpu().numpy())
        
        prediction[current_idx : current_idx + bs] = out_real
        current_idx += bs

# ==========================================
# 5. EXPORT PREDICTIONS
# ==========================================
os.makedirs(cfg_infer.paths.output_loc, exist_ok=True)
out_file = os.path.join(cfg_infer.paths.output_loc, 'preds.npy')

np.save(out_file, prediction)
print(f"Success! Predictions saved to: {out_file}")
print(f"Final array shape: {prediction.shape}")