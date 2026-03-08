import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

from models.baseline_model import FNO2D
from src.utils.config import load_config

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Load both configs to ensure test paths and model architecture match perfectly
cfg_infer = load_config("configs/infer.yaml")
cfg_train = load_config("configs/train.yaml")

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------
# Load Grid-Wise Robust Stats 
# -----------------------
print("Loading grid-wise normalization stats...")
stats_path = '/kaggle/working/grid_robust_stats.npy'
stats = np.load(stats_path, allow_pickle=True).item()

pm_median = stats['cpm25']['median'].reshape(1, cfg_infer.data.S1, cfg_infer.data.S2, 1)
pm_iqr = stats['cpm25']['iqr'].reshape(1, cfg_infer.data.S1, cfg_infer.data.S2, 1)

def denorm(x):
    """ Inverse the robust scaling """
    return (x * pm_iqr) + pm_median

# Compute Static Topography Proxy using the training stats
psfc_median = stats['psfc']['median']
topo_proxy = (psfc_median - np.mean(psfc_median)) / (np.std(psfc_median) + 1e-5)

# ==========================================
# 2. DATA LOADER
# ==========================================
class TestDataLoader(torch.utils.data.Dataset):
    def __init__(self, cfg_infer, cfg_train, stats_dict, topo_proxy):
        self.time_in = cfg_train.data.time_input    # 10
        self.total_time = cfg_train.data.total_time # 26
        self.S1 = cfg_train.data.S1
        self.S2 = cfg_train.data.S2
        
        # Pull raw feature requirements from train config to match training logic
        self.met_variables = cfg_train.features.met_variables
        self.emi_variables = cfg_train.features.emission_variables
        self.raw_features = self.met_variables + self.emi_variables
        self.derived_features = cfg_train.features.derived_variables
        self.all_features = self.raw_features + self.derived_features
        
        self.stats = stats_dict
        self.topo_proxy = topo_proxy

        # Load raw files lazily
        self.arrs = {}
        for feat in self.raw_features:
            path = os.path.join(cfg_infer.paths.input_loc, f"{feat}.npy")
            self.arrs[feat] = np.load(path, mmap_mode="r")

        self.N = self.arrs[self.raw_features[0]].shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # 1. Extract Raw Sequence Data
        seq_raw = {}
        for feat in self.raw_features:
            if feat == "cpm25":
                seq_raw[feat] = np.array(self.arrs[feat][idx, :self.time_in], dtype=np.float32)
            else:
                seq_raw[feat] = np.array(self.arrs[feat][idx, :self.total_time], dtype=np.float32)
                
        # 2. Compute Derived Features on the fly
        ws = np.sqrt(seq_raw['u10']**2 + seq_raw['v10']**2)
        vc = np.log1p(ws * seq_raw['pblh']) # Log transform applied immediately
        rm = (seq_raw['rain'] > 0).astype(np.float32)
        
        # 3. Apply Log Transforms to Skewed Raw Data
        skewed_features = ['rain', 'bio', 'NMVOC_finn']
        for feat in skewed_features:
            seq_raw[feat] = np.log1p(seq_raw[feat])

        # 4. Normalize and Separate PM2.5 from Weather
        pm25_hist = None
        weather_feats_list = []
        
        for feat in self.all_features:
            # Route to the correct array
            if feat in seq_raw: arr = seq_raw[feat]
            elif feat == 'wind_speed': arr = ws
            elif feat == 'vent_coef': arr = vc
            elif feat == 'rain_mask': arr = rm

            # Apply robust scaling
            f_median = self.stats[feat]['median']
            f_iqr = self.stats[feat]['iqr']
            arr = (arr - f_median) / f_iqr
            
            if feat == "cpm25":
                pm25_hist = torch.from_numpy(arr).permute(1, 2, 0) # (140, 124, 10)
            else:
                weather_feats_list.append(arr)

        # 5. Append Topography Map
        topo_time = np.broadcast_to(self.topo_proxy[None, :, :], (self.total_time, self.S1, self.S2))
        weather_feats_list.append(topo_time)

        # 6. Stack weather features
        weather_stack = np.stack(weather_feats_list, axis=0)
        
        # Reshape to (140, 124, Channels * Time) -> (140, 124, 19 * 26 = 494)
        weather_tensor = torch.from_numpy(weather_stack).permute(2, 3, 1, 0).reshape(self.S1, self.S2, -1)
        
        # Combine PM2.5 (10) + Weather (494) = 504 Channels
        x = torch.cat((pm25_hist, weather_tensor), dim=-1)

        return x

test_dataset = TestDataLoader(cfg_infer, cfg_train, stats, topo_proxy)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# ==========================================
# 3. MODEL INITIALIZATION
# ==========================================
weather_channels = (cfg_train.features.V - 1) # 20 raw/derived - 1 pm25
in_channels = cfg_train.data.time_input + (weather_channels * cfg_train.data.total_time)

print(f"Building WNO Model with {in_channels} input channels...")
model = FNO2D(
    in_channels=in_channels,
    time_out=cfg_train.data.time_out,
    width=cfg_train.model.width,
    modes=cfg_train.model.modes
).to(device)

checkpoint_path = cfg_train.paths.model_save_path.replace(".pt", "_best.pt")
print(f"Loading best weights from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
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
        
        out = model(x)
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