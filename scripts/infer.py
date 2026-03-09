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
stats_path = cfg.paths.stats_path
stats = np.load(stats_path, allow_pickle=True).item()

pm_median = stats['cpm25']['median'].reshape(1, cfg_infer.data.S1, cfg_infer.data.S2, 1)
pm_iqr = stats['cpm25']['iqr'].reshape(1, cfg_infer.data.S1, cfg_infer.data.S2, 1)

def denorm(x):
    return (x * pm_iqr) + pm_median

# Compute Static Topography Proxy using training stats
# Note: Since psfc isn't in train.yaml features anymore, we load it directly from test_in just for this map
psfc_test = np.load(os.path.join(cfg_infer.paths.input_loc, "psfc.npy"))
psfc_median = np.median(psfc_test, axis=(0, 1)) # <-- FIX: Median over BOTH sample and time dimensions
topo_proxy = (psfc_median - np.mean(psfc_median)) / (np.std(psfc_median) + 1e-5)

# ==========================================
# 2. DATA LOADER (PHASE 1 ALIGNED)
# ==========================================
class TestDataLoader(torch.utils.data.Dataset):
    def __init__(self, cfg_infer, cfg_train, stats_dict, topo_proxy):
        self.time_in = cfg_train.data.time_input    # 10
        self.total_time = cfg_train.data.total_time # 26
        self.S1 = cfg_train.data.S1
        self.S2 = cfg_train.data.S2
        
        # Exact feature lists from train.yaml
        self.met_variables = cfg_train.features.met_variables
        self.emi_variables = cfg_train.features.emission_variables
        
        self.stats = stats_dict
        self.topo_proxy = topo_proxy

        # Load raw files lazily
        self.arrs = {}
        for feat in self.met_variables + self.emi_variables:
            path = os.path.join(cfg_infer.paths.input_loc, f"{feat}.npy")
            self.arrs[feat] = np.load(path, mmap_mode="r")

        self.N = self.arrs['cpm25'].shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # 1. Extract Raw Sequence Data
        seq_raw = {}
        for feat in self.met_variables + self.emi_variables:
            if feat == "cpm25":
                seq_raw[feat] = np.array(self.arrs[feat][idx, :self.time_in], dtype=np.float32)
            else:
                seq_raw[feat] = np.array(self.arrs[feat][idx, :self.total_time], dtype=np.float32)
                
        # 2. Compute Derived Features
        ws = np.sqrt(seq_raw['u10']**2 + seq_raw['v10']**2)
        vc = np.log1p(ws * seq_raw['pblh'])
        rm = (seq_raw['rain'] > 0).astype(np.float32)
        
        seq_raw['wind_speed'] = ws
        seq_raw['vent_coef'] = vc
        seq_raw['rain_mask'] = rm
        
        # 3. Apply Log Transforms
        skewed_features = ['rain', 'bio', 'NMVOC_finn', 'pblh']
        for feat in skewed_features:
            seq_raw[feat] = np.log1p(seq_raw[feat])

        # 4. Normalize and Categorize Features
        pm25_hist = None
        temporal_feats = []
        static_feats = []
        
        # Process Temporal (Weather & Derived)
        temporal_list = [f for f in self.met_variables if f != 'cpm25'] + cfg_train.features.derived_variables
        for feat in temporal_list:
            arr = (seq_raw[feat] - self.stats[feat]['median']) / self.stats[feat]['iqr']
            temporal_feats.append(arr)
            
        # Process Static (Emissions)
        for feat in self.emi_variables:
            arr = (seq_raw[feat] - self.stats[feat]['median']) / self.stats[feat]['iqr']
            static_feats.append(arr)
            
        # Get PM25
        pm25_hist = (seq_raw['cpm25'] - self.stats['cpm25']['median']) / self.stats['cpm25']['iqr']
        pm25_hist = torch.from_numpy(pm25_hist).permute(1, 2, 0) # (140, 124, 10)

        # 5. Build Tensors (Matching train.py exactly)
        # Temporal: (Channels, Time, H, W) -> (H, W, Time, Channels) -> (H, W, Time * Channels)
        temporal_stack = np.stack(temporal_feats, axis=0)
        temporal_tensor = torch.from_numpy(temporal_stack).permute(2, 3, 1, 0).reshape(self.S1, self.S2, -1)
        
        # Static: Stack -> Mean across time -> (Channels, Time, H, W) -> (Channels, H, W) -> (H, W, Channels)
        static_stack = np.stack(static_feats, axis=0)
        static_tensor = torch.from_numpy(static_stack).mean(dim=1).permute(1, 2, 0)
        
        # Topo
        topo_tensor = torch.from_numpy(self.topo_proxy).unsqueeze(-1)
        
        # Combine (10 + 260 + 7 + 1 = 278 Channels)
        x = torch.cat((pm25_hist, temporal_tensor, static_tensor, topo_tensor), dim=-1)

        return x

test_dataset = TestDataLoader(cfg_infer, cfg_train, stats, topo_proxy)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# ==========================================
# 3. MODEL INITIALIZATION
# ==========================================
pm_channels = cfg_train.data.time_input
temporal_channels = 10 * cfg_train.data.total_time 
static_channels = 7 
topo_channels = 1

in_channels = pm_channels + temporal_channels + static_channels + topo_channels

print(f"Building WNO Model with optimized {in_channels} input channels...")
model = FNO2D(
    in_channels=in_channels,
    time_out=cfg_train.data.time_out,
    width=cfg_train.model.width,
    modes=cfg_train.model.modes
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