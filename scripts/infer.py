import os
import torch
import numpy as np
from tqdm import tqdm
import warnings

from models.baseline_model import FNO2D
from src.utils.config import load_config

warnings.filterwarnings("ignore")

cfg_infer = load_config("configs/infer.yaml")
cfg_train = load_config("configs/train.yaml")
# --- OVERRIDE CONFIG PATHS TO USE GOOGLE DRIVE ---
cfg_infer.paths.stats_path = "/content/drive/MyDrive/Kaggle/AISE-Pollution_Data/stats/global_stats.npy"
cfg_infer.paths.input_loc = "/content/drive/MyDrive/Kaggle/AISE-Pollution_Data/test_in"
cfg_train.paths.model_save_path = "/content/drive/MyDrive/Kaggle/AISE-Pollution_Data/models/fno2d.pt"
cfg_infer.paths.output_loc = "/content/drive/MyDrive/Kaggle/AISE-Pollution_Data/output"
# -------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading grid-wise normalization stats...")
stats_path = cfg_infer.paths.stats_path
stats = np.load(stats_path, allow_pickle=True).item()

pm_median = stats['cpm25']['median'].reshape(1, cfg_infer.data.S1, cfg_infer.data.S2, 1)
pm_iqr = stats['cpm25']['iqr'].reshape(1, cfg_infer.data.S1, cfg_infer.data.S2, 1)

def denorm(x):
    return (x * pm_iqr) + pm_median

psfc_test = np.load(os.path.join(cfg_infer.paths.input_loc, "psfc.npy"))
psfc_median = np.median(psfc_test, axis=1) 
topo_proxy = (psfc_median - np.mean(psfc_median)) / (np.std(psfc_median) + 1e-5)

class TestDataLoader(torch.utils.data.Dataset):
    def __init__(self, cfg_infer, cfg_train, stats_dict, topo_proxy):
        self.time_in = cfg_train.data.time_input    
        self.total_time = cfg_train.data.total_time 
        self.S1 = cfg_train.data.S1
        self.S2 = cfg_train.data.S2
        
        self.met_variables = cfg_train.features.met_variables
        self.emi_variables = cfg_train.features.emission_variables
        
        self.stats = stats_dict
        self.topo_proxy = topo_proxy

        self.arrs = {}
        for feat in self.met_variables + self.emi_variables:
            path = os.path.join(cfg_infer.paths.input_loc, f"{feat}.npy")
            self.arrs[feat] = np.load(path, mmap_mode="r")

        self.N = self.arrs['cpm25'].shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        seq_raw = {}
        for feat in self.met_variables + self.emi_variables:
            if feat == "cpm25":
                seq_raw[feat] = np.array(self.arrs[feat][idx, :self.time_in], dtype=np.float32)
            else:
                seq_raw[feat] = np.array(self.arrs[feat][idx, :self.total_time], dtype=np.float32)
                
        ws = np.sqrt(seq_raw['u10']**2 + seq_raw['v10']**2)
        vc = np.log1p(ws * seq_raw['pblh'])
        rm = (seq_raw['rain'] > 0).astype(np.float32)
        
        seq_raw['wind_speed'] = ws
        seq_raw['vent_coef'] = vc
        seq_raw['rain_mask'] = rm
        
        emi_vars = ["PM25", "NH3", "SO2", "NOx", "NMVOC_e", "NMVOC_finn", "bio"]
        for feat in emi_vars:
            if feat in seq_raw:
                seq_raw[feat] = np.log1p(seq_raw[feat] * 1e11)
                
        skewed_features = ['rain', 'pblh']
        for feat in skewed_features:
            if feat in seq_raw:
                seq_raw[feat] = np.log1p(seq_raw[feat])

        temporal_feats = []
        temporal_list = [f for f in self.met_variables if f != 'cpm25'] + self.emi_variables + cfg_train.features.derived_variables
        
        for feat in temporal_list:
            if self.stats[feat].get('type') == 'minmax':
                f_min, f_max = self.stats[feat]['min'], self.stats[feat]['max']
                arr = (seq_raw[feat] - f_min) / (f_max - f_min)
            else:
                arr = (seq_raw[feat] - self.stats[feat]['median']) / self.stats[feat]['iqr']
            temporal_feats.append(arr)
            
        pm25_hist = (seq_raw['cpm25'] - self.stats['cpm25']['median']) / self.stats['cpm25']['iqr']
        pm25_hist = torch.from_numpy(pm25_hist).permute(1, 2, 0) 
        
        temporal_stack = np.stack(temporal_feats, axis=0)
        temporal_tensor = torch.from_numpy(temporal_stack).permute(2, 3, 1, 0).reshape(self.S1, self.S2, -1)
        topo_tensor = torch.from_numpy(self.topo_proxy[idx]).unsqueeze(-1)
        
        x = torch.cat((pm25_hist, temporal_tensor, topo_tensor), dim=-1)
        return x

test_dataset = TestDataLoader(cfg_infer, cfg_train, stats, topo_proxy)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
pm_channels = cfg_train.data.time_input
num_temporal_features = (len(cfg_train.features.met_variables) - 1) + len(cfg_train.features.derived_variables) + len(cfg_train.features.emission_variables)
temporal_channels = num_temporal_features * cfg_train.data.total_time 
topo_channels = 1
in_channels = pm_channels + temporal_channels + topo_channels

SEEDS = [0, 42, 2026]
# Accumulator array for the ensemble sum
prediction_sum = np.zeros((len(test_dataset), cfg_train.data.S1, cfg_train.data.S2, cfg_train.data.time_out), dtype=np.float32)

print("Starting memory-safe sequential ensemble inference...")

for seed in SEEDS:
    print(f"\nLoading and running Model Seed {seed}...")
    model = FNO2D(
        in_channels=in_channels,
        time_out=cfg_train.data.time_out,
        width=cfg_train.model.width,
        modes=cfg_train.model.modes,
        time_input=cfg_train.data.time_input
    ).to(device)
    
    checkpoint_path = cfg_train.paths.model_save_path.replace(".pt", f"_seed{seed}.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    clean_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items() if k != 'n_averaged'}
    model.load_state_dict(clean_state_dict)
    model.eval()

    current_idx = 0
    with torch.no_grad():
        for x in tqdm(test_loader):
            x = x.to(device, non_blocking=True)
            bs = x.size(0)
            
            out = model(x)
            out_real = denorm(out.cpu().numpy())
            
            # Clip physics bounds and add directly to accumulator
            prediction_sum[current_idx : current_idx + bs] += np.clip(out_real, 0, None)
            current_idx += bs
            
    # --- MEMORY SAFETY FIX ---
    # Wipe the model from VRAM before loading the next one
    del model
    torch.cuda.empty_cache()

# Calculate the final ensemble average
final_prediction = prediction_sum / len(SEEDS)

os.makedirs(cfg_infer.paths.output_loc, exist_ok=True)
out_file = os.path.join(cfg_infer.paths.output_loc, 'preds.npy')

np.save(out_file, final_prediction)
print(f"\nSuccess! Final ensemble predictions saved to: {out_file}")