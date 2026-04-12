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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading grid-wise normalization stats...")
stats_path = cfg_infer.paths.stats_path
stats = np.load(stats_path, allow_pickle=True).item()

pm_median = stats['cpm25']['median'].reshape(1, cfg_infer.data.S1, cfg_infer.data.S2, 1)
pm_iqr = stats['cpm25']['iqr'].reshape(1, cfg_infer.data.S1, cfg_infer.data.S2, 1)

def denorm(x):
    return (x * pm_iqr) + pm_median

class FastInMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split="test"):
        if split == "train":
            base_path = cfg.paths.savepath_train
        elif split == "val":
            base_path = cfg.paths.savepath_val
        elif split == "test":
            base_path = cfg.paths.savepath_test
        else:
            raise ValueError(f"Unknown split: {split}")
            
        print(f"Loading {split.capitalize()} Data into RAM...")
        self.data = torch.from_numpy(np.load(os.path.join(base_path, f"{split}_data.npy")).astype(np.float32))
        self.valid_starts = np.load(os.path.join(base_path, f"{split}_indices.npy"))
        
        self.time_in = cfg.data.time_input
        self.total_time = cfg.data.total_time
        self.S1, self.S2 = cfg.data.S1, cfg.data.S2
        
        all_features = (cfg.features.met_variables + cfg.features.emission_variables + cfg.features.derived_variables)
        self.target_idx = all_features.index('cpm25')
        self.temporal_idx = [i for i, f in enumerate(all_features) if f != 'cpm25']
        self.topo_idx = len(all_features)

    def __len__(self): 
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        window = self.data[start : start + self.total_time].clone() 
        
        pm_hist = window[:self.time_in, ..., self.target_idx].permute(1, 2, 0) 
        temporal_all = window[:, ..., self.temporal_idx]
        temporal_tensor = temporal_all.permute(1, 2, 0, 3).reshape(self.S1, self.S2, -1)
        topo = window[0, ..., self.topo_idx].unsqueeze(-1)
        
        x = torch.cat((pm_hist, temporal_tensor, topo), dim=-1)
        y = window[self.time_in:, ..., self.target_idx].permute(1, 2, 0)
        return x, y

test_dataset = FastInMemoryDataset(cfg_train, split="test")
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
        for x, _ in tqdm(test_loader):
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

# RMSE Calculation
print("Calculating Test RMSE...")
all_targets = []
# Ensure pm parameters match shapes for denorm
pm_iqr_np = stats['cpm25']['iqr'].reshape(cfg_train.data.S1, cfg_train.data.S2, 1)
pm_median_np = stats['cpm25']['median'].reshape(cfg_train.data.S1, cfg_train.data.S2, 1)

for i in range(len(test_dataset)):
    _, y = test_dataset[i]
    y_phys = (y.numpy() * pm_iqr_np) + pm_median_np
    all_targets.append(y_phys)

all_targets = np.stack(all_targets, axis=0)
test_rmse = np.sqrt(np.mean((final_prediction - all_targets) ** 2))

os.makedirs(cfg_infer.paths.output_loc, exist_ok=True)
out_file = os.path.join(cfg_infer.paths.output_loc, 'preds.npy')

np.save(out_file, final_prediction)
print(f"\nSuccess! Final ensemble predictions saved to: {out_file}")
print(f"=====================================")
print(f"Final Ensemble Test RMSE: {test_rmse:.4f}")
print(f"=====================================")