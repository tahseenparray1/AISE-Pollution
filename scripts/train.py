import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from src.utils.config import load_config
from models.baseline_model import FNO2D
from torch.optim.swa_utils import AveragedModel, SWALR 

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
cfg = load_config("configs/train.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True
np.random.seed(0)

# === FAST DEV RUN: Set to True for quick testing, False for full training ===
FAST_DEV_RUN = False
MAX_BATCHES = 10 if FAST_DEV_RUN else None  # None = use all batches
MAX_EPOCHS = 10 if FAST_DEV_RUN else cfg.training.epochs

S1, S2 = cfg.data.S1, cfg.data.S2

# ==========================================
# 2. STATS & LOSS UTILITIES
# ==========================================
print("Loading robust grid-wise stats...")
stats = np.load(cfg.paths.stats_path, allow_pickle=True).item()

pm_median_tensor = torch.tensor(stats['cpm25']['median'], dtype=torch.float32).to(device).view(1, S1, S2, 1)
pm_iqr_tensor = torch.tensor(stats['cpm25']['iqr'], dtype=torch.float32).to(device).view(1, S1, S2, 1)

def to_physical(x_norm):
    return (x_norm * pm_iqr_tensor) + pm_median_tensor

# ==========================================
# 3. DATA LOADER
# ==========================================
class FastInMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, split, cfg):
        base_path = getattr(cfg.paths, f"savepath_{split}")
        self.split = split
        
        print(f"[{split.upper()}] Loading into RAM...")
        self.data = torch.from_numpy(np.load(os.path.join(base_path, f"{split}_data.npy")).astype(np.float32))
        self.valid_starts = np.load(os.path.join(base_path, f"{split}_indices.npy"))
        
        self.time_in = cfg.data.time_input
        self.total_time = cfg.data.total_time
        self.S1, self.S2 = cfg.data.S1, cfg.data.S2
        
        # Build index mapping dynamically from feature order in prepare_dataset.yaml
        all_features = (cfg.features.met_variables 
                       + cfg.features.emission_variables 
                       + cfg.features.derived_variables)
        
        assert 'cpm25' in all_features, "Target variable 'cpm25' must be in met_variables!"
        self.target_idx = all_features.index('cpm25')
        
        self.temporal_idx = [i for i, f in enumerate(all_features) 
                           if f != 'cpm25' and f not in cfg.features.emission_variables]
        self.static_idx = [i for i, f in enumerate(all_features) 
                          if f in cfg.features.emission_variables]
        self.topo_idx = len(all_features)

    def __len__(self): 
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        window = self.data[start : start + self.total_time] 
        
        # 1. PM2.5 Historical Context (10 channels)
        pm_hist = window[:self.time_in, ..., self.target_idx].permute(1, 2, 0) 
        
        # 2. Dynamic Temporal Weather (10 features * 26 hours = 260 channels)
        temporal_weather = window[:, ..., self.temporal_idx]
        temporal_weather = temporal_weather.permute(1, 2, 0, 3).reshape(self.S1, self.S2, -1)
        
        # 3. Static Emissions (first frame — they are identical across all timesteps)
        static_emissions = window[0, ..., self.static_idx]
        
        # 4. Static Topography (1 channel)
        topo = window[0, ..., self.topo_idx].unsqueeze(-1)
        
        # Combine everything (10 + 260 + 7 + 1 = 278 Channels)
        x = torch.cat((pm_hist, temporal_weather, static_emissions, topo), dim=-1)
        
        # Target
        y = window[self.time_in:, ..., self.target_idx].permute(1, 2, 0)
        return x, y

train_ds = FastInMemoryDataset("train", cfg)
val_ds = FastInMemoryDataset("val", cfg)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ==========================================
# 4. MODEL & OPTIMIZER
# ==========================================
pm_channels = cfg.data.time_input
temporal_channels = 10 * cfg.data.total_time # 10 dynamic features * 26 hours
static_channels = 7 # 7 emission proxy maps
topo_channels = 1

in_channels = pm_channels + temporal_channels + static_channels + topo_channels
print(f"Building Model with {in_channels} input channels...")

model = FNO2D(
    in_channels=in_channels, 
    time_out=cfg.data.time_out, 
    width=cfg.model.width, 
    modes=cfg.model.modes,
    time_input=cfg.data.time_input
).to(device)

optimizer = Adam(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)


swa_model = AveragedModel(model)
swa_start = int(cfg.training.epochs * 0.75)
swa_scheduler = SWALR(optimizer, swa_lr=5e-4)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
# Create a linearly increasing weight tensor for the 16 hours
# E.g., Hour 1 gets a weight of 1.0, Hour 16 gets a weight of ~3.0
time_weights = torch.linspace(1.0, 3.0, steps=cfg.data.time_out).to(device)
# Reshape to broadcast correctly against (Batch, H, W, Time)
time_weights = time_weights.view(1, 1, 1, cfg.data.time_out)

scaler = torch.amp.GradScaler("cuda")

best_val_rmse = float('inf')
log = []

for ep in range(MAX_EPOCHS):
    model.train()
    t_start = time.time()
    train_mse_acc = 0.0
    num_batches = 0

    for x, y in tqdm(train_loader, desc=f"Epoch {ep}"):
        x, y = x.to(device), y.to(device)
        
        # Add noise, protecting topography (last channel)
        noise = torch.randn_like(x) * 0.01
        noise[..., -1] = 0.0
        x = x + noise
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast("cuda"):
            out = model(x)
            
            pred_phys = to_physical(out)
            targ_phys = to_physical(y)
            
            # --- TIME & MAGNITUDE WEIGHTED MSE LOSS ---
            # 1. Base Squared Error
            base_sq_error = (pred_phys - targ_phys) ** 2
            
            # 2. Magnitude Weights: Force the model to care about the 1% extreme spikes
            # For a pixel with 0 pollution, weight is 1.0
            # For a pixel with 300 pollution, weight is 1.0 + (300/100) = 4.0
            magnitude_weights = 1.0 + (targ_phys / 100.0)
            
            # 3. Combine penalties (time_weights is already defined outside your loop)
            weighted_sq_error = base_sq_error * time_weights * magnitude_weights
            
            # 4. Final Scalar Loss
            total_loss = torch.mean(weighted_sq_error)

        
        with torch.no_grad():
            pred_phys_clipped = F.relu(pred_phys.detach())
            train_mse_acc += torch.mean((pred_phys_clipped - targ_phys) ** 2).item()
            num_batches += 1
            
        scaler.scale(total_loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if MAX_BATCHES and num_batches >= MAX_BATCHES:
            break

    # SWA SCHEDULER STEP
    if ep >= swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()

    # VALIDATION
    eval_model = swa_model if ep >= swa_start else model
    eval_model.eval()

    val_mse_acc = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            out = eval_model(x) 
            
            pred_phys = to_physical(out)
            targ_phys = to_physical(y)
            
            pred_phys_clipped = F.relu(pred_phys)
            
            val_mse_acc += torch.mean((pred_phys_clipped - targ_phys) ** 2).item()

    train_rmse = np.sqrt(train_mse_acc / num_batches)
    val_rmse = np.sqrt(val_mse_acc / len(val_loader))
    
    duration = time.time() - t_start
    print(f"Epoch {ep} | Time: {duration:.1f}s | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        if cfg.training.save_checkpoint:
            torch.save({'model_state_dict': eval_model.state_dict()}, cfg.paths.model_save_path.replace(".pt", "_best.pt"))
        print(f"  -> New Best Val RMSE: {best_val_rmse:.4f}")

    log.append({"epoch": ep, "train_rmse": train_rmse, "val_rmse": val_rmse})
    with open(cfg.paths.save_dir, 'w') as f: json.dump(log, f)