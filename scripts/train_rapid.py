import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from src.utils.adam import Adam
from src.utils.config import load_config
from models.baseline_model import FNO2D

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
cfg = load_config("configs/train_rapid.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

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

def spatial_gradient_loss(pred_phys, target_phys):
    dx_p = pred_phys[:, 1:, :, :] - pred_phys[:, :-1, :, :]
    dx_t = target_phys[:, 1:, :, :] - target_phys[:, :-1, :, :]
    dy_p = pred_phys[:, :, 1:, :] - pred_phys[:, :, :-1, :]
    dy_t = target_phys[:, :, 1:, :] - target_phys[:, :, :-1, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)

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
        
        # ALL features (weather + emissions) except the target are now temporal!
        self.temporal_idx = [i for i, f in enumerate(all_features) if f != 'cpm25']
        self.topo_idx = len(all_features)

    def __len__(self): 
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        window = self.data[start : start + self.total_time].clone() 
        
        # 1. PM2.5 Historical Context (10 channels)
        pm_hist = window[:self.time_in, ..., self.target_idx].permute(1, 2, 0) 
        
        # 2. Dynamic Temporal Features (Weather AND Emissions: 17 features * 26 hours)
        temporal_all = window[:, ..., self.temporal_idx]
        temporal_tensor = temporal_all.permute(1, 2, 0, 3).reshape(self.S1, self.S2, -1)
        
        # 3. Static Topography (1 channel)
        topo = window[0, ..., self.topo_idx].unsqueeze(-1)
        
        # Combine (10 + 442 + 1 = 453 Channels)
        x = torch.cat((pm_hist, temporal_tensor, topo), dim=-1)
        y = window[self.time_in:, ..., self.target_idx].permute(1, 2, 0)
        return x, y

train_ds = FastInMemoryDataset("train", cfg)
val_ds = FastInMemoryDataset("val", cfg)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)

print(f"Rapid Dataset: {len(train_ds)} train samples, {len(val_ds)} val samples")

# ==========================================
# 4. MODEL & OPTIMIZER (No SWA)
# ==========================================
pm_channels = cfg.data.time_input
# Dynamically calculate based on the new temporal stack (17 * 26 = 442)
temporal_channels = len(train_ds.temporal_idx) * cfg.data.total_time
topo_channels = 1

in_channels = pm_channels + temporal_channels + topo_channels
print(f"Building Model with optimized {in_channels} input channels...")

model = FNO2D(
    in_channels=in_channels, 
    time_out=cfg.data.time_out, 
    width=cfg.model.width, 
    time_input=cfg.data.time_input
).to(device)

optimizer = Adam(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)

# ==========================================
# 5. TRAINING LOOP WITH STRATIFIED TRACKING
# ==========================================
best_val_extreme_rmse = float('inf')
log = []

for ep in range(cfg.training.epochs):
    model.train()
    t_start = time.time()
    train_mse_acc = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {ep}"):
        x, y = x.to(device), y.to(device)
        
        noise = torch.randn_like(x) * 0.01
        noise[..., -1] = 0.0  # Zero out noise for topography
        x = x + noise
        
        optimizer.zero_grad(set_to_none=True)
        
        out = model(x)
        
        pred_phys = to_physical(out)
        targ_phys = to_physical(y)
        
        mse_loss = F.mse_loss(pred_phys, targ_phys)
        loss_grad = spatial_gradient_loss(pred_phys, targ_phys)
        total_loss = mse_loss + 0.1 * loss_grad
        
        with torch.no_grad():
            pred_phys_clipped = F.relu(pred_phys.detach())
            train_mse_acc += torch.mean((pred_phys_clipped - targ_phys) ** 2).item()
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    scheduler.step()

    # VALIDATION WITH STRATIFIED TRACKING
    model.eval()
    val_mse_acc = 0.0
    extreme_mse_acc = 0.0
    extreme_count = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            out = model(x)
            
            pred_phys = to_physical(out)
            targ_phys = to_physical(y)
            pred_phys_clipped = F.relu(pred_phys)
            
            # Overall RMSE accumulation
            val_mse_acc += torch.mean((pred_phys_clipped - targ_phys) ** 2).item()
            
            # Extreme PM2.5 (>100) RMSE tracking
            extreme_mask = targ_phys > 100.0
            n_extreme = extreme_mask.sum().item()
            if n_extreme > 0:
                extreme_sq_errors = ((pred_phys_clipped - targ_phys) ** 2)[extreme_mask]
                extreme_mse_acc += extreme_sq_errors.sum().item()
                extreme_count += n_extreme

    train_rmse = np.sqrt(train_mse_acc / len(train_loader))
    val_rmse = np.sqrt(val_mse_acc / len(val_loader))
    
    if extreme_count > 0:
        val_extreme_rmse = np.sqrt(extreme_mse_acc / extreme_count)
    else:
        val_extreme_rmse = float('inf')
    
    duration = time.time() - t_start
    print(f"Epoch {ep} | Time: {duration:.1f}s | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | Extreme PM2.5 (>100) RMSE: {val_extreme_rmse:.4f}")

    # Save based on best EXTREME RMSE
    if val_extreme_rmse < best_val_extreme_rmse:
        best_val_extreme_rmse = val_extreme_rmse
        if cfg.training.save_checkpoint:
            torch.save({'model_state_dict': model.state_dict()}, cfg.paths.model_save_path.replace(".pt", "_best.pt"))
        print(f"  -> New Best Extreme RMSE: {best_val_extreme_rmse:.4f}")

    log.append({"epoch": ep, "train_rmse": train_rmse, "val_rmse": val_rmse, "val_extreme_rmse": val_extreme_rmse})
    with open(cfg.paths.save_dir, 'w') as f: json.dump(log, f)

print(f"\n{'='*50}")
print(f"RAPID RUN COMPLETE | Best Extreme RMSE: {best_val_extreme_rmse:.4f}")
print(f"{'='*50}")
