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
cfg = load_config("configs/train.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

S1, S2 = cfg.data.S1, cfg.data.S2

# ==========================================
# 2. STATS & LOSS UTILITIES
# ==========================================
print("Loading robust grid-wise stats...")
stats = np.load('/kaggle/working/grid_robust_stats.npy', allow_pickle=True).item()

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
        attr_name = f"savepath_{split}"
        base_path = getattr(cfg.paths, attr_name)
        self.split = split
        
        data_path = os.path.join(base_path, f"{split}_data.npy")
        idx_path = os.path.join(base_path, f"{split}_indices.npy")
        
        print(f"[{split.upper()}] Loading into RAM: {data_path}")
        self.data = torch.from_numpy(np.load(data_path).astype(np.float32))
        
        self.valid_starts = np.load(idx_path)
        
        self.time_in = cfg.data.time_input
        self.total_time = cfg.data.total_time

    def __len__(self): 
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        
        # CRITICAL: .clone() prevents us from permanently zeroing out the RAM cache
        window = self.data[start : start + self.total_time].clone() 
        
        # --- TEST 3: CHANNEL DROPOUT (Training Only) ---
        if self.split == 'train':
            drop_prob = 0.10  # 10% chance to drop each feature independently
            
            # Indices 1 through 18 are the physical weather/emission features.
            # (0 is PM2.5, 19 is topography, 20 & 21 are diurnal sine/cosine waves)
            for f_idx in range(1, 19):
                if torch.rand(1).item() < drop_prob:
                    # Zero out this specific feature across all 26 hours for this sample
                    window[:, ..., f_idx] = 0.0
        # -----------------------------------------------
        
        pm_hist = window[:self.time_in, ..., 0].permute(1, 2, 0)
        weather = window[:, ..., 1:].permute(1, 2, 0, 3).reshape(S1, S2, -1)
        x = torch.cat((pm_hist, weather), dim=-1)
        
        y = window[self.time_in:, ..., 0].permute(1, 2, 0)
        return x, y

train_ds = FastInMemoryDataset("train", cfg)
val_ds = FastInMemoryDataset("val", cfg)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ==========================================
# 4. MODEL & OPTIMIZER
# ==========================================
weather_channels = (cfg.features.V - 1 + 2)
in_channels = cfg.data.time_input + (weather_channels * cfg.data.total_time)

model = FNO2D(
    in_channels=in_channels, 
    time_out=cfg.data.time_out, 
    width=cfg.model.width, 
    modes=cfg.model.modes
).to(device)

optimizer = Adam(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
best_val_rmse = float('inf')
log = []

for ep in range(cfg.training.epochs):
    model.train()
    t_start = time.time()
    train_mse_acc = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {ep}"):
        x, y = x.to(device), y.to(device)
        
        x = x + torch.randn_like(x) * 0.01
        
        optimizer.zero_grad(set_to_none=True)
        
        # Single Forward Pass (Direct Multi-Step)
        # Single Forward Pass (Direct Multi-Step)
        out = model(x)
        
        pred_phys = to_physical(out)
        targ_phys = to_physical(y)
        
        # --- NEW LOSS FORMULATION ---
        # 1. Direct Physical RMSE (Matches Kaggle Metric Exactly)
        mse_loss = torch.mean((pred_phys - targ_phys) ** 2)
        rmse_loss = torch.sqrt(mse_loss + 1e-8) # 1e-8 prevents NaN gradients at 0
        
        # 2. Spatial Gradient Loss (Keeps plume edges sharp)
        loss_grad = spatial_gradient_loss(pred_phys, targ_phys)
        
        # 3. Blended Total Loss
        total_loss = rmse_loss  + 0.1 * loss_grad
        
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_mse_acc += torch.mean((pred_phys.detach() - targ_phys) ** 2).item()

    scheduler.step()

    # VALIDATION
    model.eval()
    val_mse_acc = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred_phys = to_physical(out)
            targ_phys = to_physical(y)
            val_mse_acc += torch.mean((pred_phys - targ_phys) ** 2).item()

    train_rmse = np.sqrt(train_mse_acc / len(train_loader))
    val_rmse = np.sqrt(val_mse_acc / len(val_loader))
    
    duration = time.time() - t_start
    print(f"Epoch {ep} | Time: {duration:.1f}s | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        if cfg.training.save_checkpoint:
            torch.save({'model_state_dict': model.state_dict()}, cfg.paths.model_save_path.replace(".pt", "_best.pt"))
        print(f"  -> New Best Val RMSE: {best_val_rmse:.4f}")

    log.append({"epoch": ep, "train_rmse": train_rmse, "val_rmse": val_rmse})
    with open(cfg.paths.save_dir, 'w') as f: json.dump(log, f)