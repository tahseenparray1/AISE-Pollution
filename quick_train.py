"""
Optimized training script for CPU.
Key changes from baseline:
1. Force CPU (RTX 5050 sm_120 not supported by current PyTorch)
2. Reduced model width (64 instead of 256) -> ~600K params instead of ~10M
3. Drop dead emission channels (all zeros after normalization)
4. Use only meaningful input channels
5. Smaller batch, fewer channels = much faster per epoch
"""
import os, sys, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
from src.utils.adam import Adam
from src.utils.config import load_config
from models.baseline_model import FNO2D

cfg = load_config("configs/train.yaml")

# FORCE CPU
device = torch.device("cpu")
print(f"Device: {device}")
torch.manual_seed(0); np.random.seed(0)
S1, S2 = cfg.data.S1, cfg.data.S2

# Load stats
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

class SmartDataset(torch.utils.data.Dataset):
    """Dataset that drops dead emission channels and reduces input dim."""
    def __init__(self, split, cfg):
        base_path = getattr(cfg.paths, f"savepath_{split}")
        self.split = split
        print(f"[{split.upper()}] Loading into RAM...")
        self.data = torch.from_numpy(np.load(os.path.join(base_path, f"{split}_data.npy")).astype(np.float32))
        self.valid_starts = np.load(os.path.join(base_path, f"{split}_indices.npy"))
        self.time_in = cfg.data.time_input      # 10
        self.total_time = cfg.data.total_time    # 26
        self.S1, self.S2 = cfg.data.S1, cfg.data.S2

        # Feature indices in data: 0-7: met, 8-14: emi (dead!), 15-17: derived, 18: topo
        # We SKIP emission channels 8-14 (all zeros)
        # We also skip q2 (ch 1, near-zero values)
        all_features = cfg.features.met_variables + cfg.features.emission_variables + cfg.features.derived_variables
        self.target_idx = 0  # cpm25

        # Active temporal features (drop cpm25, drop q2(idx 1), drop emissions)
        # Met (non-target, non-q2): t2(2), u10(3), v10(4), swdown(5), pblh(6), rain(7)
        # Derived: wind_speed(15), vent_coef(16), rain_mask(17)
        self.temporal_idx = [2, 3, 4, 5, 6, 7, 15, 16, 17]  # 9 features instead of 10
        self.topo_idx = 18

    def __len__(self): return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        window = self.data[start : start + self.total_time].clone()

        # 1. PM2.5 History (10 channels)
        pm_hist = window[:self.time_in, ..., self.target_idx].permute(1, 2, 0)

        # 2. Temporal weather: 9 features * 26 hours = 234 channels
        temporal_weather = window[:, ..., self.temporal_idx]
        temporal_weather = temporal_weather.permute(1, 2, 0, 3).reshape(self.S1, self.S2, -1)

        # 3. Static Topography (1 channel) - skip dead emissions
        topo = window[0, ..., self.topo_idx].unsqueeze(-1)

        # Combine: 10 + 234 + 1 = 245 channels (was 278)
        x = torch.cat((pm_hist, temporal_weather, topo), dim=-1)

        # Target
        y = window[self.time_in:, ..., self.target_idx].permute(1, 2, 0)
        return x, y

train_ds = SmartDataset("train", cfg)
val_ds = SmartDataset("val", cfg)

# Verify shapes
x_sample, y_sample = train_ds[0]
print(f"Input shape : {x_sample.shape}")  # (140, 124, 245)
print(f"Target shape: {y_sample.shape}")  # (140, 124, 16)

IN_CHANNELS = x_sample.shape[-1]
WIDTH = 64  # Reduced from 256

BS = 4
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=0)

print(f"Input channels: {IN_CHANNELS} (was 278), Model width: {WIDTH} (was 256)")
print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

model = FNO2D(in_channels=IN_CHANNELS, time_out=cfg.data.time_out, width=WIDTH,
              modes=cfg.model.modes, time_input=cfg.data.time_input).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,} (was 9,840,528)")

EPOCHS = 15
LR = 3e-3
optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

log = []
best_val_rmse = float('inf')

for ep in range(EPOCHS):
    model.train()
    t_start = time.time()
    train_mse_acc = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {ep:2d}"):
        x, y = x.to(device), y.to(device)
        
        # Light noise augmentation (skip topo = last channel)
        noise = torch.randn_like(x) * 0.01
        noise[..., -1] = 0.0
        x = x + noise

        optimizer.zero_grad(set_to_none=True)
        out = model(x)

        pred_phys = to_physical(out)
        targ_phys = to_physical(y)

        huber_loss = F.huber_loss(pred_phys, targ_phys, delta=10.0)
        loss_grad = spatial_gradient_loss(pred_phys, targ_phys)
        total_loss = huber_loss + 0.1 * loss_grad

        with torch.no_grad():
            pred_clipped = F.relu(pred_phys.detach())
            train_mse_acc += torch.mean((pred_clipped - targ_phys) ** 2).item()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    scheduler.step()

    # Validation
    model.eval()
    val_mse_acc = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred_phys = to_physical(out)
            targ_phys = to_physical(y)
            pred_clipped = F.relu(pred_phys)
            val_mse_acc += torch.mean((pred_clipped - targ_phys) ** 2).item()

    train_rmse = np.sqrt(train_mse_acc / len(train_loader))
    val_rmse = np.sqrt(val_mse_acc / len(val_loader))
    duration = time.time() - t_start
    lr_now = scheduler.get_last_lr()[0]
    print(f"Epoch {ep:2d} | {duration:.0f}s | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | LR: {lr_now:.6f}")

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save({'model_state_dict': model.state_dict()}, 'fno_baseline_best.pt')
        print(f"  -> New Best! Saved checkpoint")

    log.append({"epoch": ep, "train_rmse": float(train_rmse), "val_rmse": float(val_rmse)})
    with open('log.json', 'w') as f: json.dump(log, f)

print(f"\n{'='*60}")
print(f"TRAINING COMPLETE")
print(f"Best Val RMSE: {best_val_rmse:.4f}")
print(f"{'='*60}")
