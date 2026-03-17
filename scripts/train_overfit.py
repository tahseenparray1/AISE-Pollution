"""
train_overfit.py  (fixed)

Changes:
  FIX Critical #4a: FNO2D() call missing total_time and num_temporal_features
                    -> TypeError on model instantiation.
  FIX Critical #4b: static_channels hardcoded as 7 (single mean only).
                    main train.py uses [min, mean, max] -> 21 channels.
                    Aligned here to match.
  FIX Critical #4c: topo_idx = len(all_features) indexed into the data array,
                    but topo is no longer embedded there.  Load from file instead.
"""

import os
import time
import torch
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

pm_median_tensor = torch.tensor(
    stats['cpm25']['median'], dtype=torch.float32).to(device).view(1, S1, S2, 1)
pm_iqr_tensor = torch.tensor(
    stats['cpm25']['iqr'], dtype=torch.float32).to(device).view(1, S1, S2, 1)


def to_physical(x_norm):
    return x_norm * pm_iqr_tensor + pm_median_tensor


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
        print(f"[{split.upper()}] Loading into RAM...")
        self.data         = torch.from_numpy(
            np.load(os.path.join(base_path, f"{split}_data.npy")).astype(np.float32))
        self.valid_starts = np.load(
            os.path.join(base_path, f"{split}_indices.npy"))

        self.time_in    = cfg.data.time_input
        self.total_time = cfg.data.total_time
        self.S1, self.S2 = cfg.data.S1, cfg.data.S2

        all_features = (cfg.features.met_variables
                        + cfg.features.emission_variables
                        + cfg.features.derived_variables)
        assert 'cpm25' in all_features
        self.target_idx   = all_features.index('cpm25')
        self.temporal_idx = [i for i, f in enumerate(all_features)
                             if f != 'cpm25' and f not in cfg.features.emission_variables]
        self.static_idx   = [i for i, f in enumerate(all_features)
                             if f in cfg.features.emission_variables]

        # FIX Critical #4c: load topo from separate file
        self.topo_proxy = np.load(cfg.paths.topo_path)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start  = self.valid_starts[idx]
        window = self.data[start : start + self.total_time].clone()

        pm_hist = window[:self.time_in, ..., self.target_idx].permute(1, 2, 0)

        temporal_raw = window[:, ..., self.temporal_idx]
        temporal     = temporal_raw.permute(1, 2, 0, 3).reshape(self.S1, self.S2, -1)

        # FIX Critical #4b: use [min, mean, max] to match train.py (21 ch, not 7)
        static_raw = window[:, ..., self.static_idx]
        static     = torch.cat([
            static_raw.min(dim=0)[0],
            static_raw.mean(dim=0),
            static_raw.max(dim=0)[0],
        ], dim=-1)

        topo = torch.from_numpy(self.topo_proxy).unsqueeze(-1)

        x = torch.cat([pm_hist, temporal, static, topo], dim=-1)
        y = window[self.time_in:, ..., self.target_idx].permute(1, 2, 0)
        return x, y


train_ds = FastInMemoryDataset("train", cfg)
val_ds   = FastInMemoryDataset("val",   cfg)

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=cfg.training.batch_size,
    shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = torch.utils.data.DataLoader(
    val_ds,   batch_size=cfg.training.batch_size,
    shuffle=False, num_workers=4, pin_memory=True)

print(f"Dataset: {len(train_ds)} train | {len(val_ds)} val")

# ==========================================
# 4. MODEL & OPTIMIZER
# ==========================================
pm_channels             = cfg.data.time_input
temporal_features_count = len(train_ds.temporal_idx)
temporal_channels       = temporal_features_count * cfg.data.total_time
static_channels         = len(cfg.features.emission_variables) * 3   # FIX: 21, not 7
topo_channels           = 1
in_channels             = pm_channels + temporal_channels + static_channels + topo_channels

print(f"Building model: {in_channels} input channels")

# FIX Critical #4a: pass total_time and num_temporal_features
model = FNO2D(
    in_channels=in_channels,
    time_out=cfg.data.time_out,
    width=cfg.model.width,
    time_input=cfg.data.time_input,
    total_time=cfg.data.total_time,
    num_temporal_features=temporal_features_count,
).to(device)

optimizer = Adam(model.parameters(), lr=float(cfg.training.lr), weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6)

# ==========================================
# 5. OVERFIT DIAGNOSTIC LOOP
# ==========================================
print("Isolating a single batch for overfitting test...")
x_single, y_single = next(iter(train_loader))
x_single, y_single = x_single.to(device), y_single.to(device)

for ep in range(100):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    out       = model(x_single)
    pred_phys = to_physical(out)
    targ_phys = to_physical(y_single)

    mse_loss   = F.mse_loss(pred_phys, targ_phys)
    loss_grad  = spatial_gradient_loss(pred_phys, targ_phys)
    total_loss = mse_loss + 0.1 * loss_grad

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    print(f"Epoch {ep:3d} | Loss: {total_loss.item():.4f}")

print("\nOVERFIT DIAGNOSTIC COMPLETE")
print("If loss is not approaching ~0 by epoch 100, model capacity or "
      "learning rate needs adjusting.")