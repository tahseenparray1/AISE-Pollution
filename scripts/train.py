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
from torch.optim.swa_utils import AveragedModel, SWALR 

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
cfg = load_config("configs/train.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

# Hardware optimization for static grid sizes
torch.backends.cudnn.benchmark = True

S1, S2 = cfg.data.S1, cfg.data.S2

# ==========================================
# 2. STATS & LOSS UTILITIES
# ==========================================
print("Loading robust grid-wise stats...")
stats = np.load(cfg.paths.stats_path, allow_pickle=True).item()

pm_median_tensor = torch.tensor(stats['cpm25']['median'], dtype=torch.float32).to(device).view(1, S1, S2, 1)
pm_iqr_tensor = torch.tensor(stats['cpm25']['iqr'], dtype=torch.float32).to(device).view(1, S1, S2, 1)

def to_physical(x_norm):
    """Reverse robust normalization then reverse log1p to get raw µg/m³."""
    log_val = (x_norm * pm_iqr_tensor) + pm_median_tensor
    return torch.expm1(log_val)

def sobolev_loss(pred_log, targ_log):
    """
    Sobolev H1 gradient loss: penalizes blurry predictions by matching spatial derivatives.
    Assumes inputs are (B, H, W, T). H is dim 1, W is dim 2.
    """
    # Spatial gradients along H (dim 1)
    dx_p = pred_log[:, 1:, :, :] - pred_log[:, :-1, :, :]
    dx_t = targ_log[:, 1:, :, :] - targ_log[:, :-1, :, :]
    # Spatial gradients along W (dim 2)
    dy_p = pred_log[:, :, 1:, :] - pred_log[:, :, :-1, :]
    dy_t = targ_log[:, :, 1:, :] - targ_log[:, :, :-1, :]
    return F.mse_loss(dx_p, dx_t) + F.mse_loss(dy_p, dy_t)

# ==========================================
# 3. DATA LOADER (3D Spatiotemporal)
# ==========================================
class FastInMemoryDataset(torch.utils.data.Dataset):
    """
    3D Spatiotemporal data loader.
    Output x shape: (C, T, H, W) where C=19, T=26
    Output y shape: (H, W, T_out) where T_out=16
    """
    def __init__(self, split, cfg):
        base_path = getattr(cfg.paths, f"savepath_{split}")
        self.split = split
        
        print(f"[{split.upper()}] Loading into RAM...")
        self.data = torch.from_numpy(np.load(os.path.join(base_path, f"{split}_data.npy")).astype(np.float32))
        self.valid_starts = np.load(os.path.join(base_path, f"{split}_indices.npy"))
        
        self.time_in = cfg.data.time_input
        self.total_time = cfg.data.total_time
        self.S1, self.S2 = cfg.data.S1, cfg.data.S2
        
        # Build index mapping dynamically from feature order
        all_features = (cfg.features.met_variables 
                       + cfg.features.emission_variables 
                       + cfg.features.derived_variables)
        
        assert 'cpm25' in all_features, "Target variable 'cpm25' must be in met_variables!"
        self.target_idx = all_features.index('cpm25')
        
        # Temporal weather feature indices (met vars except cpm25, plus derived vars)
        self.temporal_idx = [i for i, f in enumerate(all_features) 
                           if f != 'cpm25' and f not in cfg.features.emission_variables]
        # Static emission feature indices
        self.static_idx = [i for i, f in enumerate(all_features) 
                          if f in cfg.features.emission_variables]
        # Topography is the last channel
        self.topo_idx = len(all_features)

        self.n_temporal = len(self.temporal_idx)
        self.n_static = len(self.static_idx)

    def __len__(self): 
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        window = self.data[start : start + self.total_time].clone()
        # window shape: (26, H, W, V+1)
        
        # 1. PM2.5 across all 26 hours: mask hours 10-25 (future)
        # CRITICAL: Use normalized zero, not literal 0.0!
        # In normalized space, 0.0 physical PM2.5 = (log1p(0) - median) / iqr = -median/iqr
        zero_val = torch.tensor(
            (0.0 - stats['cpm25']['median']) / stats['cpm25']['iqr'],
            dtype=torch.float32
        )  # (H, W) grid of normalized zeros
        
        # CRITICAL: Must .clone() so we don't overwrite the original window (which y is extracted from)
        pm_full = window[:, ..., self.target_idx].clone()  # (26, H, W)
        pm_full[self.time_in:] = zero_val  # fill with true normalized zero
        pm_full = pm_full.unsqueeze(0)  # (1, 26, H, W)
        
        # 2. Dynamic temporal weather: 10 features across 26 hours
        # Shape: (10, 26, H, W)
        temporal = window[:, ..., self.temporal_idx]  # (26, H, W, 10)
        temporal = temporal.permute(3, 0, 1, 2)  # (10, 26, H, W)
        
        # 3. Static emissions: mean across time, broadcast to T=26
        # Shape: (7, 26, H, W)
        static_mean = window[:, ..., self.static_idx].mean(dim=0)  # (H, W, 7)
        static_mean = static_mean.permute(2, 0, 1)  # (7, H, W)
        static_3d = static_mean.unsqueeze(1).expand(-1, self.total_time, -1, -1)  # (7, 26, H, W)
        
        # 4. Topography: constant, broadcast to T=26
        # Shape: (1, 26, H, W)
        topo = window[0, ..., self.topo_idx]  # (H, W)
        topo_3d = topo.unsqueeze(0).unsqueeze(0).expand(1, self.total_time, -1, -1)  # (1, 26, H, W)
        
        # Combine: (1 + 10 + 7 + 1 = 19, 26, H, W)
        x = torch.cat((pm_full, temporal, static_3d, topo_3d), dim=0)
        
        # Target: PM2.5 for hours 10-25 → (H, W, 16) — same format as before
        y = window[self.time_in:, ..., self.target_idx].permute(1, 2, 0)  # (H, W, 16)
        return x, y

train_ds = FastInMemoryDataset("train", cfg)
val_ds = FastInMemoryDataset("val", cfg)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ==========================================
# 4. MODEL & OPTIMIZER (3D Spatiotemporal WNO)
# ==========================================
# Channel count: 1 PM2.5 + 10 temporal + 7 static + 1 topo = 19
in_channels = 1 + len(train_ds.temporal_idx) + len(train_ds.static_idx) + 1
print(f"Building 3D Spatiotemporal WNO with {in_channels} input channels...")

model = FNO2D(
    in_channels=in_channels, 
    time_out=cfg.data.time_out, 
    width=cfg.model.width, 
    modes=cfg.model.modes,
    time_input=cfg.data.time_input,
    time_steps=cfg.data.total_time
).to(device)

# ==========================================
# DEBUG: Startup Diagnostics
# ==========================================
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n{'='*60}")
print(f"3D SPATIOTEMPORAL WNO DIAGNOSTICS")
print(f"{'='*60}")
print(f"  Total Parameters:     {total_params:,}")
print(f"  Trainable Parameters: {trainable_params:,}")
print(f"  Device:               {device}")
print(f"  Input Channels:       {in_channels} (PM:1 + Temporal:{len(train_ds.temporal_idx)} + Static:{len(train_ds.static_idx)} + Topo:1)")
print(f"  Time Steps (T):       {cfg.data.total_time}")
print(f"  Output Steps:         {cfg.data.time_out}")
print(f"  Grid Size:            {S1}x{S2}")
print(f"  Batch Size:           {cfg.training.batch_size}")
print(f"  Epochs:               {cfg.training.epochs}")
print(f"  SWA Start Epoch:      {int(cfg.training.epochs * 0.75)}")
print(f"  Train Samples:        {len(train_ds)}")
print(f"  Val Samples:          {len(val_ds)}")
print(f"  PM25 Median Stats:    min={stats['cpm25']['median'].min():.4f}  max={stats['cpm25']['median'].max():.4f}")
print(f"  PM25 IQR Stats:       min={stats['cpm25']['iqr'].min():.4f}  max={stats['cpm25']['iqr'].max():.4f}")
print(f"{'='*60}\n")

# Compile model for massive speedup if PyTorch 2.0+
try:
    print("Compiling model graph with torch.compile (mode='reduce-overhead')...")
    # Use "reduce-overhead" mode to optimize the graph without
    # triggering advanced SM-heavy auto-tuners that crash older GPUs like T4.
    model = torch.compile(model, mode="reduce-overhead")
except Exception as e:
    print(f"torch.compile failed (normal on older PyTorch): {e}")

optimizer = Adam(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)

swa_model = AveragedModel(model)
swa_start = int(cfg.training.epochs * 0.75)
swa_scheduler = SWALR(optimizer, swa_lr=5e-5)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
scaler = torch.amp.GradScaler('cuda')
best_val_rmse = float('inf')
log = []

for ep in range(cfg.training.epochs):
    model.train()
    t_start = time.time()
    train_mse_acc = 0.0
    train_wmse_acc = 0.0
    train_sobolev_acc = 0.0
    batch_count = 0

    for x, y in tqdm(train_loader, desc=f"Epoch {ep}"):
        x, y = x.to(device), y.to(device)
        
        # --- DEBUG: First-batch diagnostic on epoch 0 ---
        if ep == 0 and batch_count == 0:
            print(f"\n{'='*60}")
            print(f"FIRST BATCH DIAGNOSTICS (Epoch 0, Batch 0)")
            print(f"{'='*60}")
            print(f"  x shape: {x.shape}  |  y shape: {y.shape}")
            print(f"  x range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            print(f"  y range: [{y.min().item():.4f}, {y.max().item():.4f}]")
            print(f"  x has NaN: {torch.isnan(x).any().item()}")
            y_phys_check = to_physical(y)
            print(f"  y physical range: [{y_phys_check.min().item():.1f}, {y_phys_check.max().item():.1f}] µg/m³")
            print(f"{'='*60}\n")
        
        # Apply noise (skip topo channel = last in dim 1)
        noise = torch.randn_like(x) * 0.01
        noise[:, -1, :, :, :] = 0.0  # don't noise topography (last channel)
        x = x + noise
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            out = model(x)  # out: (B, H, W, 16)
            
            # --- PHYSICALLY-WEIGHTED LATENT LOSS ---
            with torch.no_grad():
                targ_phys = to_physical(y)
            
            weights = torch.clamp(targ_phys / 30.0, min=1.0)
            raw_latent_mse = F.mse_loss(out, y, reduction='none')
            weighted_mse = (raw_latent_mse * weights).mean()
            
            # --- SOBOLEV GRADIENT LOSS ---
            sob_loss = sobolev_loss(out, y)
            
            # Combined: weighted MSE + Sobolev
            total_loss = weighted_mse + 0.1 * sob_loss

        # Diagnostics (no gradients)
        with torch.no_grad():
            pred_phys = to_physical(out)
            pred_phys_clipped = F.relu(pred_phys)
            train_mse_acc += torch.mean((pred_phys_clipped - targ_phys) ** 2).item()
            train_wmse_acc += weighted_mse.item()
            train_sobolev_acc += sob_loss.item()

        # Backpropagate
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        batch_count += 1

    # --- SWA SCHEDULER STEP ---
    if ep >= swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()

    # ==========================================
    # VALIDATION
    # ==========================================
    eval_model = swa_model if ep >= swa_start else model
    eval_model.eval()

    val_mse_acc = 0.0
    val_sharpness_pred = 0.0
    val_sharpness_targ = 0.0
    val_temporal_shift = 0.0
    val_batches = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = eval_model(x)  # (B, H, W, 16)
            
            pred_phys = to_physical(out)
            targ_phys = to_physical(y)
            pred_phys_clipped = F.relu(pred_phys)
            
            val_mse_acc += torch.mean((pred_phys_clipped - targ_phys) ** 2).item()
            
            # --- NEW TELEMETRY ---
            # 1. Plume Sharpness Ratio (gradient norm)
            pred_sharp = torch.abs(pred_phys[..., 1:] - pred_phys[..., :-1]).mean()
            targ_sharp = torch.abs(targ_phys[..., 1:] - targ_phys[..., :-1]).mean()
            val_sharpness_pred += pred_sharp.item()
            val_sharpness_targ += targ_sharp.item()
            
            # 2. Temporal Advection Shift (H1 vs H16)
            # out: (B, H, W, 16) — compare first and last predicted hours
            temporal_shift = F.mse_loss(out[..., 0], out[..., 15]).item()
            val_temporal_shift += temporal_shift
            val_batches += 1

    train_rmse = np.sqrt(train_mse_acc / len(train_loader))
    val_rmse = np.sqrt(val_mse_acc / len(val_loader))
    avg_wmse = train_wmse_acc / len(train_loader)
    avg_sobolev = train_sobolev_acc / len(train_loader)
    
    # Telemetry averages
    sharpness_ratio = val_sharpness_pred / (val_sharpness_targ + 1e-5) if val_batches > 0 else 0.0
    avg_temporal_shift = val_temporal_shift / val_batches if val_batches > 0 else 0.0
    
    # --- Gradient norm ---
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    
    current_lr = optimizer.param_groups[0]['lr']
    duration = time.time() - t_start
    
    # Output stats from last val batch
    out_mean = out.mean().item()
    out_std = out.std().item()
    pred_neg_pct = (pred_phys < 0).float().mean().item() * 100
    
    print(f"\nEpoch {ep} | Time: {duration:.1f}s | LR: {current_lr:.2e}")
    print(f"  Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")
    print(f"  Losses -> Weighted MSE: {avg_wmse:.6f} | Sobolev: {avg_sobolev:.6f}")
    print(f"  Grad Norm: {grad_norm:.4f} | Scaler Scale: {scaler.get_scale():.0f}")
    print(f"  Val Out (norm): mean={out_mean:.4f} std={out_std:.4f}")
    print(f"  Val Pred (phys): mean={pred_phys.mean().item():.1f} max={pred_phys.max().item():.1f} neg%={pred_neg_pct:.1f}%")
    print(f"  Sharpness Ratio: {sharpness_ratio:.4f} | Temporal Shift (H1 vs H16): {avg_temporal_shift:.4f}")

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        if cfg.training.save_checkpoint:
            torch.save({'model_state_dict': eval_model.state_dict()}, cfg.paths.model_save_path.replace(".pt", "_best.pt"))
        print(f"  ✅ New Best Val RMSE: {best_val_rmse:.4f}")

    epoch_log = {
        "epoch": ep,
        "train_rmse": round(train_rmse, 4),
        "val_rmse": round(val_rmse, 4),
        "weighted_mse": round(avg_wmse, 6),
        "sobolev_loss": round(avg_sobolev, 6),
        "grad_norm": round(grad_norm, 4),
        "lr": current_lr,
        "scaler_scale": scaler.get_scale(),
        "sharpness_ratio": round(sharpness_ratio, 4),
        "temporal_shift": round(avg_temporal_shift, 4),
        "val_pred_neg_pct": round(pred_neg_pct, 2),
        "duration_s": round(duration, 1)
    }
    log.append(epoch_log)
    with open(cfg.paths.save_dir, 'w') as f: json.dump(log, f, indent=2)