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
        # Feature order: met_variables_raw + emission_variables_raw + derived_features + [topo]
        all_features = (cfg.features.met_variables 
                       + cfg.features.emission_variables 
                       + cfg.features.derived_variables)
        
        # Target variable index (robust against YAML reordering)
        assert 'cpm25' in all_features, "Target variable 'cpm25' must be in met_variables!"
        self.target_idx = all_features.index('cpm25')
        
        # Temporal weather feature indices (met vars except cpm25, plus derived vars)
        self.temporal_idx = [i for i, f in enumerate(all_features) 
                           if f != 'cpm25' and f not in cfg.features.emission_variables]
        # Static emission feature indices
        self.static_idx = [i for i, f in enumerate(all_features) 
                          if f in cfg.features.emission_variables]
        # Topography is the last channel appended in prepare_dataset.py
        self.topo_idx = len(all_features)

    def __len__(self): 
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        window = self.data[start : start + self.total_time].clone() 
        
        # 1. PM2.5 Historical Context (10 channels)
        pm_hist = window[:self.time_in, ..., self.target_idx].permute(1, 2, 0) 
        
        # 2. Dynamic Temporal Weather (10 features * 26 hours = 260 channels)
        temporal_weather = window[:, ..., self.temporal_idx]
        temporal_weather = temporal_weather.permute(1, 2, 0, 3).reshape(self.S1, self.S2, -1)
        
        # 3. Static Emissions (Mean across the 26 hours to collapse to 7 spatial channels)
        static_emissions = window[:, ..., self.static_idx].mean(dim=0)
        
        # 4. Static Topography (1 channel)
        topo = window[0, ..., self.topo_idx].unsqueeze(-1)
        
        # Combine everything (10 + 260 + 7 + 1 = 278 Channels)
        x = torch.cat((pm_hist, temporal_weather, static_emissions, topo), dim=-1)
        
        # Target (uses same config-driven index as input)
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
temporal_channels = 10 * cfg.data.total_time # 10 dynamic features
static_channels = 7 # 7 emission proxy maps
topo_channels = 1

in_channels = pm_channels + temporal_channels + static_channels + topo_channels
print(f"Building Model with optimized {in_channels} input channels...")

model = FNO2D(
    in_channels=in_channels, 
    time_out=cfg.data.time_out, 
    width=cfg.model.width, 
    modes=cfg.model.modes,
    time_input=cfg.data.time_input
).to(device)

# ==========================================
# DEBUG: Startup Diagnostics
# ==========================================
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n{'='*60}")
print(f"MODEL DIAGNOSTICS")
print(f"{'='*60}")
print(f"  Total Parameters:     {total_params:,}")
print(f"  Trainable Parameters: {trainable_params:,}")
print(f"  Device:               {device}")
print(f"  Input Channels:       {in_channels} (PM:{pm_channels} + Temporal:{temporal_channels} + Static:{static_channels} + Topo:{topo_channels})")
print(f"  Output Steps:         {cfg.data.time_out}")
print(f"  Grid Size:            {S1}x{S2}")
print(f"  Batch Size:           {cfg.training.batch_size}")
print(f"  Epochs:               {cfg.training.epochs}")
print(f"  SWA Start Epoch:      {int(cfg.training.epochs * 0.75)}")
print(f"  Train Samples:        {len(train_ds)}")
print(f"  Val Samples:          {len(val_ds)}")
print(f"  PM25 Median Stats:    min={stats['cpm25']['median'].min():.4f}  max={stats['cpm25']['median'].max():.4f}  mean={stats['cpm25']['median'].mean():.4f}")
print(f"  PM25 IQR Stats:       min={stats['cpm25']['iqr'].min():.4f}  max={stats['cpm25']['iqr'].max():.4f}  mean={stats['cpm25']['iqr'].mean():.4f}")
print(f"{'='*60}\n")

optimizer = Adam(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)


swa_model = AveragedModel(model)
swa_start = int(cfg.training.epochs * 0.75)
swa_scheduler = SWALR(optimizer, swa_lr=5e-4)

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
    train_latent_loss_acc = 0.0
    batch_count = 0

    for x, y in tqdm(train_loader, desc=f"Epoch {ep}"):
        x, y = x.to(device), y.to(device)
        
        # --- DEBUG: First-batch deep diagnostic on epoch 0 ---
        if ep == 0 and batch_count == 0:
            print(f"\n{'='*60}")
            print(f"FIRST BATCH DIAGNOSTICS (Epoch 0, Batch 0)")
            print(f"{'='*60}")
            print(f"  x shape: {x.shape}  |  y shape: {y.shape}")
            print(f"  x range: [{x.min().item():.4f}, {x.max().item():.4f}]  mean={x.mean().item():.4f}  std={x.std().item():.4f}")
            print(f"  y range: [{y.min().item():.4f}, {y.max().item():.4f}]  mean={y.mean().item():.4f}  std={y.std().item():.4f}")
            print(f"  y has NaN: {torch.isnan(y).any().item()}  |  x has NaN: {torch.isnan(x).any().item()}")
            y_phys_check = to_physical(y)
            print(f"  y physical range: [{y_phys_check.min().item():.1f}, {y_phys_check.max().item():.1f}] µg/m³")
            print(f"  y physical mean:  {y_phys_check.mean().item():.1f} µg/m³")
            print(f"  y physical % negative: {(y_phys_check < 0).float().mean().item()*100:.2f}%")
            print(f"{'='*60}\n")
        
        # Apply noise only to continuous features, protecting topography
        noise = torch.randn_like(x) * 0.01
        noise[..., -1] = 0.0
        x = x + noise
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            out = model(x)
            
            # --- THE ULTIMATE LOSS FIX ---
            # Calculate pure MSE loss strictly in the log-normalized space.
            # The network never sees the massive physical numbers during backprop!
            total_loss = F.mse_loss(out, y)
            
        # Detach and convert to physical space ONLY for diagnostic printouts
        with torch.no_grad():
            pred_phys = to_physical(out)
            targ_phys = to_physical(y)
            
            pred_phys_clipped = F.relu(pred_phys)
            train_mse_acc += torch.mean((pred_phys_clipped - targ_phys) ** 2).item()
            train_latent_loss_acc += total_loss.item()

        # Backpropagate the highly stable latent loss
        scaler.scale(total_loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        batch_count += 1

    # --- NEW: SWA SCHEDULER STEP ---
    if ep >= swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()

    # VALIDATION
    # Use the SWA model if we have crossed the threshold
    eval_model = swa_model if ep >= swa_start else model
    eval_model.eval()

    val_mse_acc = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            # ✅ FIXED BUG: This must be eval_model, otherwise SWA does nothing!
            out = eval_model(x) 
            
            pred_phys = to_physical(out)
            targ_phys = to_physical(y)
            
            # Apply ReLU to mimic the physical world constraint for our final score
            pred_phys_clipped = F.relu(pred_phys)
            
            val_mse_acc += torch.mean((pred_phys_clipped - targ_phys) ** 2).item()

    train_rmse = np.sqrt(train_mse_acc / len(train_loader))
    val_rmse = np.sqrt(val_mse_acc / len(val_loader))
    avg_latent_loss = train_latent_loss_acc / len(train_loader)
    
    # --- DEBUG: Compute gradient norm across all parameters ---
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    
    current_lr = optimizer.param_groups[0]['lr']
    duration = time.time() - t_start
    
    # --- DEBUG: Output distribution stats from last validation batch ---
    out_mean = out.mean().item()
    out_std = out.std().item()
    out_min = out.min().item()
    out_max = out.max().item()
    pred_neg_pct = (pred_phys < 0).float().mean().item() * 100
    
    print(f"\nEpoch {ep} | Time: {duration:.1f}s | LR: {current_lr:.2e}")
    print(f"  Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | Latent MSE: {avg_latent_loss:.6f}")
    print(f"  Grad Norm: {grad_norm:.4f} | Scaler Scale: {scaler.get_scale():.0f}")
    print(f"  Val Out (normalized): mean={out_mean:.4f} std={out_std:.4f} range=[{out_min:.4f}, {out_max:.4f}]")
    print(f"  Val Pred (physical):  mean={pred_phys.mean().item():.1f} max={pred_phys.max().item():.1f} neg%={pred_neg_pct:.1f}%")

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        if cfg.training.save_checkpoint:
            torch.save({'model_state_dict': eval_model.state_dict()}, cfg.paths.model_save_path.replace(".pt", "_best.pt"))
        print(f"  ✅ New Best Val RMSE: {best_val_rmse:.4f}")

    epoch_log = {
        "epoch": ep,
        "train_rmse": round(train_rmse, 4),
        "val_rmse": round(val_rmse, 4),
        "latent_mse": round(avg_latent_loss, 6),
        "grad_norm": round(grad_norm, 4),
        "lr": current_lr,
        "scaler_scale": scaler.get_scale(),
        "val_out_mean": round(out_mean, 4),
        "val_out_std": round(out_std, 4),
        "val_pred_neg_pct": round(pred_neg_pct, 2),
        "duration_s": round(duration, 1)
    }
    log.append(epoch_log)
    with open(cfg.paths.save_dir, 'w') as f: json.dump(log, f, indent=2)