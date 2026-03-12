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

# ==========================================
# 3. DATA LOADER (Time-as-Channel)
# ==========================================
class FastInMemoryDataset(torch.utils.data.Dataset):
    """
    Outputs x as (C, T=26, H, W) with DYNAMIC emissions.
    The model internally slices to T=10 and flattens time into channels.
    No zero-masking needed — the model never sees hours 10+.
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
        
        # Build index mapping dynamically
        all_features = (cfg.features.met_variables 
                       + cfg.features.emission_variables 
                       + cfg.features.derived_variables)
        
        assert 'cpm25' in all_features, "Target variable 'cpm25' must be in met_variables!"
        self.target_idx = all_features.index('cpm25')
        
        self.temporal_idx = [i for i, f in enumerate(all_features) 
                           if f != 'cpm25' and f not in cfg.features.emission_variables]
        self.emission_idx = [i for i, f in enumerate(all_features) 
                            if f in cfg.features.emission_variables]
        self.topo_idx = len(all_features)

    def __len__(self): 
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        window = self.data[start : start + self.total_time]
        
        # 1. PM2.5: all 26 hours (model slices to :10 internally)
        pm = window[:, ..., self.target_idx].unsqueeze(0)  # (1, 26, H, W)
        
        # 2. Dynamic weather: 10 features × 26 hours
        temporal = window[:, ..., self.temporal_idx].permute(3, 0, 1, 2)  # (10, 26, H, W)
        
        # 3. DYNAMIC emissions: 7 features × 26 hours (preserves diurnal cycle)
        emissions = window[:, ..., self.emission_idx].permute(3, 0, 1, 2)  # (7, 26, H, W)
        
        # 4. Topography: broadcast to T=26
        topo = window[0, ..., self.topo_idx]
        topo_3d = topo.unsqueeze(0).unsqueeze(0).expand(1, self.total_time, -1, -1)
        
        # Combine: (19, 26, H, W)
        x = torch.cat((pm, temporal, emissions, topo_3d), dim=0)
        
        # Target: PM2.5 for hours 10-25 → (H, W, 16)
        y = window[self.time_in:, ..., self.target_idx].permute(1, 2, 0)
        return x, y

train_ds = FastInMemoryDataset("train", cfg)
val_ds = FastInMemoryDataset("val", cfg)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ==========================================
# 4. MODEL & OPTIMIZER (Time-as-Channel U-Net)
# ==========================================
in_channels = 1 + len(train_ds.temporal_idx) + len(train_ds.emission_idx) + 1
print(f"Building Time-as-Channel U-Net with {in_channels} features × {cfg.data.time_input} hours = {in_channels * cfg.data.time_input} flattened channels...")

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
print(f"TIME-AS-CHANNEL U-NET DIAGNOSTICS")
print(f"{'='*60}")
print(f"  Total Parameters:     {total_params:,}")
print(f"  Trainable Parameters: {trainable_params:,}")
print(f"  Device:               {device}")
print(f"  Input Features:       {in_channels} (PM:1 + Weather:{len(train_ds.temporal_idx)} + Emissions:{len(train_ds.emission_idx)} + Topo:1)")
print(f"  Flattened Channels:   {in_channels * cfg.data.time_input}")
print(f"  U-Net Base Width:     {cfg.model.width}")
print(f"  Output Steps:         {cfg.data.time_out}")
print(f"  Grid Size:            {S1}x{S2}")
print(f"  Batch Size:           {cfg.training.batch_size}")
print(f"  Epochs:               {cfg.training.epochs}")
print(f"  Train Samples:        {len(train_ds)}")
print(f"  Val Samples:          {len(val_ds)}")
print(f"{'='*60}\n")

optimizer = Adam(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))

# OneCycleLR: warm up to max_lr in first 30%, then smooth decay.
# Prevents the sharp spikes from CosineAnnealing.
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-3,
    steps_per_epoch=len(train_loader),
    epochs=cfg.training.epochs,
    pct_start=0.3
)

swa_model = AveragedModel(model)
swa_start = int(cfg.training.epochs * 0.75)
swa_scheduler = SWALR(optimizer, swa_lr=5e-5)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
scaler = torch.amp.GradScaler('cuda')
best_val_rmse = float('inf')
log = []

accumulation_steps = 4

for ep in range(cfg.training.epochs):
    model.train()
    t_start = time.time()
    train_mse_acc = 0.0
    train_loss_acc = 0.0
    batch_count = 0
    last_grad_norm = 0.0  # Store grad norm before zero_grad wipes it

    optimizer.zero_grad(set_to_none=True)

    for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {ep}")):
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
            print(f"  y physical mean:  {y_phys_check.mean().item():.1f} µg/m³")
            print(f"{'='*60}\n")
        
        # Apply noise (skip topo channel)
        noise = torch.randn_like(x) * 0.01
        noise[:, -1, :, :, :] = 0.0
        x = x + noise
        
        with torch.amp.autocast('cuda'):
            out = model(x)  # out: (B, H, W, 16)
            
            # --- PHYSICAL HUBER LOSS ---
            # Smooth L1 with beta=10: linear penalty for errors >10 µg/m³
            with torch.no_grad():
                targ_phys = to_physical(y)
            pred_phys = to_physical(out)
            huber_loss = F.smooth_l1_loss(pred_phys, targ_phys, beta=10.0)
            
            # --- SOBOLEV GRADIENT LOSS (latent space) ---
            # Maintains sharp plume boundaries
            sob_loss = sobolev_loss(out, y)
            
            total_loss = (huber_loss + 0.1 * sob_loss) / accumulation_steps

        # Diagnostics (no gradients)
        with torch.no_grad():
            pred_phys_clipped = F.relu(pred_phys)
            train_mse_acc += torch.mean((pred_phys_clipped - targ_phys) ** 2).item()
            train_loss_acc += huber_loss.item()

        # Backpropagate (accumulates gradients)
        scaler.scale(total_loss).backward()
        
        # Step optimizer every accumulation_steps batches
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # FIX: Capture grad norm BEFORE zero_grad wipes it
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            last_grad_norm = grad_norm ** 0.5
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        # OneCycleLR steps per batch (not per epoch)
        if ep < swa_start:
            scheduler.step()
            
        batch_count += 1

    # --- SWA ---
    if ep >= swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()

    # ==========================================
    # VALIDATION
    # ==========================================
    eval_model = swa_model if ep >= swa_start else model
    eval_model.eval()

    val_mse_acc = 0.0
    val_sharpness_pred = 0.0
    val_sharpness_targ = 0.0
    val_batches = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = eval_model(x)
            
            pred_phys = to_physical(out)
            targ_phys = to_physical(y)
            pred_phys_clipped = F.relu(pred_phys)
            
            val_mse_acc += torch.mean((pred_phys_clipped - targ_phys) ** 2).item()
            
            # Sharpness Ratio
            pred_sharp = torch.abs(pred_phys[:, 1:, :, :] - pred_phys[:, :-1, :, :]).mean()
            targ_sharp = torch.abs(targ_phys[:, 1:, :, :] - targ_phys[:, :-1, :, :]).mean()
            val_sharpness_pred += pred_sharp.item()
            val_sharpness_targ += targ_sharp.item()
            val_batches += 1

    train_rmse = np.sqrt(train_mse_acc / len(train_loader))
    val_rmse = np.sqrt(val_mse_acc / len(val_loader))
    avg_loss = train_loss_acc / len(train_loader)
    
    sharpness_ratio = val_sharpness_pred / (val_sharpness_targ + 1e-5) if val_batches > 0 else 0.0
    
    current_lr = optimizer.param_groups[0]['lr']
    duration = time.time() - t_start
    
    # Output stats from last val batch
    pred_neg_pct = (pred_phys < 0).float().mean().item() * 100
    
    print(f"\nEpoch {ep} | Time: {duration:.1f}s | LR: {current_lr:.2e}")
    print(f"  Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")
    print(f"  Huber Loss: {avg_loss:.4f}")
    print(f"  Grad Norm: {last_grad_norm:.4f} | Scaler Scale: {scaler.get_scale():.0f}")
    print(f"  Val Pred (phys): mean={pred_phys.mean().item():.1f} max={pred_phys.max().item():.1f} neg%={pred_neg_pct:.1f}%")
    print(f"  Sharpness Ratio: {sharpness_ratio:.4f}")

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        if cfg.training.save_checkpoint:
            torch.save({'model_state_dict': eval_model.state_dict()}, cfg.paths.model_save_path.replace(".pt", "_best.pt"))
        print(f"  ✅ New Best Val RMSE: {best_val_rmse:.4f}")

    epoch_log = {
        "epoch": ep,
        "train_rmse": round(train_rmse, 4),
        "val_rmse": round(val_rmse, 4),
        "huber_loss": round(avg_loss, 4),
        "grad_norm": round(last_grad_norm, 4),
        "lr": current_lr,
        "scaler_scale": scaler.get_scale(),
        "sharpness_ratio": round(sharpness_ratio, 4),
        "val_pred_neg_pct": round(pred_neg_pct, 2),
        "duration_s": round(duration, 1)
    }
    log.append(epoch_log)
    with open(cfg.paths.save_dir, 'w') as f: json.dump(log, f, indent=2)