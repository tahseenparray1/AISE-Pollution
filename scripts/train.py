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
print(f"Building Model with optimized {in_channels} input channels (Massive memory saving!)...")

model = FNO2D(
    in_channels=in_channels, 
    time_out=cfg.data.time_out, 
    width=cfg.model.width, 
    modes=cfg.model.modes,
    time_input=cfg.data.time_input
).to(device)

# Enable cuDNN autotuner for fixed grid sizes (140x124)
torch.backends.cudnn.benchmark = True

optimizer = Adam(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)


swa_model = AveragedModel(model)
swa_start = int(cfg.training.epochs * 0.75)
swa_scheduler = SWALR(optimizer, swa_lr=5e-5)  # Gentle exploration, not a shock

# ==========================================
# 5. TRAINING LOOP
# ==========================================
best_val_rmse = float('inf')
log = []

# AMP: Mixed Precision for memory-efficient PDE-RNN training
scaler = torch.amp.GradScaler('cuda')

for ep in range(cfg.training.epochs):
    model.train()
    t_start = time.time()
    train_mse_acc = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {ep}"):
        x, y = x.to(device), y.to(device)
        
        # Noise augmentation: protect static & binary channels
        noise = torch.randn_like(x) * 0.01
        noise[..., -1] = 0.0  # Topography (last channel)
        # Vectorized rain_mask protection: feature index 9 in each of 26 timesteps
        rain_mask_indices = torch.arange(26, device=x.device) * 10 + 10 + 9
        noise[..., rain_mask_indices] = 0.0
        x = x + noise
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            out = model(x)
            
            pred_phys = to_physical(out)
            targ_phys = to_physical(y)
            
            # MSE loss directly optimizes RMSE (Kaggle metric)
            # but weighted to fix the long-tail imbalance + temporal decay
            
            # 1. VALUE-WEIGHTED MSE: penalize spike errors 5x more
            raw_mse = (pred_phys - targ_phys) ** 2
            spatial_weights = torch.where(targ_phys > 80.0, 5.0, 1.0)
            
            # 2. TEMPORAL WEIGHTING: H1=1.0x → H16=2.5x (fight blurring)
            # pred_phys is (B, H, W, 16) — time at dim=3
            temp_weights = torch.linspace(1.0, 2.5, steps=16, device=x.device).view(1, 1, 1, 16)
            
            # 3. COMBINED WEIGHTED LOSS
            weighted_mse = (raw_mse * spatial_weights * temp_weights).mean()
            
            loss_grad = spatial_gradient_loss(pred_phys, targ_phys)
            total_loss = weighted_mse + 0.05 * loss_grad

        with torch.no_grad():
            pred_phys_clipped = F.relu(pred_phys.detach())
            train_mse_acc += torch.mean((pred_phys_clipped - targ_phys) ** 2).item()
        
        # AMP: scaled backward + unscale before grad clipping
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

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
            
            pred_phys_clipped = F.relu(pred_phys)
            
            val_mse_acc += torch.mean((pred_phys_clipped - targ_phys) ** 2).item()
            
            # Accumulate per-hour and spike/bg diagnostics (every 5 epochs)
            if ep % 5 == 0 or ep == cfg.training.epochs - 1:
                per_hour = torch.mean((pred_phys_clipped - targ_phys) ** 2, dim=(0, 1, 2))
                if 'hour_mse_acc' not in dir():
                    hour_mse_acc = torch.zeros(16, device=device)
                    spike_se_list = [0.0, 0.0, 0.0]  # >50, >100, >200
                    spike_n_list = [0, 0, 0]
                    bg_se, bg_n = 0.0, 0
                    early_mse, late_mse = 0.0, 0.0
                hour_mse_acc += per_hour
                
                # Multi-threshold spike breakdown
                for thresh, acc_idx in [(50.0, 0), (100.0, 1), (200.0, 2)]:
                    mask = targ_phys > thresh
                    if mask.any():
                        spike_se_list[acc_idx] += torch.sum((pred_phys_clipped[mask] - targ_phys[mask]) ** 2).item()
                        spike_n_list[acc_idx] += mask.sum().item()
                bg_mask = targ_phys <= 50.0
                if bg_mask.any():
                    bg_se += torch.sum((pred_phys_clipped[bg_mask] - targ_phys[bg_mask]) ** 2).item()
                    bg_n += bg_mask.sum().item()
                
                # Early (H1-H4) vs Late (H13-H16) temporal breakdown
                early_mse += torch.mean((pred_phys_clipped[..., :4] - targ_phys[..., :4]) ** 2).item()
                late_mse += torch.mean((pred_phys_clipped[..., 12:] - targ_phys[..., 12:]) ** 2).item()
                
                # Prediction range stats
                if 'pred_min' not in dir():
                    pred_min, pred_max = float('inf'), float('-inf')
                    extreme_correct, extreme_total = 0, 0
                pred_min = min(pred_min, pred_phys_clipped.min().item())
                pred_max = max(pred_max, pred_phys_clipped.max().item())
                # Extreme accuracy: for pixels > 200, is prediction within 50% of truth?
                ext_mask = targ_phys > 200.0
                if ext_mask.any():
                    ext_pred = pred_phys_clipped[ext_mask]
                    ext_true = targ_phys[ext_mask]
                    within_50pct = (ext_pred > ext_true * 0.5) & (ext_pred < ext_true * 1.5)
                    extreme_correct += within_50pct.sum().item()
                    extreme_total += ext_mask.sum().item()

    train_rmse = np.sqrt(train_mse_acc / len(train_loader))
    val_rmse = np.sqrt(val_mse_acc / len(val_loader))
    gap = train_rmse - val_rmse
    
    duration = time.time() - t_start
    print(f"Epoch {ep} | Time: {duration:.1f}s | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | Gap: {gap:+.2f}")
    
    # Deep diagnostic logs every 5 epochs
    if ep % 5 == 0 or ep == cfg.training.epochs - 1:
        try:
            hour_rmses = torch.sqrt(hour_mse_acc / len(val_loader)).cpu().numpy()
            print(f"  Per-Hour RMSE: " + " | ".join([f"H{h+1}:{v:.1f}" for h, v in enumerate(hour_rmses)]))
            e_rmse = np.sqrt(early_mse / len(val_loader))
            l_rmse = np.sqrt(late_mse / len(val_loader))
            print(f"  Early(H1-4): {e_rmse:.2f} | Late(H13-16): {l_rmse:.2f} | Decay: {l_rmse - e_rmse:+.2f}")
            if bg_n > 0:
                print(f"  Background RMSE (<=50):  {np.sqrt(bg_se / bg_n):.2f} ({bg_n:,} px)")
            for i, thresh in enumerate([50, 100, 200]):
                if spike_n_list[i] > 0:
                    print(f"  Spike RMSE (>{thresh}):{'  ' if thresh < 100 else ' '}{np.sqrt(spike_se_list[i] / spike_n_list[i]):.2f} ({spike_n_list[i]:,} px)")
            if extreme_total > 0:
                print(f"  Extreme Accuracy (>200, within 50%): {100*extreme_correct/extreme_total:.1f}% ({extreme_correct:,}/{extreme_total:,})")
            print(f"  Pred Range: [{pred_min:.1f}, {pred_max:.1f}] | LR: {optimizer.param_groups[0]['lr']:.2e}")
            del hour_mse_acc, spike_se_list, spike_n_list, bg_se, bg_n, early_mse, late_mse
            del pred_min, pred_max, extreme_correct, extreme_total
        except:
            pass

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        if cfg.training.save_checkpoint:
            # Save the currently evaluating model's dict (which will be SWA if active)
            torch.save({'model_state_dict': eval_model.state_dict()}, cfg.paths.model_save_path.replace(".pt", "_best.pt"))
        print(f"  -> New Best Val RMSE: {best_val_rmse:.4f}")

    log.append({"epoch": ep, "train_rmse": train_rmse, "val_rmse": val_rmse})
    with open(cfg.paths.save_dir, 'w') as f: json.dump(log, f)

# Finalize SWA batch normalization statistics
print("Finalizing SWA batch norm statistics...")
torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

# Save final SWA model
if cfg.training.save_checkpoint:
    torch.save({'model_state_dict': swa_model.state_dict()}, cfg.paths.model_save_path.replace(".pt", "_best.pt"))
    print(f"Final SWA model saved. Best Val RMSE: {best_val_rmse:.4f}")