import os
import time
import json
import logging
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
import sys

cfg = load_config("configs/train.yaml")

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(getattr(cfg.paths, 'log_path', 'training.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

S1, S2 = cfg.data.S1, cfg.data.S2

# ==========================================
# 2. STATS & LOSS UTILITIES
# ==========================================
logging.info("Loading robust grid-wise stats...")
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

def skewed_mse_loss(pred_phys, target_phys, threshold=100.0, alpha=0.05, max_penalty=5.0):
    """
    Custom MSE that exponentially penalizes under-predictions on high-pollution days.
    If the target is a hotspot (>100) and the model under-predicts, the MSE is scaled 
    exponentially by the error magnitude, bounded to prevent exploding gradients.
    """
    mse = (pred_phys - target_phys) ** 2
    
    weights = torch.ones_like(target_phys)
    
    # Identify under-predictions strictly on hotspots
    under_mask = (target_phys > threshold) & (pred_phys < target_phys)
    
    if under_mask.any():
        error = target_phys[under_mask] - pred_phys[under_mask]
        # Calculate exponential penalty multiplier, clamped to prevent NaN gradients
        penalty_weight = torch.exp(torch.clamp(alpha * error, max=max_penalty))
        
        weights[under_mask] = penalty_weight
        
    # ALWAYS return the weighted calculation so scaling is identical
    return torch.sum(weights * mse) / torch.sum(weights)

def log_layer_dynamics(model, epoch, grad_acc=None, num_steps=1):
    """ Logs the average absolute weight and gradient magnitude of major network blocks """
    logging.info(f"--- Epoch {epoch} Layer Dynamics (Mean L1 Norm) ---")
    
    blocks_to_track = {
        'Temporal Encoder': 'temporal_encoder',
        'Input Encoder': 'input_encoder',
        'WNO Block 0': 'block0',
        'WNO Block 1': 'block1',
        'WNO Block 2': 'block2',
        'WNO Block 3': 'block3',
        'Output FC1': 'fc1',
        'Output FC2': 'fc2'
    }

    for block_name, prefix in blocks_to_track.items():
        weight_sum, grad_sum, count = 0.0, 0.0, 0
        for name, param in model.named_parameters():
            if name.startswith(prefix) and param.requires_grad:
                weight_sum += param.abs().mean().item()
                if grad_acc is not None and name in grad_acc:
                    grad_sum += grad_acc[name] / max(1, num_steps)
                elif param.grad is not None:
                    grad_sum += param.grad.abs().mean().item()
                count += 1
                
        if count > 0:
            avg_weight = weight_sum / count
            avg_grad = grad_sum / count
            logging.info(f" -> {block_name:18s} | Weight: {avg_weight:.6f} | Gradient: {avg_grad:.6f}")
    
    logging.info("--------------------------------------------------")

# ==========================================
# 3. DATA LOADER
# ==========================================
class FastInMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, split, cfg):
        base_path = getattr(cfg.paths, f"savepath_{split}")
        self.split = split
        
        logging.info(f"[{split.upper()}] Loading into RAM...")
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
        # Topography is no longer embedded in train_data to prevent bloat.
        topo_path = cfg.paths.topo_path
        self.topo_proxy = np.load(topo_path)

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
        
        # 3. Static Emissions (Extract [min, mean, max] along the time dimension to preserve daily profile)
        static_emissions_raw = window[:, ..., self.static_idx]
        static_emissions = torch.cat([
            static_emissions_raw.min(dim=0)[0],
            static_emissions_raw.mean(dim=0),
            static_emissions_raw.max(dim=0)[0]
        ], dim=-1)
        
        # 4. Static Topography (1 channel)
        topo = torch.from_numpy(self.topo_proxy).unsqueeze(-1)
        
        # Combine everything (10 + 260 + 21 + 1 = 292 Channels)
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
temporal_features_count = len([f for f in cfg.features.met_variables if f != 'cpm25'] + cfg.features.derived_variables)
temporal_channels = temporal_features_count * cfg.data.total_time
static_channels = len(cfg.features.emission_variables) * 3
topo_channels = 1

in_channels = pm_channels + temporal_channels + static_channels + topo_channels
logging.info(f"Building Model with optimized {in_channels} input channels (Massive memory saving!)...")

model = FNO2D(
    in_channels=in_channels, 
    time_out=cfg.data.time_out, 
    width=cfg.model.width, 
    modes=cfg.model.modes,
    time_input=cfg.data.time_input,
    total_time=cfg.data.total_time,
    num_temporal_features=temporal_features_count
).to(device)

def log_model_summary(model):
    logging.info("========================================")
    logging.info("       WNO MODEL ARCHITECTURE SIZES      ")
    logging.info("========================================")
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        params_count = param.numel()
        total_params += params_count
        if param.requires_grad:
            trainable_params += params_count
            logging.info(f" -> {name:40s} | {params_count:,} params | Shape: {list(param.shape)}")
        else:
            logging.info(f" -> {name:40s} | {params_count:,} params | Shape: {list(param.shape)} (FROZEN)")
            
    logging.info("----------------------------------------")
    logging.info(f" Total Parameters:     {total_params:,}")
    logging.info(f" Trainable Parameters: {trainable_params:,}")
    logging.info("========================================")

log_model_summary(model)

optimizer = Adam(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)


swa_model = AveragedModel(model)
swa_start = int(cfg.training.epochs * 0.75)
swa_scheduler = SWALR(optimizer, swa_lr=5e-4)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
best_val_rmse = float('inf')
log = []
use_amp = (device.type == 'cuda')
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

for ep in range(cfg.training.epochs):
    model.train()
    t_start = time.time()
    train_mse_acc = 0.0
    grad_ratio_acc = 0.0
    grad_acc = {name: 0.0 for name, p in model.named_parameters() if p.requires_grad}
    
    current_lr = optimizer.param_groups[0]['lr']
    logging.info(f"--- Starting Epoch {ep} | LR: {current_lr:.6f} ---")

    for step, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {ep}")):
        step_start = time.time()
        x, y = x.to(device), y.to(device)
        
        # FIX #2: Apply noise only to continuous features, protecting:
        # - Topography (last channel): static elevation proxy from PSFC
        c0 = cfg.data.time_input
        c1 = c0 + (cfg.data.total_time * temporal_features_count)

        noise = torch.randn_like(x) * 0.01
        noise[..., -1] = 0.0  # Protect topography

        # Protect ONLY the binary rain_mask
        rain_mask_name = 'rain_mask'
        all_features = cfg.features.met_variables + cfg.features.emission_variables + cfg.features.derived_variables
        temporal_vars = [f for f in all_features if f != 'cpm25' and f not in cfg.features.emission_variables]
        if rain_mask_name in temporal_vars:
            rm_idx = temporal_vars.index(rain_mask_name)
            rain_mask_indices = torch.arange(c0 + rm_idx, c1, temporal_features_count)
            noise[..., rain_mask_indices] = 0.0
            
        x = x + noise
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            # Single Forward Pass (Direct Multi-Step)
            out = model(x)
            
        # EXIT AUTOCAST: Convert to float32 to prevent MSE from overflowing float16
        out_f32 = out.float()
        
        pred_phys = to_physical(out_f32)
        targ_phys = to_physical(y.float())
        
        # --- NEW LOSS FORMULATION ---
        # 1. Skewed MSE (Exponentially penalizes hotspot under-predictions instead of smoothing them)
        skewed_loss = skewed_mse_loss(pred_phys, targ_phys, threshold=100.0, alpha=0.05)
        
        # 2. Spatial Gradient Loss (Keeps plume edges sharp)
        loss_grad = spatial_gradient_loss(pred_phys, targ_phys)
        
        # 3. Blended Total Loss
        total_loss = skewed_loss + 0.1 * loss_grad

        
        with torch.no_grad():
            pred_phys_clipped = F.relu(pred_phys.detach())
            train_mse_acc += torch.mean((pred_phys_clipped - targ_phys) ** 2).item()
            
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        
        # Gradient Flow Ratio Check
        grad_out = model.fc2.weight.grad.abs().mean().item() if (model.fc2.weight.grad is not None) else 0.0
        grad_in = model.input_encoder[0].weight.grad.abs().mean().item() if (model.input_encoder[0].weight.grad is not None) else 0.0
        grad_ratio_acc += grad_in / (grad_out + 1e-8)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate gradients continuously for layer dynamics logging
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_acc[name] += param.grad.abs().mean().item()
        
        step_duration = time.time() - step_start
        if step % 20 == 0:
            mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0
            log_msg = (
                f"Epoch {ep:02d} | Step {step:04d}/{len(train_loader)} | "
                f"Time/Batch: {step_duration:.3f}s | "
                f"Loss Total: {total_loss.item():.4f} | "
                f"SkewedMSE: {skewed_loss.item():.4f} | "
                f"Grad: {loss_grad.item():.4f} | "
                f"GPU Mem: {mem_allocated:.1f}MB"
            )
            logging.info(log_msg)

    # --- NEW: LOG LAYER DYNAMICS ---
    # Captures the state of the network weights and gradients across the whole epoch
    log_layer_dynamics(model, ep, grad_acc=grad_acc, num_steps=len(train_loader))

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
    val_high_pm_mse_acc = 0.0
    val_low_pm_mse_acc = 0.0
    high_pm_count = 0
    low_pm_count = 0
    
    val_skewed_acc, val_grad_acc = 0.0, 0.0
    val_t1_mse_acc, val_t8_mse_acc, val_t16_mse_acc = 0.0, 0.0, 0.0
    val_igp_mse_acc, val_ocean_mse_acc = 0.0, 0.0
    val_max_pred, val_max_targ = 0.0, 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            # ✅ FIXED BUG: This must be eval_model, otherwise SWA does nothing!
            out = eval_model(x) 
            
            pred_phys = to_physical(out)
            targ_phys = to_physical(y)
            
            # Apply ReLU to mimic the physical world constraint for our final score
            pred_phys_clipped = F.relu(pred_phys)
            
            # Global Validation Step
            val_mse_acc += torch.mean((pred_phys_clipped - targ_phys) ** 2).item()
            
            # Additive Loss Components
            val_skewed_acc += skewed_mse_loss(pred_phys, targ_phys, threshold=100.0, alpha=0.05).item()
            val_grad_acc += spatial_gradient_loss(pred_phys, targ_phys).item()
            
            # Forecast Horizon Degradation
            val_t1_mse_acc += torch.mean((pred_phys_clipped[..., 0] - targ_phys[..., 0]) ** 2).item()
            val_t8_mse_acc += torch.mean((pred_phys_clipped[..., 7] - targ_phys[..., 7]) ** 2).item()
            val_t16_mse_acc += torch.mean((pred_phys_clipped[..., 15] - targ_phys[..., 15]) ** 2).item()
            
            # Regional Stratification
            val_igp_mse_acc += torch.mean((pred_phys_clipped[:, 60:90, 40:100, :] - targ_phys[:, 60:90, 40:100, :]) ** 2).item()
            val_ocean_mse_acc += torch.mean((pred_phys_clipped[:, 0:30, 0:50, :] - targ_phys[:, 0:30, 0:50, :]) ** 2).item()
            
            # Peak Capture
            val_max_pred = max(val_max_pred, pred_phys.max().item())
            val_max_targ = max(val_max_targ, targ_phys.max().item())
            
            # Data Specific Analysis Step (Extreme Spikes vs Clear Conditions)
            # Define high pollution threshold > 100 ug/m3, Low pollution < 30 ug/m3
            high_mask = targ_phys > 100.0
            low_mask = targ_phys < 30.0
            
            if high_mask.any():
                val_high_pm_mse_acc += torch.mean((pred_phys_clipped[high_mask] - targ_phys[high_mask]) ** 2).item() * high_mask.sum().item()
                high_pm_count += high_mask.sum().item()
                
            if low_mask.any():
                val_low_pm_mse_acc += torch.mean((pred_phys_clipped[low_mask] - targ_phys[low_mask]) ** 2).item() * low_mask.sum().item()
                low_pm_count += low_mask.sum().item()

    train_rmse = np.sqrt(train_mse_acc / len(train_loader))
    val_rmse = np.sqrt(val_mse_acc / len(val_loader))
    
    val_high_rmse = np.sqrt(val_high_pm_mse_acc / max(1, high_pm_count))
    val_low_rmse = np.sqrt(val_low_pm_mse_acc / max(1, low_pm_count))
    
    val_t1_rmse = np.sqrt(val_t1_mse_acc / len(val_loader))
    val_t8_rmse = np.sqrt(val_t8_mse_acc / len(val_loader))
    val_t16_rmse = np.sqrt(val_t16_mse_acc / len(val_loader))
    val_igp_rmse = np.sqrt(val_igp_mse_acc / len(val_loader))
    val_ocean_rmse = np.sqrt(val_ocean_mse_acc / len(val_loader))
    
    duration = time.time() - t_start
    msg = (f"Epoch {ep} | Time: {duration:.1f}s | Train RMSE: {train_rmse:.4f} | "
           f"Val RMSE: {val_rmse:.4f} | High PM Val RMSE: {val_high_rmse:.4f} | Low PM Val RMSE: {val_low_rmse:.4f}")
    logging.info(msg)
    
    peak_capture = val_max_pred / (val_max_targ + 1e-5)
    grad_ratio_avg = grad_ratio_acc / len(train_loader)
    
    logging.info(f"--- Epoch {ep} Advanced Telemetry ---")
    logging.info(f" [When]  T+1 RMSE: {val_t1_rmse:.4f} | T+8 RMSE: {val_t8_rmse:.4f} | T+16 RMSE: {val_t16_rmse:.4f}")
    logging.info(f" [Where] IGP RMSE: {val_igp_rmse:.4f} | Ocean RMSE: {val_ocean_rmse:.4f}")
    logging.info(f" [Physics] Peak Capture Ratio: {peak_capture:.4f} (Pred Max: {val_max_pred:.1f} / Actual Max: {val_max_targ:.1f})")
    logging.info(f" [Arch] Gradient Flow Ratio (In/Out): {grad_ratio_avg:.6f}")
    logging.info(f" [Loss] Val SkewedMSE: {(val_skewed_acc / len(val_loader)):.4f} | Val Grad: {(val_grad_acc / len(val_loader)):.4f}")
    logging.info("-------------------------------------")

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        if cfg.training.save_checkpoint:
            # Save the currently evaluating model's dict (which will be SWA if active)
            torch.save({'model_state_dict': eval_model.state_dict()}, cfg.paths.model_save_path.replace(".pt", "_best.pt"))
        best_msg = f"  -> New Best Val RMSE: {best_val_rmse:.4f}"
        logging.info(best_msg)

    log.append({"epoch": ep, "train_rmse": train_rmse, "val_rmse": val_rmse})
    with open(cfg.paths.save_dir, 'w') as f: json.dump(log, f)