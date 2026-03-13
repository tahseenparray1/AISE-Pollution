"""
TPU-Optimized Training Script for PM2.5 WNO Forecasting
========================================================
Optimized for TPU v5e-8 (8 cores) via PyTorch XLA.

Key TPU optimizations:
  1. xmp.spawn() for multi-core parallelism across all 8 TPU cores
  2. XLA-specific optimizer step (xm.optimizer_step) for lazy graph execution
  3. Native bfloat16 via XLA env flags (no GradScaler needed)
  4. MpDeviceLoader for async host-to-device prefetching
  5. xm.mark_step() barriers for correct graph compilation
  6. DistributedSampler for correct data sharding across cores
  7. xm.master_print() for clean single-process logging
  8. xm.save() for safe checkpoint serialization
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# TPU / XLA Imports
# ==========================================
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from src.utils.config import load_config
from models.baseline_model import FNO2D
from torch.optim.swa_utils import AveragedModel, SWALR

# ==========================================
# Enable bfloat16 for TPU (compiler-level, no GradScaler needed)
# ==========================================
os.environ['XLA_USE_BF16'] = '1'


def train_fn(index):
    """
    Main training function. Called once per TPU core by xmp.spawn().
    `index` is the core ID (0-7 for v5e-8).
    """
    # ==========================================
    # 1. SETUP & CONFIGURATION
    # ==========================================
    cfg = load_config("configs/train.yaml")
    
    # XLA device for this specific core
    device = xm.xla_device()
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    S1, S2 = cfg.data.S1, cfg.data.S2
    
    # ==========================================
    # 2. STATS & LOSS UTILITIES
    # ==========================================
    xm.master_print("Loading robust grid-wise stats...")
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
    # 3. DATA LOADER (with DistributedSampler for multi-core)
    # ==========================================
    class FastInMemoryDataset(torch.utils.data.Dataset):
        def __init__(self, split, cfg):
            base_path = getattr(cfg.paths, f"savepath_{split}")
            self.split = split
            
            xm.master_print(f"[{split.upper()}] Loading into RAM...")
            self.data = torch.from_numpy(np.load(os.path.join(base_path, f"{split}_data.npy")).astype(np.float32))
            self.valid_starts = np.load(os.path.join(base_path, f"{split}_indices.npy"))
            
            self.time_in = cfg.data.time_input
            self.total_time = cfg.data.total_time
            self.S1, self.S2 = cfg.data.S1, cfg.data.S2
            
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
            window = self.data[start : start + self.total_time].clone()
            
            # 1. PM2.5 Historical Context (10 channels)
            pm_hist = window[:self.time_in, ..., self.target_idx].permute(1, 2, 0)
            
            # 2. Dynamic Temporal Weather (10 features * 26 hours = 260 channels)
            temporal_weather = window[:, ..., self.temporal_idx]
            temporal_weather = temporal_weather.permute(1, 2, 0, 3).reshape(self.S1, self.S2, -1)
            
            # 3. DYNAMIC Emissions (7 features * 26 hours = 182 channels)
            dynamic_emissions = window[:, ..., self.static_idx]
            dynamic_emissions = dynamic_emissions.permute(1, 2, 0, 3).reshape(self.S1, self.S2, -1)
            
            # 4. Static Topography (1 channel)
            topo = window[0, ..., self.topo_idx].unsqueeze(-1)
            
            # Combine (10 + 260 + 182 + 1 = 453 Channels)
            x = torch.cat((pm_hist, temporal_weather, dynamic_emissions, topo), dim=-1)
            
            # Target
            y = window[self.time_in:, ..., self.target_idx].permute(1, 2, 0)
            return x, y
    
    train_ds = FastInMemoryDataset("train", cfg)
    val_ds = FastInMemoryDataset("val", cfg)
    
    # DistributedSampler: each TPU core gets a unique shard of data
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_ds,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )
    
    # TPU prefers num_workers=0 (data is in RAM, no host bottleneck)
    # drop_last=True ensures all cores get the same number of batches
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        sampler=train_sampler, num_workers=0, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.training.batch_size,
        sampler=val_sampler, num_workers=0, drop_last=False
    )
    
    # ==========================================
    # 4. MODEL & OPTIMIZER
    # ==========================================
    pm_channels = cfg.data.time_input
    temporal_channels = 10 * cfg.data.total_time
    emission_channels = 7 * cfg.data.total_time
    topo_channels = 1
    
    in_channels = pm_channels + temporal_channels + emission_channels + topo_channels
    xm.master_print(f"Building WNO with {in_channels} input channels...")
    
    model = FNO2D(
        in_channels=in_channels, 
        time_out=cfg.data.time_out, 
        width=cfg.model.width, 
        modes=cfg.model.modes,
        time_input=cfg.data.time_input
    ).to(device)
    
    # Use standard AdamW for TPU (XLA-compatible)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(cfg.training.lr), 
        weight_decay=float(cfg.training.weight_decay)
    )
    
    # Diagnostics (master core only)
    total_params = sum(p.numel() for p in model.parameters())
    xm.master_print(f"\n{'='*60}")
    xm.master_print(f"TPU TRAINING DIAGNOSTICS")
    xm.master_print(f"{'='*60}")
    xm.master_print(f"  Total Parameters:     {total_params:,}")
    xm.master_print(f"  Device:               TPU v5e-8 ({xm.xrt_world_size()} cores)")
    xm.master_print(f"  Input Channels:       {in_channels}")
    xm.master_print(f"  WNO Base Width:       {cfg.model.width}")
    xm.master_print(f"  Output Steps:         {cfg.data.time_out}")
    xm.master_print(f"  Grid Size:            {S1}x{S2}")
    xm.master_print(f"  Batch Size/Core:      {cfg.training.batch_size}")
    xm.master_print(f"  Effective Batch Size: {cfg.training.batch_size * xm.xrt_world_size()}")
    xm.master_print(f"  Epochs:               {cfg.training.epochs}")
    xm.master_print(f"  Train Samples:        {len(train_ds)}")
    xm.master_print(f"  Val Samples:          {len(val_ds)}")
    xm.master_print(f"  Precision:            bfloat16 (XLA native)")
    xm.master_print(f"{'='*60}\n")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=1e-6
    )
    
    swa_model = AveragedModel(model)
    swa_start = int(cfg.training.epochs * 0.40)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-4)
    
    # ==========================================
    # 5. TRAINING LOOP
    # ==========================================
    best_val_rmse = float('inf')
    log = []
    
    for ep in range(cfg.training.epochs):
        model.train()
        train_sampler.set_epoch(ep)  # Shuffle differently each epoch
        t_start = time.time()
        train_mse_acc = 0.0
        train_loss_acc = 0.0
        batch_count = 0
        
        # MpDeviceLoader: async host→TPU data prefetching pipeline
        para_train_loader = pl.MpDeviceLoader(train_loader, device)
        
        for batch_idx, (x, y) in enumerate(para_train_loader):
            # First-batch diagnostic (master core only)
            if ep == 0 and batch_idx == 0:
                xm.master_print(f"\n{'='*60}")
                xm.master_print(f"FIRST BATCH DIAGNOSTICS (Epoch 0, Core {xm.get_ordinal()})")
                xm.master_print(f"{'='*60}")
                xm.master_print(f"  x shape: {x.shape}  |  y shape: {y.shape}")
                xm.master_print(f"  x dtype: {x.dtype}")
                xm.master_print(f"{'='*60}\n")
            
            # 3% input noise (infinite data multiplier)
            noise = torch.randn_like(x) * 0.03
            noise[..., -1] = 0.0  # Protect topography channel
            x = x + noise
            
            optimizer.zero_grad(set_to_none=True)
            
            out = model(x)
            
            pred_phys = to_physical(out)
            targ_phys = to_physical(y)
            
            # 1. Huber Loss (delta=40, acts like MSE for Kaggle RMSE optimization)
            huber_loss = F.huber_loss(pred_phys, targ_phys, delta=40.0)
            
            # 2. Temporal Consistency (relaxed weight for spike prediction)
            temp_pred = pred_phys[:, :, :, 1:] - pred_phys[:, :, :, :-1]
            temp_targ = targ_phys[:, :, :, 1:] - targ_phys[:, :, :, :-1]
            temporal_loss = F.l1_loss(temp_pred, temp_targ)
            
            # 3. Spatial Gradient (plume edges)
            loss_grad = spatial_gradient_loss(pred_phys, targ_phys)
            
            total_loss = huber_loss + (0.2 * temporal_loss) + (0.5 * loss_grad)
            
            # Accumulate diagnostics (no grad)
            with torch.no_grad():
                pred_phys_clipped = F.relu(pred_phys.detach())
                train_mse_acc += torch.mean((pred_phys_clipped - targ_phys) ** 2).item()
                train_loss_acc += total_loss.item()
            
            total_loss.backward()
            
            # Gradient clipping (compatible with XLA)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # XLA optimizer step: triggers lazy graph execution on TPU
            xm.optimizer_step(optimizer)
            
            batch_count += 1
        
        # SWA handling
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
        val_batches = 0
        
        para_val_loader = pl.MpDeviceLoader(val_loader, device)
        
        with torch.no_grad():
            for x, y in para_val_loader:
                out = eval_model(x)
                
                pred_phys = to_physical(out)
                targ_phys = to_physical(y)
                pred_phys_clipped = F.relu(pred_phys)
                
                val_mse_acc += torch.mean((pred_phys_clipped - targ_phys) ** 2).item()
                
                pred_sharp = torch.abs(pred_phys[:, :, :, 1:] - pred_phys[:, :, :, :-1]).mean()
                targ_sharp = torch.abs(targ_phys[:, :, :, 1:] - targ_phys[:, :, :, :-1]).mean()
                val_sharpness_pred += pred_sharp.item()
                val_sharpness_targ += targ_sharp.item()
                val_batches += 1
        
        # Aggregate metrics across all TPU cores via xm.mesh_reduce
        train_mse_total = xm.mesh_reduce('train_mse', train_mse_acc, lambda x: sum(x) / len(x))
        val_mse_total = xm.mesh_reduce('val_mse', val_mse_acc, lambda x: sum(x) / len(x))
        train_loss_total = xm.mesh_reduce('train_loss', train_loss_acc, lambda x: sum(x) / len(x))
        
        train_rmse = np.sqrt(train_mse_total / max(batch_count, 1))
        val_rmse = np.sqrt(val_mse_total / max(val_batches, 1))
        avg_loss = train_loss_total / max(batch_count, 1)
        
        sharpness_ratio = val_sharpness_pred / (val_sharpness_targ + 1e-5) if val_batches > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        duration = time.time() - t_start
        
        # --- Logging (master core only) ---
        xm.master_print(f"\nEpoch {ep} | Time: {duration:.1f}s | LR: {current_lr:.2e}")
        xm.master_print(f"  Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")
        xm.master_print(f"  Total Loss: {avg_loss:.4f}")
        xm.master_print(f"  Sharpness Ratio: {sharpness_ratio:.4f}")
        
        # Save best model (master core only, using xm.save for safety)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            if cfg.training.save_checkpoint:
                xm.save(
                    {'model_state_dict': eval_model.state_dict()}, 
                    cfg.paths.model_save_path.replace(".pt", "_best.pt")
                )
            xm.master_print(f"  -> ✅ New Best Val RMSE: {best_val_rmse:.4f}")
        
        # Log to JSON (master core only)
        if xm.get_ordinal() == 0:
            epoch_log = {
                "epoch": ep,
                "train_rmse": round(train_rmse, 4),
                "val_rmse": round(val_rmse, 4),
                "total_loss": round(avg_loss, 4),
                "lr": current_lr,
                "sharpness_ratio": round(sharpness_ratio, 4),
                "duration_s": round(duration, 1)
            }
            log.append(epoch_log)
            with open(cfg.paths.save_dir, 'w') as f:
                json.dump(log, f, indent=2)
    
    xm.master_print(f"\n{'='*60}")
    xm.master_print(f"TRAINING COMPLETE! Best Val RMSE: {best_val_rmse:.4f}")
    xm.master_print(f"{'='*60}")


# ==========================================
# ENTRY POINT: Spawn across all 8 TPU cores
# ==========================================
if __name__ == '__main__':
    # nprocs=8 for TPU v5e-8 (all 8 cores)
    xmp.spawn(train_fn, args=(), nprocs=8)
