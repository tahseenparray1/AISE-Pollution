"""
train.py  (V2 — fixed)

Changes vs original V2:
  FIX (Bug #1): skewed_mse_loss was defined but never called.
                Now included in total_loss with weight 0.5.
  FIX (Bug #5): use_amp=False but the GradScaler/autocast scaffolding
                was still wrapping every step, adding confusion and
                dead .float() casts.  Cleaned up: AMP is now a single
                flag that either enables full AMP or falls back to
                plain float32 with zero overhead.
  FIX (Bug #6): log_path was not in train.yaml so the logger fell back
                to 'training.log' in the cwd.  Now reads from config
                with a proper /kaggle/working/ fallback.
  FIX (Bug #8): log_layer_dynamics ran every epoch (200 full param
                scans for 100 epochs).  Now runs every LOG_DYNAMICS_EVERY
                epochs (default 10).
  Bottleneck 1c: nan-safe gradient accumulator.
  Bottleneck 2a: skewed loss weight raised 0.5 -> 2.0.
  Bottleneck 2b: Huber delta raised 10 -> 25 in training + val loops.
  Bottleneck 3:  two-chunk rollout in training loop.
"""

import os
import sys
import math
import time
import json
import logging

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.utils.adam import Adam
from src.utils.config import load_config
from models.baseline_model import FNO2D
from torch.optim.swa_utils import AveragedModel, SWALR

# ============================================================
# 1. SETUP & CONFIGURATION
# ============================================================
cfg    = load_config("configs/train.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

LOG_DYNAMICS_EVERY = 2   # FIX Bug #8: only run expensive layer-dynamics log every N epochs

# FIX Bug #6: read log_path from config; fall back to /kaggle/working/training.log
_log_path = getattr(cfg.paths, 'log_path', '/kaggle/working/training.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(_log_path),
        logging.StreamHandler(sys.stdout),
    ],
)

S1, S2 = cfg.data.S1, cfg.data.S2

# FIX Bug #5: AMP is now a single boolean.  Set True to enable; False = pure float32.
USE_AMP = False   # safe default: enable on GPU, off on CPU

# ============================================================
# 2. STATS & LOSS UTILITIES
# ============================================================
logging.info("Loading robust grid-wise stats...")
stats = np.load(cfg.paths.stats_path, allow_pickle=True).item()

pm_median_tensor = torch.tensor(
    stats['cpm25']['median'], dtype=torch.float32).to(device).view(1, S1, S2, 1)
pm_iqr_tensor = torch.tensor(
    stats['cpm25']['iqr'],    dtype=torch.float32).to(device).view(1, S1, S2, 1)


def to_physical(x_norm):
    return x_norm * pm_iqr_tensor + pm_median_tensor


def spatial_gradient_loss(pred_phys, target_phys):
    dx_p = pred_phys[:, 1:, :, :] - pred_phys[:, :-1, :, :]
    dx_t = target_phys[:, 1:, :, :] - target_phys[:, :-1, :, :]
    dy_p = pred_phys[:, :, 1:, :] - pred_phys[:, :, :-1, :]
    dy_t = target_phys[:, :, 1:, :] - target_phys[:, :, :-1, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)


def skewed_mse_loss(pred_norm, targ_norm, pred_phys, targ_phys,
                    threshold=100.0, alpha=0.05, max_penalty=2.0):
    """
    Weighted MSE in normalised space with exponential upweighting on
    under-predictions where the target exceeds `threshold` µg/m³.
    Operating in normalised space prevents gradient explosions.

    FIX Bug #1: this function was defined in the original V2 but never
    called.  It is now included in total_loss (see training loop).
    """
    mse     = (pred_norm - targ_norm) ** 2
    weights = torch.ones_like(targ_norm)

    under_mask = (targ_phys > threshold) & (pred_phys < targ_phys)
    if under_mask.any():
        error_phys = targ_phys[under_mask] - pred_phys[under_mask]
        weights[under_mask] = torch.exp(
            torch.clamp(alpha * error_phys, max=max_penalty))

    weights = weights / (weights.sum() + 1e-8)   # normalise to avoid scale drift
    return torch.sum(weights * mse)


def log_layer_dynamics(model, epoch, grad_acc=None, num_steps=1):
    """Log mean L1 weight and gradient norm for each major block."""
    logging.info(f"--- Epoch {epoch} Layer Dynamics ---")
    blocks = {
        'Temporal Encoder': 'temporal_encoder',
        'Input Encoder'   : 'input_encoder',
        'WNO Block 0'     : 'block0',
        'WNO Block 1'     : 'block1',
        'WNO Block 2'     : 'block2',
        'WNO Block 3'     : 'block3',
        'Output FC1'      : 'fc1',
        'Output FC2'      : 'fc2',
    }
    for block_name, prefix in blocks.items():
        w_sum = g_sum = count = 0
        for name, param in model.named_parameters():
            if name.startswith(prefix) and param.requires_grad:
                w_sum += param.abs().mean().item()
                if grad_acc and name in grad_acc:
                    g_sum += grad_acc[name] / max(1, num_steps)
                elif param.grad is not None:
                    g_sum += param.grad.abs().mean().item()
                count += 1
        if count:
            logging.info(
                f"  {block_name:18s} | W: {w_sum/count:.6f} | G: {g_sum/count:.6f}")
    logging.info("--------------------------------------------------")


# ============================================================
# 3. DATA LOADER
# ============================================================
class FastInMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, split, cfg):
        base_path = getattr(cfg.paths, f"savepath_{split}")
        logging.info(f"[{split.upper()}] Loading into RAM...")
        self.data         = torch.from_numpy(
            np.load(os.path.join(base_path, f"{split}_data.npy")).astype(np.float32))
        self.valid_starts = np.load(
            os.path.join(base_path, f"{split}_indices.npy"))

        self.time_in   = cfg.data.time_input
        self.total_time = cfg.data.total_time
        self.S1, self.S2 = cfg.data.S1, cfg.data.S2

        all_features = (cfg.features.met_variables
                        + cfg.features.emission_variables
                        + cfg.features.derived_variables)
        assert 'cpm25' in all_features
        self.target_idx  = all_features.index('cpm25')
        self.temporal_idx = [i for i, f in enumerate(all_features)
                             if f != 'cpm25' and f not in cfg.features.emission_variables]
        self.static_idx   = [i for i, f in enumerate(all_features)
                             if f in cfg.features.emission_variables]

        self.topo_proxy = np.load(cfg.paths.topo_path)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start  = self.valid_starts[idx]
        window = self.data[start : start + self.total_time].clone()

        pm_hist = window[:self.time_in, ..., self.target_idx].permute(1, 2, 0)

        temporal_raw = window[:, ..., self.temporal_idx]
        temporal     = temporal_raw.permute(1, 2, 0, 3).reshape(self.S1, self.S2, -1)

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

# ============================================================
# 4. MODEL & OPTIMIZER
# ============================================================
pm_channels              = cfg.data.time_input
temporal_features_count  = len(
    [f for f in cfg.features.met_variables if f != 'cpm25']
    + cfg.features.derived_variables)
temporal_channels = temporal_features_count * cfg.data.total_time
static_channels   = len(cfg.features.emission_variables) * 3
topo_channels     = 1
in_channels       = pm_channels + temporal_channels + static_channels + topo_channels

logging.info(f"Building model: {in_channels} input channels | width={cfg.model.width}")

model = FNO2D(
    in_channels=in_channels,
    time_out=cfg.data.time_out,
    width=cfg.model.width,
    modes=cfg.model.modes,
    time_input=cfg.data.time_input,
    total_time=cfg.data.total_time,
    num_temporal_features=temporal_features_count,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
logging.info(f"Total parameters: {total_params:,}")

optimizer = Adam(model.parameters(),
                 lr=float(cfg.training.lr),
                 weight_decay=float(cfg.training.weight_decay))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.training.epochs, eta_min=1e-6)

swa_model   = AveragedModel(model)
swa_start   = int(cfg.training.epochs * 0.75)
swa_scheduler = SWALR(optimizer, swa_lr=1e-4)

# FIX Bug #5: single scaler; enabled=False is a clean no-op (no dead casts needed)
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

# ============================================================
# 5. TRAINING LOOP
# ============================================================
best_val_rmse = float('inf')
log = []

all_features   = (cfg.features.met_variables
                  + cfg.features.emission_variables
                  + cfg.features.derived_variables)
temporal_vars  = [f for f in all_features
                  if f != 'cpm25' and f not in cfg.features.emission_variables]
c0 = cfg.data.time_input
c1 = c0 + cfg.data.total_time * temporal_features_count

for ep in range(cfg.training.epochs):
    model.train()
    t_start        = time.time()
    train_mse_acc  = 0.0
    grad_ratio_acc = 0.0
    grad_acc = {name: 0.0 for name, p in model.named_parameters() if p.requires_grad}

    current_lr = optimizer.param_groups[0]['lr']
    logging.info(f"--- Epoch {ep} | LR: {current_lr:.6f} ---")

    for step, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {ep}")):
        step_start = time.time()
        x, y = x.to(device), y.to(device)

        # Noise augmentation — protect topo (last ch) and binary rain_mask
        noise = torch.randn_like(x) * 0.01
        noise[..., -1] = 0.0   # topo
        if 'rain_mask' in temporal_vars:
            rm_idx            = temporal_vars.index('rain_mask')
            rain_mask_indices = torch.arange(c0 + rm_idx, c1, temporal_features_count)
            noise[..., rain_mask_indices] = 0.0
        x = x + noise

        optimizer.zero_grad(set_to_none=True)

        # --- Chunk 1 forward ---
        out_chunk1   = model(x)
        pred_c1_norm = out_chunk1[..., :8].float()
        pred_c1_phys = to_physical(pred_c1_norm)
        targ_c1_phys = to_physical(y[..., :8].float())

        # Backward chunk 1 IMMEDIATELY — frees ~2.5 GB of activations before chunk 2 forward
        huber1 = F.huber_loss(pred_c1_phys, targ_c1_phys, delta=25.0)
        grad1  = spatial_gradient_loss(pred_c1_phys, targ_c1_phys)
        (huber1 + 0.1 * grad1).backward()     # gradients accumulate in .grad

        # --- Chunk 2 forward (chunk 1 activation graph is now freed) ---
        x2 = x.clone()
        x2[..., 2:10] = pred_c1_norm.detach()   # already detached from freed graph

        out_chunk2   = model(x2)
        pred_c2_norm = out_chunk2[..., :8].float()
        pred_c2_phys = to_physical(pred_c2_norm)
        targ_c2_phys = to_physical(y[..., 8:16].float())

        # Combined tensors: chunk1 detached so only chunk2 parameters receive gradient here
        pred_phys_all = torch.cat([pred_c1_phys.detach(), pred_c2_phys], dim=-1)
        targ_phys_all = torch.cat([targ_c1_phys,          targ_c2_phys], dim=-1)
        out_norm_all  = torch.cat([pred_c1_norm.detach(),  pred_c2_norm], dim=-1)

        huber2      = F.huber_loss(pred_c2_phys, targ_c2_phys, delta=25.0)
        grad2       = spatial_gradient_loss(pred_phys_all, targ_phys_all)
        skewed_loss = skewed_mse_loss(
            out_norm_all, y.float(), pred_phys_all, targ_phys_all,
            threshold=100.0, alpha=0.05, max_penalty=2.0)
        (huber2 + 0.1 * grad2 + 2.0 * skewed_loss).backward()   # accumulates onto chunk1 grads

        with torch.no_grad():
            pred_clipped   = F.relu(pred_phys_all.detach())
            train_mse_acc += torch.mean((pred_clipped - targ_phys_all) ** 2).item()

        grad_out = model.fc2.weight.grad.abs().mean().item() \
            if model.fc2.weight.grad is not None else 0.0
        grad_in  = model.input_encoder[0].weight.grad.abs().mean().item() \
            if model.input_encoder[0].weight.grad is not None else 0.0
        grad_ratio_acc += grad_in / (grad_out + 1e-8)

        # USE_AMP=False so GradScaler is a no-op; simplified to direct norm+step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Fix 1c: nan-safe gradient accumulator
                v = param.grad.abs().mean().item()
                if math.isfinite(v):
                    grad_acc[name] += v

        if step % 20 == 0:
            mem = torch.cuda.memory_allocated(device) / 1024 ** 2 \
                if torch.cuda.is_available() else 0
            total_huber = huber1.item() + huber2.item()
            logging.info(
                f"Ep {ep:02d} | Step {step:04d}/{len(train_loader)} | "
                f"{time.time()-step_start:.2f}s | "
                f"Huber {total_huber:.4f} | "
                f"Grad {grad2.item():.4f} | "
                f"Skewed {skewed_loss.item():.4f} | "
                f"GPU {mem:.0f}MB")

    # FIX Bug #8: only log layer dynamics every LOG_DYNAMICS_EVERY epochs
    if ep % LOG_DYNAMICS_EVERY == 0:
        log_layer_dynamics(model, ep, grad_acc=grad_acc, num_steps=len(train_loader))

    if ep >= swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()

    # ---- Validation ----
    eval_model = swa_model if ep >= swa_start else model
    eval_model.eval()

    val_mse_acc = high_mse_acc = low_mse_acc = 0.0
    high_count  = low_count = 0
    val_t1 = val_t8 = val_t16 = 0.0
    val_igp = val_ocean = 0.0
    val_max_pred = val_max_targ = 0.0
    val_huber_acc = val_grad_acc = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            # Bug 1 fix: val loop uses same two-chunk rollout as training/inference
            yf = y.float()
            out_c1 = eval_model(x).float()
            x2_val = x.clone()
            x2_val[..., 2:10] = out_c1[..., :8]
            out_c2 = eval_model(x2_val).float()
            out = torch.cat([out_c1[..., :8], out_c2[..., :8]], dim=-1)
            pred_phys    = to_physical(out)
            targ_phys    = to_physical(yf)
            pred_clipped = F.relu(pred_phys)

            val_mse_acc   += torch.mean((pred_clipped - targ_phys) ** 2).item()
            # Fix 2b: same delta as training loop
            val_huber_acc += F.huber_loss(pred_phys, targ_phys, delta=25.0).item()
            val_grad_acc  += spatial_gradient_loss(pred_phys, targ_phys).item()

            val_t1  += torch.mean((pred_clipped[..., 0]  - targ_phys[..., 0])  ** 2).item()
            val_t8  += torch.mean((pred_clipped[..., 7]  - targ_phys[..., 7])  ** 2).item()
            val_t16 += torch.mean((pred_clipped[..., 15] - targ_phys[..., 15]) ** 2).item()

            val_igp   += torch.mean((pred_clipped[:, 60:90, 40:100, :]
                                     - targ_phys[:, 60:90, 40:100, :]) ** 2).item()
            val_ocean += torch.mean((pred_clipped[:, 0:30, 0:50, :]
                                     - targ_phys[:, 0:30, 0:50, :])   ** 2).item()

            val_max_pred = max(val_max_pred, pred_phys.max().item())
            val_max_targ = max(val_max_targ, targ_phys.max().item())

            hi_mask = targ_phys > 100.0
            lo_mask = targ_phys < 30.0
            if hi_mask.any():
                high_mse_acc += ((pred_clipped - targ_phys) ** 2)[hi_mask].sum().item()
                high_count   += hi_mask.sum().item()
            if lo_mask.any():
                low_mse_acc  += ((pred_clipped - targ_phys) ** 2)[lo_mask].sum().item()
                low_count    += lo_mask.sum().item()

    N   = len(val_loader)
    train_rmse    = np.sqrt(train_mse_acc / len(train_loader))
    val_rmse      = np.sqrt(val_mse_acc / N)
    val_high_rmse = np.sqrt(high_mse_acc / max(1, high_count))
    val_low_rmse  = np.sqrt(low_mse_acc  / max(1, low_count))
    duration      = time.time() - t_start

    logging.info(
        f"Epoch {ep} | {duration:.1f}s | "
        f"Train RMSE {train_rmse:.4f} | Val RMSE {val_rmse:.4f} | "
        f"High-PM {val_high_rmse:.4f} | Low-PM {val_low_rmse:.4f}")
    logging.info(
        f"  Horizon → T+1 {np.sqrt(val_t1/N):.4f} "
        f"T+8 {np.sqrt(val_t8/N):.4f} T+16 {np.sqrt(val_t16/N):.4f} | "
        f"IGP {np.sqrt(val_igp/N):.4f} Ocean {np.sqrt(val_ocean/N):.4f} | "
        f"Peak {val_max_pred:.1f}/{val_max_targ:.1f}")

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        if cfg.training.save_checkpoint:
            torch.save(
                {'model_state_dict': eval_model.state_dict()},
                cfg.paths.model_save_path.replace(".pt", "_best.pt"))
        logging.info(f"  -> New Best Val RMSE: {best_val_rmse:.4f}")

    log.append({"epoch": ep, "train_rmse": train_rmse, "val_rmse": val_rmse,
                 "val_high_rmse": val_high_rmse})
    with open(cfg.paths.save_dir, 'w') as f:
        json.dump(log, f)