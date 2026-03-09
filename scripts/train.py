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

cfg = load_config("configs/train.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

S1, S2 = cfg.data.S1, cfg.data.S2
os.makedirs(os.path.dirname(cfg.paths.save_dir), exist_ok=True)
os.makedirs(os.path.dirname(cfg.paths.model_save_path), exist_ok=True)
print("Loading robust grid-wise stats...")
stats = np.load(cfg.paths.stats_path, allow_pickle=True).item()

pm_median_tensor = torch.tensor(stats['cpm25']['median'], dtype=torch.float32).to(device).view(1, S1, S2, 1)
pm_iqr_tensor = torch.tensor(stats['cpm25']['iqr'], dtype=torch.float32).to(device).view(1, S1, S2, 1)

def to_physical(x_norm):
    """ Reverses the robust scaling, then reverses the Log1p transform """
    log_space = (x_norm * pm_iqr_tensor) + pm_median_tensor
    # 🚀 torch.expm1 reverses the np.log1p applied in prep
    physical = torch.expm1(log_space) 
    return F.relu(physical) # Ensure no negative PM2.5

class FastInMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, split, cfg):
        base_path = getattr(cfg.paths, f"savepath_{split}")
        print(f"[{split.upper()}] Loading into RAM...")
        self.data = torch.from_numpy(np.load(os.path.join(base_path, f"{split}_data.npy")).astype(np.float32))
        self.valid_starts = np.load(os.path.join(base_path, f"{split}_indices.npy"))
        self.time_in = cfg.data.time_input
        self.total_time = cfg.data.total_time
        self.S1, self.S2 = cfg.data.S1, cfg.data.S2

    def __len__(self): 
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        window = self.data[start : start + self.total_time].clone() 
        
        # Keep ALL features in the temporal loop (fixes the Dynamic Emission bug)
        pm_hist = window[:self.time_in, ..., 0].permute(1, 2, 0) 
        weather = window[:, ..., 1:].permute(1, 2, 0, 3).reshape(self.S1, self.S2, -1)
        x = torch.cat((pm_hist, weather), dim=-1)
        
        # y is now naturally normalized log-space PM2.5
        y = window[self.time_in:, ..., 0].permute(1, 2, 0)
        return x, y

train_ds = FastInMemoryDataset("train", cfg)
val_ds = FastInMemoryDataset("val", cfg)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=2, pin_memory=True)

# 10 pm_hist + (19 features * 26 hours) + 26 Topo = 530 channels
in_channels = 10 + (19 * 26) 
print(f"Building Model with {in_channels} input channels...")

model = FNO2D(in_channels=in_channels, time_out=cfg.data.time_out, width=cfg.model.width, modes=cfg.model.modes).to(device)
optimizer = Adam(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)

swa_model = AveragedModel(model)
swa_start = int(cfg.training.epochs * 0.75)
swa_scheduler = SWALR(optimizer, swa_lr=5e-4)

best_val_rmse = float('inf')
log = []

for ep in range(cfg.training.epochs):
    model.train()
    t_start = time.time()
    train_mse_acc = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {ep}"):
        x, y = x.to(device), y.to(device)
        x = x + torch.randn_like(x) * 0.01 # Input noise regularization
        
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        
        # 🚀 Optimize directly in normalized log-space. 
        # This completely neutralizes the 4,000,000 extreme MSE penalties.
        loss = F.mse_loss(out, y)
        
        # Track physical Kaggle metric silently
        with torch.no_grad():
            pred_phys = to_physical(out)
            targ_phys = to_physical(y)
            train_mse_acc += torch.mean((pred_phys - targ_phys) ** 2).item()
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    if ep >= swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()

    eval_model = swa_model if ep >= swa_start else model
    eval_model.eval()
    val_mse_acc = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = eval_model(x) 
            
            # Evaluate using Kaggle's Physical RMSE Metric
            pred_phys = to_physical(out)
            targ_phys = to_physical(y)
            val_mse_acc += torch.mean((pred_phys - targ_phys) ** 2).item()

    train_rmse = np.sqrt(train_mse_acc / len(train_loader))
    val_rmse = np.sqrt(val_mse_acc / len(val_loader))
    
    duration = time.time() - t_start
    print(f"Epoch {ep} | Time: {duration:.1f}s | Train Phys RMSE: {train_rmse:.4f} | Val Phys RMSE: {val_rmse:.4f}")

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        if cfg.training.save_checkpoint:
            torch.save({'model_state_dict': eval_model.state_dict()}, cfg.paths.model_save_path.replace(".pt", "_best.pt"))
        print(f"  -> New Best Val RMSE: {best_val_rmse:.4f}")

    log.append({"epoch": ep, "train_rmse": train_rmse, "val_rmse": val_rmse})
    with open(cfg.paths.save_dir, 'w') as f: json.dump(log, f)