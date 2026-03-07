import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from src.utils.adam import Adam
from models.baseline_model import FNO2D
from src.utils.config import load_config

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
cfg = load_config("configs/train.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

V = len(cfg.features.met_variables) + len(cfg.features.emission_variables)
S1, S2 = cfg.data.S1, cfg.data.S2

# ==========================================
# 2. LOSS FUNCTIONS & METRICS
# ==========================================
print("Loading grid-wise normalization stats for true RMSE tracking...")
stats = np.load('/kaggle/working/grid_robust_stats.npy', allow_pickle=True).item()

pm_iqr_np = stats['cpm25']['iqr'].reshape(1, S1, S2, 1)
pm_iqr_tensor = torch.tensor(pm_iqr_np, dtype=torch.float32).to(device)

class ExactMSELoss(nn.Module):
    """Calculates the Average Domain MSE in the original physical space."""
    def __init__(self, iqr_tensor):
        super().__init__()
        self.iqr = iqr_tensor
        
    def forward(self, pred, target):
        real_diff = (pred - target) * self.iqr
        return torch.mean(real_diff ** 2)

# Notice we track MSE now, not RMSE!
metric_loss = ExactMSELoss(pm_iqr_tensor)

# ==========================================
# 3. FAST IN-MEMORY DATALOADER 
# ==========================================
class FastInMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, split, savepath_train, savepath_val):
        base_path = savepath_train if split == "train" else savepath_val
        file_name = "train_data.npy" if split == "train" else "val_data.npy"
        file_path = os.path.join(base_path, file_name)
        
        print(f"[{split.upper()}] Loading dataset into RAM from {file_path}...")
        raw_data = np.load(file_path).astype(np.float32)
        self.data = torch.from_numpy(raw_data)
        
        ram_usage = (self.data.element_size() * self.data.nelement()) / (1024 ** 3)
        print(f"[{split.upper()}] Shared Memory Footprint: {self.data.shape} ({ram_usage:.2f} GB)")
        
        self.time_in = cfg.data.time_input 
        self.time_out = cfg.data.time_out   
        self.total_time = cfg.data.total_time
        self.window_size = getattr(cfg.data, 'horizon', 26) 
        
        self.stride = getattr(cfg.data, 'stride', 1) 
        
        self.n_samples = (self.data.shape[0] - self.window_size) // self.stride + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        window = self.data[start_idx:end_idx]
        
        # PM2.5 History (First 10 hours)
        pm25_hist = window[:self.time_in, ..., 0]   
        pm25_hist = pm25_hist.permute(1, 2, 0)      
        
        # Future Weather Forcing 
        other_feats = window[:, ..., 1:]            
        other_feats = other_feats.permute(1, 2, 0, 3).reshape(140, 124, -1) 
        
        x = torch.cat((pm25_hist, other_feats), dim=-1)
        
        # Target (Future 16 hours of PM2.5)
        y = window[self.time_in:, ..., 0].permute(1, 2, 0) 
        
        return x, y

train_dataset = FastInMemoryDataset("train", cfg.paths.savepath_train, cfg.paths.savepath_val)
test_dataset  = FastInMemoryDataset("val",   cfg.paths.savepath_train, cfg.paths.savepath_val)

batch_size = cfg.training.batch_size 

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, 
    num_workers=1, pin_memory=True, persistent_workers=True 
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, 
    num_workers=1, pin_memory=True, persistent_workers=True
)

# ==========================================
# 4. MODEL & OPTIMIZER
# ==========================================
in_channels = cfg.data.time_input + (V - 1) * cfg.data.total_time 

print(f"Building Dual-Encoder FNO2D with Geographic Injection...")
model = FNO2D(
    in_channels=in_channels,
    time_out=cfg.data.time_out,
    width=cfg.model.width,
    modes=cfg.model.modes, 
).to(device)

optimizer = Adam(
    model.parameters(), 
    lr=float(cfg.training.lr), 
    weight_decay=float(cfg.training.weight_decay)
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=cfg.training.epochs, 
    eta_min=1e-6
)

os.makedirs(os.path.dirname(cfg.paths.save_dir), exist_ok=True)
os.makedirs(os.path.dirname(cfg.paths.model_save_path), exist_ok=True)

# ==========================================
# 5. THE TRAINING LOOP
# ==========================================
epochs = cfg.training.epochs 
log = []

best_val_rmse = float('inf') 
patience = 6                
epochs_without_improvement = 0

print("\nStarting Training...")
for ep in range(epochs):
    model.train()
    t_epoch_start = time.time()
    
    train_mse_total = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {ep}/{epochs-1}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # Input Jitter Regularization
        if model.training:
            noise = torch.randn_like(x) * 0.02
            x = x + noise
            
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        
        # --- NEW: PHYSICAL SPACE LOSS (HOTSPOT FIX) ---
        # The model is now penalized heavily for missing large pollution spikes
        real_diff = (out - y) * pm_iqr_tensor
        loss = torch.mean(real_diff ** 2)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate MSE
        train_mse_total += metric_loss(out.detach(), y).item()

    scheduler.step()

    # VALIDATION
    model.eval()
    test_mse_total = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            out = model(x)
            test_mse_total += metric_loss(out, y).item()

    # --- JENSEN'S INEQUALITY FIX ---
    # Average the MSE across batches, then take the square root to get true RMSE
    train_rmse_metric = np.sqrt(train_mse_total / len(train_loader))
    test_rmse_metric  = np.sqrt(test_mse_total / len(test_loader))

    epoch_duration = time.time() - t_epoch_start
    log.append({
        "epoch": ep,
        "duration": epoch_duration,
        "train_rmse": train_rmse_metric,
        "val_rmse": test_rmse_metric
    })

    print(f"Epoch {ep} | Time: {epoch_duration:.1f}s | Train RMSE: {train_rmse_metric:.4f} | Val RMSE: {test_rmse_metric:.4f}")

    # ==========================================
    # SMART CHECKPOINTING & EARLY STOPPING
    # ==========================================
    save_ckpt = getattr(cfg.training, 'save_checkpoint', True)
    use_early_stop = getattr(cfg.training, 'early_stopping', True)

    if test_rmse_metric < best_val_rmse:
        best_val_rmse = test_rmse_metric
        epochs_without_improvement = 0
        
        if save_ckpt:
            ckpt_path = cfg.paths.model_save_path.replace(".pt", f"_best.pt")
            print(f"  -> New best validation score! Saving model to {ckpt_path}")
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_rmse': best_val_rmse
            }, ckpt_path)
        else:
            print(f"  -> New best validation score! (Checkpoint saving is disabled in config)")
    else:
        epochs_without_improvement += 1
        print(f"  -> No improvement for {epochs_without_improvement} epoch(s).")
        
    with open(cfg.paths.save_dir, "w") as f:
        json.dump(log, f, indent=4)
        
    if use_early_stop and epochs_without_improvement >= patience:
        print(f"\nEarly stopping triggered at epoch {ep}! Validation RMSE hasn't improved in {patience} epochs.")
        break

print("\nTraining Complete!")