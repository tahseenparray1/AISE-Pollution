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
# 2. LOAD SPATIAL STATS FOR TRUE RMSE
# ==========================================
print("Loading grid-wise normalization stats for true RMSE loss...")
stats = np.load('/kaggle/working/grid_robust_stats.npy', allow_pickle=True).item()

# Extract the IQR for the target variable (cpm25)
pm_iqr_np = stats['cpm25']['iqr'].reshape(1, S1, S2, 1)
pm_iqr_tensor = torch.tensor(pm_iqr_np, dtype=torch.float32).to(device)

class ExactRMSELoss(nn.Module):
    """ 
    Calculates the exact Average Domain RMSE in the original physical space (ug/m3)
    by scaling the error residuals by the grid-wise IQR. 
    """
    def __init__(self, iqr_tensor, eps=1e-8):
        super().__init__()
        self.iqr = iqr_tensor
        self.eps = eps
        
    def forward(self, pred, target):
        real_diff = (pred - target) * self.iqr
        mse = torch.mean(real_diff ** 2)
        return torch.sqrt(mse + self.eps)

myloss = ExactRMSELoss(pm_iqr_tensor)

# ==========================================
# 3. FAST IN-MEMORY DATALOADER 
# ==========================================
class FastInMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, split, savepath_train, savepath_val):
        base_path = savepath_train if split == "train" else savepath_val
        file_name = "train_data.npy" if split == "train" else "val_data.npy"
        file_path = os.path.join(base_path, file_name)
        
        print(f"[{split.upper()}] Loading entire dataset into RAM from {file_path}...")
        
        # 1. Load fully into RAM (no mmap_mode="r"). This avoids slow Kaggle SSD random reads.
        raw_data = np.load(file_path).astype(np.float32)
        
        # 2. Convert to PyTorch Tensor immediately. 
        # This allows num_workers to read from Shared Memory without RAM duplication!
        self.data = torch.from_numpy(raw_data)
        
        ram_usage = (self.data.element_size() * self.data.nelement()) / (1024 ** 3)
        print(f"[{split.upper()}] Loaded into Shared Memory: {self.data.shape} ({ram_usage:.2f} GB)")
        
        self.time_in = cfg.data.time_input 
        self.time_out = cfg.data.time_out   
        self.window_size = self.time_in + self.time_out 
        self.stride = getattr(cfg.data, 'stride', 1) 
        
        self.n_samples = (self.data.shape[0] - self.window_size) // self.stride + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        # Slice directly from the in-memory tensor (0 disk I/O)
        window = self.data[start_idx:end_idx]
        
        x = window[:self.time_in]
        y = window[self.time_in:, ..., 0].permute(1, 2, 0)
        
        return x, y

train_dataset = FastInMemoryDataset("train", cfg.paths.savepath_train, cfg.paths.savepath_val)
test_dataset  = FastInMemoryDataset("val",   cfg.paths.savepath_train, cfg.paths.savepath_val)

batch_size = cfg.training.batch_size 

# Dataloader is now CPU-bound instead of I/O bound.
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True,
    persistent_workers=True 
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True,
    persistent_workers=True
)

# ==========================================
# 4. MODEL & OPTIMIZER
# ==========================================
print(f"Building FNO2D with {V} features...")
model = FNO2D(
    time_in=cfg.data.time_input,
    features=V,
    time_out=cfg.data.time_out,
    width=cfg.model.width,
    modes=cfg.model.modes,
).to(device)

optimizer = Adam(
    model.parameters(), 
    lr=float(cfg.training.lr), 
    weight_decay=float(cfg.training.weight_decay)
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=cfg.training.scheduler_step, 
    gamma=cfg.training.scheduler_gamma
)

os.makedirs(os.path.dirname(cfg.paths.save_dir), exist_ok=True)
os.makedirs(os.path.dirname(cfg.paths.model_save_path), exist_ok=True)

# ==========================================
# 5. THE TRAINING LOOP
# ==========================================
epochs = cfg.training.epochs 
log = []

print("\nStarting Training...")
for ep in range(epochs):
    model.train()
    t_epoch_start = time.time()
    train_rmse = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {ep}/{epochs-1}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        out = model(x).view(x.size(0), S1, S2, cfg.data.time_out)
        
        loss = myloss(out, y)
        loss.backward()
        optimizer.step()
        
        train_rmse += loss.item()

    scheduler.step()

    # VALIDATION
    model.eval()
    test_rmse = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            out = model(x).view(x.size(0), S1, S2, cfg.data.time_out)
            test_rmse += myloss(out, y).item()

    train_rmse /= len(train_loader)
    test_rmse  /= len(test_loader)

    epoch_duration = time.time() - t_epoch_start
    log.append({
        "epoch": ep,
        "duration": epoch_duration,
        "train_rmse": train_rmse,
        "val_rmse": test_rmse
    })

    print(f"Epoch {ep} | Time: {epoch_duration:.1f}s | Train RMSE: {train_rmse:.4f} | Val RMSE: {test_rmse:.4f}")

    # CHECKPOINTING
    if (ep + 1) % cfg.training.checkpoint_every == 0:
        ckpt_path = cfg.paths.model_save_path.replace(".pt", f"_ep{ep}.pt")
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, ckpt_path)
        
        with open(cfg.paths.save_dir, "w") as f:
            json.dump(log, f, indent=4)

print("\nTraining Complete!")