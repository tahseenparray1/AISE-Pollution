import os
import time
import json
import torch
import numpy as np
from tqdm import tqdm
from src.utils.utilities3 import LpLoss
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
S1, S2 = cfg.data.S1, cfg.data.S2 # Pulling grid size directly from config

# ==========================================
# 2. THE DATALOADER 
# ==========================================
class DataLoaders(torch.utils.data.Dataset):
    def __init__(self, split, savepath_train, savepath_val):
        # Determine paths based on split
        base_path = savepath_train if split == "train" else savepath_val
        file_name = "train_data.npy" if split == "train" else "val_data.npy"
        
        # Load lazily using mmap_mode
        self.data = np.load(os.path.join(base_path, file_name), mmap_mode="r")
        
        # Setup dimensions and sliding window logic
        self.time_in = cfg.data.time_input 
        self.time_out = cfg.data.time_out   
        self.window_size = self.time_in + self.time_out 
        self.stride = getattr(cfg.data, 'stride', 1) 
        
        # Calculate total valid windows
        self.n_samples = (self.data.shape[0] - self.window_size) // self.stride + 1

    def __len__(self):
        # PyTorch needs this to know how many batches to make
        return self.n_samples

    def __getitem__(self, idx):
        # PyTorch uses this to grab a specific window
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        # Slice the window from the SSD
        window = self.data[start_idx:end_idx].astype(np.float32)
        
        # Split into Input (10 hours, all features)
        x = torch.from_numpy(window[:self.time_in])
        
        # Split into Target (16 hours, PM2.5 only, reshaped to Lat/Lon/Time)
        y = torch.from_numpy(window[self.time_in:, ..., 0]).permute(1, 2, 0)
        
        return x, y

# FIXED: Pass the exact path names from your YAML
train_dataset = DataLoaders("train", cfg.paths.savepath_train, cfg.paths.savepath_val)
test_dataset  = DataLoaders("val",   cfg.paths.savepath_train, cfg.paths.savepath_val)

batch_size = cfg.training.batch_size #

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True,
    persistent_workers=True # Added for cluster speed
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
# 3. MODEL, OPTIMIZER, & LOSS
# ==========================================
print(f"Building FNO2D with {V} features...")
model = FNO2D(
    time_in=cfg.data.time_input,
    features=V,
    time_out=cfg.data.time_out,
    width=cfg.model.width,
    modes=cfg.model.modes,
).to(device)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_params(model)}")

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

myloss = LpLoss(size_average=False)

# Setup directories
os.makedirs(os.path.dirname(cfg.paths.save_dir), exist_ok=True)
os.makedirs(os.path.dirname(cfg.paths.model_save_path), exist_ok=True)

# ==========================================
# 4. THE TRAINING LOOP
# ==========================================
epochs = cfg.training.epochs #
log = []

print("\nStarting Training...")
for ep in range(epochs):
    model.train()
    t_epoch_start = time.time()
    train_l2 = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {ep}/{epochs-1}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass and reshape
        out = model(x).view(x.size(0), S1, S2, cfg.data.time_out)
        
        l2 = myloss(out, y)
        l2.backward()
        optimizer.step()
        
        train_l2 += l2.item()

    scheduler.step()

    # VALIDATION
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x).view(x.size(0), S1, S2, cfg.data.time_out)
            test_l2 += myloss(out, y).item()

    train_l2 /= len(train_dataset)
    test_l2  /= len(test_dataset)

    epoch_duration = time.time() - t_epoch_start
    log.append({
        "epoch": ep,
        "duration": epoch_duration,
        "train_data_loss": train_l2,
        "val_data_loss": test_l2
    })

    print(f"Epoch {ep} | Time: {epoch_duration:.1f}s | Train Loss: {train_l2:.4f} | Val Loss: {test_l2:.4f}")

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