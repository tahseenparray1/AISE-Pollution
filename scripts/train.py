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

# Set up the GPU and ensure reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

# FIX: Use keys from train.yaml (removed '_raw')
V = len(cfg.features.met_variables) + len(cfg.features.emission_variables)
S1, S2 = cfg.data.S1, cfg.data.S2 # Latitude and Longitude dimensions

# ==========================================
# 2. THE DATALOADER (On-the-Fly Windowing)
# ==========================================
class DataLoaders(torch.utils.data.Dataset):
    """
    Reads the continuous timeline from the SSD and creates overlapping
    26-hour windows instantly without wasting RAM.
    """
    def __init__(self, split, savepath_train, savepath_val):
        # Determine path based on split
        base_path = savepath_train if split == "train" else savepath_val
        
        # Load features based on the list in train.yaml
        self.met_vars = cfg.features.met_variables
        self.emi_vars = cfg.features.emission_variables
        self.all_features = self.met_vars + self.emi_vars
        
        # FIX: Ensure we load files correctly from the base_path
        self.arrs = {
            feat: np.load(os.path.join(base_path, f"{split}_{feat}.npy"), mmap_mode="r")
            for feat in self.all_features
        }
        
        self.time_in = cfg.data.time_input  # 10 hours
        self.time_out = cfg.data.time_out   # 16 hours
        self.T = self.time_in + self.time_out # 26 hours total
        
        # Use the first feature to determine total available time steps
        self.N = self.arrs[self.all_features[0]].shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Initialize an empty array for the sample: (Time, Lat, Lon, Features)
        X_full = np.empty((self.T, S1, S2, len(self.all_features)), dtype=np.float32)

        # Fill the array with data for each feature
        for i, f in enumerate(self.all_features):
            X_full[..., i] = self.arrs[f][idx, :self.T]

        # Split into Input (X) and Target (Y)
        # x: First 10 hours, all features
        x = torch.from_numpy(X_full[:self.time_in])
        
        # y: Remaining 16 hours, only PM2.5 (index 0), permuted to (Lat, Lon, Time)
        y = torch.from_numpy(X_full[self.time_in:, ..., 0]).permute(1, 2, 0)
        
        return x, y

# Initialize Datasets and Loaders
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