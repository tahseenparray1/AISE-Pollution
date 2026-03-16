import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from src.utils.adam import Adam
from src.utils.config import load_config
from models.baseline_model import FNO2D
from torch.optim.swa_utils import AveragedModel, SWALR 

cfg = load_config("configs/train.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S1, S2 = cfg.data.S1, cfg.data.S2

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

horizon_weights = torch.linspace(1.0, 1.5, cfg.data.time_out).to(device)
horizon_weights = horizon_weights / horizon_weights.mean()
horizon_weights = horizon_weights.view(1, 1, 1, -1)

class FastInMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        base_path = cfg.paths.savepath_train
        print("Loading 100% Train Data into RAM...")
        self.data = torch.from_numpy(np.load(os.path.join(base_path, "train_data.npy")).astype(np.float32))
        self.valid_starts = np.load(os.path.join(base_path, "train_indices.npy"))
        
        self.time_in = cfg.data.time_input
        self.total_time = cfg.data.total_time
        self.S1, self.S2 = cfg.data.S1, cfg.data.S2
        
        all_features = (cfg.features.met_variables + cfg.features.emission_variables + cfg.features.derived_variables)
        self.target_idx = all_features.index('cpm25')
        self.temporal_idx = [i for i, f in enumerate(all_features) if f != 'cpm25']
        self.topo_idx = len(all_features)

    def __len__(self): 
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        window = self.data[start : start + self.total_time].clone() 
        
        pm_hist = window[:self.time_in, ..., self.target_idx].permute(1, 2, 0) 
        temporal_all = window[:, ..., self.temporal_idx]
        temporal_tensor = temporal_all.permute(1, 2, 0, 3).reshape(self.S1, self.S2, -1)
        topo = window[0, ..., self.topo_idx].unsqueeze(-1)
        
        x = torch.cat((pm_hist, temporal_tensor, topo), dim=-1)
        y = window[self.time_in:, ..., self.target_idx].permute(1, 2, 0)
        return x, y

train_ds = FastInMemoryDataset(cfg)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4, pin_memory=True)

pm_channels = cfg.data.time_input
temporal_channels = len(train_ds.temporal_idx) * cfg.data.total_time
topo_channels = 1
in_channels = pm_channels + temporal_channels + topo_channels

# --- MULTI-SEED ENSEMBLE TRAINING LOOP ---
SEEDS = [0, 42, 2026] # 3 models for ensembling

for seed in SEEDS:
    print(f"\n{'='*40}")
    print(f"STARTING TRAINING FOR SEED: {seed}")
    print(f"{'='*40}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FNO2D(
        in_channels=in_channels, 
        time_out=cfg.data.time_out, 
        width=cfg.model.width, 
        modes=cfg.model.modes,
        time_input=cfg.data.time_input
    ).to(device)

    optimizer = Adam(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)

    swa_model = AveragedModel(model)
    swa_start = int(cfg.training.epochs * 0.75)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-4)

    for ep in range(cfg.training.epochs):
        model.train()
        t_start = time.time()
        train_mse_acc = 0.0

        for x, y in tqdm(train_loader, desc=f"Seed {seed} | Epoch {ep}"):
            x, y = x.to(device), y.to(device)
            
            noise = torch.randn_like(x) * 0.01
            noise[..., -1] = 0.0  
            x = x + noise
            
            optimizer.zero_grad(set_to_none=True)
            
            out = model(x)
            pred_phys = to_physical(out)
            targ_phys = to_physical(y)
            
            mse_loss = ((pred_phys - targ_phys) ** 2 * horizon_weights).mean()
            loss_grad = spatial_gradient_loss(pred_phys, targ_phys)
            total_loss = mse_loss + 0.1 * loss_grad

            with torch.no_grad():
                pred_phys_clipped = F.relu(pred_phys.detach())
                train_mse_acc += torch.mean((pred_phys_clipped - targ_phys) ** 2).item()
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if ep >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        train_rmse = np.sqrt(train_mse_acc / len(train_loader))
        duration = time.time() - t_start
        print(f"Epoch {ep} | Time: {duration:.1f}s | Train RMSE: {train_rmse:.4f}")

    # Save the final SWA model for this seed
    save_path = cfg.paths.model_save_path.replace(".pt", f"_seed{seed}.pt")
    torch.save({'model_state_dict': swa_model.state_dict()}, save_path)
    print(f"Saved ensemble member to: {save_path}")