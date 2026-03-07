import os
import torch
import numpy as np
from tqdm import tqdm
import warnings
import glob
from models.baseline_model import FNO2D
from src.utils.config import load_config

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
cfg = load_config("configs/infer.yaml")

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------
# Load Grid-Wise Robust Stats 
# -----------------------
print("Loading grid-wise normalization stats...")
stats = np.load('/kaggle/working/grid_robust_stats.npy', allow_pickle=True).item()

# Extract target (cpm25) stats and reshape for broadcasting over (batch, S1, S2, time_out)
pm_median = stats['cpm25']['median'].reshape(1, cfg.data.S1, cfg.data.S2, 1)
pm_iqr = stats['cpm25']['iqr'].reshape(1, cfg.data.S1, cfg.data.S2, 1)

def denorm(x):
    """ Inverse the robust scaling: X_raw = (X_scaled * IQR) + Median """
    return (x * pm_iqr) + pm_median

# ==========================================
# 2. DATA LOADER
# ==========================================
class TestDataLoader(torch.utils.data.Dataset):
    def __init__(self, cfg, stats_dict):
        self.time_in = cfg.data.time_input    # 10
        self.total_time = cfg.data.total_time # 26
        self.S1 = cfg.data.S1
        self.S2 = cfg.data.S2
        
        self.met_variables = cfg.features.met_variables
        self.emi_variables = cfg.features.emission_variables
        self.all_features = self.met_variables + self.emi_variables
        self.stats = stats_dict

        # Load all 16 files lazily
        self.arrs = {}
        for feat in self.all_features:
            path = os.path.join(cfg.paths.input_loc, f"{feat}.npy")
            self.arrs[feat] = np.load(path, mmap_mode="r")

        # Get total number of samples (996)
        self.N = self.arrs[self.all_features[0]].shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        pm25_hist = None
        other_feats_list = []

        for feat in self.all_features:
            # 1. Handle the temporal asymmetry of the Kaggle test set
            if feat == "cpm25":
                # Only 10 hours available for PM2.5 in test_in
                arr = np.array(self.arrs[feat][idx, :self.time_in], dtype=np.float32)
            else:
                # Full 26 hours available for weather/emissions in test_in
                arr = np.array(self.arrs[feat][idx, :self.total_time], dtype=np.float32)
            
            # 2. Grid-wise Normalize
            f_median = self.stats[feat]['median']
            f_iqr = self.stats[feat]['iqr']
            arr = (arr - f_median) / f_iqr
            
            # 3. Permute Time to the last dimension
            if feat == "cpm25":
                # (10, 140, 124) -> (140, 124, 10)
                pm25_hist = torch.from_numpy(arr).permute(1, 2, 0)
            else:
                # (26, 140, 124) -> (140, 124, 26)
                arr_t = torch.from_numpy(arr).permute(1, 2, 0)
                other_feats_list.append(arr_t)

        # 4. Stack and reshape the other features
        # 15 features of shape (140, 124, 26) -> (140, 124, 15, 26) -> (140, 124, 390)
        other_feats = torch.stack(other_feats_list, dim=-1)
        other_feats = other_feats.reshape(self.S1, self.S2, -1)
        
        # 5. Concatenate PM2.5 (10) with others (390) to create 400 input channels
        x = torch.cat((pm25_hist, other_feats), dim=-1)

        return x

test_dataset = TestDataLoader(cfg, stats)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=4, 
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ==========================================
# 3. MODEL INITIALIZATION
# ==========================================
V = len(cfg.features.met_variables) + len(cfg.features.emission_variables)
in_channels = cfg.data.time_input + (V - 1) * cfg.data.total_time 

print(f"Building FNO2D Model with {in_channels} input channels...")
model = FNO2D(
    in_channels=in_channels,
    time_out=cfg.data.time_out,
    width=cfg.model.width,
    modes=12,  # Ensure this matches the updated training parameter to prevent overfitting!
).to(device)

print(f"Loading weights from: {cfg.paths.checkpoint}")
checkpoint_path = cfg.paths.checkpoint

if not os.path.exists(checkpoint_path):
    print(f"Warning: Specific checkpoint {checkpoint_path} not found.")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    available_ckpts = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    
    if available_ckpts:
        checkpoint_path = max(available_ckpts, key=os.path.getmtime)
        print(f"Found alternative: Loading latest weights from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No .pt files found in {checkpoint_dir}. Did training save anything?")

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ==========================================
# 4. INFERENCE LOOP
# ==========================================
prediction = np.zeros((len(test_dataset), cfg.data.S1, cfg.data.S2, cfg.data.time_out), dtype=np.float32)

print("Starting inference...")
current_idx = 0

with torch.no_grad():
    for x in tqdm(test_loader):
        x = x.to(device, non_blocking=True)
        bs = x.size(0)
        
        # Forward pass (model now outputs exactly batch, nx, ny, time_out)
        out = model(x)
        
        # Un-scale the data back to true PM2.5 levels
        out_real = denorm(out.cpu().numpy())
        
        # Insert into the final prediction array
        prediction[current_idx : current_idx + bs] = out_real
        current_idx += bs

# ==========================================
# 5. EXPORT PREDICTIONS
# ==========================================
os.makedirs(cfg.paths.output_loc, exist_ok=True)
out_file = os.path.join(cfg.paths.output_loc, 'preds.npy')

np.save(out_file, prediction)
print(f"Success! Predictions saved to: {out_file}")
print(f"Final array shape: {prediction.shape}")