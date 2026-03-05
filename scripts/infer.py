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
# Original shape is (140, 124), target shape needs to be (1, 140, 124, 1)
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
        self.time_in = cfg.data.time_input
        self.time_out = cfg.data.time_out
        self.S1 = cfg.data.S1
        self.S2 = cfg.data.S2
        
        self.met_variables = cfg.features.met_variables
        self.emi_variables = cfg.features.emission_variables
        self.all_features = self.met_variables + self.emi_variables
        self.V = cfg.features.V
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
        # Create an empty block for the 10-hour history: (10, 140, 124, 16)
        x = np.empty((self.time_in, self.S1, self.S2, self.V), dtype=np.float32)

        for c, feat in enumerate(self.all_features):
            # Grab the 10 hours for this specific sample and feature
            # Shape is (10, 140, 124)
            arr = np.array(self.arrs[feat][idx, :self.time_in], dtype=np.float32)
            
            # --- Grid-wise Normalize ---
            # numpy automatically broadcasts the (140, 124) stats against the (10, 140, 124) array
            f_median = self.stats[feat]['median']
            f_iqr = self.stats[feat]['iqr']
            
            arr = (arr - f_median) / f_iqr
            x[..., c] = arr

        return torch.from_numpy(x)

test_dataset = TestDataLoader(cfg, stats)

# Batch size of 4 or 8 is safe here since we don't store gradients
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
print(f"Building FNO2D Model...")
model = FNO2D(
    time_in=cfg.data.time_input,
    features=cfg.features.V,
    time_out=cfg.data.time_out,
    width=cfg.model.width,
    modes=cfg.model.modes,
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

checkpoint = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # Switch to evaluation mode

# ==========================================
# 4. INFERENCE LOOP
# ==========================================
# Prepare the final array to hold predictions (996, 140, 124, 16)
prediction = np.zeros((len(test_dataset), cfg.data.S1, cfg.data.S2, cfg.data.time_out), dtype=np.float32)

print("Starting inference...")
current_idx = 0

with torch.no_grad():
    for x in tqdm(test_loader):
        x = x.to(device, non_blocking=True)
        bs = x.size(0) # Get current batch size
        
        # Forward pass and reshape
        out = model(x).view(bs, cfg.data.S1, cfg.data.S2, cfg.data.time_out)
        
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