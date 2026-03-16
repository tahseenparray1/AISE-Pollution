import os
import torch
import numpy as np
from tqdm import tqdm
import warnings

from models.baseline_model import FNO2D
from src.utils.config import load_config

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
cfg_infer = load_config("configs/infer.yaml")
cfg_train = load_config("configs/train.yaml")

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading grid-wise normalization stats...")
stats_path = cfg_infer.paths.stats_path
stats = np.load(stats_path, allow_pickle=True).item()

pm_median = stats['cpm25']['median'].reshape(1, cfg_infer.data.S1, cfg_infer.data.S2, 1)
pm_iqr    = stats['cpm25']['iqr'].reshape(1, cfg_infer.data.S1, cfg_infer.data.S2, 1)

def denorm(x):
    return (x * pm_iqr) + pm_median

topo_proxy_path = getattr(
    cfg_infer.paths, "topo_path",
    os.path.join(os.path.dirname(cfg_infer.paths.stats_path), "topo_proxy.npy")
)
topo_proxy = np.load(topo_proxy_path)

# ==========================================
# 2. MEMORY-EFFICIENT DATA LOADER
# ==========================================
class TestDataLoader(torch.utils.data.Dataset):
    def __init__(self, cfg_infer, cfg_train, stats_dict, topo_proxy):
        self.time_in    = cfg_train.data.time_input
        self.total_time = cfg_train.data.total_time
        self.S1         = cfg_train.data.S1
        self.S2         = cfg_train.data.S2

        self.met_variables = cfg_train.features.met_variables
        self.emi_variables = cfg_train.features.emission_variables

        self.stats      = stats_dict
        self.topo_proxy = topo_proxy
        self.input_loc  = cfg_infer.paths.input_loc

        # --- FIX 1: Use memory-mapped arrays instead of loading into RAM ---
        print("Memory-mapping test arrays (no RAM load)...")
        self.arrs = {}
        for feat in self.met_variables + self.emi_variables:
            path = os.path.join(self.input_loc, f"{feat}.npy")
            self.arrs[feat] = np.load(path, mmap_mode='r')  # <-- key change

        self.N = self.arrs['cpm25'].shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Copy slice out of mmap immediately so worker doesn't hold mmap handle
        seq_raw = {}
        for feat in self.met_variables + self.emi_variables:
            if feat == "cpm25":
                seq_raw[feat] = np.array(
                    self.arrs[feat][idx, :self.time_in], dtype=np.float32)
            else:
                seq_raw[feat] = np.array(
                    self.arrs[feat][idx, :self.total_time], dtype=np.float32)

        ws = np.sqrt(seq_raw['u10']**2 + seq_raw['v10']**2)
        vc = np.log1p(ws * seq_raw['pblh'])
        rm = (seq_raw['rain'] > 0).astype(np.float32)

        seq_raw['wind_speed'] = ws
        seq_raw['vent_coef']  = vc
        seq_raw['rain_mask']  = rm

        skewed_features = ['rain', 'bio', 'NMVOC_finn', 'pblh']
        for feat in skewed_features:
            seq_raw[feat] = np.log1p(seq_raw[feat])

        temporal_list = (
            [f for f in self.met_variables if f != 'cpm25']
            + cfg_train.features.derived_variables
        )

        temporal_feats = []
        for feat in temporal_list:
            arr = (seq_raw[feat] - self.stats[feat]['median']) / self.stats[feat]['iqr']
            temporal_feats.append(arr)

        static_feats = []
        for feat in self.emi_variables:
            arr = (seq_raw[feat] - self.stats[feat]['median']) / self.stats[feat]['iqr']
            static_feats.append(arr)

        pm25_hist = (seq_raw['cpm25'] - self.stats['cpm25']['median']) / self.stats['cpm25']['iqr']
        pm25_hist = torch.from_numpy(pm25_hist).permute(1, 2, 0)

        temporal_stack  = np.stack(temporal_feats, axis=0)
        temporal_tensor = torch.from_numpy(temporal_stack).permute(2, 3, 1, 0).reshape(self.S1, self.S2, -1)

        static_stack      = np.stack(static_feats, axis=0)
        static_tensor_raw = torch.from_numpy(static_stack)
        static_tensor_min  = static_tensor_raw.min(dim=1)[0].permute(1, 2, 0)
        static_tensor_mean = static_tensor_raw.mean(dim=1).permute(1, 2, 0)
        static_tensor_max  = static_tensor_raw.max(dim=1)[0].permute(1, 2, 0)
        static_tensor = torch.cat([static_tensor_min, static_tensor_mean, static_tensor_max], dim=-1)

        topo_tensor = torch.from_numpy(self.topo_proxy).unsqueeze(-1)

        x = torch.cat((pm25_hist, temporal_tensor, static_tensor, topo_tensor), dim=-1)
        return x


test_dataset = TestDataLoader(cfg_infer, cfg_train, stats, topo_proxy)

# --- FIX 2: Reduce batch_size and num_workers ---
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=2,       # was 4 — halves peak RAM per batch
    shuffle=False,
    num_workers=2,      # was 4 — each worker holds its own mmap slice
    pin_memory=False,   # disable to avoid extra GPU-pinned RAM
)

# ==========================================
# 3. MODEL INITIALIZATION
# ==========================================
pm_channels = cfg_train.data.time_input
temporal_features_count = len(
    [f for f in cfg_train.features.met_variables if f != 'cpm25']
    + cfg_train.features.derived_variables
)
temporal_channels = temporal_features_count * cfg_train.data.total_time
static_channels   = len(cfg_train.features.emission_variables) * 3
topo_channels     = 1

in_channels = pm_channels + temporal_channels + static_channels + topo_channels

print(f"Building WNO Model with optimized {in_channels} input channels...")
model = FNO2D(
    in_channels=in_channels,
    time_out=cfg_train.data.time_out,
    width=cfg_train.model.width,
    modes=cfg_train.model.modes,
    time_input=cfg_train.data.time_input,
    total_time=cfg_train.data.total_time,
    num_temporal_features=temporal_features_count,
).to(device)

checkpoint_path = cfg_infer.paths.checkpoint
print(f"Loading best weights from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)

clean_state_dict = {
    k.replace('module.', ''): v
    for k, v in checkpoint['model_state_dict'].items()
    if k != 'n_averaged'
}
model.load_state_dict(clean_state_dict)
model.eval()

# ==========================================
# 4. INFERENCE LOOP — CHUNKED DISK WRITE
# ==========================================
os.makedirs(cfg_infer.paths.output_loc, exist_ok=True)
out_file   = os.path.join(cfg_infer.paths.output_loc, 'preds.npy')
total_N    = len(test_dataset)
time_out   = cfg_train.data.time_out

# --- FIX 3: Write predictions to a memmap on disk, never hold full array in RAM ---
pred_mmap = np.lib.format.open_memmap(
    out_file,
    mode='w+',
    dtype=np.float32,
    shape=(total_N, cfg_train.data.S1, cfg_train.data.S2, time_out),
)

print("Starting inference...")
current_idx = 0

with torch.no_grad():
    for x in tqdm(test_loader):
        x   = x.to(device, non_blocking=True)
        bs  = x.size(0)

        out     = model(x)
        out_cpu = denorm(out.cpu().numpy())

        pred_mmap[current_idx : current_idx + bs] = out_cpu
        current_idx += bs

        # --- FIX 4: Flush every 50 batches so OS doesn't buffer too much ---
        if current_idx % (50 * test_loader.batch_size) == 0:
            pred_mmap.flush()

pred_mmap.flush()
print(f"Success! Predictions saved to: {out_file}")
print(f"Final array shape: {pred_mmap.shape}")