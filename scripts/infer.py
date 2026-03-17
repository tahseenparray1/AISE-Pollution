"""
infer.py  (V2 — fixed)

Changes vs original V2:
  FIX (Bug #2): Predictions were saved without non-negativity clipping.
                PM2.5 is physically non-negative; negative predictions
                corrupt the submission.  Added np.clip(out_real, 0, None).
  FIX (Bug #4): Emission scaling (* 1e11 before log1p) restored for
                PM25, NH3, SO2, NOx, NMVOC_e — matching the fixed
                prepare_dataset.py so train/infer normalization is
                consistent.
  FIX (Bug #7): pin_memory=True added to test DataLoader.
"""

import os
import warnings

import torch
import numpy as np
from tqdm import tqdm

from models.baseline_model import FNO2D
from src.utils.config import load_config

warnings.filterwarnings("ignore")

# ============================================================
# 1. SETUP
# ============================================================
cfg_infer = load_config("configs/infer.yaml")
cfg_train = load_config("configs/train.yaml")

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# FIX Bug #4: same set as prepare_dataset.py fixed version
EMI_SCALE_VARS   = {"PM25", "NH3", "SO2", "NOx", "NMVOC_e"}
EMI_LOGONLY_VARS = {"bio", "NMVOC_finn"}

# ============================================================
# 2. NORMALISATION STATS
# ============================================================
print("Loading grid-wise normalisation stats...")
stats = np.load(cfg_infer.paths.stats_path, allow_pickle=True).item()

pm_median = stats['cpm25']['median'].reshape(
    1, cfg_infer.data.S1, cfg_infer.data.S2, 1)
pm_iqr = stats['cpm25']['iqr'].reshape(
    1, cfg_infer.data.S1, cfg_infer.data.S2, 1)


def denorm(x):
    return x * pm_iqr + pm_median


topo_proxy_path = getattr(
    cfg_infer.paths, "topo_path",
    os.path.join(os.path.dirname(cfg_infer.paths.stats_path), "topo_proxy.npy"))
topo_proxy = np.load(topo_proxy_path)


# ============================================================
# 3. TEST DATA LOADER
# ============================================================
class TestDataLoader(torch.utils.data.Dataset):
    def __init__(self, cfg_infer, cfg_train, stats_dict, topo_proxy):
        self.time_in     = cfg_train.data.time_input
        self.total_time  = cfg_train.data.total_time
        self.S1          = cfg_train.data.S1
        self.S2          = cfg_train.data.S2
        self.met_vars    = cfg_train.features.met_variables
        self.emi_vars    = cfg_train.features.emission_variables
        self.stats       = stats_dict
        self.topo_proxy  = topo_proxy

        print("Loading test arrays into memory...")
        self.arrs = {}
        for feat in self.met_vars + self.emi_vars:
            path = os.path.join(cfg_infer.paths.input_loc, f"{feat}.npy")
            self.arrs[feat] = np.load(path)

        self.N = self.arrs['cpm25'].shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # ---- 1. Extract raw sequences ----
        seq_raw = {}
        for feat in self.met_vars + self.emi_vars:
            if feat == 'cpm25':
                seq_raw[feat] = np.array(
                    self.arrs[feat][idx, :self.time_in], dtype=np.float32)
            else:
                seq_raw[feat] = np.array(
                    self.arrs[feat][idx, :self.total_time], dtype=np.float32)

        # ---- 2. Derived meteorological features ----
        ws = np.sqrt(seq_raw['u10'] ** 2 + seq_raw['v10'] ** 2)
        # vent_coef uses RAW pblh (before log1p) — must match prepare_dataset.py
        vc = np.log1p(ws * seq_raw['pblh'])
        rm = (seq_raw['rain'] > 0).astype(np.float32)
        seq_raw['wind_speed'] = ws
        seq_raw['vent_coef']  = vc
        seq_raw['rain_mask']  = rm

        # ---- 3. Log transforms — MUST match prepare_dataset.py ----
        for feat in ('rain', 'bio', 'NMVOC_finn', 'pblh'):
            if feat in seq_raw:
                seq_raw[feat] = np.log1p(seq_raw[feat])

        # FIX Bug #4: apply 1e11 scaling before log1p for small-scale emissions
        for feat in EMI_SCALE_VARS:
            if feat in seq_raw:
                seq_raw[feat] = np.log1p(seq_raw[feat] * 1e11)

        # ---- 4. Normalise ----
        # Temporal weather features (met except cpm25 + derived)
        temporal_list = [f for f in self.met_vars if f != 'cpm25'] \
                        + list(cfg_train.features.derived_variables)
        temporal_feats = []
        for feat in temporal_list:
            arr = (seq_raw[feat] - self.stats[feat]['median']) \
                  / self.stats[feat]['iqr']
            temporal_feats.append(arr)

        # Static emission features → [min, mean, max] aggregation
        static_feats = []
        for feat in self.emi_vars:
            arr = (seq_raw[feat] - self.stats[feat]['median']) \
                  / self.stats[feat]['iqr']
            static_feats.append(arr)

        # PM2.5 history
        pm25_hist = ((seq_raw['cpm25'] - self.stats['cpm25']['median'])
                     / self.stats['cpm25']['iqr'])
        pm25_hist = torch.from_numpy(pm25_hist).permute(1, 2, 0)   # (H, W, 10)

        # ---- 5. Build tensors (must match FastInMemoryDataset exactly) ----
        temporal_stack  = np.stack(temporal_feats, axis=0)
        temporal_tensor = torch.from_numpy(temporal_stack).permute(2, 3, 1, 0) \
                              .reshape(self.S1, self.S2, -1)               # (H, W, T*F)

        static_stack = np.stack(static_feats, axis=0)   # (n_emi, T, H, W)
        static_raw   = torch.from_numpy(static_stack)
        static_tensor = torch.cat([
            static_raw.min(dim=1)[0].permute(1, 2, 0),
            static_raw.mean(dim=1).permute(1, 2, 0),
            static_raw.max(dim=1)[0].permute(1, 2, 0),
        ], dim=-1)                                                        # (H, W, 3*n_emi)

        topo_tensor = torch.from_numpy(self.topo_proxy).unsqueeze(-1)     # (H, W, 1)

        x = torch.cat([pm25_hist, temporal_tensor, static_tensor, topo_tensor], dim=-1)
        return x


test_dataset = TestDataLoader(cfg_infer, cfg_train, stats, topo_proxy)

# FIX Bug #7: pin_memory=True speeds up host→GPU transfer during inference
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=4, shuffle=False,
    num_workers=4, pin_memory=True)

# ============================================================
# 4. MODEL
# ============================================================
pm_channels             = cfg_train.data.time_input
temporal_features_count = len(
    [f for f in cfg_train.features.met_variables if f != 'cpm25']
    + list(cfg_train.features.derived_variables))
temporal_channels = temporal_features_count * cfg_train.data.total_time
static_channels   = len(cfg_train.features.emission_variables) * 3
topo_channels     = 1
in_channels       = pm_channels + temporal_channels + static_channels + topo_channels

print(f"Model: {in_channels} input channels")
model = FNO2D(
    in_channels=in_channels,
    time_out=cfg_train.data.time_out,
    width=cfg_train.model.width,
    modes=cfg_train.model.modes,
    time_input=cfg_train.data.time_input,
    total_time=cfg_train.data.total_time,
    num_temporal_features=temporal_features_count,
).to(device)

print(f"Loading checkpoint: {cfg_infer.paths.checkpoint}")
checkpoint      = torch.load(cfg_infer.paths.checkpoint, map_location=device)
clean_state_dict = {k.replace('module.', ''): v
                    for k, v in checkpoint['model_state_dict'].items()
                    if k != 'n_averaged'}
model.load_state_dict(clean_state_dict)
model.eval()

# ============================================================
# 5. INFERENCE LOOP
# ============================================================
prediction  = np.zeros(
    (len(test_dataset), cfg_train.data.S1, cfg_train.data.S2, cfg_train.data.time_out),
    dtype=np.float32)
current_idx = 0

print("Starting inference...")
with torch.no_grad():
    for x in tqdm(test_loader):
        x  = x.to(device, non_blocking=True)
        bs = x.size(0)

        out      = model(x)
        out_real = denorm(out.cpu().numpy())

        # FIX Bug #2: clip to non-negative — PM2.5 is physically bounded at 0
        prediction[current_idx : current_idx + bs] = np.clip(out_real, 0, None)
        current_idx += bs

# ============================================================
# 6. SAVE
# ============================================================
os.makedirs(cfg_infer.paths.output_loc, exist_ok=True)
out_file = os.path.join(cfg_infer.paths.output_loc, 'preds.npy')
np.save(out_file, prediction)
print(f"Success! Saved to: {out_file}  shape: {prediction.shape}")