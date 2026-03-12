"""Quick dataset distribution analysis for debugging."""
import numpy as np
import os

# Load train data and stats
train_data = np.load("data/train/train_data.npy", mmap_mode='r')
train_idx = np.load("data/train/train_indices.npy")
val_data = np.load("data/val/val_data.npy", mmap_mode='r')
val_idx = np.load("data/val/val_indices.npy")
stats = np.load("grid_robust_stats.npy", allow_pickle=True).item()

print("=" * 60)
print("DATASET SHAPE ANALYSIS")
print("=" * 60)
print(f"Train data:    {train_data.shape}  dtype={train_data.dtype}")
print(f"Train indices: {train_idx.shape}   range=[{train_idx.min()}, {train_idx.max()}]")
print(f"Val data:      {val_data.shape}    dtype={val_data.dtype}")
print(f"Val indices:   {val_idx.shape}     range=[{val_idx.min()}, {val_idx.max()}]")
print(f"Train samples: {len(train_idx)}")
print(f"Val samples:   {len(val_idx)}")

print("\n" + "=" * 60)
print("NORMALIZATION STATS (grid-wise robust: median/IQR)")
print("=" * 60)
for feat, st in stats.items():
    med = st['median']
    iqr = st['iqr']
    print(f"  {feat:>15s}  median=[{med.min():.4f}, {med.max():.4f}]  iqr=[{iqr.min():.4f}, {iqr.max():.4f}]")

print("\n" + "=" * 60)
print("FEATURE CHANNEL DISTRIBUTIONS (sampled from train)")
print("=" * 60)
feat_names = ['cpm25','q2','t2','u10','v10','swdown','pblh','rain',
              'PM25','NH3','SO2','NOx','NMVOC_e','NMVOC_finn','bio',
              'wind_speed','vent_coef','rain_mask','topo']

sample = train_data[::10, :, :, :]
for i, name in enumerate(feat_names):
    ch = sample[:, :, :, i]
    print(f"  Ch {i:2d} {name:>15s}  min={ch.min():10.4f}  max={ch.max():10.4f}  mean={ch.mean():10.4f}  std={ch.std():10.4f}")

print("\n" + "=" * 60)
print("PM2.5 TARGET DISTRIBUTION (physical units)")
print("=" * 60)
pm_med = stats['cpm25']['median']
pm_iqr = stats['cpm25']['iqr']
cpm25_norm = sample[:, :, :, 0]
cpm25_phys = (cpm25_norm * pm_iqr) + pm_med
print(f"  Normalized: min={cpm25_norm.min():.4f}, max={cpm25_norm.max():.4f}, mean={cpm25_norm.mean():.4f}")
print(f"  Physical:   min={cpm25_phys.min():.2f}, max={cpm25_phys.max():.2f}, mean={cpm25_phys.mean():.2f}")
print(f"  Percentiles (physical): p5={np.percentile(cpm25_phys, 5):.2f}, p50={np.percentile(cpm25_phys, 50):.2f}, p95={np.percentile(cpm25_phys, 95):.2f}, p99={np.percentile(cpm25_phys, 99):.2f}")

extreme_mask = np.abs(cpm25_norm) > 5
print(f"  Extreme values (|norm| > 5): {extreme_mask.sum()} out of {cpm25_norm.size} ({100*extreme_mask.sum()/cpm25_norm.size:.4f}%)")

print("\n" + "=" * 60)
print("WINDOW SAMPLING CHECK")
print("=" * 60)
for i in range(min(5, len(train_idx))):
    start = train_idx[i]
    window = train_data[start:start+26]
    pm_in = window[:10, :, :, 0]
    pm_out = window[10:, :, :, 0]
    pm_in_phys = (pm_in * pm_iqr) + pm_med
    pm_out_phys = (pm_out * pm_iqr) + pm_med
    print(f"  Sample {i}: start={start}, input_pm_mean={pm_in_phys.mean():.2f}, target_pm_mean={pm_out_phys.mean():.2f}, delta={pm_out_phys.mean()-pm_in_phys.mean():.2f}")
