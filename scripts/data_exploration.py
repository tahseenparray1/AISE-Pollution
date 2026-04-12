import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.utils.config import load_config
import warnings

warnings.filterwarnings("ignore")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_data_month(month, feat, raw_path):
    path = os.path.join(raw_path, month, f"{feat}.npy")
    if os.path.exists(path):
        return np.load(path).astype(np.float32)
    return None

def plot_seasonal_mean(cfg, outdir):
    """Plot mean PM2.5 across the India grid for each of the 4 representative months."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    months = cfg.data.months
    raw_path = cfg.paths.raw_path
    
    global_min, global_max = float('inf'), float('-inf')
    monthly_means = []
    
    for i, month in enumerate(months):
        pm_data = load_data_month(month, "cpm25", raw_path)
        if pm_data is not None:
            mean_map = pm_data.mean(axis=0)
            monthly_means.append(mean_map)
            global_min = min(global_min, mean_map.min())
            global_max = max(global_max, np.percentile(mean_map, 99))
        else:
            monthly_means.append(None)
            
    for i, month in enumerate(months):
        mean_map = monthly_means[i]
        if mean_map is not None:
            im = axes[i].imshow(mean_map, cmap="YlOrRd", origin="upper", vmin=0, vmax=global_max)
            axes[i].set_title(f"Mean PM2.5 Concentration - {month}")
            axes[i].set_xlabel("Longitude Index")
            axes[i].set_ylabel("Latitude Index")
            fig.colorbar(im, ax=axes[i], shrink=0.7)
        else:
            axes[i].set_title(f"{month} Data Unreachable")
    
    plt.suptitle("Seasonal Variation of PM2.5 (Mean Spatial Distribution)", fontsize=16)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "seasonal_cpm25_map.png"), dpi=150)
    plt.close(fig)

def plot_emissions_hotspots(cfg, outdir):
    """Plot correlation/distribution of man-made and bio emissions across April."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    raw_path = cfg.paths.raw_path
    month = cfg.data.months[0]  # Just use the first month to observe footprint
    
    emissions = ["PM25", "NOx", "SO2"]
    out_titles = ["Primary PM2.5 Emissions", "NOx Emissions", "SO2 Emissions"]
    
    for i, emi in enumerate(emissions):
        data = load_data_month(month, emi, raw_path)
        if data is not None:
            # Aggregate log emissions for better viewing
            mean_emi = np.log1p(data.mean(axis=0) * 1e11) 
            im = axes[i].imshow(mean_emi, cmap="magma", origin="upper")
            axes[i].set_title(f"{out_titles[i]} Hotspots ({month})")
            fig.colorbar(im, ax=axes[i], shrink=0.8, label="Log1p(Emit)")
            
    fig.suptitle(f"Common Emission Sources Spatial Profiles - {month}", fontsize=15)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "emission_profiles.png"), dpi=150)
    plt.close()

def plot_weather_correlation(cfg, outdir):
    """Extract a single grid cell (a hotspot) and map weather changes over a week."""
    raw_path = cfg.paths.raw_path
    month = cfg.data.months[3] # Using Winter month (Index 3)
    
    pm_data = load_data_month(month, "cpm25", raw_path)
    if pm_data is None: return
    
    # Extract hotspot
    mean_pm = pm_data.mean(axis=0)
    flat_idx = np.argmax(mean_pm)
    r, c = np.unravel_index(flat_idx, mean_pm.shape)
    
    t2_data = load_data_month(month, "t2", raw_path)
    pblh_data = load_data_month(month, "pblh", raw_path)
    
    time_window = slice(0, 24*7) # 1 week
    
    pm_trace = pm_data[time_window, r, c]
    t2_trace = t2_data[time_window, r, c] if t2_data is not None else None
    pblh_trace = pblh_data[time_window, r, c] if pblh_data is not None else None
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(pm_trace, color="red", label="PM2.5", linewidth=2)
    ax1.set_xlabel("Hours (1 Week)")
    ax1.set_ylabel("PM2.5", color="red")
    
    if pblh_trace is not None:
        ax2 = ax1.twinx()
        ax2.plot(pblh_trace, color="blue", label="PBLH", alpha=0.6, linestyle="--")
        ax2.set_ylabel("PBLH (m)", color="blue")
        
    plt.title(f"PM2.5 vs Boundary Layer Height at Hotspot ({r}, {c})")
    fig.savefig(os.path.join(outdir, "pm25_vs_pblh_timeseries.png"), dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser("AISE-Pollution Data Exploration")
    parser.add_argument("--outdir", type=str, default="data_exploration_output", help="Directory to save output plots.")
    args = parser.parse_args()
    
    ensure_dir(args.outdir)
    cfg = load_config("configs/prepare_dataset.yaml")
    
    print("Generating Seasonal Map...")
    plot_seasonal_mean(cfg, args.outdir)
    
    print("Generating Emission Profile Map...")
    plot_emissions_hotspots(cfg, args.outdir)
    
    print("Generating Timeseries Weather Map...")
    plot_weather_correlation(cfg, args.outdir)
    
    print(f"Dataset exploration successful. Plots saved in '{args.outdir}'.")

if __name__ == "__main__":
    main()
