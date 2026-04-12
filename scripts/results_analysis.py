import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings

# Use the train configurator to fetch test splits
from configs.config import load_config
# We can dynamically import the dataset builder from train
import importlib.util

warnings.filterwarnings("ignore")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_train_module():
    """Dynamically import train module to get FastInMemoryDataset to avoid circular CLI execution."""
    spec = importlib.util.spec_from_file_location("train_module", "scripts/train.py")
    train_module = importlib.util.module_from_spec(spec)
    # Patch torch.utils.data.DataLoader so train.py doesn't crash on load if it initiates stuff
    original_loader = torch.utils.data.DataLoader
    torch.utils.data.DataLoader = lambda *args, **kwargs: None
    spec.loader.exec_module(train_module)
    torch.utils.data.DataLoader = original_loader
    return train_module.FastInMemoryDataset

def build_rmse_bar_chart(preds, truth, outdir):
    N, H, W, T = preds.shape
    
    # Calculate RMSE per horizon
    rmse_per_h = np.array([
        np.sqrt(((preds[:, :, :, t] - truth[:, :, :, t]) ** 2).mean(axis=(1, 2))).mean()
        for t in range(T)
    ])
    
    hours = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(hours, rmse_per_h, color="indianred", edgecolor="white")
    ax.set_xlabel("Forecast Hour")
    ax.set_ylabel("RMSE (µg/m³)")
    ax.set_title("Forecast Accuracy Degradation over Horizon", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "forecast_rmse_barchart.png"), dpi=150)
    plt.close()

def build_spatial_error_map(preds, truth, outdir):
    """Spatial error map highlighting where models struggle most."""
    mae_map = np.abs(preds - truth).mean(axis=(0, 3))
    
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mae_map, cmap="hot", origin="upper", aspect="auto")
    ax.set_title("Mean Absolute Error (Spatial View)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, shrink=0.8, label="MAE Error (µg/m³)")
    
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "spatial_mae_map.png"), dpi=150)
    plt.close()

def build_actuals_vs_pred_curves(preds, truth, outdir):
    """Average domain prediction vs actual line plot."""
    spatial_avg_pred = preds.mean(axis=(1, 2)).mean(axis=0)
    spatial_avg_truth = truth.mean(axis=(1, 2)).mean(axis=0)
    
    hours = np.arange(1, preds.shape[-1] + 1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hours, spatial_avg_pred, "o-", color="tab:blue", label="Average Predicted")
    ax.plot(hours, spatial_avg_truth, "s--", color="tab:green", label="Average Truth", alpha=0.8)
    
    ax.set_xlabel("Forecast Hour")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Test Set Prediction Curves vs Ground Truth Actuals")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "actuals_vs_preds.png"), dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser("AISE-Pollution Fast Results Analysis")
    parser.add_argument("--preds", type=str, default="/kaggle/working/preds.npy", help="Path to predictions.")
    parser.add_argument("--outdir", type=str, default="results_analysis_output", help="Save directory.")
    args = parser.parse_args()
    
    ensure_dir(args.outdir)
    print(f"Loading predictions from {args.preds}...")
    if not os.path.exists(args.preds):
        print("ERROR: preds.npy not found! Run `python scripts/infer.py` first.")
        return
        
    preds = np.load(args.preds)
    
    print("Loading underlying FastInMemoryDataset parameters to extract True Ground Truth labels...")
    # Get config and test datasets
    from src.utils.config import load_config
    cfg = load_config("configs/train.yaml")
    
    # We load FastInMemoryDataset carefully or reimplement extraction.
    # To avoid train.py loading entirely, let's copy the load logic purely:
    base_path = cfg.paths.savepath_test
    print("Loading test arrays...")
    import torch
    data = torch.from_numpy(np.load(os.path.join(base_path, "test_data.npy")).astype(np.float32))
    starts = np.load(os.path.join(base_path, "test_indices.npy"))
    
    stats_path = cfg.paths.stats_path
    stats = np.load(stats_path, allow_pickle=True).item()
    pm_median = stats['cpm25']['median'].reshape(1, 140, 124, 1)
    pm_iqr = stats['cpm25']['iqr'].reshape(1, 140, 124, 1)
    
    all_features = (cfg.features.met_variables + cfg.features.emission_variables + cfg.features.derived_variables)
    target_idx = all_features.index('cpm25')
    
    print("Constructing truth array...")
    truth_list = []
    time_in = cfg.data.time_input
    total_time = cfg.data.total_time
    
    for idx in range(len(starts)):
        start = starts[idx]
        window = data[start : start + total_time]
        y_norm = window[time_in:, ..., target_idx].permute(1, 2, 0).numpy()
        y_phys = (y_norm * pm_iqr[0]) + pm_median[0]
        truth_list.append(y_phys)
        
    truth = np.stack(truth_list, axis=0)
    
    print("Generating RMSE Error Distributions...")
    build_rmse_bar_chart(preds, truth, args.outdir)
    
    print("Generating Spatial Error Maps...")
    build_spatial_error_map(preds, truth, args.outdir)
    
    print("Generating Temporal Actual vs Prediction Plot...")
    build_actuals_vs_pred_curves(preds, truth, args.outdir)
    
    overall_rmse = np.sqrt(((preds - truth) ** 2).mean())
    print("\n===============================")
    print(f"Overall Testing RMSE: {overall_rmse:.4f}")
    print("===============================")
    print(f"Results generated internally and saved to `{args.outdir}` directory.")

if __name__ == "__main__":
    main()
