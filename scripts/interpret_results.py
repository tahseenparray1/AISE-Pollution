"""
=============================================================================
 PM2.5 Prediction Result Interpretation Framework
=============================================================================
 Loads preds.npy and produces a comprehensive analysis:
   1. Summary statistics & sanity checks
   2. Spatial heatmaps at selected forecast horizons
   3. Temporal evolution curves (domain-avg & hotspot)
   4. Distribution analysis (histograms + box plots per horizon)
   5. Hotspot & coldspot identification
   6. Spatial anomaly maps (deviation from temporal mean)
   7. Hour-to-hour change-rate analysis
   8. (Optional) Comparison against ground-truth if available

 Usage:
   python scripts/interpret_results.py                                # defaults
   python scripts/interpret_results.py --preds /path/to/preds.npy     # custom path
   python scripts/interpret_results.py --truth /path/to/truth.npy     # with ground truth
=============================================================================
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec


# ──────────────────────────────────────────────────────────────────────
# 0.  CLI & Setup
# ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Interpret PM2.5 forecast predictions")
    p.add_argument("--preds", type=str, default="/kaggle/working/preds.npy",
                   help="Path to preds.npy  (N, H, W, 16)")
    p.add_argument("--truth", type=str, default=None,
                   help="Optional ground-truth .npy file (same shape as preds)")
    p.add_argument("--outdir", type=str, default="results_analysis",
                   help="Directory to save all plots and reports")
    p.add_argument("--sample_ids", type=int, nargs="*", default=None,
                   help="Specific sample indices to visualize (default: auto-picks)")
    return p.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# 1.  Summary Statistics & Sanity Checks
# ──────────────────────────────────────────────────────────────────────
def summary_statistics(preds, outdir):
    """Print and save global summary statistics."""
    N, H, W, T = preds.shape
    report = []
    report.append("=" * 60)
    report.append("  PM2.5 PREDICTION — SUMMARY REPORT")
    report.append("=" * 60)
    report.append(f"  Shape          : {preds.shape}  (samples, H, W, forecast_hours)")
    report.append(f"  Samples        : {N}")
    report.append(f"  Spatial grid   : {H} × {W}")
    report.append(f"  Forecast hours : {T}")
    report.append("-" * 60)
    report.append(f"  Global min     : {preds.min():.4f}")
    report.append(f"  Global max     : {preds.max():.4f}")
    report.append(f"  Global mean    : {preds.mean():.4f}")
    report.append(f"  Global median  : {np.median(preds):.4f}")
    report.append(f"  Global std     : {preds.std():.4f}")
    report.append("-" * 60)

    neg_frac = (preds < 0).sum() / preds.size * 100
    zero_frac = (preds == 0).sum() / preds.size * 100
    nan_count = np.isnan(preds).sum()
    inf_count = np.isinf(preds).sum()
    report.append(f"  Negative vals  : {neg_frac:.4f}%")
    report.append(f"  Exact zeros    : {zero_frac:.4f}%")
    report.append(f"  NaN count      : {nan_count}")
    report.append(f"  Inf count      : {inf_count}")
    report.append("-" * 60)

    # Per-horizon statistics
    report.append("  Per-Horizon Statistics (mean ± std):")
    for t in range(T):
        h_data = preds[:, :, :, t]
        report.append(f"    Hour +{t+1:02d}  :  {h_data.mean():8.2f} ± {h_data.std():8.2f}"
                      f"   [min={h_data.min():.2f}, max={h_data.max():.2f}]")
    report.append("=" * 60)

    text = "\n".join(report)
    print(text)
    with open(os.path.join(outdir, "summary_report.txt"), "w") as f:
        f.write(text)


# ──────────────────────────────────────────────────────────────────────
# 2.  Spatial Heatmaps at Selected Forecast Horizons
# ──────────────────────────────────────────────────────────────────────
def plot_spatial_heatmaps(preds, outdir, sample_ids=None):
    """Plot PM2.5 spatial heatmaps at forecast hours 1, 4, 8, 12, 16."""
    N, H, W, T = preds.shape
    horizons = [0, 3, 7, 11, 15]  # 0-indexed → hours 1, 4, 8, 12, 16
    horizons = [h for h in horizons if h < T]

    if sample_ids is None:
        # Pick 3 representative samples: low, median, high mean PM2.5
        sample_means = preds.mean(axis=(1, 2, 3))
        sorted_idx = np.argsort(sample_means)
        sample_ids = [sorted_idx[0], sorted_idx[N // 2], sorted_idx[-1]]

    vmin = max(0, np.percentile(preds, 1))
    vmax = np.percentile(preds, 99)

    for sid in sample_ids:
        fig, axes = plt.subplots(1, len(horizons), figsize=(4 * len(horizons), 4))
        if len(horizons) == 1:
            axes = [axes]
        for ax, h in zip(axes, horizons):
            im = ax.imshow(preds[sid, :, :, h], cmap="YlOrRd", vmin=vmin, vmax=vmax,
                           origin="upper", aspect="auto")
            ax.set_title(f"Hour +{h + 1}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Longitude idx")
            ax.set_ylabel("Latitude idx")
        fig.suptitle(f"Sample {sid} — PM2.5 Forecast (µg/m³)", fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=axes, shrink=0.8, label="PM2.5 (µg/m³)")
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, f"spatial_heatmap_sample{sid}.png"), dpi=150)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 3.  Temporal Evolution Curves
# ──────────────────────────────────────────────────────────────────────
def plot_temporal_evolution(preds, outdir):
    """Domain-averaged PM2.5 across the 16 forecast hours, with spread."""
    N, H, W, T = preds.shape
    # Per-sample spatial average  → (N, T)
    spatial_avg = preds.mean(axis=(1, 2))
    hours = np.arange(1, T + 1)

    mean_curve = spatial_avg.mean(axis=0)
    p10 = np.percentile(spatial_avg, 10, axis=0)
    p25 = np.percentile(spatial_avg, 25, axis=0)
    p75 = np.percentile(spatial_avg, 75, axis=0)
    p90 = np.percentile(spatial_avg, 90, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(hours, p10, p90, alpha=0.15, color="tab:blue", label="10th–90th pctl")
    ax.fill_between(hours, p25, p75, alpha=0.30, color="tab:blue", label="25th–75th pctl")
    ax.plot(hours, mean_curve, "o-", color="tab:blue", linewidth=2, label="Mean")
    ax.set_xlabel("Forecast Hour", fontsize=12)
    ax.set_ylabel("Domain-Avg PM2.5 (µg/m³)", fontsize=12)
    ax.set_title("Temporal Evolution of Domain-Averaged PM2.5", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "temporal_evolution.png"), dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 4.  Distribution Analysis — Histograms & Box Plots
# ──────────────────────────────────────────────────────────────────────
def plot_distributions(preds, outdir):
    """Histograms and box-plots for selected forecast horizons."""
    T = preds.shape[-1]
    horizons = [0, 7, 15]
    horizons = [h for h in horizons if h < T]

    # --- Histograms ---
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4), sharey=True)
    if len(horizons) == 1:
        axes = [axes]
    for ax, h in zip(axes, horizons):
        vals = preds[:, :, :, h].ravel()
        ax.hist(vals, bins=100, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.axvline(vals.mean(), color="red", linestyle="--", label=f"mean={vals.mean():.1f}")
        ax.axvline(np.median(vals), color="orange", linestyle="--", label=f"median={np.median(vals):.1f}")
        ax.set_title(f"Hour +{h + 1}", fontsize=12, fontweight="bold")
        ax.set_xlabel("PM2.5 (µg/m³)")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Count")
    fig.suptitle("PM2.5 Value Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "distributions_histogram.png"), dpi=150)
    plt.close(fig)

    # --- Box plots per horizon ---
    spatial_means = preds.mean(axis=(1, 2))  # (N, T)
    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot([spatial_means[:, t] for t in range(T)],
                    labels=[f"+{t+1}" for t in range(T)],
                    patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
    ax.set_xlabel("Forecast Hour", fontsize=12)
    ax.set_ylabel("Sample-wise Spatial Mean PM2.5", fontsize=12)
    ax.set_title("Box Plot — PM2.5 Spatial Mean per Forecast Hour", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "distributions_boxplot.png"), dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 5.  Hotspot & Coldspot Identification
# ──────────────────────────────────────────────────────────────────────
def plot_hotspot_coldspot(preds, outdir):
    """Identify persistent high/low PM2.5 regions across all samples and horizons."""
    # Time-mean then sample-mean  →  (H, W)
    mean_map = preds.mean(axis=(0, 3))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Hotspot map
    im0 = axes[0].imshow(mean_map, cmap="YlOrRd", origin="upper", aspect="auto")
    axes[0].set_title("Mean PM2.5 — Hotspot Map", fontsize=13, fontweight="bold")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="µg/m³")

    # Std-dev map (variability)
    std_map = preds.std(axis=(0, 3))
    im1 = axes[1].imshow(std_map, cmap="viridis", origin="upper", aspect="auto")
    axes[1].set_title("Std Dev PM2.5 — Variability Map", fontsize=13, fontweight="bold")
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="µg/m³")

    for ax in axes:
        ax.set_xlabel("Longitude idx")
        ax.set_ylabel("Latitude idx")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "hotspot_coldspot.png"), dpi=150)
    plt.close(fig)

    # Top-10 hotspot cells
    flat_idx = np.argsort(mean_map.ravel())[::-1][:10]
    rows, cols = np.unravel_index(flat_idx, mean_map.shape)
    print("\n  Top-10 Hotspot Grid Cells (highest mean PM2.5):")
    for rank, (r, c) in enumerate(zip(rows, cols), 1):
        print(f"    #{rank:2d}  ({r:3d}, {c:3d})  mean={mean_map[r, c]:.2f} µg/m³")


# ──────────────────────────────────────────────────────────────────────
# 6.  Spatial Anomaly Maps
# ──────────────────────────────────────────────────────────────────────
def plot_anomaly_maps(preds, outdir, sample_ids=None):
    """Show deviation of each forecast hour from the sample's temporal mean."""
    N, H, W, T = preds.shape
    if sample_ids is None:
        sample_ids = [N // 2]

    for sid in sample_ids[:2]:
        sample = preds[sid]  # (H, W, T)
        tmean = sample.mean(axis=2, keepdims=True)  # (H, W, 1)
        anomaly = sample - tmean

        horizons = [0, 7, 15]
        horizons = [h for h in horizons if h < T]
        vlim = np.percentile(np.abs(anomaly), 95)

        fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4))
        if len(horizons) == 1:
            axes = [axes]
        for ax, h in zip(axes, horizons):
            im = ax.imshow(anomaly[:, :, h], cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                           origin="upper", aspect="auto")
            ax.set_title(f"Hour +{h+1}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Longitude idx")
            ax.set_ylabel("Latitude idx")
        fig.suptitle(f"Sample {sid} — PM2.5 Anomaly (deviation from temporal mean)",
                     fontsize=13, fontweight="bold")
        fig.colorbar(im, ax=axes, shrink=0.8, label="Δ PM2.5 (µg/m³)")
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, f"anomaly_sample{sid}.png"), dpi=150)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 7.  Hour-to-Hour Change Rate
# ──────────────────────────────────────────────────────────────────────
def plot_change_rate(preds, outdir):
    """Analyse how fast PM2.5 changes between consecutive forecast hours."""
    # Δ between consecutive hours: (N, H, W, T-1)
    delta = np.diff(preds, axis=3)
    spatial_abs_delta = np.abs(delta).mean(axis=(1, 2))  # (N, T-1)

    hours = np.arange(1, delta.shape[3] + 1)
    mean_delta = spatial_abs_delta.mean(axis=0)
    p25 = np.percentile(spatial_abs_delta, 25, axis=0)
    p75 = np.percentile(spatial_abs_delta, 75, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(hours, p25, p75, alpha=0.3, color="tab:orange", label="25th–75th pctl")
    ax.plot(hours, mean_delta, "s-", color="tab:orange", linewidth=2, label="Mean |Δ|")
    ax.set_xlabel("Hour Transition (t → t+1)", fontsize=12)
    ax.set_ylabel("Mean |Δ PM2.5| (µg/m³)", fontsize=12)
    ax.set_title("Hour-to-Hour Change Rate", fontsize=14, fontweight="bold")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"+{h}→+{h+1}" for h in hours], rotation=45, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "change_rate.png"), dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 8.  Ground-Truth Comparison (optional)
# ──────────────────────────────────────────────────────────────────────
def compare_with_truth(preds, truth, outdir):
    """If ground truth is provided, compute and visualise error metrics."""
    from src.utils.metrics import rmse, mfb, smape

    N, H, W, T = preds.shape
    assert truth.shape == preds.shape, (
        f"Shape mismatch: preds {preds.shape} vs truth {truth.shape}")

    # Per-horizon RMSE  (spatially averaged per sample, then mean over samples)
    rmse_per_h = np.array([
        np.sqrt(((preds[:, :, :, t] - truth[:, :, :, t]) ** 2).mean(axis=(1, 2))).mean()
        for t in range(T)
    ])
    mfb_per_h = np.array([
        mfb(truth[:, :, :, t], preds[:, :, :, t]).mean() for t in range(T)
    ])

    hours = np.arange(1, T + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # RMSE
    axes[0].bar(hours, rmse_per_h, color="indianred", edgecolor="white")
    axes[0].set_xlabel("Forecast Hour")
    axes[0].set_ylabel("RMSE (µg/m³)")
    axes[0].set_title("RMSE per Forecast Hour", fontweight="bold")
    axes[0].grid(True, axis="y", alpha=0.3)

    # MFB
    axes[1].bar(hours, mfb_per_h, color="steelblue", edgecolor="white")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("Forecast Hour")
    axes[1].set_ylabel("MFB")
    axes[1].set_title("Mean Fractional Bias per Forecast Hour", fontweight="bold")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "ground_truth_metrics.png"), dpi=150)
    plt.close(fig)

    # Spatial error map (mean absolute error across all samples and hours)
    mae_map = np.abs(preds - truth).mean(axis=(0, 3))
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mae_map, cmap="hot", origin="upper", aspect="auto")
    ax.set_title("Mean Absolute Error — Spatial Map", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude idx")
    ax.set_ylabel("Latitude idx")
    fig.colorbar(im, ax=ax, shrink=0.8, label="MAE (µg/m³)")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "ground_truth_mae_map.png"), dpi=150)
    plt.close(fig)

    # Overall RMSE
    overall_rmse = np.sqrt(((preds - truth) ** 2).mean())
    print(f"\n  Overall RMSE : {overall_rmse:.4f} µg/m³")
    print(f"  Per-horizon RMSE: {np.array2string(rmse_per_h, precision=2)}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    ensure_dir(args.outdir)

    print(f"\nLoading predictions from: {args.preds}")
    preds = np.load(args.preds)
    print(f"  Shape: {preds.shape}")

    # ---- Run all analyses ----
    summary_statistics(preds, args.outdir)
    plot_spatial_heatmaps(preds, args.outdir, sample_ids=args.sample_ids)
    plot_temporal_evolution(preds, args.outdir)
    plot_distributions(preds, args.outdir)
    plot_hotspot_coldspot(preds, args.outdir)
    plot_anomaly_maps(preds, args.outdir, sample_ids=args.sample_ids)
    plot_change_rate(preds, args.outdir)

    if args.truth is not None:
        print(f"\nLoading ground truth from: {args.truth}")
        truth = np.load(args.truth)
        compare_with_truth(preds, truth, args.outdir)

    print(f"\n✓ All outputs saved to: {args.outdir}/")
    print("  Files generated:")
    for f in sorted(os.listdir(args.outdir)):
        print(f"    • {f}")


if __name__ == "__main__":
    main()
