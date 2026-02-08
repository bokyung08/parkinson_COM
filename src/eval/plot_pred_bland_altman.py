"""
Quick plotting utility:
- Prediction vs Ground Truth scatter
- Bland-Altman plot

Usage (example):
python -m src.eval.plot_pred_bland_altman --tsv results/hybrid_ablation/20260108_003809/abl_D/holdout_predictions.tsv --seconds 10 --fps 30
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(tsv_path, seconds=None, fps=30.0):
    df = pd.read_csv(tsv_path, sep="\t")
    if seconds is not None:
        if "frame" in df.columns:
            df = df[df["frame"] < seconds * fps]
        elif "time" in df.columns:
            df = df[df["time"] < seconds]
    y_true = df.iloc[:, 0].astype(float).to_numpy()
    y_pred = df.iloc[:, 1].astype(float).to_numpy()
    return y_true, y_pred, df


def plot_scatter(y_true, y_pred, out_path):
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot(lims, lims, "r--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Prediction vs Ground Truth")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bland_altman(y_true, y_pred, out_path):
    diff = y_pred - y_true
    mean = (y_pred + y_true) / 2
    md = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_upper = md + 1.96 * sd
    loa_lower = md - 1.96 * sd

    plt.figure()
    plt.scatter(mean, diff, alpha=0.7)
    plt.axhline(md, color="red", linestyle="--", label=f"Mean diff={md:.3f}")
    plt.axhline(loa_upper, color="gray", linestyle="--", label=f"+1.96 SD={loa_upper:.3f}")
    plt.axhline(loa_lower, color="gray", linestyle="--", label=f"-1.96 SD={loa_lower:.3f}")
    plt.xlabel("Mean of True & Pred")
    plt.ylabel("Pred - True")
    plt.title("Bland-Altman")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot scatter and Bland-Altman from predictions TSV.")
    parser.add_argument("--tsv", required=True, help="Path to predictions TSV (true, pred, ...).")
    parser.add_argument("--seconds", type=float, default=None, help="Use only first N seconds if time/frame column exists.")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for frame->time conversion when seconds is set.")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save plots (default: same as TSV).")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(args.tsv)
    os.makedirs(out_dir, exist_ok=True)

    y_true, y_pred, _ = load_data(args.tsv, seconds=args.seconds, fps=args.fps)
    scatter_path = os.path.join(out_dir, "scatter.png")
    ba_path = os.path.join(out_dir, "bland_altman.png")
    plot_scatter(y_true, y_pred, scatter_path)
    plot_bland_altman(y_true, y_pred, ba_path)
    print(f"[INFO] Saved plots to {scatter_path} and {ba_path}")


if __name__ == "__main__":
    main()
