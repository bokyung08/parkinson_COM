import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def save_bar(df, metric, out_path, ylabel=None):
    plt.figure(figsize=(10, 5))
    labels = df["model"] + "_" + df["ablation"]
    values = df[metric]
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel or metric.upper())
    plt.title(f"{metric.upper()} by Model/Ablation")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot model comparison summary (MAE/RMSE + additional regression metrics).")
    parser.add_argument("--summary_json", type=str, default="results/model_comparison_summary.json")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for plots")
    args = parser.parse_args()

    df = load_summary(args.summary_json)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join("results", "comparison_plots", ts)
    os.makedirs(out_dir, exist_ok=True)

    df = df.copy()
    metrics = ["mae", "rmse", "pearson", "kendall", "ccc", "r2", "explained_variance", "mape", "medae"]
    for col in metrics:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            save_bar(df, col, os.path.join(out_dir, f"{col}_bar.png"), ylabel=col.upper())
    print(f"[INFO] Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
