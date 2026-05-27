import argparse
import json
import os
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from .train_hybrid_gcn import (
    PoseDataset,
    list_npy_and_labels,
    train_one_fold,
    evaluate,
    plot_history,
    save_reg_plots,
    plot_error_distribution,
    compute_saliency,
)
from .hybrid_gcn import HybridCOMGCNv2

ABLA_MODES = ["A", "B", "C", "D"]  # A: 좌표, B: 좌표+속도, C: 좌표+속도+amp/var, D: 풀 9채널


def run_ablation(processed_dir, label_dir, epochs=20, batch_size=4, folds=None, seconds=13.0, fps=30.0, lr=1e-3):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("results", "hybrid_ablation", timestamp)
    os.makedirs(base_dir, exist_ok=True)
    max_len = int(seconds * fps)
    npy_files, labels = list_npy_and_labels(processed_dir, label_dir)
    if len(npy_files) == 0:
        print("No samples found for hybrid ablation.")
        return

    summary = []
    for mode in ABLA_MODES:
        print(f"[INFO] Hybrid Torch Ablation {mode} 시작")
        run_dir = os.path.join(base_dir, f"abl_{mode}")
        os.makedirs(run_dir, exist_ok=True)

        # 단일 hold-out 80/20 split
        all_idx = np.arange(len(npy_files))
        tr_idx, val_idx = train_test_split(all_idx, test_size=0.2, shuffle=True, random_state=42)
        train_files = [npy_files[i] for i in tr_idx]
        val_files = [npy_files[i] for i in val_idx]
        train_labels = [labels[i] for i in tr_idx]
        val_labels = [labels[i] for i in val_idx]

        train_ds = PoseDataset(train_files, train_labels, ablation_mode=mode, max_len=max_len)
        val_ds = PoseDataset(val_files, val_labels, ablation_mode=mode, max_len=max_len)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        model = HybridCOMGCNv2(in_channels=train_ds[0][0].shape[-1])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_path = os.path.join(run_dir, "best.pth")

        model, history = train_one_fold(model, train_loader, val_loader, device, epochs=epochs, lr=lr, save_path=best_path)
        plot_history(history, os.path.join(run_dir, "history"))

        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
        mae, rmse, pearson, kendall, ccc, r2, evs, mape, medae, y_true, y_pred, ids = evaluate(model, val_loader, device)
        save_reg_plots(y_true, y_pred, ids, run_dir, prefix="holdout")
        plot_error_distribution([np.abs(y_pred - y_true)], run_dir)
        compute_saliency(model, val_loader, device, run_dir, prefix="holdout_saliency")

        summary.append({
            "ablation": mode,
            "mae": float(mae),
            "rmse": float(rmse),
            "pearson": float(pearson),
            "kendall": float(kendall),
            "ccc": float(ccc),
            "r2": float(r2),
            "explained_variance": float(evs),
            "mape": float(mape),
            "medae": float(medae),
        })

    if summary:
        import pandas as pd
        df = pd.DataFrame(summary)
        df.to_csv(os.path.join(base_dir, "ablation_summary.csv"), index=False)
        with open(os.path.join(base_dir, "ablation_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Hybrid torch ablation summary saved to {base_dir}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid PyTorch ablation study (single hold-out)")
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=13.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    run_ablation(args.processed_dir, args.label_dir, epochs=args.epochs, batch_size=args.batch_size, folds=args.folds, seconds=args.seconds, fps=args.fps, lr=args.lr)


if __name__ == "__main__":
    main()
