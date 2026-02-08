"""
PyTorch training script for Hybrid GCN (moved under trainers package).
"""

import os
import argparse
import json
from datetime import datetime
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, kendalltau
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from ..features.feature_engineering import build_hybrid_node_features, apply_ablation
from ..models.hybrid_gcn import HybridCOMGCNv2
from .train_model import load_labels  # reuse label parsing


class PoseDataset(Dataset):
    def __init__(self, npy_files, labels, ablation_mode='D', max_len=150):
        self.npy_files = npy_files
        self.labels = labels
        self.ablation_mode = ablation_mode
        self.max_len = max_len

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        npy_path = self.npy_files[idx]
        x = np.load(npy_path)  # (T, J*9) or (T, J, 3)
        if x.ndim == 2:
            J = 33
            x = x.reshape(x.shape[0], J, -1)
        if x.shape[-1] == 3:
            x = build_hybrid_node_features(x)
        x = apply_ablation(x, self.ablation_mode)
        T, J, C = x.shape
        if T > self.max_len:
            x = x[: self.max_len]
        elif T < self.max_len:
            pad = np.zeros((self.max_len - T, J, C), dtype=x.dtype)
            x = np.concatenate([x, pad], axis=0)
        y = self.labels[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            self.npy_files[idx],
        )


def list_npy_and_labels(processed_dir, label_dir, use_only_suffix='_2_pose.npy'):
    npy_files = []
    labels = []
    label_map = load_labels(label_dir)
    for root, _, files in os.walk(processed_dir):
        for f in files:
            if not f.endswith(use_only_suffix):
                continue
            pid = f.replace('_pose.npy', '').rsplit('_', 1)[0]
            if pid not in label_map:
                continue
            gait = label_map[pid]["gait_updrs"]
            npy_files.append(os.path.join(root, f))
            labels.append(gait)
    return npy_files, labels


def train_one_fold(model, train_loader, val_loader, device, epochs=80, lr=1e-3, save_path=None):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-5, verbose=1)
    criterion = nn.L1Loss()  # MAE
    best_val = float("inf")
    history = {"train_mae": [], "val_mae": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())
        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses)
        history["train_mae"].append(float(mean_train))
        history["val_mae"].append(float(mean_val))
        scheduler.step(mean_val)
        print(f"Epoch {epoch+1}/{epochs} train_mae={mean_train:.3f} val_mae={mean_val:.3f}")
        if mean_val < best_val and save_path:
            best_val = mean_val
            torch.save(model.state_dict(), save_path)
    return model, history


def plot_history(history, out_path_prefix):
    if not history:
        return
    os.makedirs(os.path.dirname(out_path_prefix), exist_ok=True)
    if "train_mae" in history:
        plt.figure()
        plt.plot(history.get("train_mae", []), label="train_mae")
        plt.plot(history.get("val_mae", []), label="val_mae")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("Training/Validation MAE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path_prefix + "_mae.png")
        plt.close()


def save_reg_plots(y_true, y_pred, ids, out_dir, prefix="val"):
    os.makedirs(out_dir, exist_ok=True)
    residuals = y_pred - y_true
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    # Scatter
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel('True Gait UPDRS')
    plt.ylabel('Predicted Gait UPDRS')
    plt.title(f'{prefix} True vs Pred (MAE={mae:.2f}, RMSE={rmse:.2f})')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_scatter.png"))
    plt.close()
    # Residual hist
    plt.figure()
    plt.hist(residuals, bins=20, alpha=0.8)
    plt.xlabel('Residual (Pred-True)')
    plt.ylabel('Count')
    plt.title(f'{prefix} Residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_residuals.png"))
    plt.close()
    # Distribution
    plt.figure()
    plt.hist(y_true, bins=20, alpha=0.6, label='True')
    plt.hist(y_pred, bins=20, alpha=0.6, label='Pred')
    plt.xlabel('Gait UPDRS')
    plt.ylabel('Count')
    plt.title(f'{prefix} Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_dist.png"))
    plt.close()
    # Abs error CDF
    abs_err = np.abs(residuals)
    sorted_err = np.sort(abs_err)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    plt.figure()
    plt.plot(sorted_err, cdf)
    plt.xlabel('Absolute Error')
    plt.ylabel('CDF')
    plt.title(f'{prefix} Abs Error CDF')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_abs_error_cdf.png"))
    plt.close()
    # TSV dump
    with open(os.path.join(out_dir, f"{prefix}_predictions.tsv"), 'w', encoding='utf-8') as f:
        f.write("sample_id\ttrue_gait_updrs\tpred_gait_updrs\tabs_error\n")
        for sid, yt, yp in zip(ids, y_true, y_pred):
            f.write(f"{sid}\t{yt:.6f}\t{yp:.6f}\t{abs(yp-yt):.6f}\n")
    np.savez_compressed(
        os.path.join(out_dir, f"{prefix}_predictions.npz"),
        sample_ids=np.array(ids),
        y_true=y_true,
        y_pred=y_pred
    )


def plot_error_distribution(per_fold_abs_err, out_dir):
    """Box/violin plot of absolute errors per fold."""
    if not per_fold_abs_err:
        return
    os.makedirs(out_dir, exist_ok=True)
    data = per_fold_abs_err
    plt.figure()
    sns.boxplot(data=data)
    plt.xlabel("Fold")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error per Fold (Boxplot)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "abs_error_per_fold_box.png"))
    plt.close()

    plt.figure()
    sns.violinplot(data=data, cut=0)
    plt.xlabel("Fold")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error per Fold (Violin)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "abs_error_per_fold_violin.png"))
    plt.close()


def compute_saliency(model, loader, device, out_dir, prefix="saliency"):
    """Gradient-based saliency: mean |grad| over features -> heatmap (T x J) and joint bar."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    try:
        xb, yb, _ = next(iter(loader))
    except StopIteration:
        return
    xb = xb.to(device)
    xb.requires_grad = True
    pred = model(xb)
    loss = pred.mean()  # scalar
    loss.backward()
    grads = xb.grad.detach().cpu().numpy()  # (B, T, J, C)
    grad_abs = np.abs(grads).mean(axis=0)  # (T, J, C)
    heat = grad_abs.mean(axis=2)           # (T, J)
    joint_importance = heat.mean(axis=0)   # (J,)

    # Heatmap T x J
    plt.figure(figsize=(10, 4))
    sns.heatmap(heat.T, cmap="magma", cbar=True)
    plt.xlabel("Time")
    plt.ylabel("Joint")
    plt.title("Saliency Heatmap (mean |grad|)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_heatmap.png"))
    plt.close()

    # Joint bar
    plt.figure(figsize=(10, 3))
    plt.bar(np.arange(len(joint_importance)), joint_importance)
    plt.xlabel("Joint index")
    plt.ylabel("Mean |grad|")
    plt.title("Joint Importance (Saliency)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_joint_importance.png"))
    plt.close()
