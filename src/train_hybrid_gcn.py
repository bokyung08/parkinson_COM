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

from .feature_engineering import build_hybrid_node_features, apply_ablation
from .hybrid_gcn import HybridCOMGCNv2
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


def evaluate(model, loader, device):
    model.eval()
    preds, trues, ids = [], [], []
    with torch.no_grad():
        for xb, yb, paths in loader:
            xb = xb.to(device)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
            trues.append(yb.numpy())
            ids.extend(paths)
    y_pred = np.concatenate(preds).flatten()
    y_true = np.concatenate(trues).flatten()
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    # Additional metrics
    pearson = pearsonr(y_true, y_pred).statistic if len(y_true) > 1 else np.nan
    kendall = kendalltau(y_true, y_pred).statistic if len(y_true) > 1 else np.nan
    # Concordance correlation coefficient (CCC)
    mx, my = np.mean(y_true), np.mean(y_pred)
    vx, vy = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mx) * (y_pred - my))
    ccc = (2 * cov) / (vx + vy + (mx - my) ** 2 + 1e-8)
    # R2 / Explained variance
    ss_tot = np.sum((y_true - mx) ** 2) + 1e-8
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    evs = 1 - np.var(y_true - y_pred) / (np.var(y_true) + 1e-8)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    medae = np.median(np.abs(y_true - y_pred))
    return mae, rmse, pearson, kendall, ccc, r2, evs, mape, medae, y_true, y_pred, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="HospitalData/processed_pose_data")
    parser.add_argument("--label_dir", default="HospitalData/JSON")
    parser.add_argument("--ablation", default="D", choices=["A", "B", "C", "D"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=13.0, help="Clip each sequence to this duration (seconds)")
    parser.add_argument("--fps", type=float, default=30.0, help="Assumed fps for pose sequences")
    parser.add_argument("--run_dir", default=None, help="Optional output dir; default results/hybrid_runs/<timestamp>")
    args = parser.parse_args()

    run_dir = args.run_dir or os.path.join("results", "hybrid_runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    max_len = int(args.seconds * args.fps)
    npy_files, labels = list_npy_and_labels(args.processed_dir, args.label_dir)
    print(f"Found {len(npy_files)} samples for training (ablation {args.ablation}), max_len={max_len} frames (~{args.seconds}s @ {args.fps}fps).")
    if len(npy_files) == 0:
        raise SystemExit("No samples found. Ensure *_2_pose.npy exists under the processed directory.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=min(args.folds, len(npy_files)), shuffle=True, random_state=42)

    fold_metrics = []
    per_fold_abs_err = []
    overall_trues, overall_preds, overall_ids = [], [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(npy_files), 1):
        print(f"\n--- Fold {fold}/{kf.get_n_splits()} ---")
        train_files = [npy_files[i] for i in train_idx]
        val_files = [npy_files[i] for i in val_idx]
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]

        train_ds = PoseDataset(train_files, train_labels, ablation_mode=args.ablation, max_len=max_len)
        val_ds = PoseDataset(val_files, val_labels, ablation_mode=args.ablation, max_len=max_len)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

        model = HybridCOMGCNv2(in_channels=train_ds[0][0].shape[-1])
        best_path = os.path.join(run_dir, f"fold_{fold}_best.pth")
        model, history = train_one_fold(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, save_path=best_path)
        plot_history(history, os.path.join(run_dir, f"fold_{fold}_history"))

        # load best and evaluate
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
        mae, rmse, pearson, kendall, ccc, r2, evs, mape, medae, y_true, y_pred, ids = evaluate(model, val_loader, device)
        print(f"[Fold {fold}] MAE={mae:.3f} RMSE={rmse:.3f} Pear={pearson:.3f} Kend={kendall:.3f} R2={r2:.3f}")
        fold_metrics.append((mae, rmse, pearson, kendall, ccc, r2, evs, mape, medae))
        save_reg_plots(y_true, y_pred, ids, run_dir, prefix=f"fold_{fold}")
        per_fold_abs_err.append(np.abs(y_pred - y_true))
        overall_trues.append(y_true)
        overall_preds.append(y_pred)
        overall_ids.extend(ids)

    # summarize
    metrics_arr = np.array(fold_metrics)
    mean_vals = metrics_arr.mean(axis=0)
    std_vals = metrics_arr.std(axis=0)
    keys = ["mae", "rmse", "pearson", "kendall", "ccc", "r2", "explained_variance", "mape", "medae"]
    summary = {
        "folds": [
            {k: float(v) for k, v in zip(keys, vals)}
            for vals in fold_metrics
        ],
        "mean": {k: float(v) for k, v in zip(keys, mean_vals)},
        "std": {k: float(v) for k, v in zip(keys, std_vals)},
    }
    with open(os.path.join(run_dir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Overall metrics/plots across folds
    all_true = np.concatenate(overall_trues) if overall_trues else np.array([])
    all_pred = np.concatenate(overall_preds) if overall_preds else np.array([])
    if all_true.size > 0:
        overall_mae = float(np.mean(np.abs(all_pred - all_true)))
        overall_rmse = float(np.sqrt(np.mean((all_pred - all_true) ** 2)))
        overall_pearson = float(pearsonr(all_true, all_pred).statistic) if all_true.size > 1 else float("nan")
        overall_kendall = float(kendalltau(all_true, all_pred).statistic) if all_true.size > 1 else float("nan")
        mx, my = all_true.mean(), all_pred.mean()
        vx, vy = all_true.var(), all_pred.var()
        cov = np.mean((all_true - mx) * (all_pred - my))
        overall_ccc = float((2 * cov) / (vx + vy + (mx - my) ** 2 + 1e-8)) if all_true.size > 1 else float("nan")
        ss_tot = np.sum((all_true - mx) ** 2) + 1e-8
        ss_res = np.sum((all_true - all_pred) ** 2)
        overall_r2 = float(1 - ss_res / ss_tot)
        overall_evs = float(1 - np.var(all_true - all_pred) / (np.var(all_true) + 1e-8))
        overall_mape = float(np.mean(np.abs((all_true - all_pred) / (all_true + 1e-8))) * 100)
        overall_medae = float(np.median(np.abs(all_true - all_pred)))

        save_reg_plots(all_true, all_pred, overall_ids, run_dir, prefix="overall")
        with open(os.path.join(run_dir, "overall_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mae": overall_mae,
                    "rmse": overall_rmse,
                    "pearson": overall_pearson,
                    "kendall": overall_kendall,
                    "ccc": overall_ccc,
                    "r2": overall_r2,
                    "explained_variance": overall_evs,
                    "mape": overall_mape,
                    "medae": overall_medae,
                },
                f, indent=2, ensure_ascii=False
            )
        plot_error_distribution(per_fold_abs_err, run_dir)
        compute_saliency(model, val_loader, device, run_dir, prefix="overall_saliency")

    print("\n=== CV Summary ===")
    for k, v in summary["mean"].items():
        print(f"{k.upper()}: mean={v:.3f} std={summary['std'][k]:.3f}")
    print(f"[INFO] Run artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
