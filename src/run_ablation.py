import os
import csv
import json
import argparse
from datetime import datetime
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, kendalltau
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .feature_engineering import build_hybrid_node_features, apply_ablation
from .hybrid_gcn import HybridCOMGCN
from .train_model import load_labels  # 라벨 파싱 재사용


# -----------------------------
# Dataset / Data utils
# -----------------------------
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


# -----------------------------
# Model variants
# -----------------------------
def build_model(variant: str, in_channels: int):
    """
    variant:
      - 'gcn2' : 기본 2-블록 HybridCOMGCN
      - 'gcn1' : 1-블록으로 얕은 모델
    """
    if variant == "gcn1":
        from .hybrid_gcn import GCNBlock, build_mediapipe_adjacency
        class OneBlock(nn.Module):
            def __init__(self, in_ch):
                super().__init__()
                self.A = build_mediapipe_adjacency()
                self.gcn = GCNBlock(in_ch, 128)
                self.temporal_pool = nn.AdaptiveAvgPool1d(1)
                self.regressor = nn.Linear(128, 1)
            def forward(self, x):
                A = self.A.to(x.device)
                x = self.gcn(x, A)
                x = x.mean(dim=2)
                x = self.temporal_pool(x.transpose(1, 2)).squeeze(-1)
                return self.regressor(x).squeeze(-1)
        return OneBlock(in_channels)
    # default
    return HybridCOMGCN(in_channels=in_channels)


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_fold(model, train_loader, val_loader, device, epochs=80, lr=1e-3, save_path=None):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-5, verbose=False)
    criterion = nn.L1Loss()  # MAE
    best_val = float("inf")

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
        scheduler.step(mean_val)
        if mean_val < best_val and save_path:
            best_val = mean_val
            torch.save(model.state_dict(), save_path)
    return model


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(device)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
            trues.append(yb.numpy())
    y_pred = np.concatenate(preds).flatten()
    y_true = np.concatenate(trues).flatten()
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    pearson = pearsonr(y_true, y_pred).statistic if len(y_true) > 1 else np.nan
    kendall = kendalltau(y_true, y_pred).statistic if len(y_true) > 1 else np.nan
    mx, my = np.mean(y_true), np.mean(y_pred)
    vx, vy = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mx) * (y_pred - my))
    ccc = (2 * cov) / (vx + vy + (mx - my) ** 2 + 1e-8)
    ss_tot = np.sum((y_true - mx) ** 2) + 1e-8
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    evs = 1 - np.var(y_true - y_pred) / (np.var(y_true) + 1e-8)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    medae = np.median(np.abs(y_true - y_pred))
    return mae, rmse, pearson, kendall, ccc, r2, evs, mape, medae


# -----------------------------
# Ablation runner
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="HospitalData/processed_pose_data")
    parser.add_argument("--label_dir", default="HospitalData/JSON")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=13.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--suffix", default="_2_pose.npy", help="Which npy files to use (default: _2_pose.npy)")
    args = parser.parse_args()

    run_dir = args.run_dir or os.path.join("results", "ablation_runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    max_len = int(args.seconds * args.fps)
    npy_files, labels = list_npy_and_labels(args.processed_dir, args.label_dir, use_only_suffix=args.suffix)
    if len(npy_files) == 0:
        raise SystemExit("No samples found for ablation. Check processed_dir / suffix.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=min(args.folds, len(npy_files)), shuffle=True, random_state=42)

    # Define ablation settings
    feature_modes = ["A", "B", "C", "D"]  # 전처리/특징 차이
    model_variants = ["gcn1", "gcn2"]     # 모델 구조 차이

    results_csv = os.path.join(run_dir, "ablation_results.csv")
    with open(results_csv, "w", newline='', encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["feature_mode", "model_variant", "fold", "mae", "rmse", "pearson", "kendall", "ccc", "r2", "explained_variance", "mape", "medae"])

        for feat in feature_modes:
            for mv in model_variants:
                fold_metrics = []
                print(f"\n==== Ablation feat={feat}, model={mv} ====")
                for fold, (train_idx, val_idx) in enumerate(kf.split(npy_files), 1):
                    train_files = [npy_files[i] for i in train_idx]
                    val_files = [npy_files[i] for i in val_idx]
                    train_labels = [labels[i] for i in train_idx]
                    val_labels = [labels[i] for i in val_idx]

                    train_ds = PoseDataset(train_files, train_labels, ablation_mode=feat, max_len=max_len)
                    val_ds = PoseDataset(val_files, val_labels, ablation_mode=feat, max_len=max_len)
                    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
                    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

                    model = build_model(mv, in_channels=train_ds[0][0].shape[-1])
                    best_path = os.path.join(run_dir, f"{feat}_{mv}_fold{fold}_best.pth")
                    model = train_one_fold(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, save_path=best_path)
                    if os.path.exists(best_path):
                        model.load_state_dict(torch.load(best_path, map_location=device))
                    mae, rmse, pear, kend, ccc, r2, evs, mape, medae = evaluate(model, val_loader, device)
                    writer.writerow([feat, mv, fold, f"{mae:.4f}", f"{rmse:.4f}", f"{pear:.4f}", f"{kend:.4f}", f"{ccc:.4f}", f"{r2:.4f}", f"{evs:.4f}", f"{mape:.4f}", f"{medae:.4f}"])
                    fcsv.flush()
                    fold_metrics.append((mae, rmse, pear, kend, ccc, r2, evs, mape, medae))

                # summary row per (feat, mv)
                arr = np.array(fold_metrics)
                mean_vals = np.mean(arr, axis=0)
                std_vals = np.std(arr, axis=0)
                writer.writerow([feat, mv, "mean"] + [f"{v:.4f}" for v in mean_vals])
                writer.writerow([feat, mv, "std"] + [f"{v:.4f}" for v in std_vals])
                fcsv.flush()

    print(f"[INFO] Ablation results saved to: {results_csv}")


if __name__ == "__main__":
    main()
