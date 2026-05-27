"""Train the Hybrid Torch baseline under the unified rebuttal protocol."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Rebuttal.unified_data_utils import (
    deterministic_split,
    load_unified_dataset,
    regression_metrics,
)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def save_outputs(out_dir: Path, ids: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = regression_metrics(y_true, y_pred)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / "predictions.tsv").open("w", encoding="utf-8") as f:
        f.write("sample_id\ty_true\ty_pred\tabs_error\n")
        for sample_id, t, p in zip(ids, y_true, y_pred):
            f.write(f"{sample_id}\t{t:.6f}\t{p:.6f}\t{abs(p-t):.6f}\n")
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.75)
    lower = min(float(np.min(y_true)), float(np.min(y_pred)))
    upper = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([lower, upper], [lower, upper], "r--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter.png", dpi=200)
    plt.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="HospitalData/processed_pose_data")
    parser.add_argument("--label_dir", default="HospitalData/JSON")
    parser.add_argument("--out_dir", default="Rebuttal/results/unified_dl_baselines/D_item10_90_10")
    parser.add_argument("--target", choices=["item10", "gait"], default="item10")
    parser.add_argument("--ablation", choices=["A", "B", "C", "D"], default="D")
    parser.add_argument("--max_len", type=int, default=390)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    model_dir = out_dir / "hybrid_torch"
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, ids, manifest = load_unified_dataset(
        Path(args.processed_dir),
        Path(args.label_dir),
        target=args.target,
        ablation=args.ablation,
        max_len=args.max_len,
    )
    train_idx, val_idx = deterministic_split(len(y), args.test_size, args.random_state)
    manifest.to_csv(out_dir / "dataset_manifest.tsv", sep="\t", index=False)
    with (out_dir / "val_samples.txt").open("w", encoding="utf-8") as f:
        for i in val_idx:
            f.write(f"{ids[i]}\n")

    import torch
    import torch.nn as nn
    from src.hybrid_gcn import HybridCOMGCNv2

    set_seeds(args.random_state)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = HybridCOMGCNv2(in_channels=x.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.MSELoss()

    class NumpyIndexDataset(torch.utils.data.Dataset):
        def __init__(self, x_array, y_array, index_array):
            self.x_array = x_array
            self.y_array = y_array
            self.index_array = np.asarray(index_array)

        def __len__(self):
            return len(self.index_array)

        def __getitem__(self, dataset_idx):
            source_idx = self.index_array[dataset_idx]
            return (
                torch.from_numpy(self.x_array[source_idx]).float(),
                torch.tensor(self.y_array[source_idx], dtype=torch.float32),
                int(source_idx),
            )

    generator = torch.Generator().manual_seed(args.random_state)
    train_loader = torch.utils.data.DataLoader(
        NumpyIndexDataset(x, y, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = torch.utils.data.DataLoader(
        NumpyIndexDataset(x, y, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best.pth"
    best_val = float("inf")
    history = {"loss": [], "mae": [], "val_loss": [], "val_mae": [], "learning_rate": []}
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        abs_errors = []
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            abs_errors.extend(torch.abs(pred.detach() - yb).cpu().numpy().tolist())
        model.eval()
        val_losses = []
        val_abs = []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(float(loss.item()))
                val_abs.extend(torch.abs(pred - yb).cpu().numpy().tolist())
        train_loss = float(np.mean(losses))
        train_mae = float(np.mean(abs_errors))
        val_loss = float(np.mean(val_losses))
        val_mae = float(np.mean(val_abs))
        history["loss"].append(train_loss)
        history["mae"].append(train_mae)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        scheduler.step(val_loss)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"loss={train_loss:.4f} mae={train_mae:.4f} "
            f"val_loss={val_loss:.4f} val_mae={val_mae:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
    train_seconds = time.perf_counter() - start
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    preds = []
    ordered_indices = []
    with torch.no_grad():
        for xb, _, source_idx in val_loader:
            pred_batch = model(xb.to(device)).cpu().numpy().reshape(-1)
            preds.extend(pred_batch.tolist())
            ordered_indices.extend([int(i) for i in source_idx.numpy().tolist()])
    pred = np.asarray(preds, dtype=np.float32)
    y_true = np.asarray([y[i] for i in ordered_indices], dtype=np.float32)
    ordered_ids = [ids[i] for i in ordered_indices]
    metrics = save_outputs(model_dir, ordered_ids, y_true, pred)
    row = {
        "model": "HybridTorch",
        "params": parameter_count(model),
        "train_seconds": float(train_seconds),
        **metrics,
    }
    with (model_dir / "row.json").open("w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    with (model_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    pd.DataFrame([row]).to_csv(model_dir / "summary.csv", index=False)
    print(pd.DataFrame([row]))


if __name__ == "__main__":
    main()
