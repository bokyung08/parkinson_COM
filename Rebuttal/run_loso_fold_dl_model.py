"""Train one LOSO fold for one DL model under the unified rebuttal protocol."""

from __future__ import annotations

import argparse
import importlib.util
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

from Rebuttal.unified_data_utils import load_unified_dataset, regression_metrics


MODEL_DISPLAY = {
    "main_tf": "MainTF",
    "fusion_tf": "FusionTF",
    "hybrid_torch": "HybridTorch",
}


def set_python_numpy_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_main_tf_builder():
    """Load src/models/model_builder.py without running src.models.__init__."""
    module_path = REPO_ROOT / "src" / "models" / "model_builder.py"
    spec = importlib.util.spec_from_file_location("loso_main_tf_model_builder", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model builder from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_pose_model


def save_predictions(
    out_dir: Path,
    sample_ids: list[str],
    patient_ids: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.tsv").open("w", encoding="utf-8") as f:
        f.write("sample_id\tpatient_id\ty_true\ty_pred\tabs_error\n")
        for sample_id, patient_id, true_value, pred_value in zip(sample_ids, patient_ids, y_true, y_pred):
            f.write(
                f"{sample_id}\t{patient_id}\t{true_value:.6f}\t"
                f"{pred_value:.6f}\t{abs(pred_value - true_value):.6f}\n"
            )

    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.75)
    lower = min(float(np.min(y_true)), float(np.min(y_pred)))
    upper = max(float(np.max(y_true)), float(np.max(y_pred)))
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    plt.plot([lower, upper], [lower, upper], "r--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter.png", dpi=200)
    plt.close()


def build_fold_row(
    args: argparse.Namespace,
    params: int,
    train_seconds: float,
    metrics: dict[str, float],
    n_train: int,
    n_val: int,
) -> dict[str, object]:
    return {
        "model": MODEL_DISPLAY[args.model],
        "model_key": args.model,
        "target": args.target,
        "ablation": args.ablation,
        "fold": args.fold_index,
        "val_patient_id": args.val_patient_id,
        "n_train": n_train,
        "n_val": n_val,
        "params": int(params),
        "train_seconds": float(train_seconds),
        **metrics,
    }


def train_main_tf(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    model_dir: Path,
) -> tuple[dict[str, object], np.ndarray]:
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam

    set_python_numpy_seeds(args.random_state)
    tf.random.set_seed(args.random_state)

    optimizer = Adam(learning_rate=args.learning_rate, clipnorm=1.0)
    build_pose_model = load_main_tf_builder()
    model = build_pose_model(x.shape[1:], optimizer=optimizer)
    best_path = model_dir / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(best_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0,
        ),
    ]
    start = time.perf_counter()
    history = model.fit(
        x[train_idx],
        y[train_idx],
        validation_data=(x[val_idx], y[val_idx]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    train_seconds = time.perf_counter() - start
    if best_path.exists():
        model.load_weights(str(best_path))
    pred = model.predict(x[val_idx], verbose=0).reshape(-1)
    metrics = regression_metrics(y[val_idx], pred)
    write_json(model_dir / "history.json", {k: [float(v) for v in vals] for k, vals in history.history.items()})
    write_json(model_dir / "metrics.json", metrics)
    row = build_fold_row(args, int(model.count_params()), train_seconds, metrics, len(train_idx), len(val_idx))
    return row, pred


def train_fusion_tf(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    model_dir: Path,
) -> tuple[dict[str, object], np.ndarray]:
    import tensorflow as tf
    from Rebuttal.run_unified_fusion_tf import build_model

    set_python_numpy_seeds(args.random_state)
    tf.random.set_seed(args.random_state)

    model = build_model(x.shape[1:], x.shape[2], args.learning_rate)
    best_path = model_dir / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(best_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0,
        ),
    ]
    start = time.perf_counter()
    history = model.fit(
        x[train_idx],
        y[train_idx],
        validation_data=(x[val_idx], y[val_idx]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    train_seconds = time.perf_counter() - start
    if best_path.exists():
        model.load_weights(str(best_path))
    pred = model.predict(x[val_idx], verbose=0).reshape(-1)
    metrics = regression_metrics(y[val_idx], pred)
    write_json(model_dir / "history.json", {k: [float(v) for v in vals] for k, vals in history.history.items()})
    write_json(model_dir / "metrics.json", metrics)
    row = build_fold_row(args, int(model.count_params()), train_seconds, metrics, len(train_idx), len(val_idx))
    return row, pred


def train_hybrid_torch(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    model_dir: Path,
) -> tuple[dict[str, object], np.ndarray]:
    import torch
    import torch.nn as nn
    from src.hybrid_gcn import HybridCOMGCNv2

    set_python_numpy_seeds(args.random_state)
    torch.manual_seed(args.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)

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
            source_idx = int(self.index_array[dataset_idx])
            return (
                torch.from_numpy(self.x_array[source_idx]).float(),
                torch.tensor(self.y_array[source_idx], dtype=torch.float32),
                source_idx,
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
    with torch.no_grad():
        for xb, _, _ in val_loader:
            preds.extend(model(xb.to(device)).cpu().numpy().reshape(-1).tolist())
    pred = np.asarray(preds, dtype=np.float32)
    metrics = regression_metrics(y[val_idx], pred)
    write_json(model_dir / "history.json", history)
    write_json(model_dir / "metrics.json", metrics)
    params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    row = build_fold_row(args, params, train_seconds, metrics, len(train_idx), len(val_idx))
    return row, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one LOSO fold for one DL model.")
    parser.add_argument("--processed_dir", default="HospitalData/processed_pose_data")
    parser.add_argument("--label_dir", default="HospitalData/JSON")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target", choices=["item10", "gait"], default="item10")
    parser.add_argument("--ablation", choices=["A", "B", "C", "D"], default="D")
    parser.add_argument("--model", choices=["main_tf", "fusion_tf", "hybrid_torch"], required=True)
    parser.add_argument("--val_patient_id", required=True)
    parser.add_argument("--fold_index", type=int, required=True)
    parser.add_argument("--max_len", type=int, default=390)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    model_dir = out_dir / args.model
    model_dir.mkdir(parents=True, exist_ok=True)

    x, y, ids, manifest = load_unified_dataset(
        Path(args.processed_dir),
        Path(args.label_dir),
        target=args.target,
        ablation=args.ablation,
        max_len=args.max_len,
    )
    patient_ids = manifest["patient_id"].astype(str).to_numpy()
    val_idx = np.where(patient_ids == args.val_patient_id)[0]
    if len(val_idx) == 0:
        raise SystemExit(f"No validation samples found for patient {args.val_patient_id}")
    train_idx = np.where(patient_ids != args.val_patient_id)[0]
    if len(train_idx) == 0:
        raise SystemExit("LOSO fold has no training samples.")

    manifest.to_csv(out_dir / "dataset_manifest.tsv", sep="\t", index=False)
    fold_meta = {
        "target": args.target,
        "ablation": args.ablation,
        "model": args.model,
        "fold": args.fold_index,
        "val_patient_id": args.val_patient_id,
        "train_indices": train_idx.astype(int).tolist(),
        "val_indices": val_idx.astype(int).tolist(),
        "max_len": args.max_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": args.device,
    }
    write_json(model_dir / "fold_config.json", fold_meta)

    print(
        f"[INFO] LOSO fold={args.fold_index} model={args.model} "
        f"ablation={args.ablation} train={len(train_idx)} val={len(val_idx)} "
        f"val_patient={args.val_patient_id}"
    )

    if args.model == "main_tf":
        row, pred = train_main_tf(x, y, train_idx, val_idx, args, model_dir)
    elif args.model == "fusion_tf":
        row, pred = train_fusion_tf(x, y, train_idx, val_idx, args, model_dir)
    else:
        row, pred = train_hybrid_torch(x, y, train_idx, val_idx, args, model_dir)

    val_sample_ids = [ids[i] for i in val_idx]
    val_patient_ids = [str(patient_ids[i]) for i in val_idx]
    save_predictions(model_dir, val_sample_ids, val_patient_ids, y[val_idx], pred)
    write_json(model_dir / "row.json", row)
    pd.DataFrame([row]).to_csv(model_dir / "summary.csv", index=False)
    print(pd.DataFrame([row]))


if __name__ == "__main__":
    main()
