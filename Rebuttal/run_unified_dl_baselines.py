"""Unified DL baseline training for rebuttal.

Runs the two non-proposed deep-learning baselines under the same protocol:
same dataset, target, ablation feature set, temporal window, split, loss,
optimizer learning rate, epochs, and batch size.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import Identity
from tensorflow.keras.layers import (
    Add,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
    Reshape,
)
from tensorflow.keras.models import Model


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Rebuttal.run_baseline_comparison import load_configuration_d_dataset
from src.hybrid_gcn import HybridCOMGCNv2


@dataclass(frozen=True)
class UnifiedConfig:
    processed_dir: str
    label_dir: str
    out_dir: str
    target: str
    ablation: str
    max_len: int
    test_size: float
    random_state: int
    epochs: int
    batch_size: int
    learning_rate: float
    loss: str
    temporal_window: str


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AdaptiveGCNLayer(Layer):
    def __init__(self, out_channels: int, num_joints: int, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.num_joints = num_joints

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.out_channels),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.A = self.add_weight(
            shape=(self.num_joints, self.num_joints),
            initializer=Identity(),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        x = tf.einsum("btjc,ck->btjk", x, self.W)
        A = tf.nn.softmax(self.A, axis=-1)
        return tf.einsum("ij,btjk->btik", A, x)


class SpatialAttentionBlock(Layer):
    def __init__(self, num_heads: int, key_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.dropout = Dropout(dropout)
        self.add = Add()
        self.norm = LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        batch = tf.shape(x)[0]
        frames = tf.shape(x)[1]
        joints = tf.shape(x)[2]
        channels = tf.shape(x)[3]
        x_flat = tf.reshape(x, (-1, joints, channels))
        attn = self.mha(x_flat, x_flat)
        attn = self.dropout(attn, training=training)
        x_flat = self.add([x_flat, attn])
        x_flat = self.norm(x_flat)
        return tf.reshape(x_flat, (batch, frames, joints, channels))


def temporal_transformer_block(x, num_heads: int, key_dim: int, ff_dim: int, dropout: float = 0.1):
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn = Dropout(dropout)(attn)
    x = Add()([x, attn])
    x = LayerNormalization(epsilon=1e-6)(x)
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(x.shape[-1])(ff)
    ff = Dropout(dropout)(ff)
    x = Add()([x, ff])
    return LayerNormalization(epsilon=1e-6)(x)


def build_fusion_tf_model(input_shape, num_joints: int, learning_rate: float) -> Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = AdaptiveGCNLayer(64, num_joints)(inputs)
    x = AdaptiveGCNLayer(128, num_joints)(x)
    x = SpatialAttentionBlock(num_heads=4, key_dim=32)(x)
    frames, joints, _ = input_shape
    x = Reshape((frames, joints * 128))(x)
    for _ in range(2):
        x = temporal_transformer_block(x, num_heads=4, key_dim=32, ff_dim=128)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation="linear")(x)
    model = Model(inputs, output, name="FusionTFUnified")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="mse",
        metrics=["mae"],
    )
    return model


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "medae": float(median_absolute_error(y_true, y_pred)),
    }


def save_predictions(out_dir: Path, ids: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.tsv").open("w", encoding="utf-8") as f:
        f.write("sample_id\ty_true\ty_pred\tabs_error\n")
        for sample_id, true_value, pred_value in zip(ids, y_true, y_pred):
            f.write(f"{sample_id}\t{true_value:.6f}\t{pred_value:.6f}\t{abs(pred_value - true_value):.6f}\n")

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

    residuals = y_pred - y_true
    plt.figure()
    plt.hist(residuals, bins=10, alpha=0.8)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "residuals.png", dpi=200)
    plt.close()


def parameter_count_torch(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_fusion_tf(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    ids: list[str],
    config: UnifiedConfig,
    out_dir: Path,
) -> dict[str, object]:
    set_all_seeds(config.random_state)
    model = build_fusion_tf_model(x.shape[1:], x.shape[2], config.learning_rate)
    model_dir = out_dir / "fusion_tf"
    model_dir.mkdir(parents=True, exist_ok=True)
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
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    train_seconds = time.perf_counter() - start
    if best_path.exists():
        model.load_weights(str(best_path))
    pred = model.predict(x[val_idx], verbose=0).reshape(-1)
    metric = metrics_dict(y[val_idx], pred)
    save_predictions(model_dir, [ids[i] for i in val_idx], y[val_idx], pred)
    with (model_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in values] for k, values in history.history.items()}, f, indent=2)
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metric, f, indent=2)
    return {
        "model": "FusionTF",
        "train_seconds": float(train_seconds),
        "params": int(model.count_params()),
        **metric,
    }


def train_hybrid_torch(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    ids: list[str],
    config: UnifiedConfig,
    out_dir: Path,
) -> dict[str, object]:
    set_all_seeds(config.random_state)
    model_dir = out_dir / "hybrid_torch"
    model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridCOMGCNv2(in_channels=x.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_path = model_dir / "best.pth"
    history = {"loss": [], "mae": [], "val_loss": [], "val_mae": []}

    x_train = torch.tensor(x[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32)
    x_val = torch.tensor(x[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.random_state),
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_val, y_val),
        batch_size=config.batch_size,
        shuffle=False,
    )

    start = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        train_abs = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_abs.extend(torch.abs(pred.detach() - yb).cpu().numpy().tolist())

        model.eval()
        val_losses = []
        val_abs = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(float(loss.item()))
                val_abs.extend(torch.abs(pred - yb).cpu().numpy().tolist())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        train_mae = float(np.mean(train_abs))
        val_mae = float(np.mean(val_abs))
        history["loss"].append(train_loss)
        history["mae"].append(train_mae)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        scheduler.step(val_loss)
        print(
            f"[HybridTorch] Epoch {epoch + 1}/{config.epochs} "
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
    with torch.no_grad():
        pred = model(x_val.to(device)).cpu().numpy().reshape(-1)
    metric = metrics_dict(y[val_idx], pred)
    save_predictions(model_dir, [ids[i] for i in val_idx], y[val_idx], pred)
    with (model_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metric, f, indent=2)
    return {
        "model": "HybridTorch",
        "train_seconds": float(train_seconds),
        "params": parameter_count_torch(model),
        **metric,
    }


def write_summary(out_dir: Path, config: UnifiedConfig, rows: list[dict[str, object]], val_ids: list[str]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "unified_dl_baseline_summary.csv", index=False)
    with (out_dir / "unified_dl_baseline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)
    with (out_dir / "val_samples.txt").open("w", encoding="utf-8") as f:
        for sample_id in val_ids:
            f.write(f"{sample_id}\n")

    lines = [
        "# Unified DL Baseline Results",
        "",
        "## Protocol",
        "",
        f"- Target: `{config.target}`",
        f"- Ablation: `{config.ablation}`",
        f"- Split: `{1 - config.test_size:.0%}/{config.test_size:.0%}` hold-out, random_state={config.random_state}",
        f"- Max length: `{config.max_len}` frames, `{config.temporal_window}` window",
        f"- Epochs: `{config.epochs}`",
        f"- Batch size: `{config.batch_size}`",
        f"- Learning rate: `{config.learning_rate}`",
        f"- Loss: `{config.loss}`",
        "",
        "## Results",
        "",
        "| Model | Params | MAE | RMSE | MedAE | Train sec |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {int(row['params'])} | "
            f"{float(row['mae']):.3f} | {float(row['rmse']):.3f} | "
            f"{float(row['medae']):.3f} | {float(row['train_seconds']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Both non-proposed DL baselines were retrained under one unified protocol. "
            "Use these numbers when discussing the two DL baselines; do not mix them with older baseline runs that used different split/loss/lr/window settings.",
        ]
    )
    (out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train non-proposed DL baselines under unified conditions.")
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
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = UnifiedConfig(
        processed_dir=args.processed_dir,
        label_dir=args.label_dir,
        out_dir=args.out_dir,
        target=args.target,
        ablation=args.ablation,
        max_len=args.max_len,
        test_size=args.test_size,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        loss="mse",
        temporal_window="last",
    )

    set_all_seeds(config.random_state)
    x, y, ids, manifest = load_configuration_d_dataset(
        processed_dir=Path(config.processed_dir),
        label_dir=Path(config.label_dir),
        target=config.target,
        ablation=config.ablation,
        max_len=config.max_len,
    )
    train_idx, val_idx = train_test_split(
        np.arange(len(y)),
        test_size=config.test_size,
        shuffle=True,
        random_state=config.random_state,
    )
    manifest.to_csv(out_dir / "dataset_manifest.tsv", sep="\t", index=False)

    print(f"[INFO] Loaded X={x.shape}, y={y.shape}")
    print(f"[INFO] Train n={len(train_idx)}, Val n={len(val_idx)}")
    print("[INFO] Training FusionTF baseline...")
    rows = [
        train_fusion_tf(x, y, train_idx, val_idx, ids, config, out_dir),
    ]
    print("[INFO] Training HybridTorch baseline...")
    rows.append(train_hybrid_torch(x, y, train_idx, val_idx, ids, config, out_dir))
    write_summary(out_dir, config, rows, [ids[i] for i in val_idx])
    print(f"[INFO] Unified DL baseline results saved to: {out_dir}")
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
