"""Train the Fusion TF baseline under the unified rebuttal protocol."""

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

import tensorflow as tf
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

from Rebuttal.unified_data_utils import (
    deterministic_split,
    load_unified_dataset,
    regression_metrics,
)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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


def build_model(input_shape, num_joints: int, learning_rate: float) -> Model:
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
    args = parser.parse_args()

    set_seeds(args.random_state)
    out_dir = Path(args.out_dir)
    model_dir = out_dir / "fusion_tf"
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

    model = build_model(x.shape[1:], x.shape[2], args.learning_rate)
    best_path = model_dir / "best.weights.h5"
    model_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(best_path), monitor="val_loss", mode="min", save_best_only=True, save_weights_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
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
    metrics = save_outputs(model_dir, [ids[i] for i in val_idx], y[val_idx], pred)
    row = {
        "model": "FusionTF",
        "params": int(model.count_params()),
        "train_seconds": float(train_seconds),
        **metrics,
    }
    with (model_dir / "row.json").open("w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    with (model_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    pd.DataFrame([row]).to_csv(model_dir / "summary.csv", index=False)
    print(pd.DataFrame([row]))


if __name__ == "__main__":
    main()
