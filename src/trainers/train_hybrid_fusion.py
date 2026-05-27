"""
Hybrid COM-Anchored Spatio-Temporal GCN
for Parkinsonian Gait Severity Estimation

- Vision-based skeleton input
- COM-normalized coordinates
- Hybrid node features
- Regression only (MDS-UPDRS gait)
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Add,
    GlobalAveragePooling2D, GlobalAveragePooling1D,
    Conv2D, BatchNormalization, ReLU, Layer,
    LayerNormalization, MultiHeadAttention, Reshape
)
from tensorflow.keras.initializers import Identity
from scipy.stats import pearsonr, kendalltau
from .train_model import load_labels  # gait UPDRS labels (items 10-14 only)


# ============================================================
# 1. Skeleton Adjacency Matrix
# ============================================================

def get_skeleton_adjacency(num_joints):
    """
    Anatomical adjacency (MediaPipe-like skeleton)
    """
    edges = [
        (11,13),(13,15),   # left arm
        (12,14),(14,16),   # right arm
        (23,25),(25,27),   # left leg
        (24,26),(26,28),   # right leg
        (11,12),           # shoulders
        (23,24),           # hips
        (11,23),(12,24)    # trunk
    ]

    A = np.eye(num_joints)
    for i, j in edges:
        if i < num_joints and j < num_joints:
            A[i, j] = 1
            A[j, i] = 1

    # Row-normalization
    D = np.sum(A, axis=1, keepdims=True) + 1e-6
    A = A / D
    return A.astype(np.float32)


# ============================================================
# 2. Feature Engineering (Hybrid Node Features)
# ============================================================

def compute_relative_velocity(joints):
    """
    joints: (T, J, 3)
    """
    return np.diff(joints, axis=0, prepend=joints[:1])


def compute_amplitude(joints):
    norms = np.linalg.norm(joints, axis=2)
    return norms.max(axis=0) - norms.min(axis=0)


def compute_variability(joints):
    norms = np.linalg.norm(joints, axis=2)
    return norms.std(axis=0)


def compute_bone_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.sum(ba * bc, axis=-1) / (
        np.linalg.norm(ba, axis=-1) *
        np.linalg.norm(bc, axis=-1) + 1e-6
    )
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def build_hybrid_features(joints):
    """
    joints: (T, J, 3)  # COM-normalized
    return: (T, J, 9)
    """
    T, J, _ = joints.shape

    vel = compute_relative_velocity(joints)
    amp = compute_amplitude(joints)
    var = compute_variability(joints)

    angles = np.zeros((T, J))

    # Example: left leg angle (hip-knee-ankle)
    if J > 27:
        angles[:,25] = compute_bone_angle(
            joints[:,23], joints[:,25], joints[:,27]
        )

    features = np.concatenate([
        joints,                                # (x, y, z)
        vel,                                   # (dx, dy, dz)
        amp[None,:,None].repeat(T, axis=0),    # amplitude
        var[None,:,None].repeat(T, axis=0),    # variability
        angles[...,None]                       # angle
    ], axis=-1)

    return features.astype(np.float32)


# ============================================================
# 3. Graph Convolution Layer
# ============================================================

class GCNLayer(Layer):
    def __init__(self, out_channels, adj_matrix, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.A = tf.constant(adj_matrix)

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        self.W = self.add_weight(
            shape=(self.in_channels, self.out_channels),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, x):
        # x: (B, T, J, C)
        x = tf.einsum("btjc,ck->btjk", x, self.W)
        x = tf.einsum("ij,btjk->btik", self.A, x)
        return x


# ============================================================
# 4. Temporal Convolution
# ============================================================

class TemporalConvBlock(Layer):
    def __init__(self, channels, kernel_size=9, **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv2D(
            channels,
            kernel_size=(kernel_size, 1),
            padding="same"
        )
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


# ============================================================
# 5. ST-GCN Block
# ============================================================

class STGCNBlock(Layer):
    def __init__(self, out_channels, adj_matrix, **kwargs):
        super().__init__(**kwargs)
        self.gcn = GCNLayer(out_channels, adj_matrix)
        self.tcn = TemporalConvBlock(out_channels)

    def call(self, x):
        res = x
        x = self.gcn(x)
        x = self.tcn(x)

        if res.shape[-1] == x.shape[-1]:
            x = Add()([x, res])
        return x


# ============================================================
# 6. Hybrid COM-Anchored ST-GCN Model
# ============================================================

def build_hybrid_com_stgcn(
    input_shape,      # (T, J, 9)
    num_joints,
    optimizer="adam"
):
    """
    Regression-only Hybrid COM-Anchored ST-GCN
    """
    A = get_skeleton_adjacency(num_joints)

    inputs = Input(shape=input_shape)

    x = STGCNBlock(64, A)(inputs)
    x = STGCNBlock(128, A)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    output = Dense(1, activation="linear")(x)

    model = Model(inputs, output)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )
    return model


# ============================================================
# New Fusion v2: Adaptive GCN + Spatial Attention + Temporal Transformer
# ============================================================

class AdaptiveGCNLayer(Layer):
    def __init__(self, out_channels, num_joints, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.num_joints = num_joints

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.out_channels),
            initializer="glorot_uniform",
            trainable=True
        )
        self.A = self.add_weight(
            shape=(self.num_joints, self.num_joints),
            initializer=Identity(),
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B, T, J, C)
        x = tf.einsum("btjc,ck->btjk", x, self.W)
        A = tf.nn.softmax(self.A, axis=-1)
        return tf.einsum("ij,btjk->btik", A, x)


class SpatialAttentionBlock(Layer):
    def __init__(self, num_heads, key_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.dropout = Dropout(dropout)
        self.add = Add()
        self.norm = LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        # x: (B, T, J, C)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        J = tf.shape(x)[2]
        C = tf.shape(x)[3]

        x_flat = tf.reshape(x, (-1, J, C))  # (B*T, J, C)
        attn = self.mha(x_flat, x_flat)
        attn = self.dropout(attn, training=training)
        x_flat = self.add([x_flat, attn])
        x_flat = self.norm(x_flat)
        return tf.reshape(x_flat, (B, T, J, C))


def temporal_transformer_block(x, num_heads, key_dim, ff_dim, dropout=0.1):
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn = Dropout(dropout)(attn)
    x = Add()([x, attn])
    x = LayerNormalization(epsilon=1e-6)(x)

    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(x.shape[-1])(ff)
    ff = Dropout(dropout)(ff)
    x = Add()([x, ff])
    return LayerNormalization(epsilon=1e-6)(x)


def build_fusion_v2_model(
    input_shape,            # (T, J, C)
    num_joints,
    num_heads=4,
    key_dim=32,
    ff_dim=128,
    num_temporal_blocks=2,
    optimizer="adam"
):
    inputs = Input(shape=input_shape)

    # Adaptive GCN (spatial)
    x = AdaptiveGCNLayer(64, num_joints)(inputs)
    x = AdaptiveGCNLayer(128, num_joints)(x)

    # Spatial attention (joint-wise)
    x = SpatialAttentionBlock(num_heads, key_dim)(x)

    # Flatten joints for temporal modeling
    T, J, _ = input_shape
    x = Reshape((T, J * 128))(x)

    # Temporal transformers
    for _ in range(num_temporal_blocks):
        x = temporal_transformer_block(x, num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation="linear")(x)

    model = Model(inputs, output)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


# ============================================================
# 7. Training / Evaluation Helpers (keeps model intact)
# ============================================================

def pad_or_clip(x, max_len):
    T = x.shape[0]
    if T > max_len:
        return x[:max_len]
    if T < max_len:
        pad = np.repeat(x[-1:], max_len - T, axis=0)
        return np.concatenate([x, pad], axis=0)
    return x


def ensure_tensor_shape(raw: np.ndarray) -> np.ndarray:
    if raw.ndim == 3:
        return raw
    if raw.ndim == 2 and raw.shape[1] % 33 == 0:
        C = raw.shape[1] // 33
        return raw.reshape(raw.shape[0], 33, C)
    raise ValueError(f"Unsupported pose shape {raw.shape}, expected (T,J,C) or (T,33*C).")


def load_dataset(processed_dir, label_dir, max_seconds=13.0, fps=30.0):
    labels = load_labels(label_dir)
    max_len = int(max_seconds * fps)
    X_list, y_list, ids = [], [], []
    for root, _, files in os.walk(processed_dir):
        for f in files:
            if not f.endswith("_pose.npy"):
                continue
            pid = f.split("_")[0]
            if pid not in labels:
                continue
            raw = np.load(os.path.join(root, f))
            raw = ensure_tensor_shape(raw)
            raw = pad_or_clip(raw, max_len)
            feats = build_hybrid_features(raw)
            X_list.append(feats)
            y_list.append(labels[pid]["gait_updrs"])
            ids.append(f)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, ids, max_len


def plot_and_save(y_true, y_pred, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    pear = float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")
    kend = float(kendalltau(y_true, y_pred).correlation) if len(y_true) > 1 else float("nan")
    mx, my = y_true.mean(), y_pred.mean()
    vx, vy = y_true.var(), y_pred.var()
    cov = ((y_true - mx) * (y_pred - my)).mean()
    ccc = float((2 * cov) / (vx + vy + (mx - my) ** 2 + 1e-8)) if len(y_true) > 1 else float("nan")
    r2 = float(r2_score(y_true, y_pred))
    evs = float(explained_variance_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))

    with open(os.path.join(out_dir, "regression_errors.txt"), "w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    f"MAE: {mae:.4f}",
                    f"RMSE: {rmse:.4f}",
                    f"Pearson: {pear:.4f}",
                    f"Kendall: {kend:.4f}",
                    f"CCC: {ccc:.4f}",
                    f"R2: {r2:.4f}",
                    f"EVS: {evs:.4f}",
                    f"MAPE: {mape:.4f}",
                    f"MedAE: {medae:.4f}",
                ]
            )
            + "\n"
        )
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mae": float(mae),
                "rmse": float(rmse),
                "pearson": pear,
                "kendall": kend,
                "ccc": ccc,
                "r2": r2,
                "explained_variance": evs,
                "mape": mape,
                "medae": medae,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(os.path.join(out_dir, "predictions.tsv"), "w", encoding="utf-8") as f:
        f.write("true\tpred\tabs_err\n")
        for t, p in zip(y_true, y_pred):
            f.write(f"{t:.6f}\t{p:.6f}\t{abs(p-t):.6f}\n")

    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("True Gait UPDRS")
    plt.ylabel("Predicted")
    plt.title(f"Scatter (MAE={mae:.2f}, RMSE={rmse:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "regression_scatter.png"))
    plt.close()

    residuals = y_pred - y_true
    plt.figure()
    plt.hist(residuals, bins=20, alpha=0.8)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "regression_residuals.png"))
    plt.close()

    plt.figure()
    plt.hist(y_true, bins=20, alpha=0.6, label="True")
    plt.hist(y_pred, bins=20, alpha=0.6, label="Pred")
    plt.xlabel("Gait UPDRS")
    plt.ylabel("Count")
    plt.title("True vs Pred Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "distribution_true_vs_pred.png"))
    plt.close()

    abs_err = np.abs(residuals)
    sorted_err = np.sort(abs_err)
    cdf = np.arange(1, len(sorted_err)+1) / len(sorted_err)
    plt.figure()
    plt.plot(sorted_err, cdf)
    plt.xlabel("Absolute Error")
    plt.ylabel("CDF")
    plt.title("Absolute Error CDF")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "abs_error_cdf.png"))
    plt.close()


def run_cv(X, y, epochs=20, batch_size=4, folds=5, lr=1e-3):
    # Single hold-out split (80/20) for consistent environment
    from sklearn.model_selection import train_test_split

    tr_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.2, shuffle=True, random_state=42)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", "fusion_tf_runs", ts)
    os.makedirs(run_dir, exist_ok=True)

    model = build_fusion_v2_model(
        input_shape=X.shape[1:],
        num_joints=X.shape[2],
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
    )
    history = model.fit(
        X[tr_idx], y[tr_idx],
        validation_data=(X[val_idx], y[val_idx]),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, monitor="val_loss")]
    )
    preds = model.predict(X[val_idx]).flatten()
    mae = mean_absolute_error(y[val_idx], preds)
    rmse = mean_squared_error(y[val_idx], preds, squared=False)

    plot_and_save(y[val_idx], preds, run_dir)
    # Save weights only to avoid serialization issues with custom layers
    model.save_weights(os.path.join(run_dir, "best_weights.weights.h5"))
    hist_safe = {k: [float(vv) for vv in vals] for k, vals in history.history.items()}
    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(hist_safe, f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"mae": float(mae), "rmse": float(rmse)}, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Run artifacts saved to: {run_dir}")


# ============================================================
# 8. CLI Entry
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Hybrid COM ST-GCN (TF) with visualization")
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max_seconds", type=float, default=13.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    X, y, ids, max_len = load_dataset(args.processed_dir, args.label_dir, args.max_seconds, args.fps)
    if len(X) == 0:
        print("No samples found. Check processed_dir/label_dir.")
        return
    run_cv(X, y, epochs=args.epochs, batch_size=args.batch_size, folds=args.folds, lr=args.lr)


if __name__ == "__main__":
    main()
