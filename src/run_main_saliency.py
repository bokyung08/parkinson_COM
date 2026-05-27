"""
Grad-based saliency for the main TF model.
Creates joint/time importance heatmaps from input gradients.
"""

import argparse
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.feature_engineering import build_features_from_npy
from src.model_builder import build_pose_model
from src.train_model import load_labels


def ensure_tensor_shape(raw: np.ndarray) -> np.ndarray:
    if raw.ndim == 3:
        return raw
    if raw.ndim == 2 and raw.shape[1] % 33 == 0:
        c = raw.shape[1] // 33
        return raw.reshape(raw.shape[0], 33, c)
    raise ValueError(f"Unexpected pose shape {raw.shape}")


def pad_or_clip(x: np.ndarray, max_len: int) -> np.ndarray:
    t = x.shape[0]
    if t > max_len:
        return x[:max_len]
    if t < max_len:
        pad = np.repeat(x[-1:], max_len - t, axis=0)
        return np.concatenate([x, pad], axis=0)
    return x


def detect_input_channels(weights_path: str, default_c: int = 9) -> int:
    if not os.path.exists(weights_path):
        return default_c
    try:
        with h5py.File(weights_path, "r") as f:
            for k in f.keys():
                group = f[k]
                if "dense" in group:
                    if "kernel:0" in group:
                        shape = group["kernel:0"].shape
                        return int(shape[0])
        return default_c
    except OSError:
        return default_c


def compute_saliency(model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    x_tf = tf.convert_to_tensor(x[None, ...])
    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        pred = model(x_tf, training=False)
    grads = tape.gradient(pred, x_tf).numpy()[0]  # (T, J, C)
    sal = np.mean(np.abs(grads), axis=-1)  # (T, J)
    return sal


def save_heatmap(sal: np.ndarray, out_path: str, title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.imshow(sal.T, aspect="auto", origin="lower")
    plt.colorbar(label="Saliency")
    plt.xlabel("Frame")
    plt.ylabel("Joint")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_joint_bar(joint_imp: np.ndarray, out_path: str, title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(joint_imp)), joint_imp)
    plt.xlabel("Joint")
    plt.ylabel("Mean saliency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Main TF saliency (grad-based)")
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--model_weights", required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--suffix", default="_2_pose.npy")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    max_len = int(args.seconds * args.fps)
    out_dir = args.out_dir or os.path.join("results", "main_saliency")
    os.makedirs(out_dir, exist_ok=True)

    labels = load_labels(args.label_dir)
    npy_files = [p for p in glob.glob(os.path.join(args.processed_dir, "**", f"*{args.suffix}"), recursive=True)]
    npy_files = [
        p for p in npy_files
        if os.path.basename(p).replace(args.suffix, "") in labels
    ]
    if not npy_files:
        raise SystemExit("No matching npy files with labels found.")

    target_c = detect_input_channels(args.model_weights, default_c=9)

    first_raw = ensure_tensor_shape(np.load(npy_files[0]))
    first_raw = pad_or_clip(first_raw, max_len)
    first_feats = build_features_from_npy(first_raw, ablation="D")
    if first_feats.shape[-1] < target_c:
        pad_c = target_c - first_feats.shape[-1]
        first_feats = np.pad(first_feats, ((0, 0), (0, 0), (0, pad_c)), mode="constant")
    elif first_feats.shape[-1] > target_c:
        first_feats = first_feats[..., :target_c]

    input_shape = (first_feats.shape[0], first_feats.shape[1], target_c)
    model = build_pose_model(input_shape)
    # Ensure all sublayers are built before loading weights
    model(tf.zeros((1,) + input_shape))
    model.load_weights(args.model_weights)

    all_sal = []
    records = []
    for i, p in enumerate(npy_files):
        if args.max_samples and i >= args.max_samples:
            break
        pid = os.path.basename(p).replace("_pose.npy", "").rsplit("_", 1)[0]
        raw = ensure_tensor_shape(np.load(p))
        raw = pad_or_clip(raw, max_len)
        feats = build_features_from_npy(raw, ablation="D")
        if feats.shape[-1] < target_c:
            pad_c = target_c - feats.shape[-1]
            feats = np.pad(feats, ((0, 0), (0, 0), (0, pad_c)), mode="constant")
        elif feats.shape[-1] > target_c:
            feats = feats[..., :target_c]

        sal = compute_saliency(model, feats)
        all_sal.append(sal)

        heatmap_path = os.path.join(out_dir, f"{pid}_saliency_heatmap.png")
        save_heatmap(sal, heatmap_path, f"Saliency Heatmap ({pid})")

        joint_imp = sal.mean(axis=0)
        bar_path = os.path.join(out_dir, f"{pid}_joint_importance.png")
        save_joint_bar(joint_imp, bar_path, f"Joint Importance ({pid})")

        records.append({"sample_id": pid})

    if all_sal:
        avg_sal = np.mean(np.stack(all_sal, axis=0), axis=0)
        save_heatmap(avg_sal, os.path.join(out_dir, "avg_saliency_heatmap.png"), "Average Saliency Heatmap")
        save_joint_bar(avg_sal.mean(axis=0), os.path.join(out_dir, "avg_joint_importance.png"), "Average Joint Importance")

    if records:
        pd.DataFrame(records).to_csv(os.path.join(out_dir, "saliency_samples.csv"), index=False)

    print(f"[INFO] Saliency outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
