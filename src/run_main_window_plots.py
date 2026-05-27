"""
Sliding-window and frame-step inference plots for the main TF model.
This does NOT modify existing code.
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


def detect_input_channels(weights_path: str, default_c: int = 9) -> int:
    if not os.path.exists(weights_path):
        return default_c
    try:
        with h5py.File(weights_path, "r") as f:
            for k in f.keys():
                group = f[k]
                if "dense" in group and "kernel:0" in group:
                    shape = group["kernel:0"].shape
                    return int(shape[0])
        return default_c
    except OSError:
        return default_c


def pad_or_clip(x: np.ndarray, max_len: int) -> np.ndarray:
    t = x.shape[0]
    if t > max_len:
        return x[:max_len]
    if t < max_len:
        pad = np.repeat(x[-1:], max_len - t, axis=0)
        return np.concatenate([x, pad], axis=0)
    return x


def build_model(weights_path: str, sample_shape: tuple) -> tf.keras.Model:
    model = build_pose_model(sample_shape)
    model(tf.zeros((1,) + sample_shape))
    model.load_weights(weights_path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Sliding-window plots for main TF model.")
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--model_weights", required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stride_frames", type=int, default=1)
    parser.add_argument("--suffix", default="_2_pose.npy")
    parser.add_argument("--out_dir", default="results/main_window_plots/latest")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    labels = load_labels(args.label_dir)
    npy_files = [p for p in glob.glob(os.path.join(args.processed_dir, "**", f"*{args.suffix}"), recursive=True)]
    npy_files = [p for p in npy_files if os.path.basename(p).replace(args.suffix, "") in labels]
    if not npy_files:
        raise SystemExit("No matching npy files with labels found.")

    window_len = int(args.seconds * args.fps)
    target_c = detect_input_channels(args.model_weights, default_c=9)

    first_raw = ensure_tensor_shape(np.load(npy_files[0]))
    first_raw = pad_or_clip(first_raw, window_len)
    first_feats = build_features_from_npy(first_raw, ablation="D")
    if first_feats.shape[-1] < target_c:
        pad_c = target_c - first_feats.shape[-1]
        first_feats = np.pad(first_feats, ((0, 0), (0, 0), (0, pad_c)), mode="constant")
    elif first_feats.shape[-1] > target_c:
        first_feats = first_feats[..., :target_c]

    sample_shape = (first_feats.shape[0], first_feats.shape[1], target_c)
    model = build_model(args.model_weights, sample_shape)

    records = []
    series_records = []
    for idx, p in enumerate(npy_files):
        if args.max_samples and idx >= args.max_samples:
            break
        pid = os.path.basename(p).replace("_pose.npy", "").rsplit("_", 1)[0]
        raw = ensure_tensor_shape(np.load(p))
        true = labels[pid]["gait_updrs"]

        if raw.shape[0] < window_len:
            raw = pad_or_clip(raw, window_len)
            windows = [raw]
        else:
            windows = []
            for s in range(0, raw.shape[0] - window_len + 1, args.stride_frames):
                windows.append(raw[s:s + window_len])

        preds = []
        for w_idx, w in enumerate(windows):
            feats = build_features_from_npy(w, ablation="D")
            if feats.shape[-1] < target_c:
                pad_c = target_c - feats.shape[-1]
                feats = np.pad(feats, ((0, 0), (0, 0), (0, pad_c)), mode="constant")
            elif feats.shape[-1] > target_c:
                feats = feats[..., :target_c]
            pred = model.predict(feats[None, ...], verbose=0).flatten()[0]
            preds.append(pred)
            records.append({"sample_id": pid, "window_idx": w_idx, "true": float(true), "pred": float(pred)})
            series_records.append({"sample_id": pid, "frame_idx": w_idx * args.stride_frames, "pred": float(pred)})

        # Per-sample trend plot
        if preds:
            plt.figure(figsize=(8, 3))
            x = [i * args.stride_frames for i in range(len(preds))]
            plt.plot(x, preds, marker="o", linewidth=1)
            plt.axhline(true, color="red", linestyle="--", label="True")
            plt.xlabel("Frame index (window start)")
            plt.ylabel("Predicted score")
            plt.title(f"Window predictions over time ({pid})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"{pid}_window_trend.png"), dpi=200)
            plt.close()

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.out_dir, "window_predictions.tsv"), sep="\t", index=False)

    if not df.empty:
        # Scatter with more points
        plt.figure(figsize=(6, 5))
        plt.scatter(df["true"], df["pred"], alpha=0.4)
        lims = [min(df["true"].min(), df["pred"].min()), max(df["true"].max(), df["pred"].max())]
        plt.plot(lims, lims, "r--")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title("Sliding-window scatter (more points)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "window_scatter.png"), dpi=200)
        plt.close()

    print(f"[INFO] Saved window plots to {args.out_dir}")


if __name__ == "__main__":
    main()
