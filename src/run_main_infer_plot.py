"""
Run inference with an existing main (TF) model and plot:
- Prediction vs Ground Truth scatter
- Bland-Altman plot

This does NOT modify any existing code; it is a standalone utility.

Example:
python -m src.run_main_infer_plot \
  --processed_dir HospitalData/processed_pose_data \
  --label_dir HospitalData/JSON \
  --model_weights results/models/20260105_202631/best_pose_model.weights.h5 \
  --seconds 10 --fps 30
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf

from src.model_builder import build_pose_model  # main TF model builder
from src.feature_engineering import build_features_from_npy
from src.train_model import load_labels  # label loader


def ensure_tensor_shape(raw: np.ndarray) -> np.ndarray:
    if raw.ndim == 3:
        return raw
    if raw.ndim == 2 and raw.shape[1] % 33 == 0:
        C = raw.shape[1] // 33
        return raw.reshape(raw.shape[0], 33, C)
    raise ValueError(f"Unexpected pose shape {raw.shape}")


def pad_or_clip(x: np.ndarray, max_len: int) -> np.ndarray:
    T = x.shape[0]
    if T > max_len:
        return x[:max_len]
    if T < max_len:
        pad = np.repeat(x[-1:], max_len - T, axis=0)
        return np.concatenate([x, pad], axis=0)
    return x


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


def plot_scatter(y_true, y_pred, out_path):
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot(lims, lims, "r--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Prediction vs Ground Truth (Main-D)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bland_altman(y_true, y_pred, out_path):
    diff = y_pred - y_true
    mean = (y_pred + y_true) / 2
    md = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_upper = md + 1.96 * sd
    loa_lower = md - 1.96 * sd

    plt.figure()
    plt.scatter(mean, diff, alpha=0.7)
    plt.axhline(md, color="red", linestyle="--", label=f"Mean diff={md:.3f}")
    plt.axhline(loa_upper, color="gray", linestyle="--", label=f"+1.96 SD={loa_upper:.3f}")
    plt.axhline(loa_lower, color="gray", linestyle="--", label=f"-1.96 SD={loa_lower:.3f}")
    plt.xlabel("Mean of True & Pred")
    plt.ylabel("Pred - True")
    plt.title("Bland-Altman (Main-D)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Infer with saved main model and plot scatter/Bland-Altman.")
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--model_weights", required=True, help="Path to main TF weights (.weights.h5 or .h5)")
    parser.add_argument("--seconds", type=float, default=10.0, help="Use only first N seconds")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for clip length")
    parser.add_argument("--suffix", default="_2_pose.npy", help="Which npy files to use (default: *_2_pose.npy)")
    parser.add_argument("--out_dir", default=None, help="Output directory for plots/TSV")
    args = parser.parse_args()

    max_len = int(args.seconds * args.fps)
    out_dir = args.out_dir or os.path.join("results", "main_infer_plots")
    os.makedirs(out_dir, exist_ok=True)

    labels = load_labels(args.label_dir)
    npy_files = [p for p in glob.glob(os.path.join(args.processed_dir, "**", f"*{args.suffix}"), recursive=True)]
    npy_files = [
        p for p in npy_files
        if os.path.basename(p).replace(args.suffix, "") in labels
    ]
    if not npy_files:
        raise SystemExit("No matching npy files with labels found.")

    # Build model from first sample
    first_raw = ensure_tensor_shape(np.load(npy_files[0]))
    first_raw = pad_or_clip(first_raw, max_len)
    first_feats = build_features_from_npy(first_raw, ablation="D")
    # 저장된 가중치가 21채널(좌표+속도+가속도 기반)로 학습된 케이스 대응: 21채널로 패딩
    target_c = detect_input_channels(args.model_weights, default_c=9)
    if first_feats.shape[-1] < target_c:
        pad_c = target_c - first_feats.shape[-1]
        first_feats = np.pad(first_feats, ((0, 0), (0, 0), (0, pad_c)), mode="constant")
    elif first_feats.shape[-1] > target_c:
        first_feats = first_feats[..., :target_c]
    input_shape = (first_feats.shape[0], first_feats.shape[1], target_c)
    model = build_pose_model(input_shape)
    model(tf.zeros((1,) + input_shape))
    model.load_weights(args.model_weights)

    records = []
    for p in npy_files:
        pid = os.path.basename(p).replace("_pose.npy", "").rsplit("_", 1)[0]
        raw = ensure_tensor_shape(np.load(p))
        raw = pad_or_clip(raw, max_len)
        feats = build_features_from_npy(raw, ablation="D")
        if feats.shape[-1] < target_c:
            pad_c = target_c - feats.shape[-1]
            feats = np.pad(feats, ((0, 0), (0, 0), (0, pad_c)), mode="constant")
        elif feats.shape[-1] > target_c:
            feats = feats[..., :target_c]
        pred = model.predict(feats[None, ...], verbose=0).flatten()[0]
        if pid in labels:
            true = labels[pid]["gait_updrs"]
            records.append({"sample_id": pid, "true": float(true), "pred": float(pred), "abs_err": abs(pred - true)})

    df = pd.DataFrame(records)
    tsv_path = os.path.join(out_dir, "main_D_predictions.tsv")
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"[INFO] Saved predictions to {tsv_path} (n={len(df)})")

    if len(df) > 0:
        y_true = df["true"].to_numpy()
        y_pred = df["pred"].to_numpy()
        scatter_png = os.path.join(out_dir, "scatter.png")
        ba_png = os.path.join(out_dir, "bland_altman.png")
        plot_scatter(y_true, y_pred, scatter_png)
        plot_bland_altman(y_true, y_pred, ba_png)
        print(f"[INFO] Saved plots to {scatter_png}, {ba_png}")


if __name__ == "__main__":
    main()
