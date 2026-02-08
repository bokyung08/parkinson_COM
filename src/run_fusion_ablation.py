import argparse
import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, kendalltau

from .train_model import load_labels
from .train_hybrid_fusion import build_hybrid_features, ensure_tensor_shape, pad_or_clip, build_fusion_v2_model
import matplotlib.pyplot as plt

# Ablation 모드 정의
# A: 좌표 3ch, B: 좌표+속도 6ch, C: 좌표+속도+amp/var 8ch, D: 풀 9ch
ABLA_MODES = ["A", "B", "C", "D"]


def load_dataset(processed_dir, label_dir, max_seconds=13.0, fps=30.0, ablation="D"):
    labels = load_labels(label_dir)
    max_len = int(max_seconds * fps)
    X_list, y_list, ids = [], [], []
    for root, _, files in os.walk(processed_dir):
        for f in files:
            if not f.endswith("_pose.npy"):
                continue
            pid = f.replace("_pose.npy", "").rsplit("_", 1)[0]
            if pid not in labels:
                continue
            raw = np.load(os.path.join(root, f))
            raw = ensure_tensor_shape(raw)
            raw = pad_or_clip(raw, max_len)
            feats = build_hybrid_features(raw)  # (T, J, 9)
            if ablation == "A":
                feats = feats[..., :3]
            elif ablation == "B":
                feats = feats[..., :6]
            elif ablation == "C":
                feats = feats[..., :8]
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
    pearson = float(pearsonr(y_true, y_pred).statistic) if len(y_true) > 1 else float("nan")
    kendall = float(kendalltau(y_true, y_pred).statistic) if len(y_true) > 1 else float("nan")
    ccc = 0.0
    if len(y_true) > 1:
        mx, my = y_true.mean(), y_pred.mean()
        vx, vy = y_true.var(), y_pred.var()
        cov = ((y_true - mx) * (y_pred - my)).mean()
        ccc = float((2 * cov) / (vx + vy + (mx - my) ** 2 + 1e-8))
    r2 = float(r2_score(y_true, y_pred))
    evs = float(explained_variance_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    medae = float(median_absolute_error(y_true, y_pred))

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mae": float(mae),
                "rmse": float(rmse),
                "pearson": pearson,
                "kendall": kendall,
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
    with open(os.path.join(out_dir, "regression_errors.txt"), "w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    f"MAE: {mae:.4f}",
                    f"RMSE: {rmse:.4f}",
                    f"Pearson: {pearson:.4f}",
                    f"Kendall: {kendall:.4f}",
                    f"CCC: {ccc:.4f}",
                    f"R2: {r2:.4f}",
                    f"EVS: {evs:.4f}",
                    f"MAPE: {mape:.4f}",
                    f"MedAE: {medae:.4f}",
                ]
            )
            + "\n"
        )
    with open(os.path.join(out_dir, "predictions.tsv"), "w", encoding="utf-8") as f:
        f.write("true\tpred\tabs_err\n")
        for t, p in zip(y_true, y_pred):
            f.write(f"{t:.6f}\t{p:.6f}\t{abs(p-t):.6f}\n")

    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(f"Scatter (MAE={mae:.2f}, RMSE={rmse:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter.png"))
    plt.close()


def run_ablation(processed_dir, label_dir, epochs, batch_size, folds=None, max_seconds=13.0, fps=30.0):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("results", "fusion_tf_ablation", timestamp)
    os.makedirs(base_dir, exist_ok=True)
    summary = []

    for mode in ABLA_MODES:
        print(f"[INFO] Fusion TF Ablation {mode} 시작")
        X, y, ids, max_len = load_dataset(processed_dir, label_dir, max_seconds=max_seconds, fps=fps, ablation=mode)
        if len(X) == 0:
            print("[WARN] 데이터 없음")
            continue
        print(f"[INFO] Samples for {mode}: {len(X)} (hold-out 80/20), max_len={max_len} frames")
        # 단일 hold-out 80/20 split
        tr_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.2, shuffle=True, random_state=42)
        run_dir = os.path.join(base_dir, f"abl_{mode}")
        os.makedirs(run_dir, exist_ok=True)

        model = build_fusion_v2_model(
            input_shape=X.shape[1:],
            num_joints=X.shape[2],
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        )
        history = model.fit(
            X[tr_idx], y[tr_idx],
            validation_data=(X[val_idx], y[val_idx]),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)],
        )
        preds = model.predict(X[val_idx]).flatten()
        y_true = y[val_idx]
        mae = mean_absolute_error(y_true, preds)
        rmse = mean_squared_error(y_true, preds, squared=False)
        pearson = float(pearsonr(y_true, preds).statistic) if len(y_true) > 1 else float("nan")
        kendall = float(kendalltau(y_true, preds).statistic) if len(y_true) > 1 else float("nan")
        mx, my = y_true.mean(), preds.mean()
        vx, vy = y_true.var(), preds.var()
        cov = ((y_true - mx) * (preds - my)).mean()
        ccc = float((2 * cov) / (vx + vy + (mx - my) ** 2 + 1e-8)) if len(y_true) > 1 else float("nan")
        r2 = float(r2_score(y_true, preds))
        evs = float(explained_variance_score(y_true, preds))
        mape = float(np.mean(np.abs((y_true - preds) / (y_true + 1e-8))) * 100)
        medae = float(median_absolute_error(y_true, preds))

        plot_and_save(y_true, preds, run_dir)
        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history.history, f, indent=2, ensure_ascii=False)
        summary.append({
            "ablation": mode,
            "mae": float(mae),
            "rmse": float(rmse),
            "pearson": float(pearson),
            "kendall": float(kendall),
            "ccc": float(ccc),
            "r2": float(r2),
            "explained_variance": float(evs),
            "mape": float(mape),
            "medae": float(medae),
        })

    if summary:
        import pandas as pd
        df = pd.DataFrame(summary)
        df.to_csv(os.path.join(base_dir, "ablation_summary.csv"), index=False)
        with open(os.path.join(base_dir, "ablation_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Fusion TF ablation summary saved to {base_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fusion TF ablation study (single hold-out split)")
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=13.0)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    run_ablation(
        args.processed_dir,
        args.label_dir,
        args.epochs,
        args.batch_size,
        args.folds,
        max_seconds=args.seconds,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
