"""Classical ML baselines for rebuttal experiments.

This script evaluates simple tabular regressors on Configuration D features.
It intentionally lives outside src/ so rebuttal experiments do not change the
main training code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


NUM_NODES = 33
DEFAULT_GAIT_KEYS = ("10", "11", "12", "13", "14")
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28


@dataclass(frozen=True)
class RunConfig:
    processed_dir: str
    label_dir: str
    out_dir: str
    target: str
    ablation: str
    max_len: int
    folds: int
    random_state: int
    holdout_test_size: float
    feature_summary: str


def scalar_score(value) -> float:
    if isinstance(value, list):
        return float(sum(value))
    if value is None:
        return 0.0
    return float(value)


def load_labels(label_dir: Path, target: str) -> dict[str, float]:
    labels: dict[str, float] = {}
    for path in sorted(label_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for patient in data.get("patient", []):
            pid = patient.get("id")
            items_list = (
                patient.get("mds_updrs_part3", {})
                .get("itmes", [])
            )
            if not pid or not items_list:
                continue
            items = items_list[0]
            if target == "gait":
                labels[pid] = sum(scalar_score(items.get(k)) for k in DEFAULT_GAIT_KEYS)
            elif target == "item10":
                labels[pid] = scalar_score(items.get("10"))
            else:
                raise ValueError(f"Unsupported target: {target}")
    return labels


def patient_id_from_pose_file(path: Path) -> str:
    stem = path.name.replace("_pose.npy", "")
    return stem.rsplit("_", 1)[0]


def ensure_pose_tensor(raw: np.ndarray) -> np.ndarray:
    if raw.ndim == 3:
        if raw.shape[1] != NUM_NODES or raw.shape[2] < 3:
            raise ValueError(f"Unexpected 3D pose shape: {raw.shape}")
        return raw[..., :3]
    if raw.ndim == 2 and raw.shape[1] % NUM_NODES == 0:
        channels = raw.shape[1] // NUM_NODES
        if channels < 3:
            raise ValueError(f"Expected at least 3 channels, got {channels}")
        return raw.reshape(raw.shape[0], NUM_NODES, channels)[..., :3]
    raise ValueError(f"Unsupported pose shape: {raw.shape}")


def compute_relative_velocity(joints: np.ndarray) -> np.ndarray:
    return np.diff(joints, axis=0, prepend=joints[:1])


def compute_amplitude(joints: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(joints, axis=2)
    return norms.max(axis=0) - norms.min(axis=0)


def compute_variability(joints: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(joints, axis=2)
    return norms.std(axis=0)


def compute_bone_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-6
    cos_angle = np.sum(ba * bc, axis=-1) / denom
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def build_hybrid_node_features(joints: np.ndarray) -> np.ndarray:
    """Build Configuration D node features: xyz, dxyz, amplitude, variability, angle."""
    time_steps, joints_count, _ = joints.shape
    velocity = compute_relative_velocity(joints)
    amplitude = compute_amplitude(joints)
    variability = compute_variability(joints)
    amp_broadcast = np.broadcast_to(amplitude[None, :, None], (time_steps, joints_count, 1))
    var_broadcast = np.broadcast_to(variability[None, :, None], (time_steps, joints_count, 1))

    angles = np.zeros((time_steps, joints_count), dtype=np.float32)
    if joints_count > MP_LEFT_ANKLE:
        angles[:, MP_LEFT_KNEE] = compute_bone_angle(
            joints[:, MP_LEFT_HIP], joints[:, MP_LEFT_KNEE], joints[:, MP_LEFT_ANKLE]
        )
    if joints_count > MP_RIGHT_ANKLE:
        angles[:, MP_RIGHT_KNEE] = compute_bone_angle(
            joints[:, MP_RIGHT_HIP], joints[:, MP_RIGHT_KNEE], joints[:, MP_RIGHT_ANKLE]
        )
    if joints_count > MP_LEFT_WRIST:
        angles[:, MP_LEFT_ELBOW] = compute_bone_angle(
            joints[:, MP_LEFT_SHOULDER], joints[:, MP_LEFT_ELBOW], joints[:, MP_LEFT_WRIST]
        )
    if joints_count > MP_RIGHT_WRIST:
        angles[:, MP_RIGHT_ELBOW] = compute_bone_angle(
            joints[:, MP_RIGHT_SHOULDER], joints[:, MP_RIGHT_ELBOW], joints[:, MP_RIGHT_WRIST]
        )

    return np.concatenate(
        [joints, velocity, amp_broadcast, var_broadcast, angles[..., None]],
        axis=-1,
    ).astype(np.float32)


def apply_ablation(node_features: np.ndarray, mode: str) -> np.ndarray:
    if mode == "A":
        return node_features[..., :3]
    if mode == "B":
        return node_features[..., :6]
    if mode == "C":
        return node_features[..., :8]
    return node_features


def keras_like_pad_sequences(x: np.ndarray, max_len: int) -> np.ndarray:
    """Match train_model.py defaults: maxlen=390, padding='post', truncating='pre'."""
    if x.shape[0] > max_len:
        return x[-max_len:]
    if x.shape[0] < max_len:
        pad_shape = (max_len - x.shape[0],) + x.shape[1:]
        return np.concatenate([x, np.zeros(pad_shape, dtype=x.dtype)], axis=0)
    return x


def load_configuration_d_dataset(
    processed_dir: Path,
    label_dir: Path,
    target: str,
    ablation: str,
    max_len: int,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    labels = load_labels(label_dir, target=target)
    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    ids: list[str] = []
    manifest_rows = []

    for pose_path in sorted(processed_dir.rglob("*_2_pose.npy")):
        pid = patient_id_from_pose_file(pose_path)
        if pid not in labels:
            continue

        raw = np.load(pose_path)
        coords = ensure_pose_tensor(raw)
        if coords.shape[0] > max_len:
            coords = coords[-max_len:]
        feats = build_hybrid_node_features(coords.astype(np.float32))
        feats = apply_ablation(feats, ablation)
        feats = keras_like_pad_sequences(feats.astype(np.float32), max_len)

        sample_id = pose_path.name.replace("_pose.npy", "")
        x_list.append(feats)
        y_list.append(labels[pid])
        ids.append(sample_id)
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "patient_id": pid,
                "target": labels[pid],
                "raw_frames": int(raw.shape[0]),
                "raw_shape": "x".join(map(str, raw.shape)),
                "used_shape": "x".join(map(str, feats.shape)),
            }
        )

    if not x_list:
        raise ValueError("No samples found. Check --processed_dir and --label_dir.")

    return (
        np.asarray(x_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
        ids,
        pd.DataFrame(manifest_rows),
    )


def summarize_features(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "flatten":
        return x.reshape(x.shape[0], -1)
    if mode != "stats":
        raise ValueError(f"Unsupported feature summary: {mode}")

    stats = [
        np.mean(x, axis=1),
        np.std(x, axis=1),
        np.min(x, axis=1),
        np.max(x, axis=1),
        np.percentile(x, 25, axis=1),
        np.percentile(x, 75, axis=1),
    ]
    return np.concatenate([s.reshape(x.shape[0], -1) for s in stats], axis=1)


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    try:
        if method == "pearson":
            return float(pearsonr(y_true, y_pred).statistic)
        if method == "spearman":
            return float(spearmanr(y_true, y_pred).statistic)
    except Exception:
        return float("nan")
    raise ValueError(method)


def concordance_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom == 0:
        return float("nan")
    return float((2 * cov) / denom)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": rmse(y_true_arr, y_pred_arr),
        "medae": float(median_absolute_error(y_true_arr, y_pred_arr)),
        "pearson": safe_corr(y_true_arr, y_pred_arr, "pearson"),
        "spearman": safe_corr(y_true_arr, y_pred_arr, "spearman"),
        "ccc": concordance_corrcoef(y_true_arr, y_pred_arr),
        "r2": float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 1 else float("nan"),
        "explained_variance": (
            float(explained_variance_score(y_true_arr, y_pred_arr))
            if len(y_true_arr) > 1
            else float("nan")
        ),
    }


def make_models(random_state: int) -> dict[str, object]:
    pca = PCA(n_components=0.95, svd_solver="full")
    return {
        "DummyMean": DummyRegressor(strategy="mean"),
        "Ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", clone(pca)),
                ("regressor", Ridge(alpha=1.0)),
            ]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            random_state=random_state,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
        ),
        "SVR_RBF": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", clone(pca)),
                ("regressor", SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")),
            ]
        ),
        "MLP": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", clone(pca)),
                (
                    "regressor",
                    MLPRegressor(
                        hidden_layer_sizes=(16,),
                        activation="relu",
                        alpha=0.05,
                        learning_rate_init=0.001,
                        max_iter=5000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def evaluate_holdout(
    models: dict[str, object],
    x_tab: np.ndarray,
    y: np.ndarray,
    ids: list[str],
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_idx, val_idx = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
    )
    summary_rows = []
    pred_rows = []

    for model_name, model in models.items():
        estimator = clone(model)
        start = time.perf_counter()
        estimator.fit(x_tab[train_idx], y[train_idx])
        train_seconds = time.perf_counter() - start
        pred = estimator.predict(x_tab[val_idx])
        metrics = regression_metrics(y[val_idx], pred)
        summary_rows.append(
            {
                "eval": "holdout",
                "model": model_name,
                "n_train": int(len(train_idx)),
                "n_test": int(len(val_idx)),
                "train_seconds": float(train_seconds),
                **metrics,
            }
        )
        for idx, pred_value in zip(val_idx, pred):
            pred_rows.append(
                {
                    "eval": "holdout",
                    "model": model_name,
                    "fold": "",
                    "sample_id": ids[idx],
                    "y_true": float(y[idx]),
                    "y_pred": float(pred_value),
                    "abs_error": float(abs(pred_value - y[idx])),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(pred_rows)


def evaluate_cv(
    models: dict[str, object],
    x_tab: np.ndarray,
    y: np.ndarray,
    ids: list[str],
    folds: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    k = min(folds, len(y))
    splitter = KFold(n_splits=k, shuffle=True, random_state=random_state)
    summary_rows = []
    fold_rows = []
    pred_rows = []

    for model_name, model in models.items():
        all_pred = np.full(shape=len(y), fill_value=np.nan, dtype=float)
        fold_metrics = []
        train_seconds_total = 0.0

        for fold, (train_idx, val_idx) in enumerate(splitter.split(x_tab), start=1):
            estimator = clone(model)
            start = time.perf_counter()
            estimator.fit(x_tab[train_idx], y[train_idx])
            train_seconds = time.perf_counter() - start
            train_seconds_total += train_seconds
            pred = estimator.predict(x_tab[val_idx])
            all_pred[val_idx] = pred

            metrics = regression_metrics(y[val_idx], pred)
            fold_metrics.append(metrics)
            fold_rows.append(
                {
                    "eval": "cv",
                    "model": model_name,
                    "fold": int(fold),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(val_idx)),
                    "train_seconds": float(train_seconds),
                    **metrics,
                }
            )
            for idx, pred_value in zip(val_idx, pred):
                pred_rows.append(
                    {
                        "eval": "cv",
                        "model": model_name,
                        "fold": int(fold),
                        "sample_id": ids[idx],
                        "y_true": float(y[idx]),
                        "y_pred": float(pred_value),
                        "abs_error": float(abs(pred_value - y[idx])),
                    }
                )

        overall = regression_metrics(y, all_pred)
        row = {
            "eval": "cv",
            "model": model_name,
            "n_train": "",
            "n_test": int(len(y)),
            "train_seconds": float(train_seconds_total),
            **overall,
        }
        for key in ("mae", "rmse", "medae"):
            values = [m[key] for m in fold_metrics]
            row[f"{key}_fold_mean"] = float(np.nanmean(values))
            row[f"{key}_fold_std"] = float(np.nanstd(values, ddof=0))
        summary_rows.append(row)

    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)


def write_plot(summary: pd.DataFrame, out_dir: Path) -> None:
    cv = summary[summary["eval"] == "cv"].sort_values("mae")
    if cv.empty:
        return
    plt.figure(figsize=(8, 4.5))
    plt.bar(cv["model"], cv["mae"], color="#4c78a8")
    plt.ylabel("MAE")
    plt.xlabel("Model")
    plt.title("Configuration D Baseline CV MAE")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "cv_mae_bar.png", dpi=200)
    plt.close()


def format_metric(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def write_markdown(
    out_dir: Path,
    config: RunConfig,
    summary: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    cv = summary[summary["eval"] == "cv"].copy().sort_values("mae")
    holdout = summary[summary["eval"] == "holdout"].copy().sort_values("mae")

    lines: list[str] = []
    lines.append("# Rebuttal Baseline Results")
    lines.append("")
    lines.append("## Experiment setup")
    lines.append("")
    lines.append(f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Dataset: `{config.processed_dir}`")
    lines.append(f"- Labels: `{config.label_dir}`")
    lines.append(f"- Target: `{config.target}`")
    lines.append(f"- Feature configuration: `{config.ablation}`")
    lines.append(f"- Sequence length: `{config.max_len}` frames")
    lines.append(f"- Tabular feature summary: `{config.feature_summary}`")
    lines.append(f"- Samples: {len(manifest)}")
    lines.append("")
    lines.append("Configuration D was rebuilt from COM-relative coordinates and converted to pooled tabular statistics for classical ML baselines. The default split mirrors the main model hold-out seed where possible, and the 5-fold CV result is the primary small-sample estimate.")
    lines.append("")

    lines.append("## 5-fold CV summary")
    lines.append("")
    lines.append("| Model | MAE | RMSE | Spearman | Pearson | R2 | MedAE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in cv.iterrows():
        lines.append(
            "| {model} | {mae} | {rmse} | {spearman} | {pearson} | {r2} | {medae} |".format(
                model=row["model"],
                mae=format_metric(row["mae"]),
                rmse=format_metric(row["rmse"]),
                spearman=format_metric(row["spearman"]),
                pearson=format_metric(row["pearson"]),
                r2=format_metric(row["r2"]),
                medae=format_metric(row["medae"]),
            )
        )
    lines.append("")

    lines.append("## Same-seed hold-out summary")
    lines.append("")
    lines.append("| Model | MAE | RMSE | Spearman | Pearson | R2 | MedAE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in holdout.iterrows():
        lines.append(
            "| {model} | {mae} | {rmse} | {spearman} | {pearson} | {r2} | {medae} |".format(
                model=row["model"],
                mae=format_metric(row["mae"]),
                rmse=format_metric(row["rmse"]),
                spearman=format_metric(row["spearman"]),
                pearson=format_metric(row["pearson"]),
                r2=format_metric(row["r2"]),
                medae=format_metric(row["medae"]),
            )
        )
    lines.append("")

    non_dummy_cv = cv[cv["model"] != "DummyMean"]
    best = non_dummy_cv.iloc[0] if not non_dummy_cv.empty else (cv.iloc[0] if not cv.empty else None)
    if best is not None:
        lines.append("## Rebuttal draft note")
        lines.append("")
        lines.append(
            "We additionally evaluated classical machine-learning baselines "
            "(Random Forest, SVR, MLP, and Ridge) using the same Configuration D "
            "features. In 5-fold cross-validation, the best non-dummy classical baseline was "
            f"{best['model']} with MAE={format_metric(best['mae'])} and "
            f"RMSE={format_metric(best['rmse'])}. These results provide the requested "
            "simple-model benchmark and should be reported alongside the proposed model "
            "using the same target definition and split protocol."
        )
        lines.append("")

    lines.append("## Generated files")
    lines.append("")
    for name in [
        "baseline_summary.csv",
        "baseline_summary.json",
        "cv_fold_metrics.csv",
        "cv_predictions.tsv",
        "holdout_predictions.tsv",
        "dataset_manifest.tsv",
        "run_config.json",
        "cv_mae_bar.png",
    ]:
        lines.append(f"- `{name}`")
    lines.append("")

    (out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rebuttal classical ML baselines.")
    parser.add_argument("--processed_dir", default="HospitalData/processed_pose_data")
    parser.add_argument("--label_dir", default="HospitalData/JSON")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--target", choices=["item10", "gait"], default="item10")
    parser.add_argument("--ablation", choices=["A", "B", "C", "D"], default="D")
    parser.add_argument("--max_len", type=int, default=390)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--holdout_test_size", type=float, default=0.1)
    parser.add_argument("--feature_summary", choices=["stats", "flatten"], default="stats")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("Rebuttal") / "results" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        processed_dir=args.processed_dir,
        label_dir=args.label_dir,
        out_dir=str(out_dir),
        target=args.target,
        ablation=args.ablation,
        max_len=args.max_len,
        folds=args.folds,
        random_state=args.random_state,
        holdout_test_size=args.holdout_test_size,
        feature_summary=args.feature_summary,
    )

    x_seq, y, ids, manifest = load_configuration_d_dataset(
        processed_dir=Path(args.processed_dir),
        label_dir=Path(args.label_dir),
        target=args.target,
        ablation=args.ablation,
        max_len=args.max_len,
    )
    x_tab = summarize_features(x_seq, args.feature_summary)
    models = make_models(args.random_state)

    holdout_summary, holdout_predictions = evaluate_holdout(
        models,
        x_tab,
        y,
        ids,
        test_size=args.holdout_test_size,
        random_state=args.random_state,
    )
    cv_summary, cv_fold_metrics, cv_predictions = evaluate_cv(
        models,
        x_tab,
        y,
        ids,
        folds=args.folds,
        random_state=args.random_state,
    )

    summary = pd.concat([holdout_summary, cv_summary], ignore_index=True)
    summary.to_csv(out_dir / "baseline_summary.csv", index=False)
    summary.to_json(out_dir / "baseline_summary.json", orient="records", indent=2)
    cv_fold_metrics.to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    holdout_predictions.to_csv(out_dir / "holdout_predictions.tsv", sep="\t", index=False)
    cv_predictions.to_csv(out_dir / "cv_predictions.tsv", sep="\t", index=False)
    manifest.to_csv(out_dir / "dataset_manifest.tsv", sep="\t", index=False)
    (out_dir / "run_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_plot(summary, out_dir)
    write_markdown(out_dir, config, summary, manifest)

    print(f"[INFO] Loaded sequence data: X={x_seq.shape}, y={y.shape}")
    print(f"[INFO] Tabular features: X={x_tab.shape}")
    print(f"[INFO] Results saved to: {out_dir}")
    print(summary[["eval", "model", "mae", "rmse", "spearman", "pearson", "r2", "medae"]])


if __name__ == "__main__":
    main()
