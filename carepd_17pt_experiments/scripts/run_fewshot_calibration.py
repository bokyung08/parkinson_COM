from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from gait17.data import load_manifest
from gait17.training import regression_metrics


PROTOCOLS = {
    "cnuh_to_carepd": ("CNUH", "CARE-PD", "results/cross_dataset_validation/cnuh_to_carepd/predictions.tsv"),
    "carepd_to_cnuh": ("CARE-PD", "CNUH", "results/cross_dataset_validation/carepd_to_cnuh/predictions.tsv"),
}


def add_patient_ids(pred: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    mapping = manifest[["sample_id", "patient_id"]].copy()
    mapping["sample_id"] = mapping["sample_id"].astype(str)
    pred = pred.copy()
    pred["sample_id"] = pred["sample_id"].astype(str)
    out = pred.merge(mapping, on="sample_id", how="left")
    if out["patient_id"].isna().any():
        missing = int(out["patient_id"].isna().sum())
        raise ValueError(f"Missing patient_id for {missing} prediction rows.")
    return out


def fit_affine(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    if len(y_pred) < 2 or float(np.std(y_pred)) < 1e-8:
        return 0.0, float(np.mean(y_true))
    x = np.column_stack([y_pred, np.ones_like(y_pred)])
    coef, *_ = np.linalg.lstsq(x, y_true, rcond=None)
    return float(coef[0]), float(coef[1])


def apply_affine(y_pred: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.clip(a * y_pred + b, 0.0, 3.0).astype(np.float32)


def evaluate_protocol(
    protocol: str,
    train_set: str,
    test_set: str,
    pred_path: Path,
    manifest: pd.DataFrame,
    calibration_subjects: list[int],
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = add_patient_ids(pd.read_csv(pred_path, sep="\t"), manifest)
    subjects = np.array(sorted(pred["patient_id"].astype(str).unique()))
    rows = []
    pred_rows = []
    rng = np.random.default_rng(seed)
    for n_cal in calibration_subjects:
        if n_cal >= len(subjects):
            continue
        for repeat in range(1, repeats + 1):
            cal_subjects = rng.choice(subjects, size=n_cal, replace=False)
            is_cal = pred["patient_id"].astype(str).isin(set(cal_subjects))
            cal = pred[is_cal]
            test = pred[~is_cal]
            if cal.empty or test.empty:
                continue
            a, b = fit_affine(cal["y_pred"].to_numpy(), cal["y_true"].to_numpy())
            base_metrics = regression_metrics(test["y_true"].to_numpy(np.float32), test["y_pred"].to_numpy(np.float32))
            calibrated_pred = apply_affine(test["y_pred"].to_numpy(np.float32), a, b)
            calibrated_metrics = regression_metrics(test["y_true"].to_numpy(np.float32), calibrated_pred)
            rows.append(
                {
                    "protocol": protocol,
                    "train_set": train_set,
                    "test_set": test_set,
                    "repeat": repeat,
                    "n_calibration_subjects": int(n_cal),
                    "n_calibration_samples": int(len(cal)),
                    "n_test_subjects": int(test["patient_id"].astype(str).nunique()),
                    "n_test_samples": int(len(test)),
                    "affine_a": a,
                    "affine_b": b,
                    "base_mae": base_metrics["mae"],
                    "base_rmse": base_metrics["rmse"],
                    "base_medae": base_metrics["medae"],
                    "calibrated_mae": calibrated_metrics["mae"],
                    "calibrated_rmse": calibrated_metrics["rmse"],
                    "calibrated_medae": calibrated_metrics["medae"],
                    "delta_mae": calibrated_metrics["mae"] - base_metrics["mae"],
                    "delta_rmse": calibrated_metrics["rmse"] - base_metrics["rmse"],
                }
            )
            tmp = test[["sample_id", "patient_id", "y_true", "y_pred"]].copy()
            tmp["protocol"] = protocol
            tmp["repeat"] = repeat
            tmp["n_calibration_subjects"] = int(n_cal)
            tmp["y_pred_calibrated"] = calibrated_pred
            tmp["abs_error_base"] = np.abs(tmp["y_pred"].to_numpy(np.float32) - tmp["y_true"].to_numpy(np.float32))
            tmp["abs_error_calibrated"] = np.abs(calibrated_pred - tmp["y_true"].to_numpy(np.float32))
            pred_rows.append(tmp)
    return pd.DataFrame(rows), pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for keys, group in rows.groupby(["protocol", "train_set", "test_set", "n_calibration_subjects"], sort=True):
        protocol, train_set, test_set, n_cal = keys
        out.append(
            {
                "protocol": protocol,
                "train_set": train_set,
                "test_set": test_set,
                "n_calibration_subjects": int(n_cal),
                "repeats": int(len(group)),
                "mean_base_mae": float(group["base_mae"].mean()),
                "mean_calibrated_mae": float(group["calibrated_mae"].mean()),
                "mean_delta_mae": float(group["delta_mae"].mean()),
                "std_delta_mae": float(group["delta_mae"].std(ddof=0)),
                "mean_base_rmse": float(group["base_rmse"].mean()),
                "mean_calibrated_rmse": float(group["calibrated_rmse"].mean()),
                "mean_delta_rmse": float(group["delta_rmse"].mean()),
            }
        )
    return pd.DataFrame(out)


def markdown_table(df: pd.DataFrame, float_cols: set[str]) -> list[str]:
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for row in df.to_dict(orient="records"):
        values = []
        for col in df.columns:
            val = row[col]
            if col in float_cols and pd.notna(val):
                values.append(f"{float(val):.3f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot affine calibration for zero-shot transfer predictions.")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--out_dir", default="results/fewshot_calibration")
    parser.add_argument("--doc_path", default="docs/fewshot_calibration_analysis.md")
    parser.add_argument("--calibration_subjects", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_path = Path(args.doc_path)
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_preds = []
    for protocol, (train_set, test_set, pred_path) in PROTOCOLS.items():
        rows, preds = evaluate_protocol(
            protocol,
            train_set,
            test_set,
            Path(pred_path),
            manifest,
            args.calibration_subjects,
            args.repeats,
            args.random_state,
        )
        all_rows.append(rows)
        if not preds.empty:
            all_preds.append(preds)
    details = pd.concat(all_rows, ignore_index=True)
    summary = summarize(details)
    details.to_csv(out_dir / "calibration_trials.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    if all_preds:
        pd.concat(all_preds, ignore_index=True).to_csv(out_dir / "calibrated_predictions.tsv", sep="\t", index=False)

    lines = [
        "# Few-Shot Target-Site Calibration",
        "",
        "- Input: zero-shot transfer predictions",
        "- Calibration method: affine mapping `y_calibrated = a * y_pred + b`, clipped to `[0, 3]`",
        "- No model retraining or fine-tuning is performed.",
        f"- Repeats per setting: `{args.repeats}`",
        "",
        "## Summary",
        "",
        *markdown_table(
            summary.rename(
                columns={
                    "protocol": "Protocol",
                    "train_set": "Train",
                    "test_set": "Test",
                    "n_calibration_subjects": "Calibration subjects",
                    "repeats": "Repeats",
                    "mean_base_mae": "Base MAE",
                    "mean_calibrated_mae": "Calibrated MAE",
                    "mean_delta_mae": "Delta MAE",
                    "std_delta_mae": "Delta MAE SD",
                    "mean_base_rmse": "Base RMSE",
                    "mean_calibrated_rmse": "Calibrated RMSE",
                    "mean_delta_rmse": "Delta RMSE",
                }
            ),
            {"Base MAE", "Calibrated MAE", "Delta MAE", "Delta MAE SD", "Base RMSE", "Calibrated RMSE", "Delta RMSE"},
        ),
        "",
        "## Interpretation",
        "",
        "Negative Delta MAE means target-site calibration improved transfer performance. "
        "This experiment tests whether the zero-shot domain gap can be reduced with a small number of labeled target-site subjects without retraining the full model.",
    ]
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote {doc_path} and {out_dir}")


if __name__ == "__main__":
    main()
