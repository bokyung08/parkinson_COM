from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from gait17.training import regression_metrics


PROTOCOL_FILES = {
    "cnuh_to_carepd": ("CNUH", "CARE-PD", "results/cross_dataset_validation/cnuh_to_carepd/predictions.tsv"),
    "carepd_to_cnuh": ("CARE-PD", "CNUH", "results/cross_dataset_validation/carepd_to_cnuh/predictions.tsv"),
}


def class_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = df.copy()
    df["true_class"] = df["y_true"].round().clip(0, 3).astype(int)
    for cls, group in df.groupby("true_class", sort=True):
        metrics = regression_metrics(group["y_true"].to_numpy(np.float32), group["y_pred"].to_numpy(np.float32))
        rows.append({"true_class": int(cls), "n": int(len(group)), **metrics})
    return pd.DataFrame(rows)


def balanced_summary(per_class: pd.DataFrame) -> dict[str, float]:
    return {
        "balanced_mae": float(per_class["mae"].mean()),
        "balanced_rmse": float(per_class["rmse"].mean()),
        "balanced_medae": float(per_class["medae"].mean()),
        "n_classes": int(len(per_class)),
    }


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
    parser = argparse.ArgumentParser(description="Score-balanced analysis for cross-dataset transfer predictions.")
    parser.add_argument("--out_dir", default="results/score_balanced_transfer")
    parser.add_argument("--doc_path", default="docs/score_balanced_transfer_analysis.md")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_path = Path(args.doc_path)
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    per_class_frames = []
    for protocol, (train_set, test_set, pred_path) in PROTOCOL_FILES.items():
        path = Path(pred_path)
        if not path.exists():
            raise SystemExit(f"Missing prediction file: {path}")
        df = pd.read_csv(path, sep="\t")
        original = regression_metrics(df["y_true"].to_numpy(np.float32), df["y_pred"].to_numpy(np.float32))
        per_class = class_metrics(df)
        per_class.insert(0, "protocol", protocol)
        per_class.insert(1, "train_set", train_set)
        per_class.insert(2, "test_set", test_set)
        per_class_frames.append(per_class)
        balanced = balanced_summary(per_class)
        summary_rows.append(
            {
                "protocol": protocol,
                "train_set": train_set,
                "test_set": test_set,
                "n": int(len(df)),
                "original_mae": original["mae"],
                "original_rmse": original["rmse"],
                "original_medae": original["medae"],
                **balanced,
                "balanced_minus_original_mae": balanced["balanced_mae"] - original["mae"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    per_class_all = pd.concat(per_class_frames, ignore_index=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    per_class_all.to_csv(out_dir / "per_class_metrics.csv", index=False)

    lines = [
        "# Score-Balanced Cross-Dataset Transfer Analysis",
        "",
        "- Input: existing zero-shot cross-dataset prediction files",
        "- No additional training required.",
        "- Balanced metrics are the unweighted average of per-score metrics across available true classes.",
        "",
        "## Summary",
        "",
        *markdown_table(
            summary.rename(
                columns={
                    "protocol": "Protocol",
                    "train_set": "Train",
                    "test_set": "Test",
                    "n": "N",
                    "original_mae": "Original MAE",
                    "original_rmse": "Original RMSE",
                    "balanced_mae": "Balanced MAE",
                    "balanced_rmse": "Balanced RMSE",
                    "balanced_minus_original_mae": "Balanced - Original MAE",
                }
            )[
                [
                    "Protocol",
                    "Train",
                    "Test",
                    "N",
                    "Original MAE",
                    "Original RMSE",
                    "Balanced MAE",
                    "Balanced RMSE",
                    "Balanced - Original MAE",
                ]
            ],
            {"Original MAE", "Original RMSE", "Balanced MAE", "Balanced RMSE", "Balanced - Original MAE"},
        ),
        "",
        "## Per-Class Metrics",
        "",
        *markdown_table(
            per_class_all.rename(
                columns={
                    "protocol": "Protocol",
                    "train_set": "Train",
                    "test_set": "Test",
                    "true_class": "True score",
                    "n": "N",
                    "mae": "MAE",
                    "rmse": "RMSE",
                    "medae": "MedAE",
                }
            ),
            {"MAE", "RMSE", "MedAE"},
        ),
        "",
        "## Interpretation",
        "",
        "Use this analysis to separate class-imbalance effects from transfer failure. "
        "If balanced MAE is much larger than original MAE, the original transfer metric is being softened by the target score distribution.",
    ]
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote {doc_path} and {out_dir}")


if __name__ == "__main__":
    main()
