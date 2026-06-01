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


SOURCES = [
    ("Classical ML", "Ridge", "groupkfold_h36m17_all", "ridge"),
    ("Classical ML", "SVR", "groupkfold_h36m17_all", "svr"),
    ("Classical ML", "Random Forest", "groupkfold_h36m17_all", "rf"),
    ("Classical ML", "Shallow MLP", "groupkfold_h36m17_all", "mlp_shallow"),
    ("Deep Learning", "Temporal CNN", "groupkfold_h36m17_all", "temporal_cnn"),
    ("SOTA", "ST-GCN", "groupkfold_h36m17_sota_cuda", "stgcn"),
    ("SOTA", "Lu official", "groupkfold_h36m17_ours_lu_official_cuda", "lu_ofddnet_official"),
    ("Proposed", "Ours V1", "groupkfold_h36m17_ours_lu_official_cuda", "ours"),
]


def load_predictions(results_root: Path) -> pd.DataFrame:
    frames = []
    for category, display_model, result_dir, model in SOURCES:
        path = results_root / result_dir / "predictions.tsv"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        df = df[df["model"].astype(str) == model].copy()
        if df.empty:
            continue
        df["category"] = category
        df["display_model"] = display_model
        frames.append(df)
    if not frames:
        raise SystemExit(f"No prediction files found under {results_root}")
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metrics = regression_metrics(group["y_true"].to_numpy(np.float32), group["y_pred"].to_numpy(np.float32))
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n": int(len(group)),
                "n_subjects": int(group["val_patient_id"].astype(str).nunique()) if "val_patient_id" in group else np.nan,
                **metrics,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


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
    parser = argparse.ArgumentParser(description="Dataset-wise breakdown for combined GroupKFold predictions.")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--out_dir", default="results/dataset_wise_breakdown")
    parser.add_argument("--doc_path", default="docs/dataset_wise_breakdown_analysis.md")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_path = Path(args.doc_path)
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    pred = load_predictions(results_root)
    overall = summarize(pred, ["category", "display_model"]).sort_values(["mae", "rmse"])
    by_dataset = summarize(pred, ["category", "display_model", "dataset"]).sort_values(["dataset", "mae"])
    pivot = by_dataset.pivot_table(index=["category", "display_model"], columns="dataset", values="mae").reset_index()
    if {"CAREPD", "CNUH"}.issubset(pivot.columns):
        pivot["CNUH_minus_CAREPD_MAE"] = pivot["CNUH"] - pivot["CAREPD"]
    pivot = pivot.sort_values("CAREPD" if "CAREPD" in pivot.columns else pivot.columns[-1])

    overall.to_csv(out_dir / "overall_metrics.csv", index=False)
    by_dataset.to_csv(out_dir / "dataset_metrics.csv", index=False)
    pivot.to_csv(out_dir / "dataset_mae_pivot.csv", index=False)

    ours = by_dataset[by_dataset["display_model"] == "Ours V1"].copy()
    lines = [
        "# Dataset-Wise Breakdown Under Combined GroupKFold",
        "",
        "- Input: existing subject-level GroupKFold predictions",
        "- Purpose: show whether the combined-domain model performs consistently on CARE-PD and CNUH test samples.",
        "- No additional training required.",
        "",
        "## Overall Model Ranking",
        "",
        *markdown_table(
            overall[["category", "display_model", "n", "mae", "rmse", "medae"]].rename(
                columns={"category": "Category", "display_model": "Model", "n": "N", "mae": "MAE", "rmse": "RMSE", "medae": "MedAE"}
            ),
            {"MAE", "RMSE", "MedAE"},
        ),
        "",
        "## Dataset-Wise Metrics",
        "",
        *markdown_table(
            by_dataset[["category", "display_model", "dataset", "n", "n_subjects", "mae", "rmse", "medae"]].rename(
                columns={
                    "category": "Category",
                    "display_model": "Model",
                    "dataset": "Dataset",
                    "n": "N",
                    "n_subjects": "Subjects",
                    "mae": "MAE",
                    "rmse": "RMSE",
                    "medae": "MedAE",
                }
            ),
            {"MAE", "RMSE", "MedAE"},
        ),
        "",
        "## Ours V1 Dataset Breakdown",
        "",
        *markdown_table(
            ours[["dataset", "n", "n_subjects", "mae", "rmse", "medae"]].rename(
                columns={"dataset": "Dataset", "n": "N", "n_subjects": "Subjects", "mae": "MAE", "rmse": "RMSE", "medae": "MedAE"}
            ),
            {"MAE", "RMSE", "MedAE"},
        ),
        "",
        "## Interpretation",
        "",
        "This analysis separates the main combined GroupKFold result by dataset. "
        "It should be used alongside zero-shot transfer results: if combined training performs well on both datasets while zero-shot transfer is poor, "
        "the evidence supports a domain-exposure interpretation rather than an architecture failure.",
        "",
        "Source outputs:",
        "",
        "```text",
        f"{out_dir / 'overall_metrics.csv'}",
        f"{out_dir / 'dataset_metrics.csv'}",
        f"{out_dir / 'dataset_mae_pivot.csv'}",
        "```",
    ]
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote {doc_path} and {out_dir}")


if __name__ == "__main__":
    main()
