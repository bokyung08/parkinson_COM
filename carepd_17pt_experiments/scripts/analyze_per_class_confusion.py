from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_SOURCES = [
    {
        "label": "Ours V1",
        "category": "Proposed",
        "result_dir": "groupkfold_h36m17_ours_lu_official_cuda",
        "model": "ours",
    },
    {
        "label": "Lu official",
        "category": "SOTA",
        "result_dir": "groupkfold_h36m17_ours_lu_official_cuda",
        "model": "lu_ofddnet_official",
    },
    {
        "label": "ST-GCN",
        "category": "SOTA",
        "result_dir": "groupkfold_h36m17_sota_cuda",
        "model": "stgcn",
    },
    {
        "label": "Temporal CNN",
        "category": "Deep Learning",
        "result_dir": "groupkfold_h36m17_all",
        "model": "temporal_cnn",
    },
    {
        "label": "SVR",
        "category": "Classical ML",
        "result_dir": "groupkfold_h36m17_all",
        "model": "svr",
    },
    {
        "label": "Random Forest",
        "category": "Classical ML",
        "result_dir": "groupkfold_h36m17_all",
        "model": "rf",
    },
    {
        "label": "Shallow MLP",
        "category": "Classical ML",
        "result_dir": "groupkfold_h36m17_all",
        "model": "mlp_shallow",
    },
    {
        "label": "Ridge",
        "category": "Classical ML",
        "result_dir": "groupkfold_h36m17_all",
        "model": "ridge",
    },
]


def rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else float("nan")


def load_predictions(results_root: Path, source: dict) -> pd.DataFrame:
    path = results_root / source["result_dir"] / "predictions.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t")
    df = df[df["model"] == source["model"]].copy()
    if df.empty:
        raise ValueError(f"No rows for model={source['model']} in {path}")
    df["display_model"] = source["label"]
    df["display_category"] = source["category"]
    df["true_class"] = df["y_true"].round().clip(0, 3).astype(int)
    df["pred_class"] = df["y_pred"].round().clip(0, 3).astype(int)
    df["error"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = df["error"].abs()
    return df


def per_class_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (category, model, cls), group in df.groupby(["display_category", "display_model", "true_class"], sort=True):
        errors = group["error"].to_numpy(np.float32)
        rows.append(
            {
                "category": category,
                "model": model,
                "true_class": int(cls),
                "n": int(len(group)),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": rmse(errors),
                "mean_pred": float(group["y_pred"].mean()),
                "std_pred": float(group["y_pred"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def confusion_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    normalized_rows = []
    for (category, model), group in df.groupby(["display_category", "display_model"], sort=True):
        mat = pd.crosstab(group["true_class"], group["pred_class"]).reindex(index=range(4), columns=range(4), fill_value=0)
        norm = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        for true_cls in range(4):
            row = {
                "category": category,
                "model": model,
                "true_class": true_cls,
                "pred_0": int(mat.loc[true_cls, 0]),
                "pred_1": int(mat.loc[true_cls, 1]),
                "pred_2": int(mat.loc[true_cls, 2]),
                "pred_3": int(mat.loc[true_cls, 3]),
            }
            rows.append(row)
            normalized_rows.append(
                {
                    "category": category,
                    "model": model,
                    "true_class": true_cls,
                    "pred_0": float(norm.loc[true_cls, 0]),
                    "pred_1": float(norm.loc[true_cls, 1]),
                    "pred_2": float(norm.loc[true_cls, 2]),
                    "pred_3": float(norm.loc[true_cls, 3]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(normalized_rows)


def markdown_table(df: pd.DataFrame, float_cols: set[str] | None = None) -> list[str]:
    float_cols = float_cols or set()
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for row in df.to_dict(orient="records"):
        values = []
        for col in df.columns:
            value = row[col]
            if col in float_cols:
                values.append(f"{float(value):.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(out_md: Path, per_class: pd.DataFrame, confusion: pd.DataFrame, norm_confusion: pd.DataFrame) -> None:
    ours_pc = per_class[per_class["model"] == "Ours V1"][
        ["true_class", "n", "mae", "rmse", "mean_pred", "std_pred"]
    ].rename(
        columns={
            "true_class": "Score",
            "n": "N",
            "mae": "MAE",
            "rmse": "RMSE",
            "mean_pred": "Mean prediction",
            "std_pred": "Prediction SD",
        }
    )
    ours_cm = confusion[confusion["model"] == "Ours V1"][["true_class", "pred_0", "pred_1", "pred_2", "pred_3"]].rename(
        columns={"true_class": "True score", "pred_0": "Pred 0", "pred_1": "Pred 1", "pred_2": "Pred 2", "pred_3": "Pred 3"}
    )
    ours_norm = norm_confusion[norm_confusion["model"] == "Ours V1"][
        ["true_class", "pred_0", "pred_1", "pred_2", "pred_3"]
    ].rename(
        columns={"true_class": "True score", "pred_0": "Pred 0", "pred_1": "Pred 1", "pred_2": "Pred 2", "pred_3": "Pred 3"}
    )

    lines = [
        "# Per-Class Error and Confusion Matrix",
        "",
        "- Source: completed prediction files under `results/`",
        "- Primary model: Ours V1 from `groupkfold_h36m17_ours_lu_official_cuda`",
        "- Prediction classes: regression output rounded to nearest integer and clipped to `[0, 3]`",
        "",
        "## Ours V1 Per-Class MAE/RMSE",
        "",
        *markdown_table(ours_pc, {"MAE", "RMSE", "Mean prediction", "Prediction SD"}),
        "",
        "## Ours V1 Confusion Matrix, Counts",
        "",
        *markdown_table(ours_cm),
        "",
        "Rows are true scores and columns are rounded predictions.",
        "",
        "## Ours V1 Confusion Matrix, Row-Normalized",
        "",
        *markdown_table(ours_norm, {"Pred 0", "Pred 1", "Pred 2", "Pred 3"}),
        "",
        "## Per-Class Metrics for All Final Models",
        "",
        *markdown_table(
            per_class[["category", "model", "true_class", "n", "mae", "rmse", "mean_pred", "std_pred"]],
            {"mae", "rmse", "mean_pred", "std_pred"},
        ),
        "",
        "## Manuscript Note",
        "",
        "The model is most accurate for the dominant score-1 class. Score-3 remains the hardest class because it has fewer samples and larger prediction dispersion. When reporting the confusion matrix, state that continuous regression outputs were rounded only for interpretability; MAE/RMSE remain the primary metrics.",
        "",
        "## COM Robustness Status",
        "",
        "The current result folders do not contain saved fold checkpoints (`.pt`, `.pth`, or `.ckpt`). Therefore COM robustness cannot be recomputed from the existing prediction tables alone, because scale/translation perturbations require rerunning the trained model on modified joint coordinates. Future runs should save the best fold checkpoint if COM robustness is required without retraining.",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-class metrics and confusion matrices from predictions.tsv files.")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--out_dir", default="docs")
    parser.add_argument("--mirror_results", default="results")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    pred_df = pd.concat([load_predictions(results_root, source) for source in FINAL_SOURCES], ignore_index=True)
    per_class = per_class_table(pred_df)
    confusion, norm_confusion = confusion_tables(pred_df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)
    confusion.to_csv(out_dir / "confusion_matrix_counts.csv", index=False)
    norm_confusion.to_csv(out_dir / "confusion_matrix_normalized.csv", index=False)
    write_report(out_dir / "per_class_confusion_analysis.md", per_class, confusion, norm_confusion)

    if args.mirror_results:
        mirror = Path(args.mirror_results)
        mirror.mkdir(parents=True, exist_ok=True)
        per_class.to_csv(mirror / "per_class_metrics.csv", index=False)
        confusion.to_csv(mirror / "confusion_matrix_counts.csv", index=False)
        norm_confusion.to_csv(mirror / "confusion_matrix_normalized.csv", index=False)
        write_report(mirror / "PER_CLASS_CONFUSION_ANALYSIS.md", per_class, confusion, norm_confusion)

    print(f"[INFO] Wrote per-class/confusion outputs to {out_dir}")


if __name__ == "__main__":
    main()
