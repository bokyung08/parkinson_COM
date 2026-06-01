from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FINAL_SOURCES = [
    ("Classical ML", "Ridge", "groupkfold_h36m17_all", "ridge"),
    ("Classical ML", "SVR", "groupkfold_h36m17_all", "svr"),
    ("Classical ML", "Random Forest", "groupkfold_h36m17_all", "rf"),
    ("Classical ML", "Shallow MLP", "groupkfold_h36m17_all", "mlp_shallow"),
    ("Deep Learning", "Temporal CNN", "groupkfold_h36m17_all", "temporal_cnn"),
    ("SOTA", "ST-GCN", "groupkfold_h36m17_sota_cuda", "stgcn"),
    ("SOTA", "Lu official", "groupkfold_h36m17_ours_lu_official_cuda", "lu_ofddnet_official"),
    ("Proposed", "Ours V1", "groupkfold_h36m17_ours_lu_official_cuda", "ours"),
]

ABLATION_SOURCES = {
    "A": ("coordinates only", "groupkfold_h36m17_ours_ablation_A_cuda"),
    "B": ("coordinates + velocity", "groupkfold_h36m17_ours_ablation_B_cuda"),
    "C": ("coordinates + velocity + amplitude/variability", "groupkfold_h36m17_ours_ablation_C_cuda"),
    "D": ("full hybrid feature set", "groupkfold_h36m17_ours_lu_official_cuda"),
}

MODEL_COLORS = {
    "Ridge": "#8a8a8a",
    "SVR": "#6f6f6f",
    "Random Forest": "#555555",
    "Shallow MLP": "#3f3f3f",
    "Temporal CNN": "#4c78a8",
    "ST-GCN": "#f58518",
    "Lu official": "#e45756",
    "Ours V1": "#54a24b",
}


def load_predictions(results_root: Path) -> pd.DataFrame:
    frames = []
    for category, label, result_dir, model in FINAL_SOURCES:
        path = results_root / result_dir / "predictions.tsv"
        df = pd.read_csv(path, sep="\t")
        df = df[df["model"] == model].copy()
        df["display_category"] = category
        df["display_model"] = label
        df["true_class"] = df["y_true"].round().clip(0, 3).astype(int)
        df["pred_class"] = df["y_pred"].round().clip(0, 3).astype(int)
        df["error"] = df["y_pred"] - df["y_true"]
        df["abs_error"] = df["error"].abs()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_fold_metrics(results_root: Path) -> pd.DataFrame:
    frames = []
    for category, label, result_dir, model in FINAL_SOURCES:
        path = results_root / result_dir / "fold_metrics.csv"
        df = pd.read_csv(path)
        df = df[df["model"] == model].copy()
        df["display_category"] = category
        df["display_model"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_ablation_summary(results_root: Path) -> pd.DataFrame:
    path = results_root / "ours_abcd_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for ablation, (feature_set, result_dir) in ABLATION_SOURCES.items():
        summary = pd.read_csv(results_root / result_dir / "summary.csv")
        row = summary[summary["model"] == "ours"].iloc[0]
        rows.append(
            {
                "ablation": ablation,
                "feature_set": feature_set,
                "folds": int(row["n_folds"]),
                "n": int(row["n_predictions"]),
                "params": int(row["params"]),
                "inference_ms_per_sample": float(row["inference_ms_per_sample"]),
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "medae": float(row["medae"]),
            }
        )
    return pd.DataFrame(rows)


def load_ablation_folds(results_root: Path) -> pd.DataFrame:
    frames = []
    for ablation, (feature_set, result_dir) in ABLATION_SOURCES.items():
        df = pd.read_csv(results_root / result_dir / "fold_metrics.csv")
        df = df[df["model"] == "ours"].copy()
        df["ablation"] = ablation
        df["feature_set"] = feature_set
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def metric_summary(pred_df: pd.DataFrame, fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (category, model), group in pred_df.groupby(["display_category", "display_model"], sort=False):
        err = group["error"].to_numpy(np.float32)
        abs_err = np.abs(err)
        folds = fold_df[fold_df["display_model"] == model]
        rows.append(
            {
                "category": category,
                "model": model,
                "n": int(len(group)),
                "folds": int(group["fold"].nunique()),
                "params": int(folds["params"].iloc[0]) if len(folds) else 0,
                "inference_ms_per_sample": float(folds["inference_ms_per_sample"].mean()) if len(folds) else 0.0,
                "mae": float(abs_err.mean()),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "medae": float(np.median(abs_err)),
            }
        )
    return pd.DataFrame(rows)


def ensure_out(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
        }
    )


def save(fig, out_dir: Path, name: str, descriptions: list[tuple[str, str]], desc: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / name, bbox_inches="tight")
    plt.close(fig)
    descriptions.append((name, desc))


def add_bar_labels(ax, fmt: str = "{:.3f}", rotation: int = 0) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if np.isfinite(height):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height,
                fmt.format(height),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=rotation,
            )


def heatmap(ax, matrix: np.ndarray, title: str, xlabels, ylabels, cmap: str = "Blues", fmt: str = ".0f") -> None:
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(xlabels)), labels=xlabels)
    ax.set_yticks(range(len(ylabels)), labels=ylabels)
    max_val = np.nanmax(matrix) if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if max_val and val > max_val * 0.55 else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center", color=color, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_main_metric_bars(summary: pd.DataFrame, out_dir: Path, descriptions: list[tuple[str, str]]) -> None:
    metrics = ["mae", "rmse", "medae"]
    models = summary["model"].tolist()
    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * width, summary[metric], width, label=metric.upper())
    ax.set_xticks(x, models, rotation=35, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("Main comparison: MAE, RMSE, and MedAE")
    ax.legend(ncol=3)
    save(fig, out_dir, "01_main_metric_bars.png", descriptions, "Grouped MAE/RMSE/MedAE bars for all final comparison models.")


def plot_mae_ranking(summary: pd.DataFrame, out_dir: Path, descriptions: list[tuple[str, str]]) -> None:
    data = summary.sort_values("mae", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [MODEL_COLORS.get(m, "#777777") for m in data["model"]]
    ax.barh(data["model"], data["mae"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("MAE")
    ax.set_title("MAE ranking")
    for y, value in enumerate(data["mae"]):
        ax.text(value + 0.006, y, f"{value:.3f}", va="center", fontsize=9)
    save(fig, out_dir, "02_mae_ranking.png", descriptions, "Horizontal ranking of final models by MAE.")


def plot_scatter_efficiency(summary: pd.DataFrame, out_dir: Path, descriptions: list[tuple[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in summary.iterrows():
        ax.scatter(row["params"] + 1, row["mae"], s=90, color=MODEL_COLORS.get(row["model"], "#777777"))
        ax.text(row["params"] + 1, row["mae"] + 0.007, row["model"], fontsize=8, ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters + 1, log scale")
    ax.set_ylabel("MAE")
    ax.set_title("Parameter count vs MAE")
    save(fig, out_dir, "03_params_vs_mae.png", descriptions, "Scatter plot of parameter count against MAE.")

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in summary.iterrows():
        ax.scatter(row["inference_ms_per_sample"], row["mae"], s=90, color=MODEL_COLORS.get(row["model"], "#777777"))
        ax.text(row["inference_ms_per_sample"], row["mae"] + 0.007, row["model"], fontsize=8, ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("Inference time, ms/sample, log scale")
    ax.set_ylabel("MAE")
    ax.set_title("Inference time vs MAE")
    save(fig, out_dir, "04_inference_vs_mae.png", descriptions, "Scatter plot of inference time against MAE.")


def plot_fold_lines(fold_df: pd.DataFrame, out_dir: Path, descriptions: list[tuple[str, str]]) -> None:
    selected = fold_df[fold_df["display_model"].isin(["Ours V1", "Lu official", "ST-GCN", "Temporal CNN"])].copy()
    for metric, filename, title in [
        ("mae", "05_fold_mae_by_model.png", "Fold-level MAE"),
        ("rmse", "06_fold_rmse_by_model.png", "Fold-level RMSE"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for model, group in selected.groupby("display_model", sort=False):
            group = group.sort_values("fold")
            ax.plot(group["fold"], group[metric], marker="o", linewidth=2, label=model, color=MODEL_COLORS.get(model))
        ax.set_xticks(sorted(selected["fold"].unique()))
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.legend()
        save(fig, out_dir, filename, descriptions, f"{title} for Ours V1 and key baselines.")


def plot_ablation(ablation: pd.DataFrame, ablation_folds: pd.DataFrame, out_dir: Path, descriptions: list[tuple[str, str]]) -> None:
    ablation = ablation.sort_values("ablation")
    metrics = ["mae", "rmse", "medae"]
    x = np.arange(len(ablation))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * width, ablation[metric], width, label=metric.upper())
    ax.set_xticks(x, ablation["ablation"])
    ax.set_xlabel("Ablation")
    ax.set_ylabel("Error")
    ax.set_title("Ours V1 A/B/C/D ablation metrics")
    ax.legend(ncol=3)
    save(fig, out_dir, "07_ablation_metrics.png", descriptions, "Grouped MAE/RMSE/MedAE bars for Ours V1 ablations.")

    fig, ax = plt.subplots(figsize=(9, 5))
    for ab, group in ablation_folds.groupby("ablation"):
        group = group.sort_values("fold")
        ax.plot(group["fold"], group["mae"], marker="o", linewidth=2, label=ab)
    ax.set_xticks(sorted(ablation_folds["fold"].unique()))
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE")
    ax.set_title("Ours V1 fold-level MAE by ablation")
    ax.legend(title="Ablation")
    save(fig, out_dir, "08_ablation_fold_mae.png", descriptions, "Fold-level MAE trends for Ours V1 ablations A-D.")


def plot_per_class(per_class: pd.DataFrame, out_dir: Path, descriptions: list[tuple[str, str]]) -> None:
    ours = per_class[per_class["model"] == "Ours V1"].sort_values("true_class")
    x = np.arange(len(ours))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, ours["mae"], width, label="MAE")
    ax.bar(x + width / 2, ours["rmse"], width, label="RMSE")
    ax.set_xticks(x, ours["true_class"])
    ax.set_xlabel("True score")
    ax.set_ylabel("Error")
    ax.set_title("Ours V1 per-class error")
    ax.legend()
    save(fig, out_dir, "09_ours_per_class_error.png", descriptions, "Per-class MAE/RMSE for Ours V1.")

    pivot = per_class.pivot(index="model", columns="true_class", values="mae").reindex(
        ["Ours V1", "Lu official", "ST-GCN", "Temporal CNN", "SVR", "Random Forest", "Shallow MLP", "Ridge"]
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    heatmap(ax, pivot.to_numpy(), "Per-class MAE heatmap", [str(c) for c in pivot.columns], pivot.index.tolist(), cmap="YlOrRd", fmt=".3f")
    ax.set_xlabel("True score")
    ax.set_ylabel("Model")
    save(fig, out_dir, "10_per_class_mae_heatmap.png", descriptions, "Heatmap of MAE by model and true score.")


def plot_confusion(confusion: pd.DataFrame, norm_confusion: pd.DataFrame, out_dir: Path, descriptions: list[tuple[str, str]]) -> None:
    ours = confusion[confusion["model"] == "Ours V1"].sort_values("true_class")
    mat = ours[["pred_0", "pred_1", "pred_2", "pred_3"]].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    heatmap(ax, mat, "Ours V1 confusion matrix, counts", ["0", "1", "2", "3"], ["0", "1", "2", "3"], fmt=".0f")
    ax.set_xlabel("Rounded prediction")
    ax.set_ylabel("True score")
    save(fig, out_dir, "11_ours_confusion_counts.png", descriptions, "Ours V1 4x4 confusion matrix with count values.")

    ours_norm = norm_confusion[norm_confusion["model"] == "Ours V1"].sort_values("true_class")
    mat = ours_norm[["pred_0", "pred_1", "pred_2", "pred_3"]].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    heatmap(ax, mat, "Ours V1 confusion matrix, row-normalized", ["0", "1", "2", "3"], ["0", "1", "2", "3"], fmt=".3f")
    ax.set_xlabel("Rounded prediction")
    ax.set_ylabel("True score")
    save(fig, out_dir, "12_ours_confusion_normalized.png", descriptions, "Ours V1 row-normalized 4x4 confusion matrix.")


def plot_prediction_diagnostics(pred_df: pd.DataFrame, out_dir: Path, descriptions: list[tuple[str, str]]) -> None:
    ours = pred_df[pred_df["display_model"] == "Ours V1"].copy()
    rng = np.random.default_rng(42)
    jitter_x = rng.normal(0, 0.025, size=len(ours))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ours["y_true"] + jitter_x, ours["y_pred"], s=8, alpha=0.25, color=MODEL_COLORS["Ours V1"])
    ax.plot([0, 3], [0, 3], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(-0.15, 3.15)
    ax.set_ylim(-0.05, 3.05)
    ax.set_xlabel("True score")
    ax.set_ylabel("Predicted score")
    ax.set_title("Ours V1 true vs predicted score")
    save(fig, out_dir, "13_ours_true_vs_pred.png", descriptions, "Scatter plot of true versus predicted Ours V1 scores.")

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [ours[ours["true_class"] == cls]["error"].to_numpy() for cls in range(4)]
    ax.boxplot(data, tick_labels=["0", "1", "2", "3"], showfliers=False)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("True score")
    ax.set_ylabel("Prediction error")
    ax.set_title("Ours V1 residual distribution by true score")
    save(fig, out_dir, "14_ours_residual_by_true_score.png", descriptions, "Boxplot of Ours V1 residuals by true score.")

    models = ["Ours V1", "Lu official", "ST-GCN", "Temporal CNN", "SVR", "Random Forest", "Shallow MLP", "Ridge"]
    fig, ax = plt.subplots(figsize=(11, 5))
    data = [pred_df[pred_df["display_model"] == model]["abs_error"].to_numpy() for model in models]
    ax.boxplot(data, tick_labels=models, showfliers=False)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("Absolute error")
    ax.set_title("Absolute error distribution by model")
    save(fig, out_dir, "15_abs_error_distribution_by_model.png", descriptions, "Boxplot of absolute error distributions for final models.")

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [ours[ours["true_class"] == cls]["y_pred"].to_numpy() for cls in range(4)]
    ax.boxplot(data, tick_labels=["0", "1", "2", "3"], showfliers=False)
    ax.plot([1, 2, 3, 4], [0, 1, 2, 3], color="black", linestyle="--", linewidth=1, label="ideal")
    ax.set_xlabel("True score")
    ax.set_ylabel("Predicted score")
    ax.set_title("Ours V1 prediction distribution by true score")
    ax.legend()
    save(fig, out_dir, "17_prediction_distribution_by_score.png", descriptions, "Boxplot of Ours V1 predictions grouped by true score.")


def plot_dataset_and_calibration(pred_df: pd.DataFrame, out_dir: Path, descriptions: list[tuple[str, str]]) -> None:
    selected_models = ["Ours V1", "Lu official", "ST-GCN", "Temporal CNN"]
    rows = []
    for (model, dataset), group in pred_df[pred_df["display_model"].isin(selected_models)].groupby(["display_model", "dataset"]):
        err = group["error"].to_numpy(np.float32)
        rows.append({"model": model, "dataset": dataset, "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err**2)))})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(selected_models))
    width = 0.35
    for i, dataset in enumerate(sorted(data["dataset"].unique())):
        vals = [data[(data["model"] == model) & (data["dataset"] == dataset)]["mae"].iloc[0] for model in selected_models]
        ax.bar(x + (i - 0.5) * width, vals, width, label=dataset)
    ax.set_xticks(x, selected_models, rotation=25, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("Dataset-level MAE breakdown")
    ax.legend()
    save(fig, out_dir, "16_dataset_mae_breakdown.png", descriptions, "CAREPD/CNUH MAE breakdown for Ours V1 and key baselines.")

    fig, ax = plt.subplots(figsize=(8, 5))
    for model in selected_models:
        group = pred_df[pred_df["display_model"] == model]
        cal = group.groupby("true_class")["y_pred"].mean().reindex(range(4))
        ax.plot(cal.index, cal.values, marker="o", linewidth=2, label=model, color=MODEL_COLORS.get(model))
    ax.plot([0, 1, 2, 3], [0, 1, 2, 3], color="black", linestyle="--", linewidth=1, label="ideal")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("True score")
    ax.set_ylabel("Mean predicted score")
    ax.set_title("Mean prediction by true score")
    ax.legend()
    save(fig, out_dir, "18_calibration_curve_by_model.png", descriptions, "Calibration-style curve of mean prediction by true score.")

    counts = pred_df[pred_df["display_model"] == "Ours V1"].groupby("true_class").size().reindex(range(4), fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(i) for i in counts.index], counts.values, color="#72b7b2")
    ax.set_xlabel("True score")
    ax.set_ylabel("N")
    ax.set_title("Class distribution")
    add_bar_labels(ax, "{:.0f}")
    save(fig, out_dir, "19_class_distribution.png", descriptions, "Target class distribution in the combined evaluation set.")

    baseline_mae = {}
    ours_mae = pred_df[pred_df["display_model"] == "Ours V1"]["abs_error"].mean()
    for model in ["SVR", "Temporal CNN", "ST-GCN", "Lu official"]:
        baseline_mae[model] = pred_df[pred_df["display_model"] == model]["abs_error"].mean() - ours_mae
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(baseline_mae.keys())
    values = [baseline_mae[k] for k in labels]
    ax.bar(labels, values, color=["#777777", "#4c78a8", "#f58518", "#e45756"])
    ax.set_ylabel("MAE difference vs Ours V1")
    ax.set_title("MAE advantage of Ours V1")
    ax.axhline(0, color="black", linewidth=1)
    add_bar_labels(ax, "{:.3f}")
    save(fig, out_dir, "20_mae_advantage_vs_baselines.png", descriptions, "MAE difference between selected baselines and Ours V1.")


def write_index(out_dir: Path, descriptions: list[tuple[str, str]]) -> None:
    lines = [
        "# Final Integrated Result Figures",
        "",
        "Generated from completed result tables under `results/`.",
        "",
        "| File | Description |",
        "|---|---|",
    ]
    for filename, desc in descriptions:
        lines.append(f"| `{filename}` | {desc} |")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final integrated result figures.")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--docs_root", default="docs")
    parser.add_argument("--out_dir", default="docs/final_integrated_figures")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    docs_root = Path(args.docs_root)
    out_dir = Path(args.out_dir)
    ensure_out(out_dir)

    pred_df = load_predictions(results_root)
    fold_df = load_fold_metrics(results_root)
    summary = metric_summary(pred_df, fold_df)
    ablation = load_ablation_summary(results_root)
    ablation_folds = load_ablation_folds(results_root)
    per_class = pd.read_csv(docs_root / "per_class_metrics.csv")
    confusion = pd.read_csv(docs_root / "confusion_matrix_counts.csv")
    norm_confusion = pd.read_csv(docs_root / "confusion_matrix_normalized.csv")

    descriptions: list[tuple[str, str]] = []
    plot_main_metric_bars(summary, out_dir, descriptions)
    plot_mae_ranking(summary, out_dir, descriptions)
    plot_scatter_efficiency(summary, out_dir, descriptions)
    plot_fold_lines(fold_df, out_dir, descriptions)
    plot_ablation(ablation, ablation_folds, out_dir, descriptions)
    plot_per_class(per_class, out_dir, descriptions)
    plot_confusion(confusion, norm_confusion, out_dir, descriptions)
    plot_prediction_diagnostics(pred_df, out_dir, descriptions)
    plot_dataset_and_calibration(pred_df, out_dir, descriptions)
    write_index(out_dir, descriptions)
    print(f"[INFO] Wrote {len(descriptions)} figures to {out_dir}")


if __name__ == "__main__":
    main()
