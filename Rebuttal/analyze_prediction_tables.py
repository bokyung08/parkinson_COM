"""Generate per-class, confusion-matrix, and statistical-validation tables.

This script consumes saved regression predictions and does not retrain models.
It is intended for Table 10, Table 11, and Table 12 in the AIM revision plan.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


TRUE_COL_CANDIDATES = ("y_true", "true", "true_gait_updrs")
PRED_COL_CANDIDATES = ("y_pred", "pred", "pred_gait_updrs")
ID_COL_CANDIDATES = ("sample_id", "id", "sample")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    raise ValueError(f"Missing required column. Tried: {', '.join(candidates)}")


def read_prediction_spec(spec: str) -> pd.DataFrame:
    if "=" in spec:
        model_name, path_text = spec.split("=", 1)
        model_name = model_name.strip()
    else:
        path_text = spec
        model_name = ""

    path = Path(path_text.strip())
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    true_col = find_column(df, TRUE_COL_CANDIDATES)
    pred_col = find_column(df, PRED_COL_CANDIDATES)
    id_col = find_column(df, ID_COL_CANDIDATES) if any(c.lower() in {x.lower() for x in df.columns} for c in ID_COL_CANDIDATES) else None

    if model_name:
        df["model"] = model_name
    elif "model" not in df.columns:
        df["model"] = path.parent.name if path.parent.name else path.stem

    out = pd.DataFrame(
        {
            "model": df["model"].astype(str),
            "sample_id": df[id_col].astype(str) if id_col else [f"{path.stem}_{i}" for i in range(len(df))],
            "y_true": pd.to_numeric(df[true_col], errors="raise"),
            "y_pred": pd.to_numeric(df[pred_col], errors="raise"),
        }
    )
    out["abs_error"] = (out["y_pred"] - out["y_true"]).abs()
    return out


def load_predictions(specs: list[str]) -> pd.DataFrame:
    frames = [read_prediction_spec(spec) for spec in specs]
    if not frames:
        raise ValueError("At least one --prediction input is required.")
    return pd.concat(frames, ignore_index=True)


def rounded_score(values: pd.Series, score_min: int, score_max: int) -> pd.Series:
    return values.round().clip(score_min, score_max).astype(int)


def per_class_table(df: pd.DataFrame, score_min: int, score_max: int) -> pd.DataFrame:
    rows = []
    working = df.copy()
    working["true_class"] = rounded_score(working["y_true"], score_min, score_max)
    for (model, cls), group in working.groupby(["model", "true_class"], sort=True):
        residual = group["y_pred"].to_numpy(dtype=float) - group["y_true"].to_numpy(dtype=float)
        abs_error = np.abs(residual)
        row = {
            "model": model,
            "score_class": int(cls),
            "n": int(len(group)),
            "mae": float(np.mean(abs_error)),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mean_true": float(group["y_true"].mean()),
            "mean_pred": float(group["y_pred"].mean()),
        }
        if len(group) == 1:
            row["single_true"] = float(group["y_true"].iloc[0])
            row["single_pred"] = float(group["y_pred"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def confusion_matrices(df: pd.DataFrame, score_min: int, score_max: int) -> dict[str, pd.DataFrame]:
    labels = list(range(score_min, score_max + 1))
    matrices: dict[str, pd.DataFrame] = {}
    working = df.copy()
    working["true_class"] = rounded_score(working["y_true"], score_min, score_max)
    working["pred_class"] = rounded_score(working["y_pred"], score_min, score_max)
    for model, group in working.groupby("model", sort=True):
        matrix = pd.crosstab(group["true_class"], group["pred_class"])
        matrix = matrix.reindex(index=labels, columns=labels, fill_value=0)
        matrix.index.name = "true"
        matrix.columns.name = "pred"
        matrices[model] = matrix
    return matrices


def bootstrap_ci(values: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.choice(values.size, size=values.size, replace=True)
        boot.append(float(np.mean(values[idx])))
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def aggregate_for_pairing(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "sample_id"], as_index=False)
        .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"), abs_error=("abs_error", "mean"))
    )


def statistical_table(
    df: pd.DataFrame,
    comparisons: list[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    paired = aggregate_for_pairing(df)
    for comp in comparisons:
        if ":" not in comp:
            raise ValueError(f"Comparison must use LEFT:RIGHT format: {comp}")
        left, right = [part.strip() for part in comp.split(":", 1)]
        left_df = paired[paired["model"] == left]
        right_df = paired[paired["model"] == right]
        merged = left_df.merge(
            right_df,
            on="sample_id",
            suffixes=("_left", "_right"),
            how="inner",
        )
        if merged.empty:
            rows.append(
                {
                    "comparison": comp,
                    "n_pairs": 0,
                    "wilcoxon_stat": float("nan"),
                    "p_value": float("nan"),
                    "note": "No paired sample_id overlap.",
                }
            )
            continue

        left_err = merged["abs_error_left"].to_numpy(dtype=float)
        right_err = merged["abs_error_right"].to_numpy(dtype=float)
        try:
            stat, p_value = wilcoxon(left_err, right_err, zero_method="wilcox", alternative="two-sided")
            stat = float(stat)
            p_value = float(p_value)
        except ValueError as exc:
            stat = float("nan")
            p_value = float("nan")
            note = str(exc)
        else:
            note = "p < 0.05 supports a stronger paired-error claim." if p_value < 0.05 else "Use hedged wording; paired difference is not significant at alpha=0.05."

        left_ci = bootstrap_ci(left_err, n_bootstrap, seed)
        right_ci = bootstrap_ci(right_err, n_bootstrap, seed + 1)
        delta = left_err - right_err
        delta_ci = bootstrap_ci(delta, n_bootstrap, seed + 2)
        rows.append(
            {
                "comparison": comp,
                "left_model": left,
                "right_model": right,
                "n_pairs": int(len(merged)),
                "left_mae": float(np.mean(left_err)),
                "left_mae_ci_low": left_ci[0],
                "left_mae_ci_high": left_ci[1],
                "right_mae": float(np.mean(right_err)),
                "right_mae_ci_low": right_ci[0],
                "right_mae_ci_high": right_ci[1],
                "delta_left_minus_right": float(np.mean(delta)),
                "delta_ci_low": delta_ci[0],
                "delta_ci_high": delta_ci[1],
                "wilcoxon_stat": stat,
                "p_value": p_value,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def format_float(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.3f}"


def write_markdown(
    out_dir: Path,
    per_class: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
    stats: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Per-Class and Statistical Validation Results",
        "",
        "## Protocol",
        "",
        f"- Score classes: `{args.score_min}` to `{args.score_max}` after rounding regression outputs.",
        f"- Bootstrap resamples: `{args.bootstrap}`.",
        f"- Random seed: `{args.seed}`.",
        "",
        "## Table 10. Per-Class Error",
        "",
        "| Model | Score | N | MAE | RMSE | Mean true | Mean pred | Note |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in per_class.iterrows():
        note = ""
        if int(row["n"]) == 1:
            note = f"single point: true={format_float(row.get('single_true'))}, pred={format_float(row.get('single_pred'))}"
        lines.append(
            "| {model} | {score} | {n} | {mae} | {rmse} | {mean_true} | {mean_pred} | {note} |".format(
                model=row["model"],
                score=int(row["score_class"]),
                n=int(row["n"]),
                mae=format_float(row["mae"]),
                rmse=format_float(row["rmse"]),
                mean_true=format_float(row["mean_true"]),
                mean_pred=format_float(row["mean_pred"]),
                note=note,
            )
        )

    lines.extend(["", "## Table 11. Rounded Confusion Matrices", ""])
    for model, matrix in matrices.items():
        lines.extend([f"### {model}", ""])
        header = "| True\\Pred | " + " | ".join(map(str, matrix.columns)) + " |"
        sep = "|---|" + "|".join(["---:"] * len(matrix.columns)) + "|"
        lines.extend([header, sep])
        for idx, row in matrix.iterrows():
            lines.append("| " + str(idx) + " | " + " | ".join(str(int(v)) for v in row.to_list()) + " |")
        lines.append("")

    lines.extend(
        [
            "## Table 12. Statistical Validation",
            "",
            "| Comparison | N | Left MAE [95% CI] | Right MAE [95% CI] | Delta | Wilcoxon p | Note |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in stats.iterrows():
        left = f"{format_float(row.get('left_mae'))} [{format_float(row.get('left_mae_ci_low'))}, {format_float(row.get('left_mae_ci_high'))}]"
        right = f"{format_float(row.get('right_mae'))} [{format_float(row.get('right_mae_ci_low'))}, {format_float(row.get('right_mae_ci_high'))}]"
        delta = f"{format_float(row.get('delta_left_minus_right'))} [{format_float(row.get('delta_ci_low'))}, {format_float(row.get('delta_ci_high'))}]"
        lines.append(
            f"| {row['comparison']} | {int(row['n_pairs'])} | {left} | {right} | {delta} | {format_float(row.get('p_value'))} | {row.get('note', '')} |"
        )

    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AIM Table 10/11/12 from saved predictions.")
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Prediction file, optionally named as MODEL=path/to/predictions.tsv.",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="Paired comparison in LEFT:RIGHT model-name format. Can be repeated.",
    )
    parser.add_argument("--out_dir", default="Rebuttal/results/statistical_validation")
    parser.add_argument("--score_min", type=int, default=0)
    parser.add_argument("--score_max", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_predictions(args.prediction)
    df["true_class"] = rounded_score(df["y_true"], args.score_min, args.score_max)
    df["pred_class"] = rounded_score(df["y_pred"], args.score_min, args.score_max)
    df.to_csv(out_dir / "normalized_predictions.tsv", sep="\t", index=False)

    per_class = per_class_table(df, args.score_min, args.score_max)
    per_class.to_csv(out_dir / "table10_per_class.csv", index=False)

    matrices = confusion_matrices(df, args.score_min, args.score_max)
    for model, matrix in matrices.items():
        matrix.to_csv(out_dir / f"table11_confusion_{safe_name(model)}.csv")

    comparisons = args.compare
    if not comparisons:
        models = sorted(df["model"].unique().tolist())
        comparisons = [f"{models[0]}:{models[1]}"] if len(models) >= 2 else []
    stats = statistical_table(df, comparisons, args.bootstrap, args.seed) if comparisons else pd.DataFrame()
    stats.to_csv(out_dir / "table12_statistical_validation.csv", index=False)

    metadata = {
        "predictions": args.prediction,
        "comparisons": comparisons,
        "score_min": args.score_min,
        "score_max": args.score_max,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
    }
    (out_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_markdown(out_dir, per_class, matrices, stats, args)
    print(f"[INFO] Saved Table 10/11/12 outputs to {out_dir}")


if __name__ == "__main__":
    main()
