from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RUNS = [
    ("Proposed", "Ours V1", "results/groupkfold_h36m17_ours_lu_official_cuda/predictions.tsv", "ours"),
    ("SOTA", "Lu official", "results/groupkfold_h36m17_ours_lu_official_cuda/predictions.tsv", "lu_ofddnet_official"),
    ("SOTA", "ST-GCN", "results/groupkfold_h36m17_sota_cuda/predictions.tsv", "stgcn"),
    ("SOTA", "MotionBERT-style", "results/groupkfold_h36m17_motion_encoders_cuda/predictions.tsv", "motionbert"),
    ("SOTA", "MotionAGFormer-style", "results/groupkfold_h36m17_motion_encoders_cuda/predictions.tsv", "motionagformer"),
    ("SOTA", "MotionBERT pretrained", "results/groupkfold_h36m17_motionbert_pretrained_cuda/predictions.tsv", "motionbert_pretrained"),
    ("SOTA", "MotionBERT-Lite (81-frame)", "results/groupkfold_h36m17_motionbert_lite81_cuda/predictions.tsv", "motionbert_lite_pretrained"),
    ("SOTA", "MotionAGFormer-XS", "results/groupkfold_h36m17_motionagformer_xs_pretrained_cuda/predictions.tsv", "motionagformer_xs_pretrained"),
]


def load_predictions(root: Path) -> pd.DataFrame:
    frames = []
    for category, label, rel_path, model_key in DEFAULT_RUNS:
        path = root / rel_path
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        df = df[df["model"].astype(str) == model_key].copy()
        if df.empty:
            continue
        df["category"] = category
        df["display_model"] = label
        frames.append(df)
    if not frames:
        raise SystemExit("No prediction files found for selective prediction analysis.")
    return pd.concat(frames, ignore_index=True)


def boundary_margin(y_pred: np.ndarray) -> np.ndarray:
    y = np.clip(y_pred.astype(float), 0.0, 3.0)
    boundaries = np.asarray([0.5, 1.5, 2.5], dtype=float)
    distance = np.min(np.abs(y[:, None] - boundaries[None, :]), axis=1)
    edge_distance = np.maximum.reduce([0.5 - y, y - 2.5, np.zeros_like(y)])
    return np.maximum(distance, edge_distance)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    rounded = np.rint(np.clip(y_pred, 0.0, 3.0))
    true_cls = np.rint(np.clip(y_true, 0.0, 3.0))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "medae": float(np.median(np.abs(err))),
        "rounded_accuracy": float(np.mean(rounded == true_cls)),
    }


def analyze(df: pd.DataFrame, coverages: list[float]) -> pd.DataFrame:
    rows = []
    for (category, display_model, model), group in df.groupby(["category", "display_model", "model"], sort=False):
        g = group.copy()
        g["selective_confidence"] = boundary_margin(g["y_pred"].to_numpy(float))
        g = g.sort_values("selective_confidence", ascending=False).reset_index(drop=True)
        base = metrics(g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float))
        for coverage in coverages:
            n_keep = max(1, int(round(len(g) * coverage)))
            kept = g.iloc[:n_keep]
            m = metrics(kept["y_true"].to_numpy(float), kept["y_pred"].to_numpy(float))
            rows.append(
                {
                    "category": category,
                    "model": model,
                    "display_model": display_model,
                    "coverage": float(n_keep / len(g)),
                    "n_kept": int(n_keep),
                    "n_total": int(len(g)),
                    "threshold_margin": float(kept["selective_confidence"].min()),
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "medae": m["medae"],
                    "rounded_accuracy": m["rounded_accuracy"],
                    "mae_delta_vs_full": float(m["mae"] - base["mae"]),
                    "mae_reduction_pct": float((base["mae"] - m["mae"]) / base["mae"] * 100.0) if base["mae"] > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, cols: list[str], float_cols: set[str]) -> list[str]:
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in df[cols].to_dict(orient="records"):
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{float(val):.3f}" if col in float_cols else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_doc(out_dir: Path, summary: pd.DataFrame, doc_path: Path) -> None:
    main = summary[summary["display_model"] == "Ours V1"].copy()
    model_80 = summary[np.isclose(summary["coverage"], 0.8, atol=1e-3)].copy()
    lines = [
        "# Selective Prediction Analysis",
        "",
        "- Input: existing GroupKFold prediction tables",
        "- No retraining is performed",
        "- Selection rule: keep predictions farthest from rounded-score decision boundaries `0.5, 1.5, 2.5`",
        "- Interpretation: lower retained-set MAE at lower coverage supports a clinician-review workflow where uncertain cases are flagged",
        "",
        "## Ours V1 Coverage Curve",
        "",
        *markdown_table(
            main,
            ["display_model", "coverage", "n_kept", "mae", "rmse", "medae", "rounded_accuracy", "mae_reduction_pct"],
            {"coverage", "mae", "rmse", "medae", "rounded_accuracy", "mae_reduction_pct"},
        ),
    ]
    if not model_80.empty:
        lines.extend(
            [
                "",
                "## Model Comparison at 80% Coverage",
                "",
                *markdown_table(
                    model_80.sort_values("mae"),
                    ["category", "display_model", "coverage", "n_kept", "mae", "rmse", "medae", "rounded_accuracy"],
                    {"coverage", "mae", "rmse", "medae", "rounded_accuracy"},
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Manuscript-Safe Wording",
            "",
            "> Selective prediction analysis showed that cases far from score-boundary thresholds had lower prediction error, suggesting that boundary-proximal cases can be flagged for clinician review rather than automatically accepted.",
        ]
    )
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc selective prediction analysis from saved predictions.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out_dir", default="results/selective_prediction")
    parser.add_argument("--doc_path", default="docs/selective_prediction_analysis.md")
    parser.add_argument("--coverages", nargs="+", type=float, default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_predictions(root)
    summary = analyze(df, args.coverages)
    summary.to_csv(out_dir / "summary.csv", index=False)
    write_doc(out_dir, summary, root / args.doc_path)
    print(f"[INFO] Wrote {out_dir} and {args.doc_path}")


if __name__ == "__main__":
    main()
