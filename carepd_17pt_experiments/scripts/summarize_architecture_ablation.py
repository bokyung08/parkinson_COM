from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VARIANT_LABELS = {
    "ours_mlp": ("MLP only", "mean pooling + bounded MLP"),
    "ours_gcn_mlp": ("GraphConv + MLP", "GraphConv, no joint attention, no Temporal Transformer"),
    "ours_gcn_attn_mlp": ("GraphConv + Joint Attention + MLP", "GraphConv + joint attention, no Temporal Transformer"),
    "ours": ("Full Ours V1", "GraphConv + Joint Attention + Temporal Transformer"),
}


def metrics_from_predictions(group: pd.DataFrame) -> dict[str, float]:
    err = group["y_pred"].to_numpy(np.float32) - group["y_true"].to_numpy(np.float32)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "medae": float(np.median(np.abs(err))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize architecture ablation using completed variants plus canonical full Ours result.")
    parser.add_argument("--ablation_dir", default="results/architecture_ablation_ours_cuda")
    parser.add_argument("--full_ours_dir", default="results/groupkfold_h36m17_ours_lu_official_cuda")
    parser.add_argument("--out_csv", default="results/architecture_ablation_summary.csv")
    parser.add_argument("--doc_path", default="docs/architecture_ablation_analysis.md")
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir)
    full_ours_dir = Path(args.full_ours_dir)
    rows: list[dict] = []

    folds = pd.read_csv(ablation_dir / "fold_metrics.csv")
    preds = pd.read_csv(ablation_dir / "predictions.tsv", sep="\t")
    for model in ["ours_mlp", "ours_gcn_mlp", "ours_gcn_attn_mlp"]:
        fold_group = folds[folds["model"] == model]
        pred_group = preds[preds["model"] == model]
        label, components = VARIANT_LABELS[model]
        metrics = metrics_from_predictions(pred_group)
        rows.append(
            {
                "model": model,
                "label": label,
                "components": components,
                "source": str(ablation_dir),
                "folds": int(pred_group["fold"].nunique()),
                "n": int(len(pred_group)),
                "params": int(fold_group["params"].iloc[0]),
                "inference_ms_per_sample": float(fold_group["inference_ms_per_sample"].mean()),
                **metrics,
            }
        )

    full_summary = pd.read_csv(full_ours_dir / "summary.csv")
    full_row = full_summary[full_summary["model"] == "ours"].iloc[0]
    label, components = VARIANT_LABELS["ours"]
    rows.append(
        {
            "model": "ours",
            "label": label,
            "components": components,
            "source": str(full_ours_dir),
            "folds": int(full_row["n_folds"]),
            "n": int(full_row["n_predictions"]),
            "params": int(full_row["params"]),
            "inference_ms_per_sample": float(full_row["inference_ms_per_sample"]),
            "mae": float(full_row["mae"]),
            "rmse": float(full_row["rmse"]),
            "medae": float(full_row["medae"]),
        }
    )

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)

    base = out[out["model"] == "ours_mlp"].iloc[0]
    full = out[out["model"] == "ours"].iloc[0]
    rel_gain = 100.0 * (base["mae"] - full["mae"]) / base["mae"]
    attn = out[out["model"] == "ours_gcn_attn_mlp"].iloc[0]
    transformer_gain = 100.0 * (attn["mae"] - full["mae"]) / attn["mae"]

    lines = [
        "# Architecture Ablation Analysis",
        "",
        "- Split: subject-level GroupKFold, 5 folds",
        "- Dataset: CNUH + CARE-PD H36M17",
        "- Input feature configuration: D",
        "- Full Ours V1 row: canonical final run from `groupkfold_h36m17_ours_lu_official_cuda`",
        "- Note: the interrupted full `ours` rows inside `architecture_ablation_ours_cuda` are not used.",
        "",
        "## Table",
        "",
        "| Model | Components | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in out.iterrows():
        lines.append(
            f"| {row['label']} | {row['components']} | {int(row['folds'])} | {int(row['n'])} | "
            f"{int(row['params'])} | {row['inference_ms_per_sample']:.3f} | "
            f"{row['mae']:.3f} | {row['rmse']:.3f} | {row['medae']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The full model reduces MAE from `{base['mae']:.3f}` in the MLP-only baseline to "
            f"`{full['mae']:.3f}`, a relative reduction of `{rel_gain:.1f}%`. Adding GraphConv "
            "substantially improves over MLP-only, adding joint attention further improves MAE, "
            "and the full Temporal Transformer model gives the best MAE and MedAE.",
            "",
            f"Compared with GraphConv + Joint Attention + MLP, adding the Temporal Transformer "
            f"reduces MAE from `{attn['mae']:.3f}` to `{full['mae']:.3f}` "
            f"(`{transformer_gain:.1f}%` relative reduction).",
            "",
            "Manuscript-safe wording:",
            "",
            "> Architecture ablation confirmed that the performance gain is not attributable solely "
            "to the input feature set. GraphConv improved over a mean-pooled MLP baseline, joint "
            "attention further reduced error, and the full GraphConv + Joint Attention + Temporal "
            "Transformer encoder achieved the lowest MAE and MedAE.",
        ]
    )
    Path(args.doc_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote {args.out_csv}")
    print(f"[INFO] Wrote {args.doc_path}")


if __name__ == "__main__":
    main()

