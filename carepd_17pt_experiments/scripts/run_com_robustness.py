from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from gait17.data import build_model_input, group_kfold_splits, load_manifest, load_npz_sequence
from gait17.models import make_model
from gait17.training import regression_metrics


DEFAULT_SCALE_VALUES = [0.70, 0.85, 1.00, 1.15, 1.30]
DEFAULT_TRANSLATION_VALUES = [-0.20, -0.10, 0.00, 0.10, 0.20]
DEFAULT_COMBINED_VALUES = [(0.70, -0.20), (1.30, 0.20), (0.85, 0.10), (1.15, -0.10)]


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def perturb_joints(joints: np.ndarray, scale: float = 1.0, dx: float = 0.0) -> np.ndarray:
    out = (joints * np.float32(scale)).astype(np.float32)
    out[..., 0] += np.float32(dx)
    return out


def materialize_perturbed(
    manifest_path: Path,
    df: pd.DataFrame,
    val_idx: np.ndarray,
    max_len: int,
    ablation: str,
    input_kind: str,
    normalize_com: bool,
    scale_normalization: str,
    scale: float,
    dx: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs, ys, ids = [], [], []
    val_df = df.iloc[val_idx].reset_index(drop=True)
    for _, row in val_df.iterrows():
        joints = load_npz_sequence(manifest_path, row)
        joints = perturb_joints(joints, scale=scale, dx=dx)
        xs.append(
            build_model_input(
                joints,
                max_len,
                ablation,
                input_kind,
                normalize_com=normalize_com,
                scale_normalization=scale_normalization,
            )
        )
        ys.append(float(row["target"]))
        ids.append(str(row["sample_id"]))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), ids


def predict(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(x).float()), batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            out = model(xb.to(device))
            preds.append(out.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0)


def rank_biserial_effect(a: np.ndarray, b: np.ndarray) -> float:
    diff = b - a
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return 0.0
    ranks = pd.Series(np.abs(diff)).rank(method="average").to_numpy()
    w_pos = float(ranks[diff > 0].sum())
    w_neg = float(ranks[diff < 0].sum())
    return (w_pos - w_neg) / (n * (n + 1) / 2.0)


def condition_list(scale_values: list[float], translation_values: list[float], combined_values: list[tuple[float, float]]) -> list[dict]:
    rows = [{"condition": "Original", "type": "original", "scale": 1.0, "dx": 0.0}]
    rows.extend({"condition": f"Scale {s:.2f}", "type": "scale", "scale": s, "dx": 0.0} for s in scale_values if s != 1.0)
    rows.extend({"condition": f"Shift {dx:+.2f}", "type": "translation", "scale": 1.0, "dx": dx} for dx in translation_values if dx != 0.0)
    rows.extend(
        {
            "condition": f"Scale {s:.2f} + Shift {dx:+.2f}",
            "type": "combined",
            "scale": s,
            "dx": dx,
        }
        for s, dx in combined_values
    )
    return rows


def evaluate_mode(
    manifest_path: Path,
    df: pd.DataFrame,
    splits: list[tuple[str, np.ndarray, np.ndarray]],
    checkpoint_dir: Path,
    normalize_com: bool,
    scale_normalization_arg: str,
    conditions: list[dict],
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict], list[dict]]:
    metric_rows = []
    pred_rows = []
    baseline_abs_by_fold: dict[int, np.ndarray] = {}
    for fold, (split_id, _, val_idx) in enumerate(splits, start=1):
        checkpoint_path = checkpoint_dir / f"ours_fold_{fold:02d}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, device)
        scale_normalization = (
            checkpoint.get("scale_normalization", "none")
            if scale_normalization_arg == "checkpoint"
            else scale_normalization_arg
        )
        model = make_model(checkpoint["model_name"], int(checkpoint["in_channels"])).to(device)
        model.load_state_dict(checkpoint["state_dict"])

        for condition in conditions:
            x_val, y_val, ids = materialize_perturbed(
                manifest_path,
                df,
                val_idx,
                int(checkpoint["max_len"]),
                str(checkpoint["ablation"]),
                str(checkpoint["input_kind"]),
                normalize_com=normalize_com,
                scale_normalization=str(scale_normalization),
                scale=float(condition["scale"]),
                dx=float(condition["dx"]),
            )
            y_pred = predict(model, x_val, device, batch_size)
            metrics = regression_metrics(y_val, y_pred)
            abs_err = np.abs(y_pred - y_val)
            if condition["type"] == "original":
                baseline_abs_by_fold[fold] = abs_err
            for sample_id, y_true, pred, err in zip(ids, y_val, y_pred, abs_err):
                pred_rows.append(
                    {
                        "normalization": "COM" if normalize_com else "Raw",
                        "scale_normalization": str(scale_normalization),
                        "fold": fold,
                        "split_id": split_id,
                        "condition": condition["condition"],
                        "type": condition["type"],
                        "scale": float(condition["scale"]),
                        "dx": float(condition["dx"]),
                        "sample_id": sample_id,
                        "y_true": float(y_true),
                        "y_pred": float(pred),
                        "abs_error": float(err),
                    }
                )
            metric_rows.append(
                {
                    "normalization": "COM" if normalize_com else "Raw",
                    "scale_normalization": str(scale_normalization),
                    "fold": fold,
                    "split_id": split_id,
                    "condition": condition["condition"],
                    "type": condition["type"],
                    "scale": float(condition["scale"]),
                    "dx": float(condition["dx"]),
                    "n": int(len(y_val)),
                    **metrics,
                }
            )
    return metric_rows, pred_rows


def aggregate_metrics(fold_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (norm, scale_norm, condition, ptype, scale, dx), group in pred_df.groupby(
        ["normalization", "scale_normalization", "condition", "type", "scale", "dx"],
        sort=False,
    ):
        metrics = regression_metrics(group["y_true"].to_numpy(np.float32), group["y_pred"].to_numpy(np.float32))
        rows.append(
            {
                "normalization": norm,
                "scale_normalization": scale_norm,
                "condition": condition,
                "type": ptype,
                "scale": float(scale),
                "dx": float(dx),
                "n": int(len(group)),
                **metrics,
            }
        )
    out = pd.DataFrame(rows)
    baselines = out[out["type"] == "original"][["normalization", "scale_normalization", "mae", "rmse"]].rename(
        columns={"mae": "baseline_mae", "rmse": "baseline_rmse"}
    )
    out = out.merge(baselines, on=["normalization", "scale_normalization"], how="left")
    out["delta_mae_pct"] = (out["mae"] - out["baseline_mae"]) / out["baseline_mae"] * 100.0
    out["delta_rmse_pct"] = (out["rmse"] - out["baseline_rmse"]) / out["baseline_rmse"] * 100.0
    return out.drop(columns=["baseline_mae", "baseline_rmse"])


def statistical_tests(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (norm, scale_norm), norm_df in pred_df.groupby(["normalization", "scale_normalization"], sort=False):
        original = norm_df[norm_df["type"] == "original"][["fold", "sample_id", "abs_error"]].rename(columns={"abs_error": "abs_original"})
        for (condition, ptype, scale, dx), group in norm_df[norm_df["type"] != "original"].groupby(
            ["condition", "type", "scale", "dx"],
            sort=False,
        ):
            merged = original.merge(group[["fold", "sample_id", "abs_error"]], on=["fold", "sample_id"], how="inner")
            a = merged["abs_original"].to_numpy(np.float64)
            b = merged["abs_error"].to_numpy(np.float64)
            try:
                stat, p_value = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                stat, p_value = np.nan, 1.0
            rows.append(
                {
                    "normalization": norm,
                    "scale_normalization": scale_norm,
                    "condition": condition,
                    "type": ptype,
                    "scale": float(scale),
                    "dx": float(dx),
                    "n": int(len(merged)),
                    "wilcoxon_stat": float(stat),
                    "p_value": float(p_value),
                    "rank_biserial_effect": float(rank_biserial_effect(a, b)),
                }
            )
    return pd.DataFrame(rows)


def plot_results(summary: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    scale_df = summary[summary["type"].isin(["original", "scale"])].sort_values(["normalization", "scale"])
    trans_df = summary[summary["type"].isin(["original", "translation"])].sort_values(["normalization", "dx"])

    fig, ax = plt.subplots(figsize=(7, 4))
    for norm, group in scale_df.groupby("normalization"):
        ax.plot(group["scale"], group["mae"], marker="o", linewidth=2, label=norm)
    ax.set_xlabel("Scale factor")
    ax.set_ylabel("MAE")
    ax.set_title("MAE vs scale perturbation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "mae_vs_scale.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for norm, group in scale_df.groupby("normalization"):
        ax.plot(group["scale"], group["rmse"], marker="o", linewidth=2, label=norm)
    ax.set_xlabel("Scale factor")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE vs scale perturbation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "rmse_vs_scale.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for norm, group in trans_df.groupby("normalization"):
        ax.plot(group["dx"], group["mae"], marker="o", linewidth=2, label=norm)
    ax.set_xlabel("Horizontal translation")
    ax.set_ylabel("MAE")
    ax.set_title("MAE vs translation perturbation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "mae_vs_translation.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    plot_df = summary[summary["type"] != "original"].copy()
    plot_df["label"] = plot_df["condition"].str.replace("Scale ", "S", regex=False).str.replace("Shift ", "T", regex=False)
    for norm, group in plot_df.groupby("normalization"):
        ax.plot(range(len(group)), group["delta_mae_pct"], marker="o", linewidth=2, label=f"{norm} MAE")
        ax.plot(range(len(group)), group["delta_rmse_pct"], marker="s", linewidth=2, linestyle="--", label=f"{norm} RMSE")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(plot_df[plot_df["normalization"] == "COM"])), plot_df[plot_df["normalization"] == "COM"]["label"], rotation=35, ha="right")
    ax.set_ylabel("Relative error increase (%)")
    ax.set_title("Relative degradation under perturbation")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "relative_degradation.png", bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame, float_cols: set[str]) -> list[str]:
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for row in df.to_dict(orient="records"):
        vals = []
        for col in df.columns:
            val = row[col]
            vals.append(f"{float(val):.3f}" if col in float_cols else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_report(out_dir: Path, summary: pd.DataFrame, stats: pd.DataFrame) -> None:
    com = summary[summary["normalization"] == "COM"].copy()
    display = com[["condition", "mae", "rmse", "delta_mae_pct", "delta_rmse_pct"]].rename(
        columns={
            "condition": "Condition",
            "mae": "MAE",
            "rmse": "RMSE",
            "delta_mae_pct": "Delta MAE (%)",
            "delta_rmse_pct": "Delta RMSE (%)",
        }
    )
    raw = summary[summary["normalization"] == "Raw"].copy()
    raw_display = raw[["condition", "mae", "rmse", "delta_mae_pct", "delta_rmse_pct"]].rename(
        columns={
            "condition": "Condition",
            "mae": "MAE",
            "rmse": "RMSE",
            "delta_mae_pct": "Delta MAE (%)",
            "delta_rmse_pct": "Delta RMSE (%)",
        }
    )
    stat_display = stats[stats["normalization"] == "COM"][
        ["condition", "p_value", "rank_biserial_effect"]
    ].rename(
        columns={
            "condition": "Condition",
            "p_value": "Wilcoxon p",
            "rank_biserial_effect": "Rank-biserial effect",
        }
    )

    lines = [
        "# COM Robustness Experiment",
        "",
        "- Model: Ours V1, Configuration D",
        "- Perturbations applied only at inference time",
        "- COM mode: perturb raw coordinates, then apply COM normalization before feature construction",
        "- Raw mode: perturb raw coordinates and skip COM normalization before feature construction",
        f"- Scale normalization: `{summary['scale_normalization'].iloc[0] if not summary.empty else 'unknown'}`",
        "",
        "## COM-Normalized Inference",
        "",
        *markdown_table(display, {"MAE", "RMSE", "Delta MAE (%)", "Delta RMSE (%)"}),
        "",
        "## Raw Inference Without COM Normalization",
        "",
        *markdown_table(raw_display, {"MAE", "RMSE", "Delta MAE (%)", "Delta RMSE (%)"}),
        "",
        "## Wilcoxon Tests, COM-Normalized Inference",
        "",
        *markdown_table(stat_display, {"Wilcoxon p", "Rank-biserial effect"}),
        "",
        "## Figures",
        "",
        "- `figures/mae_vs_scale.png`",
        "- `figures/rmse_vs_scale.png`",
        "- `figures/mae_vs_translation.png`",
        "- `figures/relative_degradation.png`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate COM robustness under coordinate perturbations.")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--checkpoint_dir", default="results/groupkfold_h36m17_ours_d_checkpointed_cuda/checkpoints")
    parser.add_argument("--out_dir", default="docs/com_robustness")
    parser.add_argument("--split_strategy", choices=["groupkfold"], default="groupkfold")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--fold_limit", type=int, default=None, help="Optional first-N-fold limit for smoke tests.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--scale_normalization",
        choices=["checkpoint", "none", "median_bone", "torso", "hip_width"],
        default="checkpoint",
        help="Use checkpoint metadata or override body-scale normalization at inference.",
    )
    parser.add_argument("--scale_values", nargs="+", type=float, default=DEFAULT_SCALE_VALUES)
    parser.add_argument("--translation_values", nargs="+", type=float, default=DEFAULT_TRANSLATION_VALUES)
    parser.add_argument(
        "--combined_values",
        nargs="*",
        default=[f"{s},{dx}" for s, dx in DEFAULT_COMBINED_VALUES],
        help="Combined perturbations as scale,dx pairs, e.g. 0.70,-0.20.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_manifest(manifest_path)
    splits = group_kfold_splits(df, args.n_splits)
    if args.fold_limit is not None:
        splits = splits[: args.fold_limit]
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    combined_values = []
    for item in args.combined_values:
        if not item:
            continue
        scale_str, dx_str = str(item).split(",", 1)
        combined_values.append((float(scale_str), float(dx_str)))
    conditions = condition_list(args.scale_values, args.translation_values, combined_values)

    all_metric_rows = []
    all_pred_rows = []
    for normalize_com in [True, False]:
        metric_rows, pred_rows = evaluate_mode(
            manifest_path,
            df,
            splits,
            checkpoint_dir,
            normalize_com=normalize_com,
            scale_normalization_arg=args.scale_normalization,
            conditions=conditions,
            device=device,
            batch_size=args.batch_size,
        )
        all_metric_rows.extend(metric_rows)
        all_pred_rows.extend(pred_rows)

    fold_df = pd.DataFrame(all_metric_rows)
    pred_df = pd.DataFrame(all_pred_rows)
    summary = aggregate_metrics(fold_df, pred_df)
    stats = statistical_tests(pred_df)

    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    pred_df.to_csv(out_dir / "predictions.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    stats.to_csv(out_dir / "wilcoxon_tests.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    plot_results(summary, out_dir)
    write_report(out_dir, summary, stats)
    print(f"[INFO] Wrote COM robustness outputs to {out_dir}")


if __name__ == "__main__":
    main()
