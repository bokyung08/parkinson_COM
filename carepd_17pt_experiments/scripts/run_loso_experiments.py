from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from gait17.data import group_kfold_splits, load_manifest, loso_splits
from gait17.training import run_ml_fold, run_torch_fold, regression_metrics


ML_MODELS = {"ridge", "svr", "rf", "mlp_shallow"}
TORCH_MODELS = {
    "temporal_cnn",
    "ours_mlp",
    "ours_gcn_mlp",
    "ours_gcn_attn_mlp",
    "ours",
    "stgcn",
    "motionbert",
    "motionagformer",
    "motionbert_pretrained",
    "motionbert_lite_pretrained",
    "motionagformer_xs_pretrained",
    "lu_ofddnet_official",
}
DEFAULT_MODELS = [
    "ridge",
    "svr",
    "rf",
    "mlp_shallow",
    "temporal_cnn",
    "ours",
    "stgcn",
    "lu_ofddnet_official",
]


def write_predictions(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_splits(df: pd.DataFrame, args) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if args.split_strategy == "loso":
        return loso_splits(df)
    if args.split_strategy == "groupkfold":
        return group_kfold_splits(df, args.n_splits)
    raise ValueError(f"Unsupported split strategy: {args.split_strategy}")


def summarize_split(df: pd.DataFrame, val_idx: np.ndarray) -> dict[str, object]:
    val_df = df.iloc[val_idx]
    groups = sorted(val_df["patient_id"].astype(str).unique())
    datasets = val_df.groupby("dataset").size().to_dict()
    return {
        "n_val_groups": int(len(groups)),
        "val_groups": ";".join(groups),
        "val_dataset_counts": json.dumps({str(k): int(v) for k, v in datasets.items()}, sort_keys=True),
    }


def write_partial_outputs(out_dir: Path, fold_rows: list[dict], pred_rows: list[dict], progress_rows: list[dict]) -> None:
    pd.DataFrame(progress_rows).to_csv(out_dir / "progress.csv", index=False)
    if fold_rows:
        pd.DataFrame(fold_rows).to_csv(out_dir / "fold_metrics.csv", index=False)
    if pred_rows:
        write_predictions(out_dir / "predictions.tsv", pred_rows)


def current_summary(pred_rows: list[dict], fold_rows: list[dict]) -> pd.DataFrame:
    if not pred_rows:
        return pd.DataFrame()
    pred_df = pd.DataFrame(pred_rows)
    fold_df = pd.DataFrame(fold_rows)
    summary_rows = []
    for (category, model), group in pred_df.groupby(["category", "model"], sort=True):
        metrics = regression_metrics(group["y_true"].to_numpy(np.float32), group["y_pred"].to_numpy(np.float32))
        fold_group = fold_df[fold_df["model"] == model] if not fold_df.empty else pd.DataFrame()
        summary_rows.append(
            {
                "category": category,
                "model": model,
                "n_predictions": int(len(group)),
                "n_folds": int(group["fold"].nunique()),
                "params": int(fold_group["params"].iloc[0]) if not fold_group.empty else 0,
                "inference_ms_per_sample": float(fold_group["inference_ms_per_sample"].mean()) if not fold_group.empty else 0.0,
                **metrics,
            }
        )
    return pd.DataFrame(summary_rows).sort_values(["category", "mae"])


def write_run_readme(
    out_dir: Path,
    args,
    manifest_path: Path,
    df: pd.DataFrame,
    total_jobs: int,
    fold_rows: list[dict],
    pred_rows: list[dict],
    progress_rows: list[dict],
    status: str,
    current: str | None = None,
) -> None:
    dataset_rows = []
    for dataset, group in df.groupby("dataset", sort=True):
        dataset_rows.append(
            f"| {dataset} | {len(group)} | {group['patient_id'].astype(str).nunique()} | "
            f"{group['target'].min():.0f}-{group['target'].max():.0f} |"
        )

    completed = sum(1 for row in progress_rows if row.get("status") == "completed")
    failed = sum(1 for row in progress_rows if row.get("status") == "failed")
    summary = current_summary(pred_rows, fold_rows)

    lines = [
        "# Experiment Progress",
        "",
        f"- Status: `{status}`",
        f"- Last updated: `{now()}`",
        f"- Manifest: `{manifest_path}`",
        f"- Split strategy: `{args.split_strategy}`",
        f"- Group key: `patient_id`",
        f"- N splits: `{args.n_splits if args.split_strategy == 'groupkfold' else 'LOSO'}`",
        f"- Ablation: `{args.ablation}`",
        f"- Max length: `{args.max_len}`",
        f"- Scale normalization: `{args.scale_normalization}`",
        f"- Scale augmentation: `{args.scale_aug_min:.2f}-{args.scale_aug_max:.2f}`",
        f"- Models: `{' '.join(args.models)}`",
        f"- Dataset filter: `{' '.join(args.datasets) if args.datasets else 'all'}`",
        f"- Completed jobs: `{completed}/{total_jobs}`",
        f"- Failed jobs: `{failed}`",
    ]
    if current:
        lines.append(f"- Current: `{current}`")
    lines.extend(
        [
            "",
            "## Dataset Summary",
            "",
            "| Dataset | Sequences | Patient Groups | Target Range |",
            "|---|---:|---:|---:|",
            *dataset_rows,
            "",
            "## Latest Fold Events",
            "",
            "| Time | Status | Model | Fold | Split | N train | N val | MAE | RMSE |",
            "|---|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in progress_rows[-25:]:
        lines.append(
            f"| {row.get('ended_at', row.get('started_at', ''))} | {row.get('status', '')} | "
            f"{row.get('model', '')} | {row.get('fold', '')} | {row.get('split_id', '')} | "
            f"{row.get('n_train', '')} | {row.get('n_val', '')} | "
            f"{float(row['mae']):.3f} | {float(row['rmse']):.3f} |"
            if row.get("status") == "completed"
            else f"| {row.get('started_at', '')} | {row.get('status', '')} | {row.get('model', '')} | "
            f"{row.get('fold', '')} | {row.get('split_id', '')} | {row.get('n_train', '')} | "
            f"{row.get('n_val', '')} |  |  |"
        )

    if not summary.empty:
        lines.extend(
            [
                "",
                "## Current Summary",
                "",
                "| Category | Model | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary.to_dict(orient="records"):
            lines.append(
                f"| {row['category']} | {row['model']} | {int(row['n_folds'])} | {int(row['n_predictions'])} | "
                f"{int(row['params'])} | {float(row['inference_ms_per_sample']):.3f} | "
                f"{float(row['mae']):.3f} | {float(row['rmse']):.3f} | {float(row['medae']):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `progress.csv`: live fold status log",
            "- `fold_metrics.csv`: completed fold metrics",
            "- `predictions.tsv`: completed sample predictions",
            "- `summary.csv`, `summary.json`, `RESULTS.md`: final summaries",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent H36M17 subject-level experiments.")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--out_dir", default="results/groupkfold_h36m17")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=None, help="Optional dataset filter, e.g. CNUH or CAREPD.")
    parser.add_argument("--split_strategy", choices=["loso", "groupkfold"], default="groupkfold")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--target", choices=["item10", "gait"], default="item10")
    parser.add_argument("--ablation", choices=["A", "B", "C", "D"], default="D")
    parser.add_argument("--max_len", type=int, default=390)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--scale_normalization",
        choices=["none", "median_bone", "torso", "hip_width"],
        default="none",
        help="Optional body-scale normalization after COM centering.",
    )
    parser.add_argument("--scale_aug_min", type=float, default=1.0, help="Minimum random train-time coordinate scale.")
    parser.add_argument("--scale_aug_max", type=float, default=1.0, help="Maximum random train-time coordinate scale.")
    parser.add_argument("--fold_limit", type=int, default=None, help="Optional first-N-fold limit for smoke tests.")
    parser.add_argument("--save_checkpoints", action="store_true", help="Save best torch checkpoint for each fold.")
    args = parser.parse_args()
    if args.scale_aug_min <= 0 or args.scale_aug_max <= 0 or args.scale_aug_min > args.scale_aug_max:
        raise SystemExit("--scale_aug_min and --scale_aug_max must be positive and min <= max.")

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_manifest(manifest_path)
    if args.datasets:
        requested = set(args.datasets)
        df = df[df["dataset"].astype(str).isin(requested)].reset_index(drop=True)
        if df.empty:
            raise SystemExit(f"No manifest rows left after --datasets filter: {sorted(requested)}")
        manifest_path = manifest_path.with_name(f"{manifest_path.stem}_{'_'.join(args.datasets)}{manifest_path.suffix}")
        df.to_csv(manifest_path, index=False)
    splits = build_splits(df, args)
    if args.fold_limit is not None:
        splits = splits[: args.fold_limit]
    if not splits:
        raise SystemExit("No subject-level splits available.")

    all_fold_rows = []
    all_pred_rows = []
    progress_rows = []
    total_jobs = len(args.models) * len(splits)
    write_run_readme(out_dir, args, manifest_path, df, total_jobs, all_fold_rows, all_pred_rows, progress_rows, "running")
    for model_name in args.models:
        for fold, (split_id, train_idx, val_idx) in enumerate(splits, start=1):
            split_summary = summarize_split(df, val_idx)
            current = f"model={model_name} fold={fold}/{len(splits)} split={split_id}"
            print(f"[INFO] {current}")
            progress = {
                "started_at": now(),
                "status": "running",
                "model": model_name,
                "fold": fold,
                "split_id": split_id,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                **split_summary,
            }
            progress_rows.append(progress)
            write_partial_outputs(out_dir, all_fold_rows, all_pred_rows, progress_rows)
            write_run_readme(out_dir, args, manifest_path, df, total_jobs, all_fold_rows, all_pred_rows, progress_rows, "running", current)
            try:
                if model_name in ML_MODELS:
                    row, y_true, y_pred, ids = run_ml_fold(manifest_path, train_idx, val_idx, args, model_name)
                elif model_name in TORCH_MODELS:
                    checkpoint_path = None
                    checkpoint_meta = None
                    if args.save_checkpoints:
                        checkpoint_path = out_dir / "checkpoints" / f"{model_name}_fold_{fold:02d}.pt"
                        checkpoint_meta = {
                            "fold": fold,
                            "split_id": split_id,
                            "n_train": int(len(train_idx)),
                            "n_val": int(len(val_idx)),
                            "scale_normalization": args.scale_normalization,
                            "scale_aug_min": float(args.scale_aug_min),
                            "scale_aug_max": float(args.scale_aug_max),
                            **split_summary,
                        }
                    row, y_true, y_pred, ids = run_torch_fold(
                        manifest_path,
                        train_idx,
                        val_idx,
                        args,
                        model_name,
                        checkpoint_path=checkpoint_path,
                        checkpoint_meta=checkpoint_meta,
                    )
                else:
                    raise ValueError(f"Unknown model: {model_name}")
            except KeyboardInterrupt:
                progress.update({"status": "interrupted", "ended_at": now(), "error": "KeyboardInterrupt"})
                write_partial_outputs(out_dir, all_fold_rows, all_pred_rows, progress_rows)
                write_run_readme(
                    out_dir,
                    args,
                    manifest_path,
                    df,
                    total_jobs,
                    all_fold_rows,
                    all_pred_rows,
                    progress_rows,
                    "interrupted",
                    current,
                )
                print(f"[INFO] Interrupted. Partial results saved to {out_dir}")
                raise SystemExit(130)
            except Exception as exc:
                progress.update({"status": "failed", "ended_at": now(), "error": f"{type(exc).__name__}: {exc}"})
                write_partial_outputs(out_dir, all_fold_rows, all_pred_rows, progress_rows)
                write_run_readme(out_dir, args, manifest_path, df, total_jobs, all_fold_rows, all_pred_rows, progress_rows, "failed", current)
                raise

            row.update(
                {
                    "fold": fold,
                    "split_id": split_id,
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    **split_summary,
                }
            )
            all_fold_rows.append(row)
            val_meta = df.iloc[val_idx].reset_index(drop=True)
            patient_by_sample = dict(zip(val_meta["sample_id"].astype(str), val_meta["patient_id"].astype(str)))
            dataset_by_sample = dict(zip(val_meta["sample_id"].astype(str), val_meta["dataset"].astype(str)))
            for sample_id, true_value, pred_value in zip(ids, y_true, y_pred):
                sample_id = str(sample_id)
                all_pred_rows.append(
                    {
                        "model": model_name,
                        "category": row["category"],
                        "fold": fold,
                        "split_id": split_id,
                        "val_patient_id": patient_by_sample.get(sample_id, ""),
                        "dataset": dataset_by_sample.get(sample_id, ""),
                        "sample_id": sample_id,
                        "y_true": float(true_value),
                        "y_pred": float(pred_value),
                        "abs_error": float(abs(pred_value - true_value)),
                    }
                )
            progress.update(
                {
                    "status": "completed",
                    "ended_at": now(),
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                    "medae": float(row["medae"]),
                    "train_seconds": float(row["train_seconds"]),
                    "inference_ms_per_sample": float(row["inference_ms_per_sample"]),
                }
            )
            write_partial_outputs(out_dir, all_fold_rows, all_pred_rows, progress_rows)
            write_run_readme(out_dir, args, manifest_path, df, total_jobs, all_fold_rows, all_pred_rows, progress_rows, "running", current)

    fold_df = pd.DataFrame(all_fold_rows)
    pred_df = pd.DataFrame(all_pred_rows)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    write_predictions(out_dir / "predictions.tsv", all_pred_rows)

    summary_rows = []
    for (category, model), group in pred_df.groupby(["category", "model"], sort=True):
        metrics = regression_metrics(group["y_true"].to_numpy(np.float32), group["y_pred"].to_numpy(np.float32))
        fold_group = fold_df[fold_df["model"] == model]
        summary_rows.append(
            {
                "category": category,
                "model": model,
                "n_predictions": int(len(group)),
                "n_folds": int(group["fold"].nunique()),
                "params": int(fold_group["params"].iloc[0]),
                "inference_ms_per_sample": float(fold_group["inference_ms_per_sample"].mean()),
                **metrics,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["category", "mae"])
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")

    lines = [
        "# H36M17 Dataset-Added Results",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Split strategy: `{args.split_strategy}`",
        f"- N splits: `{args.n_splits if args.split_strategy == 'groupkfold' else 'LOSO'}`",
        f"- Ablation: `{args.ablation}`",
        f"- Max length: `{args.max_len}`",
        f"- Scale normalization: `{args.scale_normalization}`",
        f"- Scale augmentation: `{args.scale_aug_min:.2f}-{args.scale_aug_max:.2f}`",
        "",
        "| Category | Model | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['category']} | {row['model']} | {int(row['n_folds'])} | {int(row['n_predictions'])} | "
            f"{int(row['params'])} | {float(row['inference_ms_per_sample']):.3f} | "
            f"{float(row['mae']):.3f} | {float(row['rmse']):.3f} | {float(row['medae']):.3f} |"
        )
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_run_readme(out_dir, args, manifest_path, df, total_jobs, all_fold_rows, all_pred_rows, progress_rows, "completed")
    print(f"[INFO] Saved results to {out_dir}")


if __name__ == "__main__":
    main()
