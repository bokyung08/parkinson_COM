from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from gait17.data import Gait17Dataset, load_manifest, materialize_arrays
from gait17.models import expected_score_from_logits, make_model, ordinal_focal_loss
from gait17.training import LU_CLASSIFIERS, parameter_count, regression_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def input_kind_for_model(model_name: str) -> str:
    return "coords" if model_name in {"stgcn", *LU_CLASSIFIERS} else "hybrid"


def train_fixed_epoch(
    manifest_path: Path,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
    args,
) -> tuple[dict, pd.DataFrame, dict]:
    input_kind = input_kind_for_model(model_name)
    x_test, y_test, ids = materialize_arrays(
        manifest_path,
        test_idx,
        args.max_len,
        args.ablation,
        input_kind,
        scale_normalization=args.scale_normalization,
    )
    if args.scale_aug_min != 1.0 or args.scale_aug_max != 1.0:
        train_ds = Gait17Dataset(
            manifest_path,
            train_idx,
            args.max_len,
            args.ablation,
            input_kind,
            scale_normalization=args.scale_normalization,
            scale_aug_min=args.scale_aug_min,
            scale_aug_max=args.scale_aug_max,
            random_state=args.random_state,
        )
        in_channels = int(x_test.shape[-1])
    else:
        x_train, y_train, _ = materialize_arrays(
            manifest_path,
            train_idx,
            args.max_len,
            args.ablation,
            input_kind,
            scale_normalization=args.scale_normalization,
        )
        train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
        in_channels = int(x_train.shape[-1])

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = make_model(model_name, in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    criterion = nn.MSELoss()
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    losses = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in loader:
            xb, yb = batch[:2]
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = ordinal_focal_loss(output, yb) if model_name in LU_CLASSIFIERS else criterion(output, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        losses.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)) if epoch_losses else np.nan})
    train_seconds = time.perf_counter() - start

    model.eval()
    test_tensor = torch.from_numpy(x_test).float()
    preds = []
    infer_start = time.perf_counter()
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(test_tensor), batch_size=args.eval_batch_size, shuffle=False):
            out = model(xb.to(device))
            pred = expected_score_from_logits(out) if model_name in LU_CLASSIFIERS else out
            preds.append(pred.detach().cpu().numpy().astype(np.float32))
    y_pred = np.concatenate(preds, axis=0)
    infer_ms = (time.perf_counter() - infer_start) * 1000.0 / max(len(y_test), 1)
    row = {
        "model": model_name,
        "params": parameter_count(model),
        "train_seconds": float(train_seconds),
        "inference_ms_per_sample": float(infer_ms),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        **regression_metrics(y_test, y_pred),
    }
    pred_df = pd.DataFrame(
        {
            "sample_id": ids,
            "y_true": y_test.astype(float),
            "y_pred": y_pred.astype(float),
            "abs_error": np.abs(y_pred - y_test).astype(float),
        }
    )
    checkpoint = {
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "input_kind": input_kind,
        "in_channels": int(in_channels),
        "ablation": args.ablation,
        "scale_normalization": args.scale_normalization,
        "max_len": int(args.max_len),
        "epochs": int(args.epochs),
    }
    return row, pred_df, {"losses": losses, "checkpoint": checkpoint}


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
    parser = argparse.ArgumentParser(description="CARE-PD leave-one-dataset-out validation.")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--out_dir", default="results/carepd_leave_one_dataset_out")
    parser.add_argument("--doc_path", default="docs/carepd_lodo_analysis.md")
    parser.add_argument("--models", nargs="+", choices=["ours", "stgcn", "lu_ofddnet_official"], default=["ours"])
    parser.add_argument("--ablation", choices=["A", "B", "C", "D"], default="D")
    parser.add_argument("--max_len", type=int, default=390)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--scale_normalization", choices=["none", "median_bone", "torso", "hip_width"], default="none")
    parser.add_argument("--scale_aug_min", type=float, default=1.0)
    parser.add_argument("--scale_aug_max", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.random_state)
    manifest_path = Path(args.manifest)
    df = load_manifest(manifest_path)
    carepd = df[df["dataset"].astype(str) == "CAREPD"].copy()
    if "source_dataset" not in carepd.columns:
        raise SystemExit("Manifest does not contain source_dataset for CARE-PD.")
    carepd = carepd[carepd["source_dataset"].notna()].copy()
    cohorts = sorted(carepd["source_dataset"].astype(str).unique())
    if len(cohorts) < 2:
        raise SystemExit("Need at least two CARE-PD source_dataset cohorts for LODO.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    doc_path = Path(args.doc_path)
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    all_preds = []
    for model_name in args.models:
        for cohort in cohorts:
            test_idx = df.index[(df["dataset"].astype(str) == "CAREPD") & (df["source_dataset"].astype(str) == cohort)].to_numpy()
            train_idx = df.index[(df["dataset"].astype(str) == "CAREPD") & (df["source_dataset"].astype(str) != cohort)].to_numpy()
            run_dir = out_dir / model_name / cohort
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] model={model_name} heldout={cohort} n_train={len(train_idx)} n_test={len(test_idx)}")
            row, pred_df, aux = train_fixed_epoch(manifest_path, train_idx, test_idx, model_name, args)
            row.update({"heldout_dataset": cohort})
            rows.append(row)
            pred_df.insert(0, "model", model_name)
            pred_df.insert(1, "heldout_dataset", cohort)
            pred_df.to_csv(run_dir / "predictions.tsv", sep="\t", index=False)
            pd.DataFrame(aux["losses"]).to_csv(run_dir / "train_loss.csv", index=False)
            torch.save(aux["checkpoint"], out_dir / "checkpoints" / f"{model_name}_{cohort}.pt")
            all_preds.append(pred_df)
            pd.DataFrame(rows).to_csv(out_dir / "fold_metrics_partial.csv", index=False)

    fold_metrics = pd.DataFrame(rows)
    predictions = pd.concat(all_preds, ignore_index=True)
    summary_rows = []
    for model_name, group in predictions.groupby("model", sort=True):
        summary_rows.append(
            {
                "model": model_name,
                "n_predictions": int(len(group)),
                "n_heldout_datasets": int(group["heldout_dataset"].nunique()),
                **regression_metrics(group["y_true"].to_numpy(np.float32), group["y_pred"].to_numpy(np.float32)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    fold_metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(out_dir / "predictions.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")

    lines = [
        "# CARE-PD Leave-One-Dataset-Out Validation",
        "",
        "- Dataset: CARE-PD only",
        "- Split: hold out one CARE-PD source dataset/cohort at a time",
        "- Fine-tuning/adaptation: none",
        f"- Epochs: `{args.epochs}`",
        "",
        "## Summary",
        "",
        *markdown_table(
            summary.rename(
                columns={
                    "model": "Model",
                    "n_predictions": "N",
                    "n_heldout_datasets": "Held-out datasets",
                    "mae": "MAE",
                    "rmse": "RMSE",
                    "medae": "MedAE",
                }
            ),
            {"MAE", "RMSE", "MedAE"},
        ),
        "",
        "## Held-Out Cohort Metrics",
        "",
        *markdown_table(
            fold_metrics[["model", "heldout_dataset", "n_train", "n_test", "mae", "rmse", "medae"]].rename(
                columns={
                    "model": "Model",
                    "heldout_dataset": "Held-out dataset",
                    "n_train": "N train",
                    "n_test": "N test",
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
        "This experiment is a stronger external generalization test within CARE-PD than random subject-level folds because the held-out cohort has no sequences represented during training.",
    ]
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote {out_dir} and {doc_path}")


if __name__ == "__main__":
    main()
