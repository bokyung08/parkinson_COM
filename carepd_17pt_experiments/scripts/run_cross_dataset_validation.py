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
from gait17.models import make_model
from gait17.training import parameter_count, regression_metrics


PROTOCOLS = {
    "cnuh_to_carepd": ("CNUH", "CAREPD", "CNUH", "CARE-PD"),
    "carepd_to_cnuh": ("CAREPD", "CNUH", "CARE-PD", "CNUH"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_indices(df: pd.DataFrame, dataset: str) -> np.ndarray:
    idx = np.where(df["dataset"].astype(str).to_numpy() == dataset)[0]
    if idx.size == 0:
        raise ValueError(f"No rows found for dataset={dataset}")
    return idx


def train_fixed_epoch(
    manifest_path: Path,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args,
) -> tuple[dict, pd.DataFrame, dict]:
    input_kind = "hybrid"
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
    model = make_model("ours", in_channels).to(device)
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
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        losses.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)) if epoch_losses else np.nan})
    train_seconds = time.perf_counter() - start

    model.eval()
    test_tensor = torch.from_numpy(x_test).float().to(device)
    preds = []
    infer_start = time.perf_counter()
    with torch.no_grad():
        for xb in DataLoader(TensorDataset(test_tensor), batch_size=args.eval_batch_size, shuffle=False):
            out = model(xb[0].to(device))
            preds.append(out.detach().cpu().numpy().astype(np.float32))
    y_pred = np.concatenate(preds, axis=0)
    infer_ms = (time.perf_counter() - infer_start) * 1000.0 / max(len(y_test), 1)

    metrics = regression_metrics(y_test, y_pred)
    row = {
        "params": parameter_count(model),
        "train_seconds": float(train_seconds),
        "inference_ms_per_sample": float(infer_ms),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        **metrics,
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
        "model_name": "ours",
        "state_dict": model.state_dict(),
        "input_kind": input_kind,
        "in_channels": int(in_channels),
        "ablation": args.ablation,
        "scale_normalization": args.scale_normalization,
        "scale_aug_min": float(args.scale_aug_min),
        "scale_aug_max": float(args.scale_aug_max),
        "max_len": int(args.max_len),
        "epochs": int(args.epochs),
    }
    return row, pred_df, {"losses": losses, "checkpoint": checkpoint}


def load_combined_summary(path: Path) -> dict:
    if not path.exists():
        return {
            "protocol": "combined_groupkfold",
            "train_set": "Combined",
            "test_set": "Combined",
            "n_train": np.nan,
            "n_test": np.nan,
            "params": np.nan,
            "train_seconds": np.nan,
            "inference_ms_per_sample": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "medae": np.nan,
            "note": f"Missing combined summary: {path}",
        }
    df = pd.read_csv(path)
    ours = df[df["model"].astype(str) == "ours"]
    if ours.empty:
        ours = df[df["category"].astype(str) == "Proposed"]
    if ours.empty:
        raise ValueError(f"No Ours row found in {path}")
    row = ours.iloc[0].to_dict()
    return {
        "protocol": "combined_groupkfold",
        "train_set": "Combined",
        "test_set": "Combined",
        "n_train": "GroupKFold",
        "n_test": int(row.get("n_predictions", 6087)),
        "params": int(row["params"]),
        "train_seconds": np.nan,
        "inference_ms_per_sample": float(row["inference_ms_per_sample"]),
        "mae": float(row["mae"]),
        "rmse": float(row["rmse"]),
        "medae": float(row["medae"]),
        "note": "Existing subject-level GroupKFold main result",
    }


def markdown_table(df: pd.DataFrame, float_cols: set[str]) -> list[str]:
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for row in df.to_dict(orient="records"):
        vals = []
        for col in df.columns:
            value = row[col]
            if col in float_cols and pd.notna(value):
                vals.append(f"{float(value):.3f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_report(out_dir: Path, summary: pd.DataFrame, domain_gap: pd.DataFrame, args) -> None:
    table = summary[["train_set", "test_set", "mae", "rmse", "medae"]].rename(
        columns={"train_set": "Train Set", "test_set": "Test Set", "mae": "MAE", "rmse": "RMSE", "medae": "MedAE"}
    )
    gap = domain_gap.rename(
        columns={
            "comparison": "Comparison",
            "delta_mae": "Delta MAE",
            "delta_rmse": "Delta RMSE",
            "relative_mae_increase_pct": "Relative MAE Increase (%)",
            "relative_rmse_increase_pct": "Relative RMSE Increase (%)",
        }
    )
    lines = [
        "# Cross-Dataset Validation",
        "",
        "- Model: Ours V1, Configuration D",
        "- Architecture: GraphConv + Joint Attention + Temporal Transformer + bounded regression",
        "- Fine-tuning/adaptation: none",
        "- External-transfer training: fixed epoch training; test labels are not used for checkpoint selection",
        f"- Epochs: `{args.epochs}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Device: `{args.device}`",
        "",
        "## Table 10. Cross-Dataset Validation",
        "",
        *markdown_table(table, {"MAE", "RMSE", "MedAE"}),
        "",
        "## Domain Gap Summary",
        "",
        *markdown_table(gap, {"Delta MAE", "Delta RMSE", "Relative MAE Increase (%)", "Relative RMSE Increase (%)"}),
        "",
        "## Section 5.8 Draft",
        "",
        "To evaluate external generalization, we conducted cross-dataset transfer experiments without fine-tuning or domain adaptation. "
        "In Protocol 1, the proposed Configuration D model was trained only on the CNUH cohort and directly evaluated on CARE-PD. "
        "In Protocol 2, the model was trained only on CARE-PD and directly evaluated on CNUH. "
        "Protocol 3 reports the subject-level GroupKFold result on the combined CNUH+CARE-PD cohort, corresponding to the main evaluation setting.",
        "",
        "[INSERT INTERPRETATION AFTER FULL RUN: compare CNUH-to-CARE-PD, CARE-PD-to-CNUH, and combined GroupKFold values.]",
        "",
        "## Discussion Notes",
        "",
        "- CNUH -> CARE-PD tests whether a small single-site clinical dataset transfers to a large multi-site benchmark.",
        "- CARE-PD -> CNUH tests whether a larger public multi-site benchmark transfers back to the local IRB-approved cohort.",
        "- Expected domain-gap factors include camera viewpoint and setup, pose-representation differences, patient severity distribution, site-specific acquisition protocols, and annotation harmonization differences.",
        "- If CARE-PD -> CNUH performs better than CNUH -> CARE-PD, the likely explanation is training-set size and diversity.",
        "- If both external-transfer directions underperform combined GroupKFold, the result supports reporting cross-site domain gap as a limitation and motivates domain adaptation or site-balanced training.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-free cross-dataset validation for Ours V1.")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--out_dir", default="results/cross_dataset_validation")
    parser.add_argument("--protocols", nargs="+", choices=list(PROTOCOLS), default=list(PROTOCOLS))
    parser.add_argument("--combined_summary", default="results/groupkfold_h36m17_ours_lu_official_cuda/summary.csv")
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    df = load_manifest(manifest_path)

    summary_rows = []
    for protocol in args.protocols:
        train_dataset, test_dataset, train_label, test_label = PROTOCOLS[protocol]
        train_idx = dataset_indices(df, train_dataset)
        test_idx = dataset_indices(df, test_dataset)
        protocol_dir = out_dir / protocol
        protocol_dir.mkdir(exist_ok=True)
        row, pred_df, aux = train_fixed_epoch(manifest_path, train_idx, test_idx, args)
        row.update(
            {
                "protocol": protocol,
                "train_set": train_label,
                "test_set": test_label,
                "note": "Fixed-epoch external transfer; no test-set checkpoint selection",
            }
        )
        summary_rows.append(row)
        pred_df.insert(0, "protocol", protocol)
        pred_df.insert(1, "train_set", train_label)
        pred_df.insert(2, "test_set", test_label)
        pred_df.to_csv(protocol_dir / "predictions.tsv", sep="\t", index=False)
        pd.DataFrame(aux["losses"]).to_csv(protocol_dir / "train_loss.csv", index=False)
        torch.save(aux["checkpoint"], out_dir / "checkpoints" / f"{protocol}.pt")

    summary_rows.append(load_combined_summary(Path(args.combined_summary)))
    summary = pd.DataFrame(summary_rows)
    summary = summary[
        [
            "protocol",
            "train_set",
            "test_set",
            "n_train",
            "n_test",
            "params",
            "train_seconds",
            "inference_ms_per_sample",
            "mae",
            "rmse",
            "medae",
            "note",
        ]
    ]

    combined = summary[summary["protocol"] == "combined_groupkfold"].iloc[0]
    gap_rows = []
    for _, row in summary[summary["protocol"] != "combined_groupkfold"].iterrows():
        delta_mae = float(row["mae"]) - float(combined["mae"])
        delta_rmse = float(row["rmse"]) - float(combined["rmse"])
        gap_rows.append(
            {
                "comparison": f"{row['train_set']} -> {row['test_set']} vs Combined GroupKFold",
                "delta_mae": delta_mae,
                "delta_rmse": delta_rmse,
                "relative_mae_increase_pct": delta_mae / float(combined["mae"]) * 100.0,
                "relative_rmse_increase_pct": delta_rmse / float(combined["rmse"]) * 100.0,
            }
        )
    if {"cnuh_to_carepd", "carepd_to_cnuh"}.issubset(set(summary["protocol"])):
        a = summary[summary["protocol"] == "cnuh_to_carepd"].iloc[0]
        b = summary[summary["protocol"] == "carepd_to_cnuh"].iloc[0]
        gap_rows.append(
            {
                "comparison": "CNUH -> CARE-PD minus CARE-PD -> CNUH",
                "delta_mae": float(a["mae"]) - float(b["mae"]),
                "delta_rmse": float(a["rmse"]) - float(b["rmse"]),
                "relative_mae_increase_pct": (float(a["mae"]) - float(b["mae"])) / max(float(b["mae"]), 1e-6) * 100.0,
                "relative_rmse_increase_pct": (float(a["rmse"]) - float(b["rmse"])) / max(float(b["rmse"]), 1e-6) * 100.0,
            }
        )
    domain_gap = pd.DataFrame(gap_rows)

    summary.to_csv(out_dir / "summary.csv", index=False)
    domain_gap.to_csv(out_dir / "domain_gap.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    write_report(out_dir, summary, domain_gap, args)
    print(f"[INFO] Wrote cross-dataset validation outputs to {out_dir}")


if __name__ == "__main__":
    main()
