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


PROTOCOLS = {
    "cnuh_to_carepd": ("CNUH", "CAREPD", "CNUH", "CARE-PD"),
    "carepd_to_cnuh": ("CAREPD", "CNUH", "CARE-PD", "CNUH"),
}

MODEL_LABELS = {
    "ours": ("Proposed", "Ours V1"),
    "stgcn": ("SOTA", "ST-GCN"),
    "lu_ofddnet_official": ("SOTA", "Lu official"),
}

COMBINED_SUMMARY_PATHS = {
    "ours": "results/groupkfold_h36m17_ours_lu_official_cuda/summary.csv",
    "lu_ofddnet_official": "results/groupkfold_h36m17_ours_lu_official_cuda/summary.csv",
    "stgcn": "results/groupkfold_h36m17_sota_cuda/summary.csv",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def input_kind_for_model(model_name: str) -> str:
    return "coords" if model_name in {"stgcn", *LU_CLASSIFIERS} else "hybrid"


def dataset_indices(df: pd.DataFrame, dataset: str) -> np.ndarray:
    idx = np.where(df["dataset"].astype(str).to_numpy() == dataset)[0]
    if idx.size == 0:
        raise ValueError(f"No rows found for dataset={dataset}")
    return idx


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

    category, display_model = MODEL_LABELS[model_name]
    row = {
        "category": category,
        "model": model_name,
        "display_model": display_model,
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
        "scale_aug_min": float(args.scale_aug_min),
        "scale_aug_max": float(args.scale_aug_max),
        "max_len": int(args.max_len),
        "epochs": int(args.epochs),
    }
    return row, pred_df, {"losses": losses, "checkpoint": checkpoint}


def load_combined_rows(models: list[str]) -> pd.DataFrame:
    rows = []
    for model_name in models:
        path = Path(COMBINED_SUMMARY_PATHS.get(model_name, ""))
        if not path.exists():
            continue
        df = pd.read_csv(path)
        match = df[df["model"].astype(str) == model_name]
        if match.empty:
            continue
        row = match.iloc[0]
        category, display_model = MODEL_LABELS[model_name]
        rows.append(
            {
                "protocol": "combined_groupkfold",
                "category": category,
                "model": model_name,
                "display_model": display_model,
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
                "note": f"Existing combined GroupKFold result from {path}",
            }
        )
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


def write_report(out_dir: Path, summary: pd.DataFrame, args) -> None:
    transfer = summary[summary["protocol"] != "combined_groupkfold"].copy()
    pivot = transfer.pivot_table(index=["category", "display_model"], columns="protocol", values="mae").reset_index()
    lines = [
        "# Cross-Dataset Model Comparison",
        "",
        "- Protocols: CNUH -> CARE-PD and CARE-PD -> CNUH",
        "- Fine-tuning/adaptation: none",
        "- Test-set checkpoint selection: none",
        f"- Epochs: `{args.epochs}`",
        f"- Device: `{args.device}`",
        "",
        "## Transfer Results",
        "",
        *markdown_table(
            transfer[["category", "display_model", "train_set", "test_set", "n_train", "n_test", "mae", "rmse", "medae"]].rename(
                columns={
                    "category": "Category",
                    "display_model": "Model",
                    "train_set": "Train",
                    "test_set": "Test",
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
        "## MAE Pivot",
        "",
        *markdown_table(pivot, {"cnuh_to_carepd", "carepd_to_cnuh"}),
        "",
        "## Interpretation",
        "",
        "Use this table to determine whether Ours is relatively more robust than ST-GCN and the Lu official-architecture baseline under strict zero-shot transfer. "
        "Even if all models degrade, Ours can be framed favorably if it has the lowest transfer MAE or the smallest degradation relative to its combined GroupKFold result.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    doc_path = Path(args.doc_path)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-shot cross-dataset comparison for Ours/ST-GCN/Lu.")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--out_dir", default="results/cross_dataset_model_comparison")
    parser.add_argument("--doc_path", default="docs/cross_dataset_model_comparison.md")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_LABELS), default=["ours", "stgcn", "lu_ofddnet_official"])
    parser.add_argument("--protocols", nargs="+", choices=list(PROTOCOLS), default=list(PROTOCOLS))
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

    rows = []
    for model_name in args.models:
        for protocol in args.protocols:
            train_dataset, test_dataset, train_label, test_label = PROTOCOLS[protocol]
            train_idx = dataset_indices(df, train_dataset)
            test_idx = dataset_indices(df, test_dataset)
            run_dir = out_dir / protocol / model_name
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] model={model_name} protocol={protocol}")
            row, pred_df, aux = train_fixed_epoch(manifest_path, train_idx, test_idx, model_name, args)
            row.update(
                {
                    "protocol": protocol,
                    "train_set": train_label,
                    "test_set": test_label,
                    "note": "Fixed-epoch external transfer; no test-set checkpoint selection",
                }
            )
            rows.append(row)
            pred_df.insert(0, "model", model_name)
            pred_df.insert(1, "protocol", protocol)
            pred_df.insert(2, "train_set", train_label)
            pred_df.insert(3, "test_set", test_label)
            pred_df.to_csv(run_dir / "predictions.tsv", sep="\t", index=False)
            pd.DataFrame(aux["losses"]).to_csv(run_dir / "train_loss.csv", index=False)
            torch.save(aux["checkpoint"], out_dir / "checkpoints" / f"{model_name}_{protocol}.pt")
            pd.DataFrame(rows).to_csv(out_dir / "transfer_summary_partial.csv", index=False)

    summary = pd.DataFrame(rows)
    combined = load_combined_rows(args.models)
    if not combined.empty:
        summary = pd.concat([summary, combined], ignore_index=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    write_report(out_dir, summary, args)
    print(f"[INFO] Wrote {out_dir}")


if __name__ == "__main__":
    main()
