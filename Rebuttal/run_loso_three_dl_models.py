"""Run LOSO-CV for MainTF, FusionTF, and HybridTorch under one protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Rebuttal.unified_data_utils import load_unified_dataset, regression_metrics


MODEL_ORDER = ["main_tf", "fusion_tf", "hybrid_torch"]
MODEL_DISPLAY = {
    "main_tf": "MainTF",
    "fusion_tf": "FusionTF",
    "hybrid_torch": "HybridTorch",
}


def normalize_models(models: list[str]) -> list[str]:
    if "all" in models:
        return MODEL_ORDER
    return models


def ablation_dir(out_root: Path, target: str, ablation: str) -> Path:
    return out_root / f"{ablation}_{target}_loso"


def short_patient(patient_id: str) -> str:
    return patient_id.replace("-", "")[:8]


def fold_dir(base_dir: Path, fold_index: int, patient_id: str) -> Path:
    return base_dir / f"fold_{fold_index:03d}_{short_patient(patient_id)}"


def row_path(base_dir: Path, fold_index: int, patient_id: str, model_key: str) -> Path:
    return fold_dir(base_dir, fold_index, patient_id) / model_key / "row.json"


def worker_command(
    args: argparse.Namespace,
    ablation: str,
    patient_id: str,
    fold_index: int,
    model_key: str,
    out_dir: Path,
) -> list[str]:
    script_path = Path(__file__).resolve().parent / "run_loso_fold_dl_model.py"
    return [
        sys.executable,
        str(script_path),
        "--processed_dir",
        args.processed_dir,
        "--label_dir",
        args.label_dir,
        "--out_dir",
        str(out_dir),
        "--target",
        args.target,
        "--ablation",
        ablation,
        "--model",
        model_key,
        "--val_patient_id",
        patient_id,
        "--fold_index",
        str(fold_index),
        "--max_len",
        str(args.max_len),
        "--random_state",
        str(args.random_state),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--device",
        args.device,
    ]


def run_command(command: list[str], dry_run: bool) -> None:
    print("[RUN]", subprocess.list2cmdline(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df


def collect_rows(out_root: Path, target: str, ablations: list[str], models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    pred_rows = []
    for ablation in ablations:
        base_dir = ablation_dir(out_root, target, ablation)
        if not base_dir.exists():
            continue
        for fold_path in sorted(base_dir.glob("fold_*")):
            if not fold_path.is_dir():
                continue
            for model_key in models:
                model_dir = fold_path / model_key
                row_file = model_dir / "row.json"
                pred_file = model_dir / "predictions.tsv"
                if not row_file.exists():
                    continue
                row = load_json(row_file)
                row["fold_dir"] = str(fold_path)
                fold_rows.append(row)
                if pred_file.exists():
                    pred_df = read_predictions(pred_file)
                    pred_df["target"] = target
                    pred_df["ablation"] = ablation
                    pred_df["model"] = row.get("model", MODEL_DISPLAY.get(model_key, model_key))
                    pred_df["model_key"] = model_key
                    pred_df["fold"] = row.get("fold")
                    pred_rows.append(pred_df)

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    return fold_df, pred_df


def overall_from_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pred_df.empty:
        return pd.DataFrame(rows)
    for (target, ablation, model_key, model), group in pred_df.groupby(["target", "ablation", "model_key", "model"]):
        y_true = group["y_true"].to_numpy(dtype=np.float32)
        y_pred = group["y_pred"].to_numpy(dtype=np.float32)
        metrics = regression_metrics(y_true, y_pred)
        rows.append(
            {
                "target": target,
                "ablation": ablation,
                "model": model,
                "model_key": model_key,
                "n_predictions": int(len(group)),
                "n_folds": int(group["fold"].nunique()),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["ablation", "model_key"]).reset_index(drop=True)


def write_summary_files(
    args: argparse.Namespace,
    out_root: Path,
    fold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    overall_df: pd.DataFrame,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    stem = f"loso_{args.target}_{'_'.join(args.ablations)}"
    fold_csv = out_root / f"{stem}_fold_results.csv"
    pred_tsv = out_root / f"{stem}_predictions.tsv"
    overall_csv = out_root / f"{stem}_overall_summary.csv"
    overall_json = out_root / f"{stem}_overall_summary.json"
    md_path = out_root / f"{stem}_RESULTS.md"

    if not fold_df.empty:
        fold_df.to_csv(fold_csv, index=False)
    if not pred_df.empty:
        pred_df.to_csv(pred_tsv, sep="\t", index=False)
    if not overall_df.empty:
        if not fold_df.empty and "inference_ms_per_sample" in fold_df.columns:
            timing = (
                fold_df.groupby(["target", "ablation", "model_key", "model"], as_index=False)
                .agg(
                    params=("params", "first"),
                    inference_ms_per_sample=("inference_ms_per_sample", "mean"),
                )
            )
            overall_df = overall_df.merge(
                timing,
                on=["target", "ablation", "model_key", "model"],
                how="left",
            )
        overall_df.to_csv(overall_csv, index=False)
        with overall_json.open("w", encoding="utf-8") as f:
            json.dump(overall_df.to_dict(orient="records"), f, indent=2)

    lines = [
        "# LOSO-CV Three-Model DL Results",
        "",
        "## Protocol",
        "",
        f"- Target: `{args.target}`",
        f"- Ablations: `{', '.join(args.ablations)}`",
        "- Split: leave-one-subject-out by `patient_id`",
        f"- Models: `{', '.join(normalize_models(args.models))}`",
        f"- Max length: `{args.max_len}` frames, last-frame window",
        f"- Epochs per fold: `{args.epochs}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Learning rate: `{args.learning_rate}`",
        f"- Torch device request: `{args.device}`",
        "- Loss: `mse`",
        "",
        "## Overall Results",
        "",
        "| Ablation | Model | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if overall_df.empty:
        lines.append("| - | - | 0 | 0 | - | - | - | - | - |")
    else:
        for row in overall_df.to_dict(orient="records"):
            params = row.get("params")
            infer_ms = row.get("inference_ms_per_sample")
            params_text = "-" if pd.isna(params) else f"{int(params)}"
            infer_text = "-" if pd.isna(infer_ms) else f"{float(infer_ms):.3f}"
            lines.append(
                f"| {row['ablation']} | {row['model']} | {int(row['n_folds'])} | "
                f"{int(row['n_predictions'])} | {params_text} | {infer_text} | {float(row['mae']):.3f} | "
                f"{float(row['rmse']):.3f} | {float(row['medae']):.3f} |"
            )
    lines.extend(
        [
            "",
            "Metrics are computed from concatenated LOSO predictions, not by averaging per-fold metrics.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Saved LOSO summary: {md_path}")


def patient_list(args: argparse.Namespace, ablation: str) -> list[str]:
    _, _, _, manifest = load_unified_dataset(
        Path(args.processed_dir),
        Path(args.label_dir),
        target=args.target,
        ablation=ablation,
        max_len=args.max_len,
    )
    patients = sorted(manifest["patient_id"].astype(str).unique().tolist())
    if args.patients:
        requested = set(args.patients)
        patients = [patient for patient in patients if patient in requested]
    if args.fold_limit is not None:
        patients = patients[: args.fold_limit]
    return patients


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOSO-CV for MainTF, FusionTF, and HybridTorch.")
    parser.add_argument("--processed_dir", default="HospitalData/processed_pose_data")
    parser.add_argument("--label_dir", default="HospitalData/JSON")
    parser.add_argument("--out_root", default="Rebuttal/results/loso_dl_models")
    parser.add_argument("--target", choices=["item10", "gait"], default="item10")
    parser.add_argument("--ablations", nargs="+", choices=["A", "B", "C", "D"], default=["D"])
    parser.add_argument("--models", nargs="+", choices=["all", "main_tf", "fusion_tf", "hybrid_torch"], default=["all"])
    parser.add_argument("--patients", nargs="*", default=None, help="optional patient_id subset for targeted reruns")
    parser.add_argument("--fold_limit", type=int, default=None, help="debug limit for the first N LOSO folds")
    parser.add_argument("--max_len", type=int, default=390)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--collect_only", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
    models = normalize_models(args.models)
    failures = []

    if not args.collect_only:
        for ablation in args.ablations:
            base_dir = ablation_dir(out_root, args.target, ablation)
            patients = patient_list(args, ablation)
            print(f"[INFO] Ablation {ablation}: {len(patients)} LOSO folds")
            for fold_index, patient_id in enumerate(patients, start=1):
                current_fold_dir = fold_dir(base_dir, fold_index, patient_id)
                for model_key in models:
                    target_row = row_path(base_dir, fold_index, patient_id, model_key)
                    if target_row.exists() and not args.overwrite:
                        print(f"[SKIP] {ablation} fold={fold_index:03d} {model_key}: row.json exists")
                        continue
                    command = worker_command(args, ablation, patient_id, fold_index, model_key, current_fold_dir)
                    try:
                        run_command(command, args.dry_run)
                    except subprocess.CalledProcessError as exc:
                        failure = {
                            "ablation": ablation,
                            "fold": fold_index,
                            "patient_id": patient_id,
                            "model": model_key,
                            "returncode": exc.returncode,
                            "command": exc.cmd,
                        }
                        failures.append(failure)
                        print(f"[ERROR] {failure}")
                        if not args.continue_on_error:
                            if not args.dry_run:
                                with (out_root / "loso_failures.json").open("w", encoding="utf-8") as f:
                                    json.dump(failures, f, indent=2)
                            raise

    if failures and not args.dry_run:
        with (out_root / "loso_failures.json").open("w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)

    if not args.dry_run:
        fold_df, pred_df = collect_rows(out_root, args.target, args.ablations, models)
        overall_df = overall_from_predictions(pred_df)
        write_summary_files(args, out_root, fold_df, pred_df, overall_df)


if __name__ == "__main__":
    main()
