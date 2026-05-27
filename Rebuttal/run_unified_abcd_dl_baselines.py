"""Run unified FusionTF and HybridTorch baselines for ablations A/B/C/D.

Each model/ablation pair is launched as a separate Python process. This keeps
TensorFlow and Torch memory lifetimes isolated during long Windows CPU runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


MODEL_SPECS = {
    "fusion_tf": {
        "script": "run_unified_fusion_tf.py",
        "row_path": Path("fusion_tf") / "row.json",
    },
    "hybrid_torch": {
        "script": "run_unified_hybrid_torch.py",
        "row_path": Path("hybrid_torch") / "row.json",
    },
}


def split_tag(test_size: float) -> str:
    train_pct = round((1.0 - test_size) * 100)
    test_pct = round(test_size * 100)
    return f"{train_pct}_{test_pct}"


def split_label(test_size: float) -> str:
    return split_tag(test_size).replace("_", "/")


def normalize_models(models: list[str]) -> list[str]:
    if "all" in models:
        return ["fusion_tf", "hybrid_torch"]
    return models


def result_dir(out_root: Path, target: str, ablation: str, test_size: float) -> Path:
    return out_root / f"{ablation}_{target}_{split_tag(test_size)}"


def row_exists(out_dir: Path, model_key: str) -> bool:
    return (out_dir / MODEL_SPECS[model_key]["row_path"]).exists()


def run_command(command: list[str], dry_run: bool) -> None:
    print("[RUN]", subprocess.list2cmdline(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def model_command(args: argparse.Namespace, model_key: str, ablation: str, out_dir: Path) -> list[str]:
    script_dir = Path(__file__).resolve().parent
    script_path = script_dir / str(MODEL_SPECS[model_key]["script"])
    command = [
        sys.executable,
        str(script_path),
        "--processed_dir",
        args.processed_dir,
        "--label_dir",
        args.label_dir,
        "--target",
        args.target,
        "--ablation",
        ablation,
        "--test_size",
        str(args.test_size),
        "--random_state",
        str(args.random_state),
        "--max_len",
        str(args.max_len),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--out_dir",
        str(out_dir),
    ]
    if model_key == "hybrid_torch":
        command.extend(["--device", args.device])
    return command


def collect_command(args: argparse.Namespace, ablation: str, out_dir: Path) -> list[str]:
    script_path = Path(__file__).resolve().parent / "collect_unified_dl_results.py"
    return [
        sys.executable,
        str(script_path),
        "--out_dir",
        str(out_dir),
        "--target",
        args.target,
        "--ablation",
        ablation,
        "--test_size",
        str(args.test_size),
        "--random_state",
        str(args.random_state),
        "--max_len",
        str(args.max_len),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        str(args.learning_rate),
    ]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_rows(args: argparse.Namespace, out_root: Path, ablations: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ablation in ablations:
        out_dir = result_dir(out_root, args.target, ablation, args.test_size)
        for model_key in ["fusion_tf", "hybrid_torch"]:
            row_path = out_dir / MODEL_SPECS[model_key]["row_path"]
            if not row_path.exists():
                continue
            row = load_json(row_path)
            rows.append(
                {
                    "target": args.target,
                    "ablation": ablation,
                    "model": row.get("model", model_key),
                    "params": row.get("params"),
                    "train_seconds": row.get("train_seconds"),
                    "mae": row.get("mae"),
                    "rmse": row.get("rmse"),
                    "medae": row.get("medae"),
                    "result_dir": str(out_dir),
                }
            )
    return rows


def write_aggregate_summary(args: argparse.Namespace, out_root: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        print("[WARN] No row.json files found for aggregate ABCD summary.")
        return

    stem = f"unified_abcd_{args.target}_{split_tag(args.test_size)}"
    csv_path = out_root / f"{stem}_summary.csv"
    json_path = out_root / f"{stem}_summary.json"
    md_path = out_root / f"{stem}_RESULTS.md"

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    lines = [
        "# Unified ABCD DL Baseline Results",
        "",
        "## Protocol",
        "",
        f"- Target: `{args.target}`",
        "- Ablations: `A`, `B`, `C`, `D`",
        f"- Split: `{split_label(args.test_size)}` hold-out, random_state={args.random_state}",
        f"- Max length: `{args.max_len}` frames, last-frame window",
        f"- Epochs: `{args.epochs}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Learning rate: `{args.learning_rate}`",
        "- Loss: `mse`",
        "",
        "## Results",
        "",
        "| Ablation | Model | Params | MAE | RMSE | MedAE | Train sec |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['ablation']} | {row['model']} | {int(row['params'])} | "
            f"{float(row['mae']):.3f} | {float(row['rmse']):.3f} | "
            f"{float(row['medae']):.3f} | {float(row['train_seconds']):.1f} |"
        )
    lines.extend(
        [
            "",
            "Each row is produced by the same unified rebuttal protocol. Completed model outputs are reused unless the runner is called with `--overwrite`.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Saved aggregate ABCD summary: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified FusionTF/HybridTorch baselines for A/B/C/D.")
    parser.add_argument("--processed_dir", default="HospitalData/processed_pose_data")
    parser.add_argument("--label_dir", default="HospitalData/JSON")
    parser.add_argument("--out_root", default="Rebuttal/results/unified_dl_baselines")
    parser.add_argument("--target", choices=["item10", "gait"], default="item10")
    parser.add_argument("--ablations", nargs="+", choices=["A", "B", "C", "D"], default=["A", "B", "C", "D"])
    parser.add_argument("--models", nargs="+", choices=["all", "fusion_tf", "hybrid_torch"], default=["all"])
    parser.add_argument("--max_len", type=int, default=390)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--overwrite", action="store_true", help="rerun model outputs even when row.json exists")
    parser.add_argument("--dry_run", action="store_true", help="print commands without running them")
    parser.add_argument("--no_collect", action="store_true", help="skip per-ablation and aggregate summary collection")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
    models = normalize_models(args.models)

    for ablation in args.ablations:
        out_dir = result_dir(out_root, args.target, ablation, args.test_size)
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Ablation {ablation}: {out_dir}")
        for model_key in models:
            if row_exists(out_dir, model_key) and not args.overwrite:
                print(f"[SKIP] {ablation} {model_key}: existing row.json found")
                continue
            run_command(model_command(args, model_key, ablation, out_dir), args.dry_run)

        if not args.no_collect and not args.dry_run:
            has_any_result = any(row_exists(out_dir, key) for key in ["fusion_tf", "hybrid_torch"])
            if has_any_result:
                run_command(collect_command(args, ablation, out_dir), args.dry_run)

    if not args.no_collect and not args.dry_run:
        write_aggregate_summary(args, out_root, aggregate_rows(args, out_root, args.ablations))


if __name__ == "__main__":
    main()
