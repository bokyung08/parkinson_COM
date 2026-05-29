from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FEATURES = {
    "A": "coordinates only",
    "B": "coordinates + velocity",
    "C": "coordinates + velocity + amplitude/variability",
    "D": "full hybrid feature set",
}


def load_row(results_root: Path, ablation: str) -> dict:
    if ablation == "D":
        result_dir = results_root / "groupkfold_h36m17_ours_lu_official_cuda"
    else:
        result_dir = results_root / f"groupkfold_h36m17_ours_ablation_{ablation}_cuda"
    summary_path = result_dir / "summary.csv"
    if not summary_path.exists():
        return {
            "model": "Ours V1",
            "ablation": ablation,
            "feature_set": FEATURES[ablation],
            "status": "pending",
            "folds": "",
            "n": "",
            "params": "",
            "inference_ms_per_sample": "",
            "mae": "",
            "rmse": "",
            "medae": "",
        }
    summary = pd.read_csv(summary_path)
    row = summary[summary["model"] == "ours"].iloc[0]
    return {
        "model": "Ours V1",
        "ablation": ablation,
        "feature_set": FEATURES[ablation],
        "status": "completed",
        "folds": int(row["n_folds"]),
        "n": int(row["n_predictions"]),
        "params": int(row["params"]),
        "inference_ms_per_sample": float(row["inference_ms_per_sample"]),
        "mae": float(row["mae"]),
        "rmse": float(row["rmse"]),
        "medae": float(row["medae"]),
    }


def fmt(value, digits: int = 3) -> str:
    if value == "":
        return "TBD"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Ours V1 A/B/C/D ablation results.")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--out_md", default="docs/ours_abcd_summary.md")
    parser.add_argument("--results_md", default="results/OURS_ABCD_SUMMARY.md")
    parser.add_argument("--out_csv", default="results/ours_abcd_summary.csv")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    rows = [load_row(results_root, ablation) for ablation in "ABCD"]
    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    lines = [
        "# Ours V1 A/B/C/D Ablation Summary",
        "",
        "| Model | Ablation | Feature set | Status | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['ablation']} | {row['feature_set']} | {row['status']} | "
            f"{fmt(row['folds'], 0)} | {fmt(row['n'], 0)} | {fmt(row['params'], 0)} | "
            f"{fmt(row['inference_ms_per_sample'])} | {fmt(row['mae'])} | "
            f"{fmt(row['rmse'])} | {fmt(row['medae'])} |"
        )

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    out_md.write_text(text, encoding="utf-8")
    if args.results_md:
        results_md = Path(args.results_md)
        results_md.parent.mkdir(parents=True, exist_ok=True)
        results_md.write_text(text, encoding="utf-8")
        print(f"[INFO] Wrote {out_md}, {results_md}, and {out_csv}")
    else:
        print(f"[INFO] Wrote {out_md} and {out_csv}")


if __name__ == "__main__":
    main()
