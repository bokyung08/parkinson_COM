import argparse
import json
import os
from glob import glob
import pandas as pd


def latest_subdir(parent):
    dirs = [d for d in glob(os.path.join(parent, "*")) if os.path.isdir(d)]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def load_ablation_rows(base_dir, model_key):
    latest = latest_subdir(base_dir)
    if not latest:
        return []
    summary_path = os.path.join(latest, "ablation_summary.json")
    if not os.path.exists(summary_path):
        return []
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    rows = []
    metrics_keys = ["mae", "rmse", "pearson", "kendall", "ccc", "r2", "explained_variance", "mape", "medae"]
    for entry in data:
        row = {
            "model": model_key,
            "run_id": os.path.basename(latest),
            "ablation": entry.get("ablation"),
        }
        for k in metrics_keys:
            row[k] = entry.get(k) or entry.get(f"overall_{k}") or entry.get(f"reg_{k}")
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect main/fusion/hybrid ablation (A-D) into one summary file.")
    parser.add_argument("--out_csv", type=str, default="results/model_comparison_summary.csv")
    parser.add_argument("--out_json", type=str, default="results/model_comparison_summary.json")
    args = parser.parse_args(argv)

    rows = []
    rows.extend(load_ablation_rows(os.path.join("results", "main_ablation"), "main_tf"))
    rows.extend(load_ablation_rows(os.path.join("results", "fusion_tf_ablation"), "fusion_tf"))
    rows.extend(load_ablation_rows(os.path.join("results", "hybrid_ablation"), "hybrid_torch"))

    if not rows:
        print("No ablation summaries found. Run the ablation scripts first.")
        return

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved comparison to {args.out_csv} and {args.out_json}")


if __name__ == "__main__":
    main()
