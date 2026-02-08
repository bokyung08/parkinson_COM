"""
아블레이션/모델 결과 요약 및 간단 시각화 스크립트
- 이미 실행된 결과 폴더를 스캔하여 순위/표/플롯을 생성
- 새 학습은 수행하지 않음
"""

import argparse
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd


def load_metric(path, keys):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: data.get(k) for k in keys}
    except Exception:
        return None


def gather_rows():
    rows = []
    # main ablation
    for mjson in glob("results/main_ablation/*/ablation_summary.json"):
        run_id = mjson.split(os.sep)[-2]
        with open(mjson, "r", encoding="utf-8") as f:
            arr = json.load(f)
        for rec in arr:
            rows.append({"model": "main_tf", "run_id": run_id, "ablation": rec.get("ablation"), "mae": rec.get("mae"), "rmse": rec.get("rmse"), "spearman": rec.get("spearman", None)})
    # fusion TF ablation
    for mjson in glob("results/fusion_tf_ablation/*/ablation_summary.json"):
        run_id = mjson.split(os.sep)[-2]
        with open(mjson, "r", encoding="utf-8") as f:
            arr = json.load(f)
        for rec in arr:
            for fold in rec.get("folds", []):
                rows.append({"model": "fusion_tf", "run_id": run_id, "ablation": rec.get("ablation"), "fold": fold.get("fold"), "mae": fold.get("mae"), "rmse": fold.get("rmse"), "spearman": fold.get("spearman")})
    # hybrid torch ablation
    for mjson in glob("results/hybrid_ablation/*/ablation_summary.json"):
        run_id = mjson.split(os.sep)[-2]
        with open(mjson, "r", encoding="utf-8") as f:
            arr = json.load(f)
        for rec in arr:
            rows.append({"model": "hybrid_torch", "run_id": run_id, "ablation": rec.get("ablation"), "mae": rec.get("overall_mae"), "rmse": rec.get("overall_rmse"), "spearman": rec.get("overall_spearman")})
    return rows


def plot_rank(df, out_dir, metric="mae"):
    os.makedirs(out_dir, exist_ok=True)
    df_sorted = df.sort_values(by=metric)
    plt.figure(figsize=(8, 4))
    plt.barh(range(len(df_sorted)), df_sorted[metric], tick_label=df_sorted["label"])
    plt.xlabel(metric.upper())
    plt.title(f"{metric.upper()} ranking (lower is better)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"rank_{metric}.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Summarize ablation/model results and visualize rankings.")
    parser.add_argument("--out_dir", type=str, default="results/ablation_report")
    args = parser.parse_args()

    rows = gather_rows()
    if not rows:
        print("No ablation results found.")
        return

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "all_results.csv"), index=False)

    # 간단 랭킹: ablation+model별 전체 평균 MAE 기준
    rank_df = df.groupby(["model", "ablation"]).agg({"mae": "mean", "rmse": "mean"}).reset_index()
    rank_df["label"] = rank_df["model"] + "_abl" + rank_df["ablation"]
    rank_df.to_csv(os.path.join(args.out_dir, "rank_summary.csv"), index=False)

    plot_rank(rank_df, args.out_dir, metric="mae")
    plot_rank(rank_df, args.out_dir, metric="rmse")

    print(f"[INFO] Report saved to {args.out_dir}")


if __name__ == "__main__":
    main()
