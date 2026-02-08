"""
통합 아블레이션 실행 스크립트
- 전처리 아블레이션(A/B/C/D) + 모델별 실행을 한 번에 수행
- 포함 대상:
  * main TF 모델 (Spatio-Temporal Transformer)
  * fusion TF ST-GCN 모델
  * hybrid Torch GCN 모델
- 실행 후 각 스크립트가 생성한 summary와 compare_models 요약을 남깁니다.
"""

import argparse
import os
import shutil
from glob import glob
from datetime import datetime

from .run_main_ablation import run_ablation as run_main_ablation
from .run_fusion_ablation import run_ablation as run_fusion_ablation
from .run_hybrid_ablation import run_ablation as run_hybrid_ablation
from .compare_models import main as compare_main


def latest_subdir(parent):
    dirs = [d for d in glob(os.path.join(parent, "*")) if os.path.isdir(d)]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="Run preprocessing+model ablation across all models.")
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for all models (main/fusion/hybrid).")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=13.0)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    print("[STEP] main TF ablation (A-D)")
    run_main_ablation(args.processed_dir, args.label_dir, epochs=args.epochs, batch_size=args.batch_size)

    print("[STEP] fusion TF ablation (A-D)")
    run_fusion_ablation(
        args.processed_dir,
        args.label_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        folds=args.folds,
    )

    print("[STEP] hybrid Torch ablation (A-D)")
    run_hybrid_ablation(
        args.processed_dir,
        args.label_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        folds=args.folds,
        seconds=args.seconds,
        fps=args.fps,
    )

    print("[STEP] Collecting model comparison summary")
    # Pass empty argv so it doesn't parse this script's args
    compare_main([])

    print("[DONE] All ablations executed.")


if __name__ == "__main__":
    main()
