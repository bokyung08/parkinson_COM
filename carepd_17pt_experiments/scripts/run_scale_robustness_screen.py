from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


VARIANTS = [
    {"name": "baseline_none", "scale_norm": "none", "aug_min": "1.00", "aug_max": "1.00"},
    {"name": "median_bone", "scale_norm": "median_bone", "aug_min": "1.00", "aug_max": "1.00"},
    {"name": "torso", "scale_norm": "torso", "aug_min": "1.00", "aug_max": "1.00"},
    {"name": "hip_width", "scale_norm": "hip_width", "aug_min": "1.00", "aug_max": "1.00"},
    {"name": "scale_aug_moderate", "scale_norm": "none", "aug_min": "0.85", "aug_max": "1.15"},
    {"name": "scale_aug_wide", "scale_norm": "none", "aug_min": "0.70", "aug_max": "1.30"},
    {"name": "median_bone_aug_moderate", "scale_norm": "median_bone", "aug_min": "0.85", "aug_max": "1.15"},
]


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    for variant in VARIANTS:
        name = variant["name"]
        result_dir = Path("results") / f"screen_scale_robustness_{name}"
        robust_dir = Path("docs") / "scale_robustness_screen" / name

        if (ROOT / result_dir / "summary.csv").exists():
            print(f"[INFO] Skipping training for {name}; summary.csv exists.", flush=True)
        else:
            print(f"[INFO] Training screen candidate {name}", flush=True)
            run(
                [
                    str(PYTHON),
                    "scripts/run_loso_experiments.py",
                    "--manifest",
                    "data/processed/manifest.csv",
                    "--out_dir",
                    str(result_dir),
                    "--models",
                    "ours",
                    "--split_strategy",
                    "groupkfold",
                    "--n_splits",
                    "5",
                    "--fold_limit",
                    "2",
                    "--ablation",
                    "D",
                    "--target",
                    "item10",
                    "--epochs",
                    "30",
                    "--batch_size",
                    "8",
                    "--device",
                    "cuda",
                    "--scale_normalization",
                    variant["scale_norm"],
                    "--scale_aug_min",
                    variant["aug_min"],
                    "--scale_aug_max",
                    variant["aug_max"],
                    "--save_checkpoints",
                ]
            )

        if (ROOT / robust_dir / "summary.csv").exists():
            print(f"[INFO] Skipping robustness for {name}; summary.csv exists.", flush=True)
        else:
            print(f"[INFO] Evaluating robustness for {name}", flush=True)
            run(
                [
                    str(PYTHON),
                    "scripts/run_com_robustness.py",
                    "--manifest",
                    "data/processed/manifest.csv",
                    "--checkpoint_dir",
                    str(result_dir / "checkpoints"),
                    "--out_dir",
                    str(robust_dir),
                    "--n_splits",
                    "5",
                    "--fold_limit",
                    "2",
                    "--batch_size",
                    "32",
                    "--device",
                    "cuda",
                    "--scale_values",
                    "0.70",
                    "0.85",
                    "0.90",
                    "0.95",
                    "1.00",
                    "1.05",
                    "1.10",
                    "1.15",
                    "1.30",
                    "--translation_values",
                    "-0.20",
                    "-0.10",
                    "0.00",
                    "0.10",
                    "0.20",
                    "--combined_values",
                    "0.70,-0.20",
                    "1.30,0.20",
                    "0.85,0.10",
                    "1.15,-0.10",
                ]
            )

    run(
        [
            str(PYTHON),
            "scripts/summarize_scale_robustness_candidates.py",
            "--root",
            "docs/scale_robustness_screen",
            "--out",
            "docs/scale_robustness_candidate_summary.md",
        ]
    )
    run(
        [
            str(PYTHON),
            "scripts/analyze_scale_feature_invariance.py",
            "--manifest",
            "data/processed/manifest.csv",
            "--out_dir",
            "docs/scale_feature_invariance",
            "--sample_limit",
            "1000",
        ]
    )


if __name__ == "__main__":
    main()
