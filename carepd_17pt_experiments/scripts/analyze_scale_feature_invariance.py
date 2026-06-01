from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from gait17.data import build_model_input, load_manifest, load_npz_sequence


GROUPS = {
    "position": slice(0, 3),
    "velocity": slice(3, 6),
    "amplitude": slice(6, 7),
    "variability": slice(7, 8),
    "angle": slice(8, 9),
}


def relative_l1(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.mean(np.abs(a)) + 1e-6
    return float(np.mean(np.abs(b - a)) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure feature-level scale sensitivity without training.")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--out_dir", default="docs/scale_feature_invariance")
    parser.add_argument("--max_len", type=int, default=390)
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--scale_values", nargs="+", type=float, default=[0.70, 0.85, 0.90, 0.95, 1.05, 1.10, 1.15, 1.30])
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_manifest(manifest_path)
    if args.sample_limit is not None:
        df = df.head(args.sample_limit).copy()

    rows = []
    for scale_norm in ["none", "median_bone", "torso", "hip_width"]:
        for scale in args.scale_values:
            group_values = {name: [] for name in GROUPS}
            for _, row in df.iterrows():
                joints = load_npz_sequence(manifest_path, row)
                base = build_model_input(
                    joints,
                    args.max_len,
                    "D",
                    "hybrid",
                    normalize_com=True,
                    scale_normalization=scale_norm,
                )
                perturbed = build_model_input(
                    (joints * np.float32(scale)).astype(np.float32),
                    args.max_len,
                    "D",
                    "hybrid",
                    normalize_com=True,
                    scale_normalization=scale_norm,
                )
                for name, slc in GROUPS.items():
                    group_values[name].append(relative_l1(base[..., slc], perturbed[..., slc]))
            for name, values in group_values.items():
                rows.append(
                    {
                        "scale_normalization": scale_norm,
                        "scale": float(scale),
                        "feature_group": name,
                        "mean_relative_l1_change": float(np.mean(values)),
                        "median_relative_l1_change": float(np.median(values)),
                    }
                )

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "feature_scale_sensitivity.csv", index=False)
    pivot = out.pivot_table(
        index=["scale_normalization", "scale"],
        columns="feature_group",
        values="mean_relative_l1_change",
        aggfunc="mean",
    ).reset_index()
    pivot.to_csv(out_dir / "feature_scale_sensitivity_pivot.csv", index=False)

    lines = [
        "# Scale Feature Invariance Diagnostic",
        "",
        "This diagnostic measures how much each feature group changes when raw coordinates are multiplied by a scale factor before feature construction.",
        "Lower values mean stronger feature-level scale invariance.",
        "",
        "## Interpretation",
        "",
        "- Angles should remain scale-invariant by construction.",
        "- COM-only coordinates remain scale-sensitive because COM centering removes translation, not scale.",
        "- Body-scale normalization should reduce scale sensitivity of position, velocity, amplitude, and variability channels.",
        "",
        "## Outputs",
        "",
        "- `feature_scale_sensitivity.csv`",
        "- `feature_scale_sensitivity_pivot.csv`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(out.to_dict(orient="records"), indent=2), encoding="utf-8")
    print(f"[INFO] Wrote feature scale-invariance diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
