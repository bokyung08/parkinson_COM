from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def candidate_decision(row: dict, baseline_mae: float | None) -> str:
    max_delta = row["max_scale_delta_mae_pct"]
    mean_delta = row["mean_scale_delta_mae_pct"]
    original_mae = row["original_mae"]
    if baseline_mae is not None and original_mae > baseline_mae * 1.15:
        return "Exclude: baseline accuracy loss"
    if max_delta <= 10.0 and mean_delta <= 7.5:
        return "Strong candidate"
    if max_delta <= 15.0 and mean_delta <= 10.0:
        return "Promising"
    if max_delta <= 25.0:
        return "Supplementary only"
    return "Exclude for scale claim"


def summarize_one(path: Path) -> dict | None:
    summary_path = path / "summary.csv"
    if not summary_path.exists():
        return None
    df = pd.read_csv(summary_path)
    if "normalization" not in df.columns or "type" not in df.columns:
        return None
    com = df[df["normalization"] == "COM"].copy()
    if com.empty:
        return None
    original = com[com["type"] == "original"].iloc[0]
    scale = com[com["type"] == "scale"].copy()
    translation = com[com["type"] == "translation"].copy()
    moderate = scale[scale["scale"].between(0.85, 1.15)]
    realistic = scale[scale["scale"].between(0.90, 1.10)]
    return {
        "candidate": path.name,
        "scale_normalization": str(original.get("scale_normalization", "unknown")),
        "original_mae": float(original["mae"]),
        "original_rmse": float(original["rmse"]),
        "mean_scale_delta_mae_pct": float(scale["delta_mae_pct"].mean()) if not scale.empty else np.nan,
        "max_scale_delta_mae_pct": float(scale["delta_mae_pct"].max()) if not scale.empty else np.nan,
        "mean_scale_delta_rmse_pct": float(scale["delta_rmse_pct"].mean()) if not scale.empty else np.nan,
        "max_scale_delta_rmse_pct": float(scale["delta_rmse_pct"].max()) if not scale.empty else np.nan,
        "mean_moderate_scale_delta_mae_pct": float(moderate["delta_mae_pct"].mean()) if not moderate.empty else np.nan,
        "max_moderate_scale_delta_mae_pct": float(moderate["delta_mae_pct"].max()) if not moderate.empty else np.nan,
        "mean_realistic_scale_delta_mae_pct": float(realistic["delta_mae_pct"].mean()) if not realistic.empty else np.nan,
        "max_realistic_scale_delta_mae_pct": float(realistic["delta_mae_pct"].max()) if not realistic.empty else np.nan,
        "mean_abs_translation_delta_mae_pct": float(translation["delta_mae_pct"].abs().mean()) if not translation.empty else np.nan,
        "max_abs_translation_delta_mae_pct": float(translation["delta_mae_pct"].abs().max()) if not translation.empty else np.nan,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize scale-robustness candidate runs.")
    parser.add_argument("--root", default="docs/scale_robustness_screen")
    parser.add_argument("--out", default="docs/scale_robustness_candidate_summary.md")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    if root.exists():
        for path in sorted(p for p in root.iterdir() if p.is_dir()):
            row = summarize_one(path)
            if row is not None:
                rows.append(row)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_path.with_suffix(".csv")
    if not rows:
        out_path.write_text("# Scale Robustness Candidate Summary\n\nNo completed candidate outputs found.\n", encoding="utf-8")
        pd.DataFrame().to_csv(csv_path, index=False)
        return

    df = pd.DataFrame(rows)
    baseline = df[df["candidate"].str.contains("baseline", case=False, na=False)]
    baseline_mae = float(baseline["original_mae"].iloc[0]) if not baseline.empty else None
    df["decision"] = [candidate_decision(row, baseline_mae) for row in df.to_dict(orient="records")]
    df = df.sort_values(["decision", "max_scale_delta_mae_pct", "original_mae"])
    df.to_csv(csv_path, index=False)

    display_cols = [
        "candidate",
        "scale_normalization",
        "original_mae",
        "original_rmse",
        "mean_scale_delta_mae_pct",
        "max_scale_delta_mae_pct",
        "mean_realistic_scale_delta_mae_pct",
        "max_realistic_scale_delta_mae_pct",
        "mean_abs_translation_delta_mae_pct",
        "decision",
    ]
    lines = [
        "# Scale Robustness Candidate Summary",
        "",
        "Candidate screening compares COM-normalized inference under scale and translation perturbations.",
        "Use this screen to decide which candidate deserves a full 5-fold manuscript run.",
        "",
        *markdown_table(df[display_cols], set(display_cols) - {"candidate", "scale_normalization", "decision"}),
        "",
        "## Reporting Rule",
        "",
        "- Use `Strong candidate` or `Promising` only after confirming the full 5-fold run.",
        "- Do not claim scale robustness from a candidate that has large baseline MAE loss.",
        "- Translation robustness should remain close to zero degradation for all COM-centered candidates.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
