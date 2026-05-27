"""Collect unified DL baseline rows into one paper-ready summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def infer_from_out_dir(out_dir: Path) -> tuple[str | None, str | None]:
    parts = out_dir.name.split("_")
    ablation = parts[0] if parts and parts[0] in {"A", "B", "C", "D"} else None
    target = parts[1] if len(parts) >= 2 else None
    return target, ablation


def split_label(test_size: float) -> str:
    train_pct = round((1.0 - test_size) * 100)
    test_pct = round(test_size * 100)
    return f"{train_pct}/{test_pct}"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def value_at(values: object, index: int) -> object:
    if not isinstance(values, list) or len(values) <= index:
        return None
    return values[index]


def final_epoch_row(model_name: str, model_dir: Path) -> dict[str, object] | None:
    history_path = model_dir / "history.json"
    if not history_path.exists():
        return None
    history = load_json(history_path)
    loss_values = history.get("loss", [])
    if not loss_values:
        return None
    epoch_index = len(loss_values) - 1
    learning_rates = history.get("learning_rate") or history.get("lr") or []
    return {
        "model": model_name,
        "final_epoch": len(loss_values),
        "loss": value_at(loss_values, epoch_index),
        "mae": value_at(history.get("mae"), epoch_index),
        "val_loss": value_at(history.get("val_loss"), epoch_index),
        "val_mae": value_at(history.get("val_mae"), epoch_index),
        "learning_rate": value_at(learning_rates, epoch_index),
    }


def metric_text(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def seconds_text(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}"


def learning_rate_text(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4e}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="Rebuttal/results/unified_dl_baselines/D_item10_90_10")
    parser.add_argument("--target", choices=["item10", "gait"], default=None)
    parser.add_argument("--ablation", choices=["A", "B", "C", "D"], default=None)
    parser.add_argument("--max_len", type=int, default=390)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    inferred_target, inferred_ablation = infer_from_out_dir(out_dir)
    target = args.target or inferred_target or "item10"
    ablation = args.ablation or inferred_ablation or "D"
    rows = []
    final_rows = []
    for model_dir, model_name in [
        (out_dir / "fusion_tf", "FusionTF"),
        (out_dir / "hybrid_torch", "HybridTorch"),
    ]:
        row_path = model_dir / "row.json"
        if row_path.exists():
            row = load_json(row_path)
            rows.append(row)
            final_row = final_epoch_row(str(row.get("model", model_name)), model_dir)
            if final_row is not None:
                final_rows.append(final_row)
    if not rows:
        raise SystemExit(f"No row.json files found under {out_dir}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "unified_dl_baseline_summary.csv", index=False)
    with (out_dir / "unified_dl_baseline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    lines = [
        "# Unified DL Baseline Results",
        "",
        "## Protocol",
        "",
        f"- Target: `{target}`",
        f"- Ablation: `{ablation}`",
        f"- Split: `{split_label(args.test_size)}` hold-out, random_state={args.random_state}",
        f"- Max length: `{args.max_len}` frames, last-frame window",
        f"- Epochs: `{args.epochs}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Learning rate: `{args.learning_rate}`",
        f"- Loss: `mse`",
        "",
        "## Results",
        "",
        "| Model | Params | MAE | RMSE | MedAE | Train sec |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {int(row['params'])} | "
            f"{metric_text(row['mae'])} | {metric_text(row['rmse'])} | "
            f"{metric_text(row['medae'])} | {seconds_text(row['train_seconds'])} |"
        )
    if final_rows:
        lines.extend(
            [
                "",
                "## Final Epoch Console Check",
                "",
                "| Model | Final epoch | Train loss | Train MAE | Val loss | Val MAE | Learning rate |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in final_rows:
            lines.append(
                f"| {row['model']} | {row['final_epoch']}/{args.epochs} | "
                f"{metric_text(row['loss'], 4)} | {metric_text(row['mae'], 4)} | "
                f"{metric_text(row['val_loss'], 4)} | {metric_text(row['val_mae'], 4)} | "
                f"{learning_rate_text(row['learning_rate'])} |"
            )
    lines.extend(
        [
            "",
            "Use this table for the two non-proposed DL baselines trained under the unified protocol. These values replace older baseline runs when discussing DL baseline fairness.",
        ]
    )
    (out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(df)
    print(f"[INFO] Saved summary to {out_dir}")


if __name__ == "__main__":
    main()
