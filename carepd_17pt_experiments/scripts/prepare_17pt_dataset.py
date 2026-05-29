from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from gait17.data import load_cnuh_json_labels
from gait17.features import ensure_h36m17


def sample_id_from_path(path: Path) -> str:
    return path.name.replace("_pose.npy", "").replace(".npy", "")


def patient_id_from_sample(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0] if "_" in sample_id else sample_id


def load_label_table(args) -> dict[str, float]:
    if args.label_csv:
        validate_path_arg("--label_csv", args.label_csv)
        if not Path(args.label_csv).is_file():
            raise SystemExit(f"--label_csv does not exist or is not a file: {args.label_csv}")
        df = pd.read_csv(args.label_csv)
        key_col = "patient_id" if "patient_id" in df.columns else "sample_id"
        return {str(row[key_col]): float(row[args.target_col]) for _, row in df.iterrows()}
    if args.label_json_dir:
        validate_path_arg("--label_json_dir", args.label_json_dir)
        if not Path(args.label_json_dir).is_dir():
            raise SystemExit(f"--label_json_dir does not exist or is not a directory: {args.label_json_dir}")
        return load_cnuh_json_labels(Path(args.label_json_dir), args.target)
    raise ValueError("Provide --label_csv or --label_json_dir.")


def validate_path_arg(name: str, value: str | None) -> None:
    if value and ("<" in value or ">" in value):
        raise SystemExit(
            f"{name} contains a placeholder path: {value}\n"
            "Replace angle-bracket examples with a real local path before running the command."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert gait pose files to the independent H36M17 .npz format.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--source_format", choices=["mediapipe33", "h36m17"], required=True)
    parser.add_argument("--label_csv", default=None)
    parser.add_argument("--label_json_dir", default=None)
    parser.add_argument("--target", choices=["item10", "gait"], default="item10")
    parser.add_argument("--target_col", default="target")
    parser.add_argument("--pattern", default="*.npy")
    args = parser.parse_args()

    validate_path_arg("--input_dir", args.input_dir)
    if not Path(args.input_dir).is_dir():
        raise SystemExit(f"--input_dir does not exist or is not a directory: {args.input_dir}")

    labels = load_label_table(args)
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    dataset_dir = output_root / args.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for src_path in sorted(input_dir.rglob(args.pattern)):
        raw = np.load(src_path)
        joints = ensure_h36m17(raw, args.source_format)
        sample_id = sample_id_from_path(src_path)
        patient_id = patient_id_from_sample(sample_id)
        target = labels.get(patient_id, labels.get(sample_id))
        if target is None:
            continue
        out_path = dataset_dir / f"{sample_id}.npz"
        np.savez_compressed(out_path, joints=joints.astype(np.float32))
        rows.append(
            {
                "dataset": args.dataset_name,
                "sample_id": sample_id,
                "patient_id": f"{args.dataset_name}:{patient_id}",
                "path": str(out_path.relative_to(output_root)),
                "target": float(target),
                "frames": int(joints.shape[0]),
            }
        )

    if not rows:
        raise SystemExit("No labeled samples were converted.")
    manifest_path = output_root / "manifest.csv"
    new_df = pd.DataFrame(rows)
    if manifest_path.exists():
        old_df = pd.read_csv(manifest_path)
        df = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(["dataset", "sample_id"], keep="last")
    else:
        df = new_df
    df.to_csv(manifest_path, index=False)
    print(f"[INFO] Converted {len(new_df)} samples. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
