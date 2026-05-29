from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd


UPDRS_DATASETS = ("3DGait", "BMCLab", "PD-GaM", "T-SDU-PD")


def iter_walks(pickle_path: Path, dataset_name: str):
    with pickle_path.open("rb") as f:
        data = pickle.load(f)
    for subject_id, walks in data.items():
        for walk_id, rec in walks.items():
            pose = rec.get("pose")
            trans = rec.get("trans")
            yield {
                "dataset": dataset_name,
                "subject_id": str(subject_id),
                "walk_id": str(walk_id),
                "sample_id": f"{dataset_name}_{subject_id}_{walk_id}",
                "UPDRS_GAIT": rec.get("UPDRS_GAIT"),
                "medication": rec.get("medication"),
                "other": rec.get("other"),
                "fps": rec.get("fps"),
                "pose_shape": "x".join(map(str, pose.shape)) if hasattr(pose, "shape") else "",
                "trans_shape": "x".join(map(str, trans.shape)) if hasattr(trans, "shape") else "",
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect CARE-PD SMPL pickles and export label metadata.")
    parser.add_argument("--carepd_root", default="data/raw/CARE-PD")
    parser.add_argument("--datasets", nargs="+", default=list(UPDRS_DATASETS))
    parser.add_argument("--out_csv", default="data/raw/CARE-PD/carepd_updrs_walk_index.csv")
    args = parser.parse_args()

    rows = []
    root = Path(args.carepd_root)
    for dataset_name in args.datasets:
        path = root / f"{dataset_name}.pkl"
        if not path.exists():
            print(f"[WARN] Missing {path}")
            continue
        rows.extend(iter_walks(path, dataset_name))

    if not rows:
        raise SystemExit("No CARE-PD rows found.")
    df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {len(df)} rows to {out_path}")
    print(df.groupby(["dataset", "UPDRS_GAIT"]).size().rename("n").reset_index())


if __name__ == "__main__":
    main()
