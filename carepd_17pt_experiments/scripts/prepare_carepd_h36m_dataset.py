from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from gait17.features import pad_or_clip


UPDRS_DATASETS = ("3DGait", "BMCLab", "PD-GaM", "T-SDU-PD")


def validate_path_arg(name: str, value: str | None) -> None:
    if value and ("<" in value or ">" in value):
        raise SystemExit(
            f"{name} contains a placeholder path: {value}\n"
            "Replace angle-bracket examples with a real local path before running the command."
        )


def load_label_map(carepd_root: Path, datasets: list[str]) -> dict[str, dict[str, object]]:
    labels = {}
    for dataset_name in datasets:
        path = carepd_root / f"{dataset_name}.pkl"
        if not path.exists():
            continue
        with path.open("rb") as f:
            data = pickle.load(f)
        for subject_id, walks in data.items():
            for walk_id, rec in walks.items():
                key_variants = {
                    str(walk_id),
                    f"{subject_id}_{walk_id}",
                    f"{dataset_name}_{subject_id}_{walk_id}",
                    f"{dataset_name}:{subject_id}:{walk_id}",
                }
                for key in key_variants:
                    labels[key] = {
                        "dataset": dataset_name,
                        "subject_id": str(subject_id),
                        "walk_id": str(walk_id),
                        "target": rec.get("UPDRS_GAIT"),
                        "medication": rec.get("medication"),
                        "fps": rec.get("fps"),
                    }
    return labels


def choose_array_key(npz, requested: str | None) -> str:
    if requested:
        if requested not in npz:
            raise KeyError(f"Requested --array_key {requested!r} not found. Keys: {list(npz.keys())}")
        return requested
    candidates = []
    for key in npz.keys():
        arr = npz[key]
        if hasattr(arr, "shape") and len(arr.shape) in {3, 4} and arr.shape[-2:] == (17, 3):
            candidates.append(key)
    if len(candidates) != 1:
        raise KeyError(f"Cannot infer H36M array key. Candidates={candidates}, all keys={list(npz.keys())}")
    return candidates[0]


def as_str_list(values) -> list[str]:
    arr = np.asarray(values)
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in arr.reshape(-1).tolist()]


def infer_ids(npz, n_samples: int, id_key: str | None, fallback_prefix: str) -> list[str]:
    if id_key:
        if id_key not in npz:
            raise KeyError(f"Requested --id_key {id_key!r} not found. Keys: {list(npz.keys())}")
        ids = as_str_list(npz[id_key])
        if len(ids) != n_samples:
            raise ValueError(f"id_key length {len(ids)} does not match n_samples {n_samples}")
        return ids
    for key in ("sample_id", "sample_ids", "walk_id", "walk_ids", "names", "ids"):
        if key in npz:
            ids = as_str_list(npz[key])
            if len(ids) == n_samples:
                return ids
    if n_samples == 1:
        return [fallback_prefix]
    return [f"{fallback_prefix}_{idx:05d}" for idx in range(n_samples)]


def normalize_h36m_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1:] == (17, 3):
        return arr[None, ...]
    if arr.ndim == 4 and arr.shape[-2:] == (17, 3):
        return arr
    raise ValueError(f"Expected (T,17,3) or (N,T,17,3), got {arr.shape}")


def load_h36m_sequences(npz_path: Path, array_key: str | None, id_key: str | None) -> list[tuple[str, np.ndarray]]:
    with np.load(npz_path, allow_pickle=True) as npz:
        if array_key:
            key = choose_array_key(npz, array_key)
            seqs = normalize_h36m_array(npz[key])
            ids = infer_ids(npz, seqs.shape[0], id_key, npz_path.stem)
            return [(sample_id, seq.astype(np.float32)) for sample_id, seq in zip(ids, seqs)]

        sample_keys = []
        for key in npz.keys():
            arr = npz[key]
            if hasattr(arr, "shape") and arr.ndim == 3 and arr.shape[1:] == (17, 3):
                sample_keys.append(key)
        if sample_keys:
            return [(key, np.asarray(npz[key], dtype=np.float32)) for key in sample_keys]

        key = choose_array_key(npz, None)
        seqs = normalize_h36m_array(npz[key])
        ids = infer_ids(npz, seqs.shape[0], id_key, npz_path.stem)
        return [(sample_id, seq.astype(np.float32)) for sample_id, seq in zip(ids, seqs)]


def find_label(sample_id: str, label_map: dict[str, dict[str, object]]):
    normalized = re.sub(r"_down\d+$", "", sample_id)
    candidates = {sample_id, normalized, normalized.replace("__", "_")}
    if "__" in normalized:
        subject_id, walk_id = normalized.split("__", 1)
        candidates.update({walk_id, f"{subject_id}_{walk_id}", f"{subject_id}__{walk_id}"})
    for candidate in candidates:
        if candidate in label_map:
            return label_map[candidate]
    if sample_id in label_map:
        return label_map[sample_id]
    for key, value in sorted(label_map.items(), key=lambda item: len(item[0]), reverse=True):
        if key in sample_id or sample_id in key:
            return value
    return None


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def convert_npz_file(
    npz_path: Path,
    output_root: Path,
    label_map: dict[str, dict[str, object]],
    array_key: str | None,
    id_key: str | None,
    max_len: int | None,
) -> list[dict[str, object]]:
    rows = []
    sequences = load_h36m_sequences(npz_path, array_key, id_key)
    for sample_id, joints in sequences:
        label = find_label(sample_id, label_map)
        if label is None or label["target"] is None:
            continue
        dataset_name = str(label["dataset"])
        clean_id = f"CAREPD_{dataset_name}_{safe_id(sample_id)}"
        out_dir = output_root / "CAREPD"
        out_dir.mkdir(parents=True, exist_ok=True)
        if max_len:
            joints = pad_or_clip(joints, max_len)
        out_path = out_dir / f"{clean_id}.npz"
        np.savez_compressed(out_path, joints=joints.astype(np.float32))
        rows.append(
            {
                "dataset": "CAREPD",
                "sample_id": clean_id,
                "patient_id": f"CAREPD:{dataset_name}:{label['subject_id']}",
                "path": str(out_path.relative_to(output_root)),
                "target": float(label["target"]),
                "frames": int(joints.shape[0]),
                "source_npz": str(npz_path),
                "source_sample_id": sample_id,
                "source_dataset": dataset_name,
                "source_walk_id": label["walk_id"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert official CARE-PD h36m preprocessed npz files to this project's H36M17 manifest format."
    )
    parser.add_argument("--carepd_root", default="data/raw/CARE-PD")
    parser.add_argument("--h36m_root", required=True, help="Official CARE-PD h36m folder or npz file.")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--datasets", nargs="+", default=list(UPDRS_DATASETS))
    parser.add_argument(
        "--filename_glob",
        default="h36m_3d_world_floorXZZplus_30f_or_longer*.npz",
        help="Which H36M files to read when --h36m_root is a directory.",
    )
    parser.add_argument("--array_key", default=None, help="Optional npz key for the H36M array.")
    parser.add_argument("--id_key", default=None, help="Optional npz key for walk/sample ids.")
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--replace_dataset", action="store_true", help="Replace existing CAREPD rows in manifest.csv.")
    parser.add_argument("--inspect_only", action="store_true")
    args = parser.parse_args()

    validate_path_arg("--carepd_root", args.carepd_root)
    validate_path_arg("--h36m_root", args.h36m_root)

    h36m_root = Path(args.h36m_root)
    npz_files = [h36m_root] if h36m_root.is_file() else sorted(h36m_root.rglob(args.filename_glob))
    if not npz_files:
        raise SystemExit(
            f"No .npz files found under {h36m_root}\n"
            "This converter expects the official CARE-PD h36m preprocessed files, not the SMPL .pkl files.\n"
            "Download h36m_preprocessed from the CARE-PD Dataverse release and place/rename it as "
            "data\\raw\\CARE-PD\\h36m, or generate it with the official TaatiTeam/CARE-PD "
            "scripts\\preprocess_smpl2h36m.sh."
        )

    if args.inspect_only:
        for path in npz_files:
            with np.load(path, allow_pickle=True) as npz:
                print(f"\n=== {path} ===")
                for key in npz.keys():
                    arr = npz[key]
                    print(f"{key}: shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}")
        return

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    label_map = load_label_map(Path(args.carepd_root), args.datasets)
    rows = []
    for path in npz_files:
        try:
            rows.extend(convert_npz_file(path, output_root, label_map, args.array_key, args.id_key, args.max_len))
        except Exception as exc:
            print(f"[WARN] Skipping {path}: {type(exc).__name__}: {exc}")

    if not rows:
        raise SystemExit("No labeled CARE-PD H36M samples were converted. Run with --inspect_only and set --array_key/--id_key.")
    manifest_path = output_root / "manifest.csv"
    new_df = pd.DataFrame(rows)
    if manifest_path.exists():
        old_df = pd.read_csv(manifest_path)
        if args.replace_dataset:
            old_df = old_df[old_df["dataset"] != "CAREPD"]
        df = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(["dataset", "sample_id"], keep="last")
    else:
        df = new_df
    df.to_csv(manifest_path, index=False)
    print(f"[INFO] Converted {len(new_df)} CARE-PD samples. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
