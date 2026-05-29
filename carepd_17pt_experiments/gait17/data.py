from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

from .constants import DEFAULT_GAIT_KEYS
from .features import apply_ablation, build_hybrid_features, com_normalize, pad_or_clip, summarize_sequence


def scalar_score(value) -> float:
    if isinstance(value, list):
        return float(sum(value))
    if value is None:
        return 0.0
    return float(value)


def load_cnuh_json_labels(label_dir: Path, target: str) -> dict[str, float]:
    labels: dict[str, float] = {}
    for path in sorted(label_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for patient in data.get("patient", []):
            patient_id = patient.get("id")
            items_list = patient.get("mds_updrs_part3", {}).get("itmes", [])
            if not patient_id or not items_list:
                continue
            items = items_list[0]
            if target == "gait":
                labels[patient_id] = sum(scalar_score(items.get(k)) for k in DEFAULT_GAIT_KEYS)
            elif target == "item10":
                labels[patient_id] = scalar_score(items.get("10"))
            else:
                raise ValueError(f"Unsupported target: {target}")
    return labels


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    required = {"dataset", "sample_id", "patient_id", "path", "target"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return df


def load_npz_sequence(manifest_path: Path, row: pd.Series) -> np.ndarray:
    path = Path(row["path"])
    if not path.is_absolute():
        path = manifest_path.parent / path
    with np.load(path) as data:
        joints = data["joints"].astype(np.float32)
    if joints.ndim != 3 or joints.shape[1:] != (17, 3):
        raise ValueError(f"Expected joints shape (T, 17, 3), got {joints.shape} in {path}")
    return joints


def build_model_input(
    joints: np.ndarray,
    max_len: int,
    ablation: str,
    input_kind: str,
    normalize_com: bool = True,
) -> np.ndarray:
    if normalize_com:
        joints = com_normalize(joints)
    if input_kind == "coords":
        return pad_or_clip(joints, max_len).astype(np.float32)
    if input_kind == "hybrid":
        if joints.shape[0] > max_len:
            joints = joints[-max_len:]
        features = apply_ablation(build_hybrid_features(joints), ablation)
        return pad_or_clip(features, max_len).astype(np.float32)
    raise ValueError(f"Unsupported input_kind: {input_kind}")


class Gait17Dataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        indices: np.ndarray,
        max_len: int,
        ablation: str,
        input_kind: str,
        normalize_com: bool = True,
    ):
        self.manifest_path = manifest_path
        self.df = load_manifest(manifest_path).iloc[indices].reset_index(drop=True)
        self.max_len = max_len
        self.ablation = ablation
        self.input_kind = input_kind
        self.normalize_com = normalize_com

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        joints = load_npz_sequence(self.manifest_path, row)
        x = build_model_input(joints, self.max_len, self.ablation, self.input_kind, self.normalize_com)
        y = np.float32(row["target"])
        return x, y, str(row["sample_id"])


def materialize_arrays(
    manifest_path: Path,
    indices: np.ndarray,
    max_len: int,
    ablation: str,
    input_kind: str,
    normalize_com: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = load_manifest(manifest_path).iloc[indices].reset_index(drop=True)
    xs, ys, ids = [], [], []
    for _, row in df.iterrows():
        joints = load_npz_sequence(manifest_path, row)
        xs.append(build_model_input(joints, max_len, ablation, input_kind, normalize_com))
        ys.append(float(row["target"]))
        ids.append(str(row["sample_id"]))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), ids


def materialize_tabular(
    manifest_path: Path,
    indices: np.ndarray,
    max_len: int,
    ablation: str,
    normalize_com: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x, y, ids = materialize_arrays(manifest_path, indices, max_len, ablation, "hybrid", normalize_com)
    return np.asarray([summarize_sequence(seq) for seq in x], dtype=np.float32), y, ids


def loso_splits(df: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    splits = []
    patient_ids = df["patient_id"].astype(str).to_numpy()
    for patient_id in sorted(df["patient_id"].astype(str).unique()):
        val_idx = np.where(patient_ids == patient_id)[0]
        train_idx = np.where(patient_ids != patient_id)[0]
        if len(train_idx) and len(val_idx):
            splits.append((patient_id, train_idx, val_idx))
    return splits


def group_kfold_splits(df: pd.DataFrame, n_splits: int = 5) -> list[tuple[str, np.ndarray, np.ndarray]]:
    patient_ids = df["patient_id"].astype(str).to_numpy()
    n_groups = len(np.unique(patient_ids))
    if n_splits < 2:
        raise ValueError("--n_splits must be at least 2 for GroupKFold.")
    if n_splits > n_groups:
        raise ValueError(f"--n_splits={n_splits} exceeds number of patient groups ({n_groups}).")
    splitter = GroupKFold(n_splits=n_splits)
    x_placeholder = np.zeros(len(df), dtype=np.float32)
    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        splitter.split(x_placeholder, df["target"].to_numpy(), groups=patient_ids),
        start=1,
    ):
        splits.append((f"groupkfold_{fold_idx:02d}", train_idx, val_idx))
    return splits
