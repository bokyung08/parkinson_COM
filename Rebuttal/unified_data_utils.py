"""Lightweight data utilities for unified rebuttal experiments.

This module intentionally avoids sklearn/scipy/tensorflow/torch imports so the
DL baseline scripts can run in low-memory Windows environments.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


NUM_NODES = 33
DEFAULT_GAIT_KEYS = ("10", "11", "12", "13", "14")
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28


def scalar_score(value) -> float:
    if isinstance(value, list):
        return float(sum(value))
    if value is None:
        return 0.0
    return float(value)


def load_labels(label_dir: Path, target: str) -> dict[str, float]:
    labels: dict[str, float] = {}
    for path in sorted(label_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for patient in data.get("patient", []):
            pid = patient.get("id")
            items_list = patient.get("mds_updrs_part3", {}).get("itmes", [])
            if not pid or not items_list:
                continue
            items = items_list[0]
            if target == "gait":
                labels[pid] = sum(scalar_score(items.get(k)) for k in DEFAULT_GAIT_KEYS)
            else:
                labels[pid] = scalar_score(items.get("10"))
    return labels


def patient_id_from_pose_file(path: Path) -> str:
    stem = path.name.replace("_pose.npy", "")
    return stem.rsplit("_", 1)[0]


def ensure_pose_tensor(raw: np.ndarray) -> np.ndarray:
    if raw.ndim == 3:
        return raw[..., :3]
    if raw.ndim == 2 and raw.shape[1] % NUM_NODES == 0:
        channels = raw.shape[1] // NUM_NODES
        return raw.reshape(raw.shape[0], NUM_NODES, channels)[..., :3]
    raise ValueError(f"Unsupported pose shape: {raw.shape}")


def compute_bone_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-6
    cos_angle = np.sum(ba * bc, axis=-1) / denom
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def build_hybrid_node_features(joints: np.ndarray) -> np.ndarray:
    time_steps, joints_count, _ = joints.shape
    velocity = np.diff(joints, axis=0, prepend=joints[:1])
    norms = np.linalg.norm(joints, axis=2)
    amplitude = norms.max(axis=0) - norms.min(axis=0)
    variability = norms.std(axis=0)
    amp_broadcast = np.broadcast_to(amplitude[None, :, None], (time_steps, joints_count, 1))
    var_broadcast = np.broadcast_to(variability[None, :, None], (time_steps, joints_count, 1))

    angles = np.zeros((time_steps, joints_count), dtype=np.float32)
    if joints_count > MP_LEFT_ANKLE:
        angles[:, MP_LEFT_KNEE] = compute_bone_angle(
            joints[:, MP_LEFT_HIP], joints[:, MP_LEFT_KNEE], joints[:, MP_LEFT_ANKLE]
        )
    if joints_count > MP_RIGHT_ANKLE:
        angles[:, MP_RIGHT_KNEE] = compute_bone_angle(
            joints[:, MP_RIGHT_HIP], joints[:, MP_RIGHT_KNEE], joints[:, MP_RIGHT_ANKLE]
        )
    if joints_count > MP_LEFT_WRIST:
        angles[:, MP_LEFT_ELBOW] = compute_bone_angle(
            joints[:, MP_LEFT_SHOULDER], joints[:, MP_LEFT_ELBOW], joints[:, MP_LEFT_WRIST]
        )
    if joints_count > MP_RIGHT_WRIST:
        angles[:, MP_RIGHT_ELBOW] = compute_bone_angle(
            joints[:, MP_RIGHT_SHOULDER], joints[:, MP_RIGHT_ELBOW], joints[:, MP_RIGHT_WRIST]
        )

    return np.concatenate(
        [joints, velocity, amp_broadcast, var_broadcast, angles[..., None]],
        axis=-1,
    ).astype(np.float32)


def apply_ablation(node_features: np.ndarray, mode: str) -> np.ndarray:
    if mode == "A":
        return node_features[..., :3]
    if mode == "B":
        return node_features[..., :6]
    if mode == "C":
        return node_features[..., :8]
    return node_features


def pad_features(features: np.ndarray, max_len: int) -> np.ndarray:
    if features.shape[0] > max_len:
        return features[-max_len:]
    if features.shape[0] < max_len:
        pad_shape = (max_len - features.shape[0],) + features.shape[1:]
        return np.concatenate([features, np.zeros(pad_shape, dtype=features.dtype)], axis=0)
    return features


def load_unified_dataset(
    processed_dir: Path,
    label_dir: Path,
    target: str,
    ablation: str,
    max_len: int,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    labels = load_labels(label_dir, target=target)
    x_list = []
    y_list = []
    ids = []
    manifest_rows = []

    for pose_path in sorted(processed_dir.rglob("*_2_pose.npy")):
        pid = patient_id_from_pose_file(pose_path)
        if pid not in labels:
            continue
        raw = np.load(pose_path)
        coords = ensure_pose_tensor(raw).astype(np.float32)
        if coords.shape[0] > max_len:
            coords = coords[-max_len:]
        features = build_hybrid_node_features(coords)
        features = apply_ablation(features, ablation)
        features = pad_features(features, max_len)
        sample_id = pose_path.name.replace("_pose.npy", "")
        x_list.append(features)
        y_list.append(labels[pid])
        ids.append(sample_id)
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "patient_id": pid,
                "target": labels[pid],
                "raw_frames": int(raw.shape[0]),
                "used_shape": "x".join(map(str, features.shape)),
            }
        )

    if not x_list:
        raise ValueError("No samples found. Check processed_dir and label_dir.")
    return (
        np.asarray(x_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
        ids,
        pd.DataFrame(manifest_rows),
    )


def deterministic_split(n_samples: int, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_test = int(math.ceil(n_samples * test_size))
    test_idx = np.sort(indices[:n_test])
    train_idx = np.sort(indices[n_test:])
    return train_idx, test_idx


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    abs_error = np.abs(residual)
    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "medae": float(np.median(abs_error)),
    }
