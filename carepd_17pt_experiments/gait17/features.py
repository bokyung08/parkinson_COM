from __future__ import annotations

import numpy as np

from .constants import H36M17_EDGES


def mediapipe33_to_h36m17(joints: np.ndarray) -> np.ndarray:
    """Convert MediaPipe 33 joints to the H36M-compatible 17-joint layout."""
    if joints.ndim != 3 or joints.shape[1] < 29 or joints.shape[2] < 3:
        raise ValueError(f"Expected (T, 33, >=3) MediaPipe input, got {joints.shape}")
    x = joints[..., :3].astype(np.float32)
    out = np.zeros((x.shape[0], 17, 3), dtype=np.float32)
    out[:, 0] = (x[:, 23] + x[:, 24]) / 2.0
    out[:, 1] = x[:, 24]
    out[:, 2] = x[:, 26]
    out[:, 3] = x[:, 28]
    out[:, 4] = x[:, 23]
    out[:, 5] = x[:, 25]
    out[:, 6] = x[:, 27]
    out[:, 7] = (x[:, 11] + x[:, 12] + x[:, 23] + x[:, 24]) / 4.0
    out[:, 8] = (x[:, 11] + x[:, 12]) / 2.0
    out[:, 9] = x[:, 0]
    out[:, 10] = (x[:, 7] + x[:, 8]) / 2.0
    out[:, 11] = x[:, 11]
    out[:, 12] = x[:, 13]
    out[:, 13] = x[:, 15]
    out[:, 14] = x[:, 12]
    out[:, 15] = x[:, 14]
    out[:, 16] = x[:, 16]
    return out


def ensure_h36m17(raw: np.ndarray, source_format: str) -> np.ndarray:
    if source_format == "h36m17":
        if raw.ndim != 3 or raw.shape[1] != 17 or raw.shape[2] < 3:
            raise ValueError(f"Expected (T, 17, >=3), got {raw.shape}")
        return raw[..., :3].astype(np.float32)
    if source_format == "mediapipe33":
        if raw.ndim == 2 and raw.shape[1] % 33 == 0:
            raw = raw.reshape(raw.shape[0], 33, raw.shape[1] // 33)
        return mediapipe33_to_h36m17(raw)
    raise ValueError(f"Unsupported source_format: {source_format}")


def com_normalize(joints: np.ndarray) -> np.ndarray:
    pelvis = joints[:, 0:1, :]
    return (joints - pelvis).astype(np.float32)


def body_scale_value(joints: np.ndarray, mode: str = "none") -> float:
    """Estimate a sequence-level body scale from H36M17 joints."""
    if mode == "none":
        return 1.0
    joints = joints.astype(np.float32)
    eps = 1e-6
    if mode == "median_bone":
        lengths = []
        for a, b in H36M17_EDGES:
            lengths.append(np.linalg.norm(joints[:, a] - joints[:, b], axis=-1))
        values = np.concatenate(lengths)
    elif mode == "torso":
        values = np.linalg.norm(joints[:, 8] - joints[:, 0], axis=-1)
    elif mode == "hip_width":
        values = np.linalg.norm(joints[:, 1] - joints[:, 4], axis=-1)
    else:
        raise ValueError(f"Unsupported scale normalization mode: {mode}")
    values = values[np.isfinite(values) & (values > eps)]
    if values.size == 0:
        return 1.0
    return float(np.median(values))


def scale_normalize(joints: np.ndarray, mode: str = "none") -> np.ndarray:
    scale = body_scale_value(joints, mode)
    if scale <= 1e-6:
        return joints.astype(np.float32)
    return (joints / np.float32(scale)).astype(np.float32)


def pad_or_clip(x: np.ndarray, max_len: int) -> np.ndarray:
    if x.shape[0] > max_len:
        return x[-max_len:]
    if x.shape[0] < max_len:
        pad = np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)
    return x


def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-6
    cos_angle = np.sum(ba * bc, axis=-1) / denom
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def build_hybrid_features(joints: np.ndarray) -> np.ndarray:
    """Build 9-channel node features on H36M17: xyz, dxyz, amplitude, variability, angle."""
    joints = joints.astype(np.float32)
    t, j, _ = joints.shape
    velocity = np.diff(joints, axis=0, prepend=joints[:1])
    norms = np.linalg.norm(joints, axis=2)
    amp = np.broadcast_to((norms.max(axis=0) - norms.min(axis=0))[None, :, None], (t, j, 1))
    var = np.broadcast_to(norms.std(axis=0)[None, :, None], (t, j, 1))
    angles = np.zeros((t, j), dtype=np.float32)
    angles[:, 2] = compute_angle(joints[:, 1], joints[:, 2], joints[:, 3])
    angles[:, 5] = compute_angle(joints[:, 4], joints[:, 5], joints[:, 6])
    angles[:, 12] = compute_angle(joints[:, 11], joints[:, 12], joints[:, 13])
    angles[:, 15] = compute_angle(joints[:, 14], joints[:, 15], joints[:, 16])
    return np.concatenate([joints, velocity, amp, var, angles[..., None]], axis=-1).astype(np.float32)


def apply_ablation(features: np.ndarray, mode: str) -> np.ndarray:
    if mode == "A":
        return features[..., :3]
    if mode == "B":
        return features[..., :6]
    if mode == "C":
        return features[..., :8]
    return features


def summarize_sequence(x: np.ndarray) -> np.ndarray:
    """Tabular summary for classical ML from a `(T, J, C)` sequence."""
    flat = x.reshape(x.shape[0], -1)
    return np.concatenate(
        [
            flat.mean(axis=0),
            flat.std(axis=0),
            flat.min(axis=0),
            flat.max(axis=0),
            np.percentile(flat, 25, axis=0),
            np.percentile(flat, 75, axis=0),
        ]
    ).astype(np.float32)


def joint_collection_distances(joints: np.ndarray) -> np.ndarray:
    diff = joints[:, :, None, :] - joints[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(joints.shape[1], k=1)
    return dist[:, iu[0], iu[1]].astype(np.float32)
