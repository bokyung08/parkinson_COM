"""
Feature engineering for COM-anchored gait severity estimation.
All features are COM-relative and deliberately exclude absolute kinematics
(absolute speed/stride) to keep the model clinically safe and small-data friendly.
"""

import numpy as np

# MediaPipe joint indices (frontal view), used for bone angle computation
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


def compute_relative_velocity(joints: np.ndarray) -> np.ndarray:
    """First-order difference; no absolute speed (safe)."""
    return np.diff(joints, axis=0, prepend=joints[:1])


def compute_amplitude(joints: np.ndarray) -> np.ndarray:
    """Per-joint motion amplitude (max-min of norm over time)."""
    norms = np.linalg.norm(joints, axis=2)  # (T, J)
    return norms.max(axis=0) - norms.min(axis=0)  # (J,)


def compute_variability(joints: np.ndarray) -> np.ndarray:
    """Per-joint temporal variability (std of norm)."""
    norms = np.linalg.norm(joints, axis=2)  # (T, J)
    return norms.std(axis=0)  # (J,)


def compute_bone_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Scale-invariant bone angle for joints (a, b, c), shape (T, 3) each.
    θ = arccos( ( (a-b)·(c-b) ) / (||a-b|| * ||c-b||) )
    """
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)) + 1e-6
    cos_angle = np.sum(ba * bc, axis=-1) / denom
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def build_hybrid_node_features(joints: np.ndarray) -> np.ndarray:
    """
    Build 9-D node features per joint per frame.
    joints: (T, J, 3) COM-relative coordinates.
    Returns: (T, J, 9) ordered as
      [x, y, z, dx, dy, dz, amplitude, variability, angle]
    """
    T, J, _ = joints.shape

    # 1) Relative velocity (safe)
    velocity = compute_relative_velocity(joints)  # (T, J, 3)

    # 2) Amplitude / Variability (time-invariant -> broadcast)
    amplitude = compute_amplitude(joints)  # (J,)
    variability = compute_variability(joints)  # (J,)
    amp_broadcast = np.broadcast_to(amplitude[None, :, None], (T, J, 1))
    var_broadcast = np.broadcast_to(variability[None, :, None], (T, J, 1))

    # 3) Bone angles (only populate knees/elbows; others remain 0)
    angles = np.zeros((T, J), dtype=np.float32)

    # Knees: hip-knee-ankle
    angles[:, MP_LEFT_KNEE] = compute_bone_angle(
        joints[:, MP_LEFT_HIP], joints[:, MP_LEFT_KNEE], joints[:, MP_LEFT_ANKLE]
    )
    angles[:, MP_RIGHT_KNEE] = compute_bone_angle(
        joints[:, MP_RIGHT_HIP], joints[:, MP_RIGHT_KNEE], joints[:, MP_RIGHT_ANKLE]
    )
    # Elbows: shoulder-elbow-wrist
    angles[:, MP_LEFT_ELBOW] = compute_bone_angle(
        joints[:, MP_LEFT_SHOULDER], joints[:, MP_LEFT_ELBOW], joints[:, MP_LEFT_WRIST]
    )
    angles[:, MP_RIGHT_ELBOW] = compute_bone_angle(
        joints[:, MP_RIGHT_SHOULDER], joints[:, MP_RIGHT_ELBOW], joints[:, MP_RIGHT_WRIST]
    )
    angles = angles[..., None]  # (T, J, 1)

    # Concatenate in mandated order
    node_features = np.concatenate(
        [joints, velocity, amp_broadcast, var_broadcast, angles],
        axis=-1
    )  # (T, J, 9)

    return node_features


def apply_ablation(node_features: np.ndarray, mode: str) -> np.ndarray:
    """
    Ablation helper to drop channel groups.
    mode:
      'A' -> coordinates only (0:3)
      'B' -> coords + velocity (0:6)
      'C' -> coords + velocity + amp/var (0:8)
      'D' -> full hybrid (0:9)
    """
    if mode == 'A':
        return node_features[..., :3]
    if mode == 'B':
        return node_features[..., :6]
    if mode == 'C':
        return node_features[..., :8]
    return node_features  # 'D' or default


# Public mapping for convenience in scripts
ABLA_MODES = {
    "A": "coords_only",
    "B": "coords_velocity",
    "C": "coords_vel_amp_var",
    "D": "full_hybrid",
}


def build_features_from_npy(joints_npy: np.ndarray, ablation: str = "D") -> np.ndarray:
    """
    Convenience wrapper: build node features then apply ablation slice.
    joints_npy: (T, J, 3) COM-relative.
    """
    feats = build_hybrid_node_features(joints_npy)
    return apply_ablation(feats, ablation)
