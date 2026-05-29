from __future__ import annotations

H36M17_JOINTS = [
    "Pelvis",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "Spine",
    "Thorax",
    "Neck_Nose",
    "Head",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
]

H36M17_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

LEFT_RIGHT_PAIRS = [
    (1, 4), (2, 5), (3, 6),
    (11, 14), (12, 15), (13, 16),
]

DEFAULT_GAIT_KEYS = ("10", "11", "12", "13", "14")

