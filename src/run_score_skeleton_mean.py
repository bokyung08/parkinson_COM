import argparse
import json
import os
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


# MediaPipe Pose indices
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# MediaPipe Pose edges (subset, same as other scripts)
EDGES = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (11, 12), (23, 24), (11, 23), (12, 24),
]


def load_score_labels(json_dir):
    labels = {}
    for fname in os.listdir(json_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(json_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for patient in data.get("patient", []):
            pid = patient["id"]
            items = patient["mds_updrs_part3"]["itmes"][0]
            item10 = items.get("10", 0)
            item10 = float(sum(item10) if isinstance(item10, list) else item10)
            score = 0.0
            for k in ["10", "11", "12", "13", "14"]:
                v = items.get(k, 0)
                score += sum(v) if isinstance(v, list) else v
            labels[pid] = {"item10": item10, "sum10_14": float(score)}
    return labels


def iter_videos(video_root):
    for root, _, files in os.walk(video_root):
        for f in files:
            base, _ = os.path.splitext(f)
            if base.endswith("_2"):
                yield os.path.join(root, f)


def extract_mean_skeleton(video_path, seconds=7.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(seconds * fps) if fps > 0 else int(seconds * 30)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.5)

    frames = []
    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            continue
        coords = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark], dtype=np.float32)
        frames.append(coords)

    cap.release()
    pose.close()

    if not frames:
        return None
    return np.mean(np.stack(frames, axis=0), axis=0)


def extract_skeleton_frames(video_path, seconds=7.0, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_total = int(seconds * fps) if fps > 0 else int(seconds * 30)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.5)

    frames = []
    frame_idx = 0
    while cap.isOpened() and frame_idx < max_total:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            continue
        coords = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark], dtype=np.float32)
        com = (coords[LEFT_HIP] + coords[RIGHT_HIP]) / 2.0
        coords = coords - com
        frames.append(coords)

    cap.release()
    pose.close()

    if not frames:
        return []
    if len(frames) <= max_frames:
        return frames
    idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
    return [frames[i] for i in idx]


def extract_frames_abs(video_path, seconds=7.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_total = int(seconds * fps) if fps > 0 else int(seconds * 30)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.5)

    frames = []
    frame_idx = 0
    while cap.isOpened() and frame_idx < max_total:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            continue
        coords = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark], dtype=np.float32)
        frames.append(coords)

    cap.release()
    pose.close()
    return frames


def plot_skeleton(ax, joints_xy, color="k", alpha=0.2):
    for i, j in EDGES:
        ax.plot([joints_xy[i, 0], joints_xy[j, 0]],
                [joints_xy[i, 1], joints_xy[j, 1]],
                color=color, alpha=alpha, linewidth=1.0)
    ax.scatter(joints_xy[:, 0], joints_xy[:, 1], s=8, color=color, alpha=alpha)


def pick_keyframes(frames):
    if not frames:
        return []
    arr = np.stack(frames, axis=0)  # (F, J, 2)
    ankle_y = arr[:, LEFT_ANKLE, 1]
    heel_idx = int(np.argmax(ankle_y))
    swing_idx = int(np.argmin(ankle_y))
    mid_idx = len(frames) // 2
    return [heel_idx, mid_idx, swing_idx]


def main():
    parser = argparse.ArgumentParser(description="Mean skeleton per item10 score (absolute coords).")
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--seconds", type=float, default=7.0)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--max_videos_per_score", type=int, default=1)
    parser.add_argument("--overlay_frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rms_window", type=int, default=5)
    args = parser.parse_args()

    labels = load_score_labels(args.label_dir)
    scores = [v["sum10_14"] for v in labels.values()]
    if not scores:
        raise SystemExit("No item10-14 scores found in JSON.")
    q1, q2, q3 = np.quantile(scores, [0.25, 0.50, 0.75])
    out_dir = args.out_dir or os.path.join("results", "score_skeleton_means", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    by_score = {0: [], 1: [], 2: [], 3: []}
    overlay = {0: [], 1: [], 2: [], 3: []}
    selected_video = {0: None, 1: None, 2: None, 3: None}
    rng = np.random.default_rng(args.seed)
    candidates = {0: [], 1: [], 2: [], 3: []}
    for vid in iter_videos(args.video_root):
        base = os.path.splitext(os.path.basename(vid))[0]
        pid = base.rsplit("_", 1)[0]
        if pid not in labels:
            continue
        mean_joints = extract_mean_skeleton(vid, seconds=args.seconds)
        if mean_joints is None:
            continue
        score = labels[pid]["sum10_14"]
        group = 0
        if score <= q1:
            group = 0
        elif score <= q2:
            group = 1
        elif score <= q3:
            group = 2
        else:
            group = 3
        candidates[group].append((mean_joints, vid))

    for group in [0, 1, 2, 3]:
        if not candidates[group]:
            continue
        picks = candidates[group]
        if len(picks) > args.max_videos_per_score:
            pick_idx = rng.choice(len(picks), size=args.max_videos_per_score, replace=False)
            picks = [picks[i] for i in pick_idx]
        for mean_joints, vid in picks:
            by_score[group].append(mean_joints)
            overlay[group].append(
                extract_skeleton_frames(vid, seconds=args.seconds, max_frames=args.overlay_frames)
            )
            if selected_video[group] is None:
                selected_video[group] = vid

    # Compute mean skeleton per score
    mean_by_score = {}
    for s, items in by_score.items():
        if items:
            mean_by_score[s] = np.mean(np.stack(items, axis=0), axis=0)

    if not mean_by_score:
        raise SystemExit("No skeletons extracted. Check video paths and labels.")

    # Global axis limits for fair comparison
    all_xy = np.concatenate(list(mean_by_score.values()), axis=0)
    xmin, ymin = np.min(all_xy[:, 0]), np.min(all_xy[:, 1])
    xmax, ymax = np.max(all_xy[:, 0]), np.max(all_xy[:, 1])

    # Save per-score figures
    for s in sorted(mean_by_score.keys()):
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_skeleton(ax, mean_by_score[s])
        ax.set_title(f"Item10 group {s}")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"mean_skeleton_item10_{s}.png"), dpi=200)
        plt.close()

    # Combined panel
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True)
    for idx, s in enumerate([0, 1, 2, 3]):
        ax = axes[idx]
        if s in mean_by_score:
            plot_skeleton(ax, mean_by_score[s])
        ax.set_title(f"Item10 {s}")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        if idx == 0:
            ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_skeleton_item10_all.png"), dpi=200)
    plt.close()

    # Overlay motion per score (tremor visualization)
    for s in [0, 1, 2, 3]:
        if not overlay[s]:
            continue
        overlay_frames = [f for seq in overlay[s] for f in seq]
        if not overlay_frames:
            continue
        overlay_arr = np.stack(overlay_frames, axis=0)  # (F, J, 2)
        overlay_xy = np.concatenate(overlay_frames, axis=0)
        xmin_o, ymin_o = np.min(overlay_xy[:, 0]), np.min(overlay_xy[:, 1])
        xmax_o, ymax_o = np.max(overlay_xy[:, 0]), np.max(overlay_xy[:, 1])
        fig, ax = plt.subplots(figsize=(4, 4))
        for frames in overlay[s]:
            for joints_xy in frames:
                plot_skeleton(ax, joints_xy, alpha=0.2)
        ax.scatter(overlay_xy[:, 0], overlay_xy[:, 1], s=2, alpha=0.05, color="tab:blue")
        ax.set_title(f"Item10 group {s}")
        ax.set_xlim(xmin_o, xmax_o)
        ax.set_ylim(ymax_o, ymin_o)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"overlay_item10_{s}.png"), dpi=200)
        plt.close()

        # Speed magnitude over time (mean across joints)
        if overlay_arr.shape[0] > 1:
            vel = np.diff(overlay_arr, axis=0)
            speed = np.linalg.norm(vel, axis=-1)  # (F-1, J)
            mean_speed = speed.mean(axis=1)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(mean_speed, linewidth=1.2)
            ax.set_title(f"Item10 group {s} mean speed")
            ax.set_xlabel("frame")
            ax.set_ylabel("speed")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"overlay_speed_item10_{s}.png"), dpi=200)
            plt.close()

            # Wrist/ankle-only speed
            tremor_joints = [LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE]
            tremor_speed = speed[:, tremor_joints].mean(axis=1)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(tremor_speed, linewidth=1.2)
            ax.set_title(f"Item10 group {s} limb speed")
            ax.set_xlabel("frame")
            ax.set_ylabel("speed")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"overlay_speed_limbs_item10_{s}.png"), dpi=200)
            plt.close()

            # Acceleration magnitude over time
            accel = np.diff(vel, axis=0)
            accel_mag = np.linalg.norm(accel, axis=-1).mean(axis=1)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(accel_mag, linewidth=1.2)
            ax.set_title(f"Item10 group {s} mean accel")
            ax.set_xlabel("frame")
            ax.set_ylabel("accel")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"overlay_accel_item10_{s}.png"), dpi=200)
            plt.close()

            # RMS smoothed speed
            w = max(1, args.rms_window)
            if len(mean_speed) >= w:
                kernel = np.ones(w) / w
                rms = np.sqrt(np.convolve(mean_speed ** 2, kernel, mode="valid"))
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.plot(rms, linewidth=1.2)
                ax.set_title(f"Item10 group {s} RMS speed (w={w})")
                ax.set_xlabel("frame")
                ax.set_ylabel("rms speed")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"overlay_speed_rms_item10_{s}.png"), dpi=200)
                plt.close()

    # Keyframe grid (heel strike / mid-stance / swing)
    keyframes_by_group = {}
    for s in [0, 1, 2, 3]:
        vid = selected_video.get(s)
        if not vid:
            continue
        frames = extract_frames_abs(vid, seconds=args.seconds)
        idxs = pick_keyframes(frames)
        if not idxs:
            continue
        keyframes_by_group[s] = [frames[i] for i in idxs]

    if keyframes_by_group:
        all_xy = np.concatenate([np.stack(fr, axis=0) for fr in keyframes_by_group.values()], axis=0)
        xmin_k, ymin_k = np.min(all_xy[:, :, 0]), np.min(all_xy[:, :, 1])
        xmax_k, ymax_k = np.max(all_xy[:, :, 0]), np.max(all_xy[:, :, 1])
        fig, axes = plt.subplots(4, 3, figsize=(9, 10), sharex=True, sharey=True)
        col_titles = ["heel strike", "mid-stance", "swing"]
        for c, t in enumerate(col_titles):
            axes[0, c].set_title(t)
        for r, s in enumerate([0, 1, 2, 3]):
            frames = keyframes_by_group.get(s)
            if not frames:
                continue
            for c in range(3):
                ax = axes[r, c]
                plot_skeleton(ax, frames[c], alpha=0.6)
                ax.set_xlim(xmin_k, xmax_k)
                ax.set_ylim(ymax_k, ymin_k)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(f"score {s}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "keyframes_item10_grid.png"), dpi=200)
        plt.close()

    print(f"[INFO] Saved mean skeletons to {out_dir}")


if __name__ == "__main__":
    main()
