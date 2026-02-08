import argparse
import os
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


NOSE = 0
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
LEFT_ANKLE = 27
RIGHT_KNEE = 26
RIGHT_ANKLE = 28


def iter_videos(root_dir):
    for root, _, files in os.walk(root_dir):
        for f in files:
            base, _ = os.path.splitext(f)
            if base.endswith("_2"):
                yield os.path.join(root, f)


def _angle(a, b, c):
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8)
    cos = np.sum(ba * bc, axis=-1) / denom
    return np.arccos(np.clip(cos, -1.0, 1.0))


def extract_landmarks(video_path, seconds=10.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(seconds * fps) if fps > 0 else int(seconds * 30)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.5)

    abs_nose = []
    rel_nose = []
    abs_coords = []
    rel_coords = []
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
        landmarks = results.pose_landmarks.landmark
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        com = (coords[LEFT_HIP] + coords[RIGHT_HIP]) / 2.0
        rel = coords - com

        abs_nose.append(coords[NOSE][:2])
        rel_nose.append(rel[NOSE][:2])
        abs_coords.append(coords)
        rel_coords.append(rel)

    cap.release()
    pose.close()

    return np.array(abs_nose), np.array(rel_nose), np.array(abs_coords), np.array(rel_coords)


def plot_traj(abs_xy, rel_xy, out_path, title):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(abs_xy[:, 0], abs_xy[:, 1], linewidth=1)
    plt.title("Absolute")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    plt.plot(rel_xy[:, 0], rel_xy[:, 1], linewidth=1, color="tab:orange")
    plt.title("COM-relative")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().invert_yaxis()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_timeseries(abs_xy, rel_xy, out_path, title):
    t = np.arange(len(abs_xy))
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.plot(t, abs_xy[:, 0], label="abs_x")
    plt.plot(t, abs_xy[:, 1], label="abs_y")
    plt.legend()
    plt.title("Absolute")
    plt.xlabel("frame")
    plt.ylabel("pos")

    plt.subplot(2, 1, 2)
    plt.plot(t, rel_xy[:, 0], label="rel_x")
    plt.plot(t, rel_xy[:, 1], label="rel_y")
    plt.legend()
    plt.title("COM-relative")
    plt.xlabel("frame")
    plt.ylabel("pos")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_distribution(abs_vals, rel_vals, out_path, title, xlabel):
    plt.figure(figsize=(6, 4))
    plt.hist(abs_vals, bins=50, alpha=0.6, label="Absolute")
    plt.hist(rel_vals, bins=50, alpha=0.6, label="COM-relative")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_xy_cloud(abs_xy, rel_xy, out_path, title):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(abs_xy[:, 0], abs_xy[:, 1], s=2, alpha=0.3)
    plt.title("Absolute")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    plt.scatter(rel_xy[:, 0], rel_xy[:, 1], s=2, alpha=0.3, color="tab:orange")
    plt.title("COM-relative")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().invert_yaxis()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare COM-normalized vs absolute pose trajectories.")
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--per_video_plots", action="store_true")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join("results", "com_compare", ts)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    all_abs_dist = []
    all_rel_dist = []
    all_abs_speed = []
    all_rel_speed = []
    all_abs_angle = []
    all_rel_angle = []
    all_abs_xy = []
    all_rel_xy = []
    for idx, path in enumerate(iter_videos(args.video_root)):
        if args.max_samples and idx >= args.max_samples:
            break
        base = os.path.splitext(os.path.basename(path))[0]
        abs_xy, rel_xy, abs_coords, rel_coords = extract_landmarks(path, seconds=args.seconds)
        if len(abs_xy) == 0:
            continue
        abs_var = np.var(abs_xy, axis=0)
        rel_var = np.var(rel_xy, axis=0)
        rows.append({
            "video": base,
            "abs_var_x": float(abs_var[0]),
            "abs_var_y": float(abs_var[1]),
            "rel_var_x": float(rel_var[0]),
            "rel_var_y": float(rel_var[1]),
        })

        if args.per_video_plots:
            plot_traj(abs_xy, rel_xy, os.path.join(out_dir, f"{base}_traj.png"), base)
            plot_timeseries(abs_xy, rel_xy, os.path.join(out_dir, f"{base}_timeseries.png"), base)

        # Aggregate metrics across all joints/frames
        abs_flat = abs_coords.reshape(-1, 3)
        rel_flat = rel_coords.reshape(-1, 3)
        all_abs_dist.append(np.linalg.norm(abs_flat[:, :2], axis=1))
        all_rel_dist.append(np.linalg.norm(rel_flat[:, :2], axis=1))

        abs_vel = np.diff(abs_coords, axis=0)
        rel_vel = np.diff(rel_coords, axis=0)
        all_abs_speed.append(np.linalg.norm(abs_vel[..., :2], axis=-1).reshape(-1))
        all_rel_speed.append(np.linalg.norm(rel_vel[..., :2], axis=-1).reshape(-1))

        abs_angle_l = _angle(abs_coords[:, LEFT_HIP], abs_coords[:, LEFT_KNEE], abs_coords[:, LEFT_ANKLE])
        abs_angle_r = _angle(abs_coords[:, RIGHT_HIP], abs_coords[:, RIGHT_KNEE], abs_coords[:, RIGHT_ANKLE])
        rel_angle_l = _angle(rel_coords[:, LEFT_HIP], rel_coords[:, LEFT_KNEE], rel_coords[:, LEFT_ANKLE])
        rel_angle_r = _angle(rel_coords[:, RIGHT_HIP], rel_coords[:, RIGHT_KNEE], rel_coords[:, RIGHT_ANKLE])
        all_abs_angle.append(np.concatenate([abs_angle_l, abs_angle_r]))
        all_rel_angle.append(np.concatenate([rel_angle_l, rel_angle_r]))

        all_abs_xy.append(abs_flat[:, :2])
        all_rel_xy.append(rel_flat[:, :2])

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        df["var_reduction_x"] = 1.0 - (df["rel_var_x"] / (df["abs_var_x"] + 1e-8))
        df["var_reduction_y"] = 1.0 - (df["rel_var_y"] / (df["abs_var_y"] + 1e-8))
        df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

        plt.figure(figsize=(8, 4))
        plt.plot(df["var_reduction_x"], label="x reduction")
        plt.plot(df["var_reduction_y"], label="y reduction")
        plt.ylabel("variance reduction")
        plt.xlabel("video index")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "variance_reduction.png"), dpi=200)
        plt.close()

        abs_dist = np.concatenate(all_abs_dist)
        rel_dist = np.concatenate(all_rel_dist)
        abs_speed = np.concatenate(all_abs_speed)
        rel_speed = np.concatenate(all_rel_speed)
        abs_angle = np.concatenate(all_abs_angle)
        rel_angle = np.concatenate(all_rel_angle)
        abs_xy_all = np.concatenate(all_abs_xy)
        rel_xy_all = np.concatenate(all_rel_xy)

        plot_distribution(abs_dist, rel_dist, os.path.join(out_dir, "dist_joint_distance.png"),
                          "Joint Distance to Origin", "distance")
        plot_distribution(abs_speed, rel_speed, os.path.join(out_dir, "dist_joint_speed.png"),
                          "Joint Speed Magnitude", "speed")
        plot_distribution(abs_angle, rel_angle, os.path.join(out_dir, "dist_joint_angle.png"),
                          "Knee Angle (rad)", "angle")
        plot_xy_cloud(abs_xy_all, rel_xy_all, os.path.join(out_dir, "xy_cloud.png"),
                      "All Joint XY Cloud (Abs vs COM)")

    print(f"[INFO] Saved COM comparison to {out_dir}")


if __name__ == "__main__":
    main()
