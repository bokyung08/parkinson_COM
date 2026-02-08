import cv2
import mediapipe as mp
import numpy as np
import os
from .utils import center_calculate  # COM 시각화용 중심 계산

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.5)


def process_video_for_pose(video_path, output_video_folder):
    """
    보행 비디오를 Pose keypoint로 변환해 COM 기준으로 정규화하고,
    위치 + 1차/2차 차분(속도/가속도) 특징을 반환합니다.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    output_video_filename = os.path.join(output_video_folder, f"{video_basename}_COM_output.mp4")
    video_writer = cv2.VideoWriter(output_video_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    all_landmarks_data = []
    prev_coords = None  # 이전 프레임 COM 기준 좌표
    prev_vel = None     # 이전 프레임 속도
    frame_idx = 0
    print(f"[INFO] 시작: {video_basename} (총 프레임: {total_frames})")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        annotated_image = frame.copy()

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_landmarks = []

            # -----------------------------------------------------------------
            # 1. COM(무게중심) 계산: 좌/우 엉덩이 평균
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

            com_x = (left_hip.x + right_hip.x) / 2
            com_y = (left_hip.y + right_hip.y) / 2
            com_z = (left_hip.z + right_hip.z) / 2

            # 2. COM 기준 좌표 + 속도/가속도 계산
            current_coords = []
            for lm in landmarks:
                current_coords.append((lm.x - com_x, lm.y - com_y, lm.z - com_z))

            if prev_coords is None:
                velocities = [(0.0, 0.0, 0.0)] * len(current_coords)
            else:
                velocities = [(c[0] - p[0], c[1] - p[1], c[2] - p[2]) for c, p in zip(current_coords, prev_coords)]

            if prev_vel is None:
                accelerations = [(0.0, 0.0, 0.0)] * len(current_coords)
            else:
                accelerations = [(v[0] - pv[0], v[1] - pv[1], v[2] - pv[2]) for v, pv in zip(velocities, prev_vel)]

            for coord, vel, acc in zip(current_coords, velocities, accelerations):
                frame_landmarks.extend([
                    coord[0], coord[1], coord[2],
                    vel[0], vel[1], vel[2],
                    acc[0], acc[1], acc[2]
                ])

            all_landmarks_data.append(frame_landmarks)
            prev_coords = current_coords
            prev_vel = velocities
            # -----------------------------------------------------------------

            if frame_idx % 100 == 0:
                print(f"[INFO] {video_basename}: {frame_idx}/{total_frames} 프레임 처리")

            # 3. 시각화용 (기존 로직 유지)
            l_s = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height)
            r_s = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width,
                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height)
            l_h = (left_hip.x * width, left_hip.y * height)
            r_h = (right_hip.x * width, right_hip.y * height)

            sh_center_x, sh_center_y = (l_s[0] + r_s[0]) / 2, (l_s[1] + r_s[1]) / 2
            hip_center_x, hip_center_y = (l_h[0] + r_h[0]) / 2, (l_h[1] + r_h[1]) / 2

            Xcom_viz, Ycom_viz = center_calculate(hip_center_x, sh_center_x, hip_center_y, sh_center_y, 52.9)
            cv2.circle(annotated_image, (int(Xcom_viz), int(Ycom_viz)), 10, (0, 255, 255), -1)
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        video_writer.write(annotated_image)

    cap.release()
    video_writer.release()
    print(f"[INFO] 완료: {video_basename}, 저장: {output_video_filename}")

    return np.array(all_landmarks_data)
