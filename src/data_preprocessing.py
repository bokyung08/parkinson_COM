import cv2
import mediapipe as mp
import numpy as np
import os
from .utils import center_calculate


# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.5)
# 영상 처리 및 Pose keypoint 추출 함수
def process_video_for_pose(video_path, output_video_folder):
    """하나의 비디오를 Pose keypoint로 변환하고 시각화 영상 생성"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    output_video_filename = os.path.join(output_video_folder, f"{video_basename}_COM_output.mp4")
    video_writer = cv2.VideoWriter(output_video_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    all_landmarks_data = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        annotated_image = frame.copy()

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_landmarks = []
            for lm in landmarks:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
            all_landmarks_data.append(frame_landmarks)

            # 무게중심 계산 (COM) 및 시각화
            l_s = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height)
            r_s = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width,
                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height)
            l_h = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height)
            r_h = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * width,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * height)

            sh_center_x, sh_center_y = (l_s[0] + r_s[0]) / 2, (l_s[1] + r_s[1]) / 2
            hip_center_x, hip_center_y = (l_h[0] + r_h[0]) / 2, (l_h[1] + r_h[1]) / 2
            Xcom, Ycom = center_calculate(hip_center_x, sh_center_x, hip_center_y, sh_center_y, 52.9)
            cv2.circle(annotated_image, (int(Xcom), int(Ycom)), 10, (0, 255, 255), -1)
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        video_writer.write(annotated_image)

    cap.release()
    video_writer.release()
    print(f"[INFO] {output_video_filename} 저장 완료")
    return np.array(all_landmarks_data)
