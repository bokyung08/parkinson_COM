import cv2
import mediapipe as mp
import numpy as np
import os
from .utils import center_calculate  # 이 파일은 시각화 용도로만 사용됩니다.

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.5)


def process_video_for_pose(video_path, output_video_folder):
    """
    하나의 비디오를 Pose keypoint로 변환하고 시각화 영상 생성
    *** COM 기반 정규화 적용 ***
    """
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

            # -----------------------------------------------------------------
            # 1. COM (무게중심) 계산 (모델 입력용 - 정규화된 좌표 사용)
            # 여기서는 두 엉덩이(LEFT_HIP, RIGHT_HIP)의 중간 지점을 COM으로 정의합니다.
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

            com_x = (left_hip.x + right_hip.x) / 2
            com_y = (left_hip.y + right_hip.y) / 2
            com_z = (left_hip.z + right_hip.z) / 2

            # 2. 모든 랜드마크를 COM 기준으로 정규화하여 저장
            for lm in landmarks:
                frame_landmarks.extend([
                    lm.x - com_x,  # 원본 x에서 COM x좌표를 뺌
                    lm.y - com_y,  # 원본 y에서 COM y좌표를 뺌
                    lm.z - com_z   # 원본 z에서 COM z좌표를 뺌
                ])
            all_landmarks_data.append(frame_landmarks)
            # -----------------------------------------------------------------

            # 3. 시각화 (이 부분은 픽셀 좌표를 사용하므로 기존 로직 유지)
            l_s = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height)
            r_s = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width,
                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height)
            l_h = (left_hip.x * width, left_hip.y * height)
            r_h = (right_hip.x * width, right_hip.y * height)

            sh_center_x, sh_center_y = (l_s[0] + r_s[0]) / 2, (l_s[1] + r_s[1]) / 2
            hip_center_x, hip_center_y = (l_h[0] + r_h[0]) / 2, (l_h[1] + r_h[1]) / 2

            # center_calculate 함수가 픽셀 좌표를 사용한다고 가정
            Xcom_viz, Ycom_viz = center_calculate(hip_center_x, sh_center_x, hip_center_y, sh_center_y, 52.9)
            cv2.circle(annotated_image, (int(Xcom_viz), int(Ycom_viz)), 10, (0, 255, 255), -1)
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        video_writer.write(annotated_image)

    cap.release()
    video_writer.release()
    print(f"[INFO] {output_video_filename} 저장 완료")

    # 이제 all_landmarks_data는 COM(엉덩이 중심) 기준 상대좌표가 됩니다.
    return np.array(all_landmarks_data)
