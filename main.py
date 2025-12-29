import os
from src.data_preprocessing import process_video_for_pose
from src.train_model import train_pose_model
from src.evaluate_model import evaluate_and_plot
import numpy as np
def main():
    dataset_path = 'hospitalwalkingdata/'
    processed_data_path = 'hospitalwalkingdata/processed_pose_data'
    output_video_folder = 'hospitalwalkingdata/results/video_outputs_pose_only'
    model_save_path = 'results/models/1106_best_pose_model.h5'
    os.makedirs(output_video_folder, exist_ok=True)
    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    # 1️⃣ Pose 데이터 전처리
    for cls in ['Normal', 'PD']:
        input_dir = os.path.join(dataset_path, cls)
        output_dir = os.path.join(processed_data_path, cls)
        os.makedirs(output_dir, exist_ok=True)
        for file in os.listdir(input_dir):
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(input_dir, file)
                pose_data = process_video_for_pose(video_path, output_video_folder)
                if pose_data is None:
                    # 두 명 이상 감지되어 스킵된 비디오
                    continue
                np.save(os.path.join(output_dir, f"{os.path.splitext(file)[0]}_pose.npy"), pose_data)

    # 2️⃣ 학습
    model, history, (X_val, y_val) = train_pose_model(processed_data_path, model_save_path)

    # 3️⃣ 평가
    evaluate_and_plot(model_save_path, X_val, y_val, ['Normal', 'PD'], 'results/plots')

if __name__ == "__main__":
    main()
