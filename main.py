import os
import numpy as np
from datetime import datetime
from src.data_preprocessing import process_video_for_pose
from src.train_model import train_pose_model
from src.evaluate_model import evaluate_and_plot


def main():
    # HospitalData 기반 경로
    dataset_path = 'HospitalData/VIDEO'
    processed_data_path = 'HospitalData/processed_pose_data'

    # 실행마다 고유 폴더 생성
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_folder = os.path.join('results', 'video_outputs_pose_only', run_id)
    model_save_path = os.path.join('results', 'models', run_id, 'best_pose_model.weights.h5')
    plots_output_dir = os.path.join('results', 'plots', run_id)

    os.makedirs(output_video_folder, exist_ok=True)
    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(plots_output_dir, exist_ok=True)

    # 1) Pose 추출/전처리 (COM 기준 + 속도/가속도)
    for patient_dir in os.listdir(dataset_path):
        full_patient_dir = os.path.join(dataset_path, patient_dir)
        if not os.path.isdir(full_patient_dir):
            continue

        output_dir = os.path.join(processed_data_path, patient_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"[STEP] 환자 {patient_dir} 처리 시작")

        for file in os.listdir(full_patient_dir):
            if not file.endswith(('.mp4', '.avi', '.mov')):
                continue
            # 요청: 파일명이 '..._2.*'로 끝나는 영상만 처리
            if not os.path.splitext(file)[0].endswith('_2'):
                print(f"[SKIP] _2 파일 아님: {file}")
                continue

            video_path = os.path.join(full_patient_dir, file)
            save_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_pose.npy")
            if os.path.exists(save_path):
                print(f"[SKIP] 이미 존재: {save_path}")
                continue
            pose_data = process_video_for_pose(video_path, output_video_folder)
            if pose_data is None:
                continue
            np.save(save_path, pose_data)
            print(f"[DONE] 저장: {save_path}")
        print(f"[STEP] 환자 {patient_dir} 처리 완료")

    # 2) 학습 (멀티태스크: HY 분류 + UPDRS 회귀)
    model, history, (X_val, y_reg_val, ids_val) = train_pose_model(processed_data_path, model_save_path)

    # 2-1) 학습 곡선 시각화
    if history is not None:
        plt_dir = plots_output_dir
        os.makedirs(plt_dir, exist_ok=True)
        hist = history.history
        # Loss
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(hist.get('loss', []), label='train_loss')
        plt.plot(hist.get('val_loss', []), label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training/Validation Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plt_dir, 'history_loss.png'))
        plt.close()
        # MAE
        if 'mae' in hist:
            plt.figure()
            plt.plot(hist.get('mae', []), label='train_mae')
            plt.plot(hist.get('val_mae', []), label='val_mae')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.title('Training/Validation MAE')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plt_dir, 'history_mae.png'))
            plt.close()

    # 3) 평가 (UPDRS 회귀 전용)
    evaluate_and_plot(model_save_path, X_val, y_reg_val, ids_val, plots_output_dir)


if __name__ == "__main__":
    main()
