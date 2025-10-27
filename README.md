# 🧠 Parkinson vs Healthy Pose Classification (LSTM 기반)

본 프로젝트는 **MediaPipe Pose**를 이용해 영상에서 인체 keypoint를 추출하고,  
**LSTM 기반 시계열 모델**을 통해 파킨슨 환자와 정상인의 동작 패턴을 분류하는 연구입니다.

---

## 📁 프로젝트 구조

parkinson_pose_lstm/
│
├── data/
│ ├── pre_final_video/ # 원본 영상
│ └── prefinal_preprocessed/ # Pose 추출 결과
├── results/
│ ├── video_outputs_pose_only/ # COM 시각화 영상
│ ├── models/ # 저장된 모델
│ └── plots/ # 학습 및 평가 그래프
├── src/
│ ├── data_preprocessing.py
│ ├── model_builder.py
│ ├── train_model.py
│ ├── evaluate_model.py
│ ├── utils.py
│ └── init.py
├── main.py
└── requirements.txt
## ⚙️ 실행 방법

1. **환경 설정**
   pip install -r requirements.txt

데이터 폴더 준비
data/pre_final_video/
    ├── healthy/
    │   ├── video1.mp4
    │   └── video2.mp4
    └── disease/
        ├── video1.mp4
        └── video2.mp4

전체 파이프라인 실행

conda env create -f environment.yml
conda activate parkinson_pose_env
python main.py

pip install -r requirements.txt

Pose npy 데이터: data/prefinal_preprocessed/

COM 시각화 영상: results/video_outputs_pose_only/

학습된 모델: results/models/best_pose_model.h5

평가 그래프: results/plots/confusion_matrix.png

