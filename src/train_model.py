import json
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from .model_builder import build_pose_model

# --- Spatio-Temporal Transformer에 맞게 차원 정의 ---
NUM_NODES = 33
NUM_FEATURES = 9  # 위치(x,y,z) + 속도(x,y,z) + 가속도(x,y,z)


def _total_updrs_score(items_dict):
    """MDS-UPDRS Part III 항목 dict에서 총점 계산"""
    total = 0
    for v in items_dict.values():
        if isinstance(v, list):
            total += sum(v)
        else:
            total += v
    return total


def _gait_updrs_score(items_dict):
    """
    보행/자세 관련 항목만 합산하여 gait UPDRS 점수를 계산.
    가정: 9=의자에서 일어서기, 10=보행, 11=동결, 12=자세 안정성, 13=자세, 14=신체 브래디키네시아.
    """
    gait_keys = ['9', '10', '11', '12', '13', '14']
    score = 0
    for k in gait_keys:
        if k not in items_dict:
            continue
        v = items_dict[k]
        score += sum(v) if isinstance(v, list) else v
    return score


def load_labels(json_dir):
    """HospitalData/JSON 폴더에서 환자별 gait/total UPDRS 라벨을 불러와 dict로 반환"""
    labels = {}
    for fname in os.listdir(json_dir):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(json_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for patient in data.get("patient", []):
            pid = patient["id"]
            items = patient["mds_updrs_part3"]["itmes"][0]
            gait = _gait_updrs_score(items)
            total = _total_updrs_score(items)
            labels[pid] = {"gait_updrs": gait, "total_updrs": total}
    return labels


def load_data(processed_data_path, label_dir):
    """
    .npy 파일을 불러와 X, y_reg, sample_ids를 생성
    - y_reg: gait UPDRS 점수
    """
    X_data = []
    y_reg = []
    sample_ids = []

    label_map = load_labels(label_dir)

    for root, _, files in os.walk(processed_data_path):
        for file_name in files:
            # 요청: '_2_pose.npy' 파일만 사용
            if not file_name.endswith('_2_pose.npy'):
                continue
            npy_path = os.path.join(root, file_name)
            pose_data = np.load(npy_path)  # (Frames, 33*9)

            stem = file_name.replace('_pose.npy', '')
            patient_id = stem.rsplit('_', 1)[0]

            label_info = label_map.get(patient_id)
            if not label_info:
                print(f"[WARN] 라벨이 없는 파일 건너뜀: {file_name}")
                continue

            X_data.append(pose_data)
            y_reg.append(label_info["gait_updrs"])
            sample_ids.append(file_name.replace('_pose.npy', ''))

    if len(X_data) == 0:
        raise ValueError("로드된 데이터가 없습니다. 경로/라벨 매핑을 확인하세요.")

    # 1. 시퀀스 패딩
    max_len = 150
    X_padded = pad_sequences(X_data, maxlen=max_len, padding='post', dtype='float32')

    # 2. 차원 변환 (Samples, Frames, 33, 9)
    expected_feat = NUM_NODES * NUM_FEATURES
    if X_padded.shape[2] != expected_feat:
        raise ValueError(f"입력 피처 수 불일치: 기대 {expected_feat}, 현재 {X_padded.shape[2]}")

    X_reshaped = X_padded.reshape(
        (X_padded.shape[0], X_padded.shape[1], NUM_NODES, NUM_FEATURES)
    )

    return X_reshaped, np.array(y_reg, dtype=np.float32), sample_ids


def train_pose_model(processed_data_path, model_save_path, label_dir="HospitalData/JSON"):
    """
    데이터 로드, 회귀 모델 빌드, 학습 실행 (gait UPDRS 예측 전용)
    """
    X, y_reg, sample_ids = load_data(processed_data_path, label_dir)

    X_train, X_val, y_reg_train, y_reg_val, ids_train, ids_val = train_test_split(
        X, y_reg, sample_ids, test_size=0.1, random_state=42
    )

    print(f"X_train shape: {X_train.shape}")  # (..., 150, 33, 9)
    print(f"X_val shape: {X_val.shape}")      # (..., 150, 33, 9)

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
    model = build_pose_model(input_shape, optimizer=optimizer)

    model.summary()

    callbacks = [
        ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        X_train, y_reg_train,
        validation_data=(X_val, y_reg_val),
        epochs=100,
        batch_size=16,
        callbacks=callbacks
    )

    print(f"모델 학습 완료. 최적 모델이 {model_save_path} 에 저장되었습니다.")

    # main.py에서 평가/추론에 활용할 수 있도록 validation set 반환
    return model, history, (X_val, y_reg_val, ids_val)
