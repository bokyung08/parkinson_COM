import json
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from ..features.feature_engineering import build_hybrid_node_features, apply_ablation
from ..models.model_builder import build_pose_model

# Spatio-Temporal Transformer에 맞는 차원 정의
NUM_NODES = 33
# 좌표(3) + 상대속도(3) + 진폭(1) + 변동성(1) + 관절각(1) = 9채널
NUM_FEATURES = 9
# 항목별 가중치(10~14). 필요 시 여기서 조정하세요.
GAIT_ITEM_WEIGHTS = {'10': 1.0, '11': 1.0, '12': 1.0, '13': 1.0, '14': 1.0}


def _total_updrs_score(items_dict):
    """MDS-UPDRS Part III 전체 항목 총점 계산"""
    total = 0
    for v in items_dict.values():
        if isinstance(v, list):
            total += sum(v)
        else:
            total += v
    return total


def _gait_updrs_score(items_dict, weights=None):
    """
    보행/자세 관련 항목(10~14)만 합산하여 gait UPDRS 계산.
    weights: dict 형태로 항목별 가중치 지정 가능. None이면 GAIT_ITEM_WEIGHTS 사용.
    """
    weights = weights or GAIT_ITEM_WEIGHTS
    gait_keys = ['10', '11', '12', '13', '14']
    score = 0
    for k in gait_keys:
        if k not in items_dict:
            continue
        v = items_dict[k]
        w = weights.get(k, 1.0)
        score += w * (sum(v) if isinstance(v, list) else v)
    return score


def _item10_score(items_dict):
    """MDS-UPDRS Part III item 10 score."""
    v = items_dict.get("10")
    if v is None:
        return 0.0
    return float(sum(v) if isinstance(v, list) else v)


def load_labels(json_dir, weights=None, target="item10"):
    """HospitalData/JSON 폴더에서 환자별 gait/total UPDRS 라벨을 불러와 dict로 반환"""
    labels = {}
    weights = weights or GAIT_ITEM_WEIGHTS
    for fname in os.listdir(json_dir):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(json_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for patient in data.get("patient", []):
            pid = patient["id"]
            items = patient["mds_updrs_part3"]["itmes"][0]
            item10 = _item10_score(items)
            if target == "gait":
                target_val = _gait_updrs_score(items, weights=weights)
            else:
                target_val = item10
            total = _total_updrs_score(items)
            labels[pid] = {"gait_updrs": target_val, "item10": item10, "total_updrs": total}
    return labels


def load_data(processed_data_path, label_dir, ablation=None, weights=None, max_len=390):
    """
    .npy 파일을 불러 X, y_reg, sample_ids를 반환
    - y_reg: 가중치가 반영된 gait UPDRS 점수
    - ablation: A/B/C/D 채널 슬라이싱
    - max_len: 시퀀스 패딩 길이(기본 390프레임 ~= 13초@30fps)
    """
    X_data = []
    y_reg = []
    sample_ids = []

    label_map = load_labels(label_dir, weights=weights)

    for root, _, files in os.walk(processed_data_path):
        for file_name in files:
            if not file_name.endswith('_2_pose.npy'):
                continue
            npy_path = os.path.join(root, file_name)
            pose_data = np.load(npy_path)  # (Frames, 33*9) 기존 가속도 포함

            stem = file_name.replace('_pose.npy', '')
            patient_id = stem.rsplit('_', 1)[0]

            label_info = label_map.get(patient_id)
            if not label_info:
                print(f"[WARN] 라벨이 없는 파일 건너뜀: {file_name}")
                continue

            # (T, 33, 9) -> 좌표만 추출 후 하이브리드 9채널(좌표+속도+amp/var+각도)로 재구성
            if pose_data.ndim != 2 or pose_data.shape[1] != NUM_NODES * NUM_FEATURES:
                # 예상치 못한 형태일 경우 스킵
                print(f"[WARN] Unexpected shape {pose_data.shape}, skipping {file_name}")
                continue
            coords_only = pose_data.reshape(-1, NUM_NODES, NUM_FEATURES)[..., :3]
            feats = build_hybrid_node_features(coords_only)  # (T, J, 9) 새 특성
            feats_flat = feats.reshape(feats.shape[0], -1)   # pad_sequences용 (T, 33*9)

            X_data.append(feats_flat)
            y_reg.append(label_info["gait_updrs"])
            sample_ids.append(file_name.replace('_pose.npy', ''))

    if len(X_data) == 0:
        raise ValueError("로드된 데이터가 없습니다. 경로/라벨 매핑을 확인하세요.")

    # 1. 시퀀스 패딩 (기본 13초)
    X_padded = pad_sequences(X_data, maxlen=max_len, padding='post', dtype='float32')

    # 2. 차원 변환 (Samples, Frames, 33, 9)
    expected_feat = NUM_NODES * NUM_FEATURES
    if X_padded.shape[2] != expected_feat:
        raise ValueError(f"입력 피처 수 불일치: 기대 {expected_feat}, 현재 {X_padded.shape[2]}")

    X_reshaped = X_padded.reshape(
        (X_padded.shape[0], X_padded.shape[1], NUM_NODES, NUM_FEATURES)
    )
    # Ablation: A/B/C/D (full=9)
    # A: 좌표 3채널, B: 좌표+속도 6채널, C: 좌표+속도+amp/var 8채널, D: 풀 9채널(좌표+속도+amp/var+각도)
    X_reshaped = apply_ablation(X_reshaped, ablation or 'D')

    return X_reshaped, np.array(y_reg, dtype=np.float32), sample_ids


def train_pose_model(
    processed_data_path,
    model_save_path,
    label_dir="HospitalData/JSON",
    ablation=None,
    weights=None,
    max_len=390,
    epochs=20,
    batch_size=4,
):
    """
    데이터 로드, 회귀 모델 빌드, 학습 실행 (gait UPDRS 예측 전용)
    """
    X, y_reg, sample_ids = load_data(processed_data_path, label_dir, ablation=ablation, weights=weights, max_len=max_len)

    X_train, X_val, y_reg_train, y_reg_val, ids_train, ids_val = train_test_split(
        X, y_reg, sample_ids, test_size=0.1, random_state=42
    )

    print(f"X_train shape: {X_train.shape}")  # (..., max_len, 33, ch)
    print(f"X_val shape: {X_val.shape}")      # (..., max_len, 33, ch)

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
    model = build_pose_model(input_shape, optimizer=optimizer)

    callbacks = [
        ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        X_train, y_reg_train,
        validation_data=(X_val, y_reg_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    print(f"모델 학습 완료. 최적 모델이 {model_save_path} 에 저장되었습니다.")

    # main.py에서 평가/추론에 활용할 수 있도록 validation set 반환
    return model, history, (X_val, y_reg_val, ids_val)
