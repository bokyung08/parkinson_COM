import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from .model_builder import build_pose_model # model_builder.py의 함수 임포트

# --- Spatio-Temporal Transformer 모델에 맞게 차원 정의 ---
NUM_NODES = 33    # MediaPipe Pose 랜드마크 33개
NUM_FEATURES = 3  # (x, y, z) 3개 좌표

def load_data(processed_data_path):
    """ .npy 파일들을 불러와 X, y 데이터셋 생성 """
    X_data = []
    y_data = []
    labels = {'healthy': 0, 'disease': 1}

    for label, value in labels.items():
        class_dir = os.path.join(processed_data_path, label)
        for file_name in os.listdir(class_dir):
            if file_name.endswith('_pose.npy'):
                npy_path = os.path.join(class_dir, file_name)
                pose_data = np.load(npy_path) # (Frames, 99)
                X_data.append(pose_data)
                y_data.append(value)

    # 1. 시퀀스 길이 맞추기 (패딩)
    # Spatio-Temporal Transformer는 입력 시퀀스 길이가 동일해야 함
    # (일반적으로는 가장 긴 시퀀스 길이를 찾지만, 여기서는 150 프레임으로 가정)
    max_len = 150 # 이 값은 데이터셋에 맞게 조절하세요.
    X_padded = pad_sequences(X_data, maxlen=max_len, padding='post', dtype='float32')
    
    # 2. *** 차원 변경 (핵심 수정 사항) ***
    # (Samples, Frames, 99) -> (Samples, Frames, 33, 3)
    try:
        X_reshaped = X_padded.reshape(
            (X_padded.shape[0], X_padded.shape[1], NUM_NODES, NUM_FEATURES)
        )
    except ValueError as e:
        print(f"Reshape 오류: {e}")
        print(f"데이터 원본 shape (padded): {X_padded.shape}")
        print(f"예상되는 특징 개수 (99)가 실제({X_padded.shape[2]})와 다른지 확인하세요.")
        raise e

    return X_reshaped, np.array(y_data)

def train_pose_model(processed_data_path, model_save_path):
    """
    데이터 로드, 모델 빌드, 학습 수행
    """
    X, y = load_data(processed_data_path)
    
    # 데이터 분리
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"X_train shape: {X_train.shape}") # (..., 150, 33, 3)
    print(f"X_val shape: {X_val.shape}")     # (..., 150, 33, 3)

    # 1. *** 모델 빌드 (핵심 수정 사항) ***
    # 모델에 2D shape가 아닌 3D shape (Frames, Nodes, Features)를 전달
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) 
    model = build_pose_model(input_shape)
    
    model.summary()

    # 2. 콜백 설정
    callbacks = [
        ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min'),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    ]

    # 3. 모델 학습
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        callbacks=callbacks
    )

    print(f"모델 학습 완료. 최적 모델이 {model_save_path} 에 저장되었습니다.")
    
    # main.py에서 평가에 사용할 수 있도록 validation set 반환
    return model, history, (X_val, y_val)