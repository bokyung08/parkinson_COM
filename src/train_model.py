import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from .model_builder import build_pose_model

def train_pose_model(processed_data_path, model_save_path):
    """Pose 데이터를 불러와 학습 및 검증"""
    class_names = ['healthy', 'disease']
    X, y = [], []
    
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(processed_data_path, class_name)
        files = [f for f in os.listdir(class_path) if f.endswith('_pose.npy')]
        for f in files:
            data = np.load(os.path.join(class_path, f))
            X.append(data)
            y.append(label)

    X_padded = pad_sequences(X, dtype='float32', padding='post', truncating='post')
    y = np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(X_padded, y, test_size=0.2, stratify=y, random_state=42)

    model = build_pose_model((X_train.shape[1], X_train.shape[2]))

    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=20,
                        batch_size=16,
                        callbacks=[checkpoint, early_stop])
    return model, history, (X_val, y_val)
