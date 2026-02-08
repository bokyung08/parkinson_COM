import json
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from ..models.model_builder import build_pose_model

# Spatio-Temporal Transformer dimensions
NUM_NODES = 33
NUM_FEATURES = 9  # coords(3) + velocity(3) + accel(3)
GAIT_ITEM_WEIGHTS = {'10': 1.0, '11': 1.0, '12': 1.0, '13': 1.0, '14': 1.0}


def _total_updrs_score(items_dict):
    total = 0
    for v in items_dict.values():
        if isinstance(v, list):
            total += sum(v)
        else:
            total += v
    return total


def _gait_updrs_score(items_dict, weights=None):
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


def load_labels(json_dir, weights=None):
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
            gait = _gait_updrs_score(items, weights=weights)
            total = _total_updrs_score(items)
            labels[pid] = {"gait_updrs": gait, "total_updrs": total}
    return labels


def load_data(processed_data_path, label_dir, ablation=None, weights=None, max_len=390):
    X_data = []
    y_reg = []
    sample_ids = []

    label_map = load_labels(label_dir, weights=weights)

    for root, _, files in os.walk(processed_data_path):
        for file_name in files:
            if not file_name.endswith('_2_pose.npy'):
                continue
            npy_path = os.path.join(root, file_name)
            pose_data = np.load(npy_path)  # (Frames, 33*9)

            stem = file_name.replace('_pose.npy', '')
            patient_id = stem.rsplit('_', 1)[0]

            label_info = label_map.get(patient_id)
            if not label_info:
                print(f"[WARN] missing label for file: {file_name}")
                continue

            X_data.append(pose_data)
            y_reg.append(label_info["gait_updrs"])
            sample_ids.append(file_name.replace('_pose.npy', ''))

    if len(X_data) == 0:
        raise ValueError("No data loaded. Check paths and labels.")

    X_padded = pad_sequences(X_data, maxlen=max_len, padding='post', dtype='float32')

    expected_feat = NUM_NODES * NUM_FEATURES
    if X_padded.shape[2] != expected_feat:
        raise ValueError(f"Feature mismatch. Expected {expected_feat}, got {X_padded.shape[2]}")

    X_reshaped = X_padded.reshape(
        (X_padded.shape[0], X_padded.shape[1], NUM_NODES, NUM_FEATURES)
    )

    if ablation == 'A':
        X_reshaped = X_reshaped[..., :3]
    elif ablation == 'B':
        X_reshaped = X_reshaped[..., :6]
    elif ablation == 'C':
        X_reshaped = X_reshaped[..., :8]

    return X_reshaped, np.array(y_reg, dtype=np.float32), sample_ids


def train_pose_model(processed_data_path, model_save_path, label_dir="HospitalData/JSON", ablation=None, weights=None, max_len=390):
    X, y_reg, sample_ids = load_data(processed_data_path, label_dir, ablation=ablation, weights=weights, max_len=max_len)

    X_train, X_val, y_reg_train, y_reg_val, ids_train, ids_val = train_test_split(
        X, y_reg, sample_ids, test_size=0.1, random_state=42
    )

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
        epochs=100,
        batch_size=16,
        callbacks=callbacks
    )

    return model, history, (X_val, y_reg_val, ids_val)
