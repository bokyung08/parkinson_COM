import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from .model_builder import SpatialTransformerBlock


def evaluate_and_plot(model_path, X_val, y_val, class_names, output_dir):

    custom_objects = {'SpatialTransformerBlock': SpatialTransformerBlock}
    model = load_model(model_path, custom_objects=custom_objects)

    print("[INFO] 모델 평가 중...")
    y_pred_probs = model.predict(X_val)
    y_pred = (y_pred_probs > 0.5).astype(int)

    # 1. Classification Report (텍스트 파일로 저장)
    print("\n--- 분류 평가 보고서 ---")
    report = classification_report(y_val, y_pred, target_names=class_names)
    print(report)
    
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"[INFO] 평가 보고서가 {report_path} 에 저장되었습니다.")

    # 2. Confusion Matrix (이미지 파일로 저장)
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path)
    print(f"[INFO] 혼동 행렬이 {plot_path} 에 저장되었습니다.")