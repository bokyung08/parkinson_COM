import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

def evaluate_and_plot(model_path, X_val, y_val, class_names, save_dir):
    model = load_model(model_path)
    y_pred_probs = model.predict(X_val)
    y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

    print("\n[Classification Report]")
    print(classification_report(y_val, y_pred, target_names=class_names))

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png")
    plt.show()
