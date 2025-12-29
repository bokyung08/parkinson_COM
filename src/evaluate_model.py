import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .model_builder import build_pose_model


def evaluate_and_plot(model_path, X_val, y_reg_val, ids_val, output_dir):
    input_shape = (X_val.shape[1], X_val.shape[2], X_val.shape[3])
    model = build_pose_model(input_shape)
    _ = model(np.zeros((1,) + input_shape, dtype=np.float32))
    model.load_weights(model_path)

    print("[INFO] 모델 검증 중...")
    y_pred_reg = model.predict(X_val)

    os.makedirs(output_dir, exist_ok=True)

    # Metrics
    mae = mean_absolute_error(y_reg_val, y_pred_reg)
    rmse = mean_squared_error(y_reg_val, y_pred_reg, squared=False)
    # R2
    ss_res = np.sum((y_reg_val - y_pred_reg.flatten()) ** 2)
    ss_tot = np.sum((y_reg_val - np.mean(y_reg_val)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')
    # Pearson r
    if len(y_reg_val) > 1:
        r = np.corrcoef(y_reg_val, y_pred_reg.flatten())[0, 1]
    else:
        r = float('nan')

    with open(os.path.join(output_dir, 'regression_errors.txt'), 'w', encoding='utf-8') as f:
        f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}\nPearson_r: {r:.4f}\n")

    # Scatter plot: 실제 vs 예측
    plt.figure()
    plt.scatter(y_reg_val, y_pred_reg, alpha=0.7)
    lims = [min(y_reg_val.min(), y_pred_reg.min()), max(y_reg_val.max(), y_pred_reg.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel('True Gait UPDRS')
    plt.ylabel('Predicted Gait UPDRS')
    plt.title(f'Gait UPDRS True vs Predicted (MAE={mae:.2f}, RMSE={rmse:.2f})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regression_scatter.png'))
    plt.close()

    # Residual histogram
    residuals = y_pred_reg.flatten() - y_reg_val
    plt.figure()
    plt.hist(residuals, bins=20, alpha=0.8)
    plt.xlabel('Residual (Pred - True)')
    plt.ylabel('Count')
    plt.title('Residual Distribution (Gait UPDRS)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regression_residuals.png'))
    plt.close()

    # True vs Pred 분포
    plt.figure()
    plt.hist(y_reg_val, bins=20, alpha=0.6, label='True')
    plt.hist(y_pred_reg, bins=20, alpha=0.6, label='Pred')
    plt.xlabel('Gait UPDRS')
    plt.ylabel('Count')
    plt.title('True vs Predicted Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_true_vs_pred.png'))
    plt.close()

    # Absolute error CDF
    abs_err = np.abs(residuals)
    sorted_err = np.sort(abs_err)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    plt.figure()
    plt.plot(sorted_err, cdf)
    plt.xlabel('Absolute Error')
    plt.ylabel('CDF')
    plt.title('Absolute Error CDF')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'abs_error_cdf.png'))
    plt.close()

    # Predictions dump (CSV-like TSV)
    pred_path = os.path.join(output_dir, 'predictions.tsv')
    with open(pred_path, 'w', encoding='utf-8') as f:
        f.write("sample_id\ttrue_gait_updrs\tpred_gait_updrs\tabs_error\n")
        for sid, yt, yp in zip(ids_val, y_reg_val, y_pred_reg.flatten()):
            f.write(f"{sid}\t{yt:.6f}\t{yp:.6f}\t{abs(yp-yt):.6f}\n")

    np.savez_compressed(
        os.path.join(output_dir, 'predictions.npz'),
        sample_ids=np.array(ids_val),
        y_reg_true=y_reg_val,
        y_reg_pred=y_pred_reg
    )

    metrics = {
        "reg_mae": float(mae),
        "reg_rmse": float(rmse),
        "reg_r2": float(r2),
        "reg_pearson_r": float(r)
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
