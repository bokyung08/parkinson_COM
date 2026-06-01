# Final Integrated Result Figures

Generated from completed result tables under `results/`.

## Recommended Manuscript Use

Use a small number of high-signal figures in the main manuscript. The strongest
combination is performance ranking, ablation, and clinical score-level
interpretability.

### Main Manuscript

| Priority | Figure | Why it helps |
|---:|---|---|
| 1 | `02_mae_ranking.png` | Shows the central claim directly: Ours V1 has the best MAE among ML, DL, SOTA, and proposed models. |
| 2 | `07_ablation_metrics.png` | Supports the design choice for the full hybrid feature set. |
| 3 | `09_ours_per_class_error.png` | Shows score-wise MAE/RMSE and answers reviewer questions about per-class behavior. |
| 4 | `12_ours_confusion_normalized.png` | Gives an interpretable clinical-score view after rounding regression outputs. |

Recommended composite main figure:

```text
Figure X:
  (a) 02_mae_ranking.png
  (b) 07_ablation_metrics.png
  (c) 09_ours_per_class_error.png
  (d) 12_ours_confusion_normalized.png
```

### Supplementary Figures

| Figure | Suggested use |
|---|---|
| `05_fold_mae_by_model.png` | Fold-level robustness/stability. |
| `18_calibration_curve_by_model.png` | Calibration-style behavior across score levels. |
| `20_mae_advantage_vs_baselines.png` | Compact rebuttal-style visualization of MAE gains over selected baselines. |
| `16_dataset_mae_breakdown.png` | Dataset-level breakdown; better as supplementary because CNUH has only 21 samples. |
| `19_class_distribution.png` | Class imbalance explanation. |

### Usually Keep Out of Main Text

| Figure | Reason |
|---|---|
| `03_params_vs_mae.png` | Useful only if model efficiency is emphasized. |
| `04_inference_vs_mae.png` | Useful only if inference speed is emphasized. |
| `13_ours_true_vs_pred.png` | Dense scatter can be visually noisy in the main text. |
| `15_abs_error_distribution_by_model.png` | Diagnostic distribution view; better as supplementary. |

| File | Description |
|---|---|
| `01_main_metric_bars.png` | Grouped MAE/RMSE/MedAE bars for all final comparison models. |
| `02_mae_ranking.png` | Horizontal ranking of final models by MAE. |
| `03_params_vs_mae.png` | Scatter plot of parameter count against MAE. |
| `04_inference_vs_mae.png` | Scatter plot of inference time against MAE. |
| `05_fold_mae_by_model.png` | Fold-level MAE for Ours V1 and key baselines. |
| `06_fold_rmse_by_model.png` | Fold-level RMSE for Ours V1 and key baselines. |
| `07_ablation_metrics.png` | Grouped MAE/RMSE/MedAE bars for Ours V1 ablations. |
| `08_ablation_fold_mae.png` | Fold-level MAE trends for Ours V1 ablations A-D. |
| `09_ours_per_class_error.png` | Per-class MAE/RMSE for Ours V1. |
| `10_per_class_mae_heatmap.png` | Heatmap of MAE by model and true score. |
| `11_ours_confusion_counts.png` | Ours V1 4x4 confusion matrix with count values. |
| `12_ours_confusion_normalized.png` | Ours V1 row-normalized 4x4 confusion matrix. |
| `13_ours_true_vs_pred.png` | Scatter plot of true versus predicted Ours V1 scores. |
| `14_ours_residual_by_true_score.png` | Boxplot of Ours V1 residuals by true score. |
| `15_abs_error_distribution_by_model.png` | Boxplot of absolute error distributions for final models. |
| `17_prediction_distribution_by_score.png` | Boxplot of Ours V1 predictions grouped by true score. |
| `16_dataset_mae_breakdown.png` | CAREPD/CNUH MAE breakdown for Ours V1 and key baselines. |
| `18_calibration_curve_by_model.png` | Calibration-style curve of mean prediction by true score. |
| `19_class_distribution.png` | Target class distribution in the combined evaluation set. |
| `20_mae_advantage_vs_baselines.png` | MAE difference between selected baselines and Ours V1. |
