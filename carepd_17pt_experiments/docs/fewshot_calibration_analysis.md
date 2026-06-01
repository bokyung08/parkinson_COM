# Few-Shot Target-Site Calibration

- Input: zero-shot transfer predictions
- Calibration method: affine mapping `y_calibrated = a * y_pred + b`, clipped to `[0, 3]`
- No model retraining or fine-tuning is performed.
- Repeats per setting: `50`

## Summary

| Protocol | Train | Test | Calibration subjects | Repeats | Base MAE | Calibrated MAE | Delta MAE | Delta MAE SD | Base RMSE | Calibrated RMSE | Delta RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| carepd_to_cnuh | CARE-PD | CNUH | 1 | 50 | 1.015 | 1.038 | 0.023 | 0.441 | 1.171 | 1.260 | 0.090 |
| carepd_to_cnuh | CARE-PD | CNUH | 3 | 50 | 1.017 | 1.077 | 0.060 | 0.265 | 1.170 | 1.290 | 0.120 |
| carepd_to_cnuh | CARE-PD | CNUH | 5 | 50 | 1.029 | 0.990 | -0.039 | 0.171 | 1.179 | 1.162 | -0.017 |
| carepd_to_cnuh | CARE-PD | CNUH | 10 | 50 | 0.999 | 0.836 | -0.163 | 0.231 | 1.149 | 0.991 | -0.158 |
| cnuh_to_carepd | CNUH | CARE-PD | 1 | 50 | 0.747 | 0.791 | 0.044 | 0.259 | 0.881 | 1.023 | 0.142 |
| cnuh_to_carepd | CNUH | CARE-PD | 3 | 50 | 0.748 | 0.672 | -0.076 | 0.117 | 0.883 | 0.845 | -0.037 |
| cnuh_to_carepd | CNUH | CARE-PD | 5 | 50 | 0.751 | 0.659 | -0.092 | 0.121 | 0.884 | 0.814 | -0.070 |
| cnuh_to_carepd | CNUH | CARE-PD | 10 | 50 | 0.748 | 0.622 | -0.126 | 0.068 | 0.882 | 0.763 | -0.119 |

## Interpretation

Negative Delta MAE means target-site calibration improved transfer performance. This experiment tests whether the zero-shot domain gap can be reduced with a small number of labeled target-site subjects without retraining the full model.
