# Selective Prediction Analysis

- Input: existing GroupKFold prediction tables
- No retraining is performed
- Selection rule: keep predictions farthest from rounded-score decision boundaries `0.5, 1.5, 2.5`
- Interpretation: lower retained-set MAE at lower coverage supports a clinician-review workflow where uncertain cases are flagged

## Ours V1 Coverage Curve

| display_model | coverage | n_kept | mae | rmse | medae | rounded_accuracy | mae_reduction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ours V1 | 1.000 | 6087 | 0.358 | 0.564 | 0.147 | 0.711 | 0.000 |
| Ours V1 | 0.900 | 5478 | 0.325 | 0.540 | 0.116 | 0.744 | 9.177 |
| Ours V1 | 0.800 | 4870 | 0.288 | 0.510 | 0.088 | 0.774 | 19.402 |
| Ours V1 | 0.700 | 4261 | 0.260 | 0.493 | 0.064 | 0.793 | 27.162 |
| Ours V1 | 0.600 | 3652 | 0.234 | 0.477 | 0.045 | 0.811 | 34.482 |
| Ours V1 | 0.500 | 3044 | 0.211 | 0.462 | 0.030 | 0.825 | 40.945 |

## Model Comparison at 80% Coverage

| category | display_model | coverage | n_kept | mae | rmse | medae | rounded_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Proposed | Ours V1 | 0.800 | 4870 | 0.288 | 0.510 | 0.088 | 0.774 |
| SOTA | MotionAGFormer-XS | 0.800 | 4870 | 0.351 | 0.612 | 0.054 | 0.700 |
| SOTA | Lu official | 0.800 | 4870 | 0.369 | 0.527 | 0.207 | 0.720 |
| SOTA | MotionBERT-Lite (81-frame) | 0.800 | 4870 | 0.384 | 0.578 | 0.149 | 0.674 |
| SOTA | ST-GCN | 0.800 | 4870 | 0.385 | 0.575 | 0.176 | 0.704 |

## Manuscript-Safe Wording

> Selective prediction analysis showed that cases far from score-boundary thresholds had lower prediction error, suggesting that boundary-proximal cases can be flagged for clinician review rather than automatically accepted.
