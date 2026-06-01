# Per-Class Error and Confusion Matrix

- Source: completed prediction files under `results/`
- Primary model: Ours V1 from `groupkfold_h36m17_ours_lu_official_cuda`
- Prediction classes: regression output rounded to nearest integer and clipped to `[0, 3]`

## Ours V1 Per-Class MAE/RMSE

| Score | N | MAE | RMSE | Mean prediction | Prediction SD |
| --- | --- | --- | --- | --- | --- |
| 0 | 2615 | 0.225 | 0.395 | 0.225 | 0.325 |
| 1 | 2183 | 0.342 | 0.482 | 0.892 | 0.470 |
| 2 | 1244 | 0.649 | 0.889 | 1.416 | 0.671 |
| 3 | 45 | 0.738 | 0.930 | 2.262 | 0.566 |

## Ours V1 Confusion Matrix, Counts

| True score | Pred 0 | Pred 1 | Pred 2 | Pred 3 |
| --- | --- | --- | --- | --- |
| 0 | 2160 | 444 | 11 | 0 |
| 1 | 492 | 1509 | 182 | 0 |
| 2 | 181 | 419 | 642 | 2 |
| 3 | 0 | 6 | 20 | 19 |

Rows are true scores and columns are rounded predictions.

## Ours V1 Confusion Matrix, Row-Normalized

| True score | Pred 0 | Pred 1 | Pred 2 | Pred 3 |
| --- | --- | --- | --- | --- |
| 0 | 0.826 | 0.170 | 0.004 | 0.000 |
| 1 | 0.225 | 0.691 | 0.083 | 0.000 |
| 2 | 0.145 | 0.337 | 0.516 | 0.002 |
| 3 | 0.000 | 0.133 | 0.444 | 0.422 |

## Per-Class Metrics for All Final Models

| category | model | true_class | n | mae | rmse | mean_pred | std_pred |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Classical ML | Random Forest | 0 | 2615 | 0.551 | 0.669 | 0.551 | 0.380 |
| Classical ML | Random Forest | 1 | 2183 | 0.334 | 0.413 | 0.953 | 0.410 |
| Classical ML | Random Forest | 2 | 1244 | 0.714 | 0.911 | 1.291 | 0.573 |
| Classical ML | Random Forest | 3 | 45 | 1.109 | 1.239 | 1.891 | 0.552 |
| Classical ML | Ridge | 0 | 2615 | 0.514 | 0.676 | 0.412 | 0.535 |
| Classical ML | Ridge | 1 | 2183 | 0.448 | 0.607 | 0.820 | 0.580 |
| Classical ML | Ridge | 2 | 1244 | 0.882 | 1.075 | 1.343 | 0.851 |
| Classical ML | Ridge | 3 | 45 | 1.060 | 1.287 | 2.646 | 1.238 |
| Classical ML | SVR | 0 | 2615 | 0.464 | 0.577 | 0.413 | 0.402 |
| Classical ML | SVR | 1 | 2183 | 0.399 | 0.509 | 0.851 | 0.487 |
| Classical ML | SVR | 2 | 1244 | 0.695 | 0.894 | 1.354 | 0.618 |
| Classical ML | SVR | 3 | 45 | 1.065 | 1.148 | 1.935 | 0.426 |
| Classical ML | Shallow MLP | 0 | 2615 | 0.520 | 0.668 | 0.448 | 0.496 |
| Classical ML | Shallow MLP | 1 | 2183 | 0.438 | 0.564 | 0.867 | 0.548 |
| Classical ML | Shallow MLP | 2 | 1244 | 0.761 | 0.941 | 1.405 | 0.729 |
| Classical ML | Shallow MLP | 3 | 45 | 1.069 | 1.378 | 2.135 | 1.073 |
| Deep Learning | Temporal CNN | 0 | 2615 | 0.367 | 0.535 | 0.316 | 0.432 |
| Deep Learning | Temporal CNN | 1 | 2183 | 0.366 | 0.487 | 0.913 | 0.479 |
| Deep Learning | Temporal CNN | 2 | 1244 | 0.622 | 0.799 | 1.457 | 0.587 |
| Deep Learning | Temporal CNN | 3 | 45 | 1.174 | 1.382 | 1.910 | 0.850 |
| Proposed | Ours V1 | 0 | 2615 | 0.225 | 0.395 | 0.225 | 0.325 |
| Proposed | Ours V1 | 1 | 2183 | 0.342 | 0.482 | 0.892 | 0.470 |
| Proposed | Ours V1 | 2 | 1244 | 0.649 | 0.889 | 1.416 | 0.671 |
| Proposed | Ours V1 | 3 | 45 | 0.738 | 0.930 | 2.262 | 0.566 |
| SOTA | Lu official | 0 | 2615 | 0.336 | 0.462 | 0.336 | 0.317 |
| SOTA | Lu official | 1 | 2183 | 0.361 | 0.450 | 0.914 | 0.442 |
| SOTA | Lu official | 2 | 1244 | 0.615 | 0.777 | 1.403 | 0.498 |
| SOTA | Lu official | 3 | 45 | 0.614 | 0.925 | 2.386 | 0.693 |
| SOTA | ST-GCN | 0 | 2615 | 0.323 | 0.475 | 0.279 | 0.385 |
| SOTA | ST-GCN | 1 | 2183 | 0.366 | 0.472 | 0.876 | 0.456 |
| SOTA | ST-GCN | 2 | 1244 | 0.790 | 0.965 | 1.283 | 0.646 |
| SOTA | ST-GCN | 3 | 45 | 1.583 | 1.657 | 1.417 | 0.491 |

## Manuscript Note

The model is most accurate for score 0 and score 1, which are also the most common classes. Score 3 remains the hardest class because it has fewer samples and larger prediction dispersion. When reporting the confusion matrix, state that continuous regression outputs were rounded only for interpretability; MAE/RMSE remain the primary metrics.

## COM Robustness Status

The current result folders do not contain saved fold checkpoints (`.pt`, `.pth`, or `.ckpt`). Therefore COM robustness cannot be recomputed from the existing prediction tables alone, because scale/translation perturbations require rerunning the trained model on modified joint coordinates. Future runs should save the best fold checkpoint if COM robustness is required without retraining.
