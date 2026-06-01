# Dataset-Wise Breakdown Under Combined GroupKFold

- Input: existing subject-level GroupKFold predictions
- Purpose: show whether the combined-domain model performs consistently on CARE-PD and CNUH test samples.
- No additional training required.

## Overall Model Ranking

| Category | Model | N | MAE | RMSE | MedAE |
| --- | --- | --- | --- | --- | --- |
| Proposed | Ours V1 | 6087 | 0.358 | 0.564 | 0.147 |
| SOTA | Lu official | 6087 | 0.404 | 0.543 | 0.307 |
| Deep Learning | Temporal CNN | 6087 | 0.425 | 0.594 | 0.287 |
| SOTA | ST-GCN | 6087 | 0.443 | 0.623 | 0.274 |
| Classical ML | SVR | 6087 | 0.492 | 0.639 | 0.386 |
| Classical ML | Random Forest | 6087 | 0.510 | 0.659 | 0.417 |
| Classical ML | Shallow MLP | 6087 | 0.544 | 0.708 | 0.423 |
| Classical ML | Ridge | 6087 | 0.570 | 0.759 | 0.446 |

## Dataset-Wise Metrics

| Category | Model | Dataset | N | Subjects | MAE | RMSE | MedAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Proposed | Ours V1 | CAREPD | 6066 | 110 | 0.356 | 0.562 | 0.146 |
| SOTA | Lu official | CAREPD | 6066 | 110 | 0.403 | 0.540 | 0.305 |
| Deep Learning | Temporal CNN | CAREPD | 6066 | 110 | 0.420 | 0.581 | 0.286 |
| SOTA | ST-GCN | CAREPD | 6066 | 110 | 0.442 | 0.621 | 0.273 |
| Classical ML | SVR | CAREPD | 6066 | 110 | 0.491 | 0.638 | 0.384 |
| Classical ML | Random Forest | CAREPD | 6066 | 110 | 0.510 | 0.658 | 0.417 |
| Classical ML | Shallow MLP | CAREPD | 6066 | 110 | 0.542 | 0.705 | 0.423 |
| Classical ML | Ridge | CAREPD | 6066 | 110 | 0.562 | 0.732 | 0.444 |
| Classical ML | Random Forest | CNUH | 21 | 21 | 0.768 | 0.953 | 0.807 |
| Proposed | Ours V1 | CNUH | 21 | 21 | 0.793 | 0.945 | 0.987 |
| Classical ML | SVR | CNUH | 21 | 21 | 0.826 | 0.965 | 0.666 |
| SOTA | Lu official | CNUH | 21 | 21 | 0.862 | 1.031 | 0.724 |
| SOTA | ST-GCN | CNUH | 21 | 21 | 0.879 | 1.008 | 0.938 |
| Classical ML | Shallow MLP | CNUH | 21 | 21 | 0.972 | 1.269 | 0.590 |
| Deep Learning | Temporal CNN | CNUH | 21 | 21 | 1.624 | 2.199 | 1.214 |
| Classical ML | Ridge | CNUH | 21 | 21 | 2.771 | 3.529 | 2.271 |

## Ours V1 Dataset Breakdown

| Dataset | N | Subjects | MAE | RMSE | MedAE |
| --- | --- | --- | --- | --- | --- |
| CAREPD | 6066 | 110 | 0.356 | 0.562 | 0.146 |
| CNUH | 21 | 21 | 0.793 | 0.945 | 0.987 |

## Interpretation

This analysis separates the main combined GroupKFold result by dataset. It should be used alongside zero-shot transfer results: if combined training performs well on both datasets while zero-shot transfer is poor, the evidence supports a domain-exposure interpretation rather than an architecture failure.

Source outputs:

```text
results\dataset_wise_breakdown\overall_metrics.csv
results\dataset_wise_breakdown\dataset_metrics.csv
results\dataset_wise_breakdown\dataset_mae_pivot.csv
```
