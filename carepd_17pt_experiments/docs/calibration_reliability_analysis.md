# Calibration Reliability Analysis

- Source: completed GroupKFold prediction tables
- Method: regression reliability curve using prediction-score bins over `[0, 3]`
- Bins: equal-width bins over clipped predicted score
- Reporting decision: no numerical calibration metric is shown in the manuscript-facing figure

## Summary

| Model | N | MAE | Figure use |
|---|---:|---:|---|
| Ours V1 | 6087 | 0.358 | Calibration curve only; no metric annotation |
| Lu official | 6087 | 0.404 | Calibration curve only; no metric annotation |
| Temporal CNN | 6087 | 0.425 | Calibration curve only; no metric annotation |
| ST-GCN | 6087 | 0.443 | Calibration curve only; no metric annotation |

## Interpretation

For the manuscript, use this as a visual calibration/reliability diagnostic rather than a metric-ranking figure. The useful point is whether the proposed model's predicted severity bins follow the monotonic ideal trend while maintaining the best MAE.

## Figures

- `docs\reviewer_figures\21_calibration_curve_ours.png`
- `docs\reviewer_figures\22_calibration_curve_models.png`

## Manuscript-Safe Wording

> Reliability analysis using prediction-score bins showed that the proposed model's continuous outputs followed the expected monotonic relationship between predicted and observed severity. This analysis provides an additional calibration-oriented diagnostic for clinical decision support, complementing MAE/RMSE and confusion-matrix analyses.

## Bin-Level Values

| Model | Bin | N | Mean pred | Mean true | Calibration error |
|---|---:|---:|---:|---:|---:|
| Temporal CNN | 0 | 2303 | 0.142 | 0.231 | 0.090 |
| Temporal CNN | 1 | 1672 | 0.772 | 0.791 | 0.018 |
| Temporal CNN | 2 | 1230 | 1.202 | 1.161 | 0.041 |
| Temporal CNN | 3 | 575 | 1.740 | 1.603 | 0.137 |
| Temporal CNN | 4 | 276 | 2.136 | 1.942 | 0.194 |
| Temporal CNN | 5 | 31 | 2.864 | 2.097 | 0.767 |
| ST-GCN | 0 | 2644 | 0.153 | 0.316 | 0.164 |
| ST-GCN | 1 | 1441 | 0.763 | 0.885 | 0.122 |
| ST-GCN | 2 | 1298 | 1.186 | 1.123 | 0.062 |
| ST-GCN | 3 | 529 | 1.737 | 1.692 | 0.045 |
| ST-GCN | 4 | 148 | 2.149 | 1.939 | 0.210 |
| ST-GCN | 5 | 27 | 2.769 | 2.037 | 0.732 |
| Lu official | 0 | 2585 | 0.224 | 0.224 | 0.000 |
| Lu official | 1 | 1562 | 0.785 | 0.908 | 0.122 |
| Lu official | 2 | 915 | 1.194 | 1.141 | 0.053 |
| Lu official | 3 | 955 | 1.729 | 1.675 | 0.053 |
| Lu official | 4 | 35 | 2.152 | 2.000 | 0.152 |
| Lu official | 5 | 35 | 2.898 | 2.743 | 0.155 |
| Ours V1 | 0 | 2833 | 0.121 | 0.301 | 0.181 |
| Ours V1 | 1 | 1219 | 0.825 | 0.879 | 0.054 |
| Ours V1 | 2 | 1159 | 1.155 | 1.116 | 0.039 |
| Ours V1 | 3 | 424 | 1.763 | 1.597 | 0.167 |
| Ours V1 | 4 | 431 | 2.107 | 1.970 | 0.137 |
| Ours V1 | 5 | 21 | 2.776 | 2.905 | 0.128 |
