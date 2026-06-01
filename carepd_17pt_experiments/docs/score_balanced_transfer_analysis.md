# Score-Balanced Cross-Dataset Transfer Analysis

- Input: existing zero-shot cross-dataset prediction files
- No additional training required.
- Balanced metrics are the unweighted average of per-score metrics across available true classes.

## Summary

| Protocol | Train | Test | N | Original MAE | Original RMSE | Balanced MAE | Balanced RMSE | Balanced - Original MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnuh_to_carepd | CNUH | CARE-PD | 6066 | 0.747 | 0.882 | 1.022 | 1.024 | 0.275 |
| carepd_to_cnuh | CARE-PD | CNUH | 21 | 1.014 | 1.170 | 1.025 | 1.034 | 0.010 |

## Per-Class Metrics

| Protocol | Train | Test | True score | N | MAE | RMSE | MedAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cnuh_to_carepd | CNUH | CARE-PD | 0 | 2608 | 1.161 | 1.162 | 1.166 |
| cnuh_to_carepd | CNUH | CARE-PD | 1 | 2175 | 0.144 | 0.148 | 0.153 |
| cnuh_to_carepd | CNUH | CARE-PD | 2 | 1239 | 0.894 | 0.896 | 0.873 |
| cnuh_to_carepd | CNUH | CARE-PD | 3 | 44 | 1.890 | 1.891 | 1.863 |
| carepd_to_cnuh | CARE-PD | CNUH | 0 | 7 | 1.730 | 1.730 | 1.744 |
| carepd_to_cnuh | CARE-PD | CNUH | 1 | 8 | 0.813 | 0.847 | 0.735 |
| carepd_to_cnuh | CARE-PD | CNUH | 2 | 5 | 0.283 | 0.285 | 0.274 |
| carepd_to_cnuh | CARE-PD | CNUH | 3 | 1 | 1.273 | 1.273 | 1.273 |

## Interpretation

Use this analysis to separate class-imbalance effects from transfer failure. If balanced MAE is much larger than original MAE, the original transfer metric is being softened by the target score distribution.
