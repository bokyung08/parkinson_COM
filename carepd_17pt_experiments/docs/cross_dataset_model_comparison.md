# Cross-Dataset Model Comparison

- Protocols: CNUH -> CARE-PD and CARE-PD -> CNUH
- Fine-tuning/adaptation: none
- Test-set checkpoint selection: none
- Epochs: `80`
- Device: `cuda`

## Transfer Results

| Category | Model | Train | Test | N train | N test | MAE | RMSE | MedAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Proposed | Ours V1 | CNUH | CARE-PD | 21 | 6066 | 0.747 | 0.882 | 0.921 |
| Proposed | Ours V1 | CARE-PD | CNUH | 6066 | 21 | 0.910 | 1.034 | 0.639 |
| SOTA | ST-GCN | CNUH | CARE-PD | 21 | 6066 | 8.346 | 9.737 | 6.734 |
| SOTA | ST-GCN | CARE-PD | CNUH | 6066 | 21 | 1.203 | 1.385 | 1.119 |
| SOTA | Lu official | CNUH | CARE-PD | 21 | 6066 | 0.898 | 1.016 | 0.596 |
| SOTA | Lu official | CARE-PD | CNUH | 6066 | 21 | 0.865 | 1.027 | 0.735 |

## MAE Pivot

| category | display_model | carepd_to_cnuh | cnuh_to_carepd |
| --- | --- | --- | --- |
| Proposed | Ours V1 | 0.910 | 0.747 |
| SOTA | Lu official | 0.865 | 0.898 |
| SOTA | ST-GCN | 1.203 | 8.346 |

## Interpretation

All models degrade under strict zero-shot transfer, but their failure modes are
different.

- CNUH -> CARE-PD: Ours V1 has the lowest MAE and RMSE (`MAE = 0.747`,
  `RMSE = 0.882`). Lu official is second (`MAE = 0.898`, `RMSE = 1.016`).
  ST-GCN is unstable in this tiny-source setting (`MAE = 8.346`), likely
  because the unbounded regression head extrapolates poorly when trained on
  only 21 CNUH samples.
- CARE-PD -> CNUH: Lu official has the lowest MAE (`0.865`), while Ours V1 is
  close (`0.910`) and has a slightly lower MedAE (`0.639` vs. `0.735`). ST-GCN
  remains worse (`MAE = 1.203`).
- Averaged across both transfer directions, Ours V1 has the lowest MAE
  (`0.829`) compared with Lu official (`0.882`) and ST-GCN (`4.774`).

Manuscript-safe wording:

> Under strict zero-shot transfer, all skeleton-based models showed substantial
> domain degradation. The proposed bounded model achieved the best average
> transfer MAE across the two directions and was most stable in the small-source
> CNUH -> CARE-PD setting, whereas Lu official was slightly better in the
> CARE-PD -> CNUH direction. These results support the use of bounded
> graph-temporal regression as a comparatively stable transfer baseline, while
> still indicating that site calibration or domain adaptation is required for
> deployment.
