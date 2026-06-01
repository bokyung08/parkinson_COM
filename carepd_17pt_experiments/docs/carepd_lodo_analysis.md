# CARE-PD Leave-One-Dataset-Out Validation

- Dataset: CARE-PD only
- Split: hold out one CARE-PD source dataset/cohort at a time
- Fine-tuning/adaptation: none
- Epochs: `80`

## Summary

| Model | N | Held-out datasets | MAE | RMSE | MedAE |
| --- | --- | --- | --- | --- | --- |
| Ours V1 | 6066 | 4 | 0.620 | 0.813 | 0.508 |

## Held-Out Cohort Metrics

| Model | Held-out dataset | N train | N test | MAE | RMSE | MedAE |
| --- | --- | --- | --- | --- | --- | --- |
| Ours V1 | 3DGait | 5976 | 90 | 0.775 | 0.947 | 0.847 |
| Ours V1 | BMCLab | 2171 | 3895 | 0.663 | 0.844 | 0.528 |
| Ours V1 | PD-GaM | 4366 | 1700 | 0.495 | 0.724 | 0.236 |
| Ours V1 | T-SDU-PD | 5685 | 381 | 0.692 | 0.836 | 0.707 |

## Interpretation

This experiment is a stronger external generalization test within CARE-PD than random subject-level folds because the held-out cohort has no sequences represented during training.

The overall CARE-PD leave-one-dataset-out result is `MAE = 0.620`, `RMSE =
0.813`, and `MedAE = 0.508`. This is worse than the CARE-PD subset result under
combined subject-level GroupKFold (`MAE = 0.356`, `RMSE = 0.562`), but better
than strict CNUH -> CARE-PD zero-shot transfer (`MAE = 0.747`, `RMSE = 0.882`).
This pattern is important:

- Subject-level GroupKFold is easiest because each CARE-PD source cohort is
  represented during training.
- CARE-PD leave-one-dataset-out is harder because an entire cohort/site is
  unseen during training.
- CNUH -> CARE-PD zero-shot is hardest because both the clinical site and pose
  representation pipeline differ.

## Comparison With Other Generalization Protocols

| Protocol | Train/Test condition | MAE | RMSE | MedAE | Interpretation |
|---|---|---:|---:|---:|---|
| Combined GroupKFold, CARE-PD subset | CARE-PD cohorts represented in train folds | 0.356 | 0.562 | 0.146 | Best expected within-distribution performance |
| CARE-PD LODO | One CARE-PD cohort fully held out | 0.620 | 0.813 | 0.508 | Moderate cohort/site shift within CARE-PD |
| CNUH -> CARE-PD zero-shot | CNUH only to CARE-PD | 0.747 | 0.882 | 0.921 | Strong cross-site and representation shift |

Relative to the CARE-PD subset GroupKFold result, LODO increases MAE by `0.264`
score units. Relative to strict CNUH -> CARE-PD zero-shot transfer, LODO reduces
MAE by `0.128` score units. Thus, multi-cohort CARE-PD training improves
generalization to unseen CARE-PD cohorts, but it does not eliminate cohort-level
domain shift.

## Cohort-Level Interpretation

| Held-out cohort | N test | MAE | RMSE | Interpretation |
|---|---:|---:|---:|---|
| 3DGait | 90 | 0.775 | 0.947 | Hardest held-out cohort; small N makes the estimate unstable |
| BMCLab | 3,895 | 0.663 | 0.844 | Largest held-out cohort; dominates the aggregate LODO metric |
| PD-GaM | 1,700 | 0.495 | 0.724 | Easiest held-out cohort |
| T-SDU-PD | 381 | 0.692 | 0.836 | Moderate-to-hard held-out cohort |

The large variation across held-out cohorts indicates that CARE-PD is not a
single homogeneous distribution. Camera setup, walking protocol, pose quality,
site population, and annotation harmonization can all change the effective
input-output mapping.

## Manuscript-Safe Wording

> In the CARE-PD leave-one-dataset-out evaluation, the proposed model achieved
> MAE = 0.620 and RMSE = 0.813 across four held-out source cohorts. Performance
> was lower than subject-level GroupKFold but better than CNUH-to-CARE-PD
> zero-shot transfer, indicating that multi-cohort training improves
> generalization to unseen CARE-PD cohorts while a measurable cohort-level
> domain gap remains.
