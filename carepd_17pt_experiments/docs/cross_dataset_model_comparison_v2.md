# Cross-Dataset Model Comparison

- Protocols: CNUH -> CARE-PD and CARE-PD -> CNUH
- Fine-tuning/adaptation: none
- Test-set checkpoint selection: none
- Epochs: `80`
- Device: `cuda`

## Transfer Results

| Category | Model | Train | Test | N train | N test | MAE | RMSE | MedAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SOTA | MotionBERT-Lite (81-frame) | CNUH | CARE-PD | 21 | 6066 | 0.889 | 1.078 | 0.841 |
| SOTA | MotionBERT-Lite (81-frame) | CARE-PD | CNUH | 6066 | 21 | 0.898 | 1.082 | 0.792 |
| SOTA | MotionAGFormer-XS pretrained | CNUH | CARE-PD | 21 | 6066 | 0.740 | 0.875 | 0.567 |
| SOTA | MotionAGFormer-XS pretrained | CARE-PD | CNUH | 6066 | 21 | 0.921 | 1.152 | 0.729 |

## MAE Pivot

| category | display_model | carepd_to_cnuh | cnuh_to_carepd |
| --- | --- | --- | --- |
| SOTA | MotionAGFormer-XS pretrained | 0.921 | 0.740 |
| SOTA | MotionBERT-Lite (81-frame) | 0.898 | 0.889 |

## Interpretation

Use this table to determine whether Ours is relatively more robust than ST-GCN and the Lu official-architecture baseline under strict zero-shot transfer. Even if all models degrade, Ours can be framed favorably if it has the lowest transfer MAE or the smallest degradation relative to its combined GroupKFold result.
