# Cross-Dataset Model Comparison

- Protocols: CNUH -> CARE-PD and CARE-PD -> CNUH
- Fine-tuning/adaptation: none
- Test-set checkpoint selection: none
- Epochs: `80`
- Device: `cuda`

## Transfer Results

| Category | Model | Train | Test | N train | N test | MAE | RMSE | MedAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SOTA | ST-GCN + bounded head | CNUH | CARE-PD | 21 | 6066 | 2.072 | 2.201 | 1.974 |

## MAE Pivot

| category | display_model | cnuh_to_carepd |
| --- | --- | --- |
| SOTA | ST-GCN + bounded head | 2.072 |

## Interpretation

Use this table to determine whether Ours is relatively more robust than ST-GCN and the Lu official-architecture baseline under strict zero-shot transfer. Even if all models degrade, Ours can be framed favorably if it has the lowest transfer MAE or the smallest degradation relative to its combined GroupKFold result.
