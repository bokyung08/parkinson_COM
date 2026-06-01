# Scale Robustness Candidate Summary

This table summarizes the completed full 5-fold scale-robustness runs.
All candidates use the same Ours V1 architecture; differences are limited to
input scale normalization and optional train-time scale augmentation.

| candidate | scale_normalization | original_mae | original_rmse | mean_scale_delta_mae_pct | max_scale_delta_mae_pct | mean_realistic_scale_delta_mae_pct | max_realistic_scale_delta_mae_pct | mean_abs_translation_delta_mae_pct | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| median_bone_aug_moderate | median_bone | 0.366 | 0.567 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | Strong candidate |
| hip_width | hip_width | 0.380 | 0.556 | -0.000 | 0.000 | -0.000 | 0.000 | 0.000 | Strong candidate |
| scale_aug_moderate | none | 0.402 | 0.605 | 0.660 | 3.399 | 0.025 | 0.568 | 0.000 | Strong candidate |

## Reporting Rule

- The recommended robustness operating point is `median_bone_aug_moderate`.
- Keep the original Ours V1 D model in the main performance table if the goal is
  best MAE.
- Use `median_bone_aug_moderate` in the COM robustness subsection when making a
  stronger camera-distance robustness claim.
- Do not describe this as a new architecture. It is the proposed architecture
  with median-bone body-scale normalization and moderate scale augmentation.
