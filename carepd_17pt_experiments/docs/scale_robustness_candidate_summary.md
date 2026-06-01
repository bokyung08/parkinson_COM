# Scale Robustness Candidate Summary

Candidate screening compares COM-normalized inference under scale and translation perturbations.
Use this screen to decide which candidate deserves a full 5-fold manuscript run.

| candidate | scale_normalization | original_mae | original_rmse | mean_scale_delta_mae_pct | max_scale_delta_mae_pct | mean_realistic_scale_delta_mae_pct | max_realistic_scale_delta_mae_pct | mean_abs_translation_delta_mae_pct | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_none | none | 0.404 | 0.559 | 52.090 | 125.772 | 25.605 | 50.380 | 0.000 | Exclude for scale claim |
| median_bone_aug_moderate | median_bone | 0.395 | 0.543 | -0.000 | 0.000 | -0.000 | 0.000 | 0.000 | Strong candidate |
| median_bone | median_bone | 0.437 | 0.585 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | Strong candidate |
| torso | torso | 0.414 | 0.603 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | Strong candidate |
| hip_width | hip_width | 0.396 | 0.560 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | Strong candidate |
| scale_aug_wide | none | 0.397 | 0.569 | 1.315 | 5.762 | 0.253 | 1.097 | 0.000 | Strong candidate |
| scale_aug_moderate | none | 0.375 | 0.522 | 3.028 | 7.950 | 0.900 | 2.030 | 0.000 | Strong candidate |

## Reporting Rule

- Use `Strong candidate` or `Promising` only after confirming the full 5-fold run.
- Do not claim scale robustness from a candidate that has large baseline MAE loss.
- Translation robustness should remain close to zero degradation for all COM-centered candidates.
