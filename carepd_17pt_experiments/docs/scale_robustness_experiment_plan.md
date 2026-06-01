# Scale Robustness Follow-up Experiments

## Why This Is Needed

The current COM-only representation is translation-invariant but not
scale-invariant. This follows directly from:

```text
P' = sP
COM' = sCOM
P'_rel = P' - COM' = s(P - COM)
```

Therefore, COM centering removes global position shifts but keeps the scale
factor. To make a defensible scale-robustness claim, we test additional
mechanisms rather than reinterpreting the COM-only result.

## Completed Immediately: Realistic Scale Range on Existing D Checkpoints

Output:

```text
docs/com_robustness_realistic/
```

COM-only D under narrower perturbations:

| Condition | MAE | RMSE | Delta MAE (%) | Delta RMSE (%) |
|---|---:|---:|---:|---:|
| Original | 0.369 | 0.564 | 0.000 | 0.000 |
| Scale 0.90 | 0.391 | 0.607 | 5.960 | 7.641 |
| Scale 0.95 | 0.374 | 0.580 | 1.339 | 2.733 |
| Scale 1.05 | 0.398 | 0.586 | 7.799 | 3.854 |
| Scale 1.10 | 0.432 | 0.622 | 17.224 | 10.207 |

Interpretation:

- COM-only is acceptable for very small scale shifts, especially around
  `0.95`.
- A `+10%` scale increase is already too damaging for a strong camera-distance
  robustness claim.
- This result can be used only as a limited realistic-range analysis, not as a
  general scale-invariance claim.

## Candidate Mechanisms to Screen

The following candidates are implemented in the screening script:

| Candidate | Mechanism | Purpose |
|---|---|---|
| `baseline_none` | COM-only | Fair 2-fold screen baseline |
| `median_bone` | COM + median bone-length normalization | Strongest scale-invariance candidate |
| `torso` | COM + pelvis-thorax normalization | Simple trunk-size normalization |
| `hip_width` | COM + hip-width normalization | Simple pelvis-size normalization |
| `scale_aug_moderate` | COM-only + train-time scale jitter 0.85-1.15 | Test augmentation-only robustness |
| `scale_aug_wide` | COM-only + train-time scale jitter 0.70-1.30 | Test stronger augmentation |
| `median_bone_aug_moderate` | median bone normalization + moderate jitter | Test combined strategy |

## Completed Diagnostic: Feature-Level Scale Sensitivity

Output:

```text
docs/scale_feature_invariance/
```

The diagnostic confirms the expected mechanism. With COM-only features
(`scale_normalization = none`), coordinate-derived channels change
approximately in proportion to the scale factor. For example, `s = 0.70`
produces roughly 30% relative change in position, velocity, amplitude, and
variability channels. Angle features are nearly unchanged because angles are
scale-invariant by construction.

With body-scale normalization (`median_bone`, `torso`, or `hip_width`), all
coordinate-derived channels show near-zero relative change under scale
perturbation. This supports the use of anthropometric normalization as the
mechanistic fix for camera-distance variation. Predictive performance still
requires the screening/full training runs below.

## Screening Command

Run:

```powershell
cd C:\Users\bokyung\Desktop\parkinson_COM\carepd_17pt_experiments
.\scripts\run_scale_robustness_screen_cuda.cmd
```

This runs 2 folds, 30 epochs per candidate, then evaluates scale and
translation perturbations.

Outputs:

```text
results/screen_scale_robustness_<candidate>/
docs/scale_robustness_screen/<candidate>/
docs/scale_robustness_candidate_summary.md
docs/scale_robustness_candidate_summary.csv
docs/scale_feature_invariance/
```

## Full Selected Run

After screening, run only the promising candidates:

```powershell
.\scripts\run_scale_robustness_full_selected_cuda.cmd
```

Default selected candidates:

```text
median_bone
scale_aug_moderate
median_bone_aug_moderate
```

Outputs:

```text
results/full_scale_robustness_<candidate>/
docs/scale_robustness_full/<candidate>/
docs/scale_robustness_full_summary.md
docs/scale_robustness_full_summary.csv
```

## Filtering Rule

Use a candidate in the manuscript only if both conditions hold:

1. Baseline original MAE does not degrade materially relative to Ours D.
2. Scale perturbation degradation is consistently below the practical
   threshold, preferably under 5-10% in the realistic range and under 10-15%
   in the broader range.

If a method improves scale robustness but worsens original MAE substantially,
report it only as supplementary or leave it out.

## Expected Most Defensible Claim

If `median_bone` or `median_bone_aug_moderate` works:

> COM centering removes global position shifts, while anthropometric
> bone-length normalization provides additional robustness to camera-distance
> induced scale variation.

If augmentation works but baseline accuracy drops:

> Scale augmentation improves robustness to simulated camera-distance changes,
> but introduces an accuracy-robustness trade-off.

If none works cleanly:

> COM centering is retained as a position-normalization mechanism; scale
> robustness remains a limitation and motivates future body-size normalization.
