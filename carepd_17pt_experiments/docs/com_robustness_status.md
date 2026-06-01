# COM Robustness Experiment

- Requested experiment: test-time scale and translation perturbation
- Model: Ours V1, Configuration D
- Perturbations:
  - Scale: `0.70, 0.85, 1.00, 1.15, 1.30`
  - Translation in x: `-0.20, -0.10, 0.00, +0.10, +0.20`
- Combined: `(0.70, -0.20)`, `(1.30, +0.20)`, `(0.85, +0.10)`, `(1.15, -0.10)`
- Metrics: MAE, RMSE, relative degradation, Wilcoxon signed-rank test, rank-biserial effect size

## Current Status

Completed.

The original completed result folders did not contain trained fold checkpoints,
so Configuration D was rerun once with checkpoint saving. The perturbation
analysis then used those saved fold weights and applied perturbations only at
inference time.

Outputs:

```text
docs\com_robustness\
  README.md
  summary.csv
  fold_metrics.csv
  wilcoxon_tests.csv
  predictions.tsv
  figures\
    mae_vs_scale.png
    rmse_vs_scale.png
    mae_vs_translation.png
    relative_degradation.png
```

## Main Finding

COM centering is robust to global horizontal translation but not to scale
changes. This means it should be described as a position-normalization method,
not as a complete camera-distance normalization method when used alone.

| Condition | MAE | RMSE | Delta MAE (%) | Delta RMSE (%) |
|---|---:|---:|---:|---:|
| Original | 0.369 | 0.564 | 0.000 | 0.000 |
| Scale 0.70 | 0.501 | 0.689 | 35.925 | 22.166 |
| Scale 0.85 | 0.416 | 0.639 | 12.733 | 13.259 |
| Scale 1.15 | 0.469 | 0.673 | 27.050 | 19.226 |
| Scale 1.30 | 0.540 | 0.780 | 46.488 | 38.201 |
| Shift -0.20 | 0.369 | 0.564 | -0.000 | 0.000 |
| Shift -0.10 | 0.369 | 0.564 | 0.000 | 0.000 |
| Shift +0.10 | 0.369 | 0.564 | -0.000 | 0.000 |
| Shift +0.20 | 0.369 | 0.564 | -0.000 | 0.000 |

Recommended manuscript wording:

> COM-centered normalization effectively removed global horizontal position
> shifts, while scale perturbations remained challenging. This indicates that
> the current representation is translation-invariant but requires additional
> body-size or bone-length normalization for camera-distance robustness.

## Scale-Robust Follow-Up

Full 5-fold candidate runs have now been completed for scale robustness. The
best trade-off is the same Ours V1 architecture with median-bone scale
normalization and moderate train-time scale augmentation.

| Variant | Scale normalization | Train-time scale augmentation | MAE | RMSE | Max scale Delta MAE (%) | Max translation Delta MAE (%) | Decision |
|---|---|---|---:|---:|---:|---:|---|
| COM-only D checkpoint | none | none | 0.369 | 0.564 | 46.488 | 0.000 | Position-robust only |
| Scale augmentation | none | 0.85-1.15 | 0.402 | 0.605 | 3.399 | 0.000 | Robust but accuracy loss |
| Hip-width normalization | hip width | none | 0.380 | 0.556 | 0.000 | 0.000 | Robust, moderate MAE loss |
| Median-bone normalization + augmentation | median bone length | 0.85-1.15 | 0.366 | 0.567 | 0.000 | 0.000 | Recommended robust operating point |

Interpretation:

- COM-only D should be used to support robustness against global position
  shifts, not full scale invariance.
- Median-bone normalization removes the residual scale sensitivity because the
  entire skeleton is divided by a sequence-level body-size estimate after COM
  centering.
- The median-bone variant is not a new architecture; it is the same proposed
  architecture with a stronger input normalization policy.

Recommended final manuscript wording:

> COM centering removed global horizontal translation effects. To address the
> remaining camera-distance sensitivity, we further evaluated a body-scale
> normalized operating point using median bone length. This variant preserved
> prediction accuracy (MAE = 0.366, RMSE = 0.567) and produced no measurable
> MAE degradation under scale factors from 0.70 to 1.30 or horizontal shifts
> from -0.20 to +0.20.

Detailed outputs:

```text
docs\scale_robustness_full_summary.md
docs\scale_robustness_full\
  scale_aug_moderate\
  median_bone_aug_moderate\
  hip_width\
```

## Step 1: Save D Fold Checkpoints

Run:

```powershell
cd C:\Users\bokyung\Desktop\parkinson_COM\carepd_17pt_experiments
.\scripts\run_ours_d_checkpointed_cuda.cmd
```

This wrote:

```text
results\groupkfold_h36m17_ours_d_checkpointed_cuda\checkpoints\
  ours_fold_01.pt
  ours_fold_02.pt
  ours_fold_03.pt
  ours_fold_04.pt
  ours_fold_05.pt
```

## Step 2: Run Perturbation Inference

This was run:

```powershell
.\scripts\run_com_robustness_cuda.cmd
```

This wrote:

```text
docs\com_robustness\
  README.md
  summary.csv
  fold_metrics.csv
  wilcoxon_tests.csv
  predictions.tsv
  figures\
    mae_vs_scale.png
    rmse_vs_scale.png
    mae_vs_translation.png
    relative_degradation.png
```

## Why Prediction Tables Are Not Enough

Per-class MAE/RMSE and confusion matrices only require:

```text
y_true, y_pred
```

COM robustness requires:

```text
trained model weights + original joint coordinates + perturbed joint coordinates
```

The perturbation is applied before feature construction and inference:

```text
joint coordinates -> perturbation -> COM or raw feature construction -> model -> perturbed prediction
```

Therefore saved predictions cannot be transformed into COM-robustness results.

## Reporting Rule

For the COM-only perturbation analysis, report the values from
`docs\com_robustness\summary.csv`. For the scale-robust operating point, report
`docs\scale_robustness_full_summary.md` and the median-bone-normalized run.
Do not report robustness from the older prediction-only result folders.
