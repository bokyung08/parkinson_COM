# COM Robustness Final Analysis

- Dataset: CNUH + CARE-PD, H36M-compatible 17-joint gait sequences
- Split: subject-level GroupKFold, 5 folds
- Model: Ours V1, Configuration D
- Perturbation timing: inference only
- Target: MDS-UPDRS item 10 gait score, range 0-3

## Question

The experiment tests whether COM-centered preprocessing makes predictions
stable under camera-induced nuisance variation:

- Horizontal translation: the subject appears shifted left or right.
- Coordinate scale: the subject appears smaller or larger due to camera
  distance or body-size variation.

## COM-Only Result

The checkpointed COM-only D run achieved baseline MAE 0.369 and RMSE 0.564.
Horizontal translation produced virtually no change, but scale perturbation
increased error.

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

Interpretation:

- COM centering successfully removes global horizontal position shifts.
- COM centering alone is not enough for camera-distance or body-scale
  robustness.
- Therefore, the safe claim for the original D model is translation robustness,
  not complete scale invariance.

## Scale-Robust Follow-Up

Full 5-fold candidate runs were then completed using the same Ours V1
architecture with stronger input normalization.

| Variant | Scale normalization | Train-time scale augmentation | MAE | RMSE | Max scale Delta MAE (%) | Max translation Delta MAE (%) |
|---|---|---|---:|---:|---:|---:|
| COM-only D checkpoint | none | none | 0.369 | 0.564 | 46.488 | 0.000 |
| Scale augmentation | none | 0.85-1.15 | 0.402 | 0.605 | 3.399 | 0.000 |
| Hip-width normalization | hip width | none | 0.380 | 0.556 | 0.000 | 0.000 |
| Median-bone normalization + augmentation | median bone length | 0.85-1.15 | 0.366 | 0.567 | 0.000 | 0.000 |

The best trade-off is median-bone normalization with moderate scale
augmentation. It keeps accuracy close to the COM-only checkpointed D run while
removing measurable degradation across all tested scale and translation
perturbations.

## Recommended Reporting Decision

- Main performance table: keep original Ours V1 D because it has the best
  overall MAE in the integrated results.
- Robustness subsection: report the median-bone-normalized operating point as
  the scale-robust variant of the same architecture.
- Do not present the median-bone variant as a new model. It is an input
  normalization and augmentation policy for the proposed architecture.

## Manuscript-Ready Section 5.3.2 Draft

To quantitatively assess robustness to camera-induced nuisance variation, we
applied controlled coordinate perturbations to the validation sequences at
inference time without retraining. The COM-only model was nearly invariant to
horizontal translation: shifting all joint coordinates by Delta x in
{-0.20, -0.10, +0.10, +0.20} produced no measurable change in MAE or RMSE.
This confirms that COM centering removes the effect of global subject position
within the image plane.

However, COM centering alone did not fully remove scale sensitivity. When
coordinates were multiplied by scale factors s in {0.70, 0.85, 1.15, 1.30},
MAE increased from 0.369 to 0.416-0.540, corresponding to a relative MAE
increase of 12.7-46.5%. This indicates that translation invariance and
camera-distance invariance are distinct properties: subtracting the COM proxy
removes global offsets but does not normalize body size or apparent camera
distance.

We therefore evaluated a scale-robust operating point using the same proposed
architecture with sequence-level body-scale normalization based on median bone
length and moderate train-time scale augmentation. This variant achieved
MAE = 0.366 and RMSE = 0.567, while producing no measurable MAE degradation
under scale factors from 0.70 to 1.30 or horizontal shifts from -0.20 to
+0.20. These results suggest that COM centering provides position invariance,
whereas median-bone body-scale normalization is required for practical
camera-distance robustness.

## Figures To Use

For the main robustness claim, use the median-bone-normalized figures:

```text
docs/scale_robustness_full/median_bone_aug_moderate/figures/mae_vs_scale.png
docs/scale_robustness_full/median_bone_aug_moderate/figures/rmse_vs_scale.png
docs/scale_robustness_full/median_bone_aug_moderate/figures/mae_vs_translation.png
docs/scale_robustness_full/median_bone_aug_moderate/figures/relative_degradation.png
```

For the limitation analysis of COM-only D, cite:

```text
docs/com_robustness/README.md
docs/com_robustness/summary.csv
```
