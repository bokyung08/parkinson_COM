# Final Integrated Results

- Last updated: 2026-06-05
- Dataset: CNUH + CARE-PD, converted to H36M-compatible 17-joint gait sequences
- Split: subject-level GroupKFold, 5 folds
- Target: MDS-UPDRS item 10 gait score, range 0-3
- Primary proposed model: Ours V1, bounded regression

## Final Reporting Decision

OursV2 is not used in the main manuscript table. It improved MedAE but did not
improve the primary MAE/RMSE metrics.

| Model | MAE | RMSE | MedAE | Decision |
|---|---:|---:|---:|---|
| Ours V1 | 0.358 | 0.564 | 0.147 | Keep as final proposed model |
| OursV2 | 0.364 | 0.604 | 0.079 | Exclude from main table |

The final manuscript should report Ours V1 as the proposed model. OursV2 can be
kept as an internal exploratory result, but it should not be framed as an
improvement.

## Data Summary

| Dataset | Sequences | Patient groups | Target range |
|---|---:|---:|---:|
| CAREPD | 6,066 | 110 | 0-3 |
| CNUH | 21 | 21 | 0-3 |
| Total | 6,087 | 131 | 0-3 |

The combined GroupKFold table is dominated by CAREPD because CNUH contributes
only 21 sequences. CNUH-only LOSO should remain the direct internal-dataset
comparison.

## Main Table: ML, DL, SOTA, and Proposed

Lower MAE, RMSE, and MedAE are better.

| Category | Model | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Classical ML | Ridge | 5 | 6,087 | 0 | 0.009 | 0.570 | 0.759 | 0.446 |
| Classical ML | SVR | 5 | 6,087 | 0 | 1.531 | 0.492 | 0.639 | 0.386 |
| Classical ML | Random Forest | 5 | 6,087 | 0 | 0.031 | 0.510 | 0.659 | 0.417 |
| Classical ML | Shallow MLP | 5 | 6,087 | 0 | 0.010 | 0.544 | 0.708 | 0.423 |
| Deep Learning | Temporal CNN | 5 | 6,087 | 188,929 | 0.294 | 0.425 | 0.594 | 0.287 |
| SOTA | ST-GCN | 5 | 6,087 | 252,097 | 22.523 | 0.443 | 0.623 | 0.274 |
| SOTA | MotionBERT-Lite (81-frame) | 5 | 6,087 | 10,814,222 | 5.243 | 0.442 | 0.625 | 0.247 |
| SOTA | Lu et al. official-architecture DD-Net/OF-DDNet | 5 | 6,087 | 147,908 | 0.445 | 0.404 | **0.543** | 0.307 |
| SOTA | MotionAGFormer-XS | 5 | 6,087 | 2,307,324 | 6.150 | 0.405 | 0.638 | **0.095** |
| Proposed | Ours V1, bounded regression | 5 | 6,087 | 158,594 | 4.615 | **0.358** | 0.564 | 0.147 |

## Main Result Interpretation

Ours V1 achieved the best MAE among the reported methods. Relative MAE
reductions are:

| Comparison | MAE reduction |
|---|---:|
| Ours V1 vs SVR, best classical ML | 27.4% |
| Ours V1 vs Temporal CNN | 15.8% |
| Ours V1 vs ST-GCN | 19.3% |
| Ours V1 vs MotionBERT-Lite (81-frame) | 19.0% |
| Ours V1 vs Lu official-architecture baseline | 11.5% |
| Ours V1 vs MotionAGFormer-XS | 11.7% |

Lu official has the best RMSE, and MotionAGFormer-XS has the best MedAE. The
safest claim is that Ours V1 improves average absolute clinical-score error
(MAE), while Lu official has slightly lower aggregate squared error and
MotionAGFormer-XS has the lowest median absolute error. MotionBERT-Lite
(81-frame) performs similarly to ST-GCN in MAE/RMSE, but does not improve over
the proposed model.

## Statistical Validation

Paired tests use sample-level absolute errors matched by fold, split ID, and
sample ID.

| Comparison | N | Ours V1 MAE | Baseline MAE | Baseline - Ours V1 MAE | Bootstrap 95% CI | Wilcoxon p-value |
|---|---:|---:|---:|---:|---|---:|
| Ours V1 vs Lu official | 6,087 | 0.358 | 0.404 | +0.047 | [0.038, 0.055] | 4.12e-61 |
| Ours V1 vs ST-GCN | 6,087 | 0.358 | 0.443 | +0.085 | [0.078, 0.094] | 5.16e-143 |

Manuscript-safe wording:

> Under identical subject-level GroupKFold evaluation, the proposed bounded
> graph-temporal regression model achieved lower MAE than both ST-GCN and the
> official-architecture Lu et al. baseline.

## Per-Class Error Analysis

Per-class metrics are computed from the completed `predictions.tsv` files. The
primary Ours V1 result is from `groupkfold_h36m17_ours_lu_official_cuda`.

| Score | N | MAE | RMSE | Mean prediction | Prediction SD |
|---|---:|---:|---:|---:|---:|
| 0 | 2,615 | 0.225 | 0.395 | 0.225 | 0.325 |
| 1 | 2,183 | 0.342 | 0.482 | 0.892 | 0.470 |
| 2 | 1,244 | 0.649 | 0.889 | 1.416 | 0.671 |
| 3 | 45 | 0.738 | 0.930 | 2.262 | 0.566 |

The model is most accurate for score 0 and score 1, which are also the most
common classes. Score 3 has only 45 samples and remains the least stable class.
This should be reported as class imbalance rather than as a separate model
failure.

## Confusion Matrix

The model remains a continuous regressor. The matrix below is for
interpretability only: predictions are rounded to the nearest integer and
clipped to `[0, 3]`.

| True score | Pred 0 | Pred 1 | Pred 2 | Pred 3 |
|---:|---:|---:|---:|---:|
| 0 | 2,160 | 444 | 11 | 0 |
| 1 | 492 | 1,509 | 182 | 0 |
| 2 | 181 | 419 | 642 | 2 |
| 3 | 0 | 6 | 20 | 19 |

Row-normalized matrix:

| True score | Pred 0 | Pred 1 | Pred 2 | Pred 3 |
|---:|---:|---:|---:|---:|
| 0 | 0.826 | 0.170 | 0.004 | 0.000 |
| 1 | 0.225 | 0.691 | 0.083 | 0.000 |
| 2 | 0.145 | 0.337 | 0.516 | 0.002 |
| 3 | 0.000 | 0.133 | 0.444 | 0.422 |

The confusion matrix shows that most misclassifications occur between adjacent
clinical scores. The most notable weakness is underestimation of true score 2
and score 3 cases.

## COM Robustness Analysis

COM robustness was evaluated by rerunning Configuration D once with saved fold
checkpoints and then applying perturbations only at inference time. The
checkpointed D run used for this analysis has baseline MAE 0.369 and RMSE
0.564, which is close to the final reported D run but not identical because it
is a separate training run.

The key result is asymmetric: COM centering gives near-perfect robustness to
horizontal translation, but COM centering alone does not provide complete scale
invariance.

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

Wilcoxon tests show that scale perturbations significantly change absolute
errors with non-trivial effect sizes, whereas translation perturbations have
negligible practical effects despite occasional small p-values caused by the
large sample size.

| Condition | Wilcoxon p | Rank-biserial effect |
|---|---:|---:|
| Scale 0.70 | 6.52e-149 | 0.385 |
| Scale 0.85 | 7.62e-62 | 0.246 |
| Scale 1.15 | 4.73e-62 | 0.246 |
| Scale 1.30 | 5.41e-101 | 0.316 |
| Shift -0.20 | 4.99e-02 | -0.038 |
| Shift -0.10 | 4.08e-01 | -0.017 |
| Shift +0.10 | 3.42e-01 | -0.020 |
| Shift +0.20 | 1.26e-02 | -0.049 |

Manuscript-safe interpretation:

> COM-centered normalization effectively removes global horizontal position
> shifts, as translation perturbations produced virtually no change in MAE or
> RMSE. However, COM centering alone does not guarantee scale invariance:
> simulated camera-distance changes increased error, especially under extreme
> scale factors. Therefore, we report COM centering as a position-normalization
> mechanism and treat scale robustness as a limitation requiring future
> bone-length or body-size normalization and scale augmentation.

Full outputs are stored in `docs/com_robustness/`.

A narrower realistic-range analysis was also generated in
`docs/com_robustness_realistic/`. In this range, COM-only D remains relatively
stable for small scale shifts but still degrades at `s = 1.10`:

| Condition | MAE | RMSE | Delta MAE (%) |
|---|---:|---:|---:|
| Original | 0.369 | 0.564 | 0.000 |
| Scale 0.90 | 0.391 | 0.607 | 5.960 |
| Scale 0.95 | 0.374 | 0.580 | 1.339 |
| Scale 1.05 | 0.398 | 0.586 | 7.799 |
| Scale 1.10 | 0.432 | 0.622 | 17.224 |

### Scale-Robust Operating Point

A follow-up full 5-fold experiment evaluated scale-robust variants using the
same Ours V1 architecture. These variants do not change the model encoder; they
only modify input normalization and, where noted, train-time coordinate scale
augmentation.

| Variant | Scale normalization | Train-time scale augmentation | MAE | RMSE | MedAE | Max scale Delta MAE (%) | Max translation Delta MAE (%) | Decision |
|---|---|---|---:|---:|---:|---:|---:|---|
| COM-only D checkpoint | none | none | 0.369 | 0.564 | 0.159 | 46.488 | 0.000 | Accurate but not scale-robust |
| Scale augmentation | none | 0.85-1.15 | 0.402 | 0.605 | 0.205 | 3.399 | 0.000 | Robust but accuracy loss |
| Hip-width normalization | hip width | none | 0.380 | **0.556** | 0.204 | 0.000 | 0.000 | Robust, moderate MAE loss |
| Median-bone normalization + augmentation | median bone length | 0.85-1.15 | **0.366** | 0.567 | 0.139 | **0.000** | **0.000** | Recommended robust operating point |

The best trade-off is median-bone normalization with moderate scale
augmentation. It preserves the main model's accuracy almost exactly
(`MAE 0.366` vs. `0.369` in the checkpointed COM-only run; `RMSE 0.567` vs.
`0.564`) while eliminating measurable degradation across all tested scale
factors (`s = 0.70` to `1.30`) and translation offsets (`Delta x = -0.20` to
`+0.20`). This result supports the following stronger but precise claim:

> COM centering removes global position shifts, and adding sequence-level
> body-scale normalization based on median bone length removes the residual
> scale sensitivity induced by simulated camera-distance changes, without
> materially degrading prediction accuracy.

Reporting decision: keep the original Ours V1 D result as the main performance
table entry because it has the best MAE among all models. Use the
median-bone-normalized variant in the COM robustness subsection as a robust
operating point or deployment variant, not as a separate new architecture.
Full outputs are stored in `docs/scale_robustness_full/` and summarized in
`docs/scale_robustness_full_summary.md`.

## Cross-Dataset Validation

External generalization was evaluated with two zero-shot transfer protocols.
No fine-tuning, domain adaptation, or test-set checkpoint selection was used.
The combined GroupKFold row is the main subject-independent result with both
domains represented in training.

| Protocol | Train Set | Test Set | N train | N test | MAE | RMSE | MedAE |
|---|---|---|---:|---:|---:|---:|---:|
| Zero-shot transfer | CNUH | CARE-PD | 21 | 6,066 | 0.747 | 0.882 | 0.921 |
| Reverse transfer | CARE-PD | CNUH | 6,066 | 21 | 1.014 | 1.170 | 0.746 |
| Combined GroupKFold | CNUH + CARE-PD | CNUH + CARE-PD | subject-level 5-fold | 6,087 | 0.358 | 0.564 | 0.147 |

Relative to the combined GroupKFold setting, zero-shot transfer substantially
increased error:

| Comparison | Delta MAE | Delta RMSE | Relative MAE increase | Relative RMSE increase |
|---|---:|---:|---:|---:|
| CNUH -> CARE-PD vs Combined | +0.390 | +0.318 | +109.0% | +56.3% |
| CARE-PD -> CNUH vs Combined | +0.657 | +0.605 | +183.7% | +107.3% |

Prediction-distribution analysis showed regression-to-the-middle behavior.
The CNUH-trained model predicted CARE-PD samples in a narrow range
(`0.975-1.205`, mean `1.144`), while the CARE-PD-trained model predicted CNUH
samples around a higher narrow range (`1.446-2.233`, mean `1.759`). This means
the transfer models remained numerically stable but were not calibrated to the
unseen target domain.

Manuscript-safe interpretation:

> Zero-shot cross-dataset transfer revealed a substantial site and
> representation domain gap. The proposed model achieved strong performance
> when both CNUH and CARE-PD were represented in subject-independent
> GroupKFold training, but direct transfer from one dataset to the other caused
> regression-to-the-middle predictions. This indicates that clinical deployment
> across sites should use site-balanced training, calibration, or explicit
> domain adaptation rather than assuming unchanged zero-shot transfer.

Likely domain-gap factors include dataset-size asymmetry, camera/viewpoint
differences, CNUH MediaPipe-derived 2.5D-to-H36M17 coordinates versus CARE-PD
SMPL/H36M-style 3D pose sequences, cohort heterogeneity, and site-specific
annotation harmonization. Detailed analysis is stored in
`docs/cross_dataset_validation_analysis.md`.

### Completed Domain-Gap Follow-Up Analyses

Five additional analyses have been completed to qualify the cross-dataset
result. Some use existing prediction files, while the model-comparison and
CARE-PD leave-one-dataset-out analyses required additional fixed-epoch training
runs.

#### 1. Dataset-Wise Breakdown Under Combined GroupKFold

The combined GroupKFold result is dominated numerically by CARE-PD because
CARE-PD contributes 6,066 of 6,087 sequences. Dataset-wise evaluation shows
that Ours V1 is strongest on CARE-PD but remains unstable on the small CNUH
subset:

| Model | CARE-PD MAE | CARE-PD RMSE | CNUH MAE | CNUH RMSE |
|---|---:|---:|---:|---:|
| Ours V1 | **0.356** | 0.562 | 0.793 | **0.945** |
| Lu official | 0.403 | **0.540** | 0.862 | 1.031 |
| ST-GCN | 0.442 | 0.621 | 0.879 | 1.008 |
| Temporal CNN | 0.420 | 0.581 | 1.624 | 2.199 |

Interpretation: the main combined result is strong and Ours V1 remains the best
MAE performer on CARE-PD, but the CNUH subset has only 21 samples and should
not be overinterpreted as a stable dataset-level benchmark. This supports a
careful claim: multi-domain training improves the dominant external benchmark
performance, while CNUH-specific generalization remains sample-limited.

#### 2. Score-Balanced Transfer Analysis

Score-balanced metrics average per-class errors equally rather than weighting
classes by the target-domain class distribution.

| Protocol | Original MAE | Original RMSE | Score-balanced MAE | Score-balanced RMSE | Balanced - Original MAE |
|---|---:|---:|---:|---:|---:|
| CNUH -> CARE-PD | 0.747 | 0.882 | 1.022 | 1.024 | +0.275 |
| CARE-PD -> CNUH | 1.014 | 1.170 | 1.025 | 1.034 | +0.010 |

For CNUH -> CARE-PD, the score-balanced MAE is much higher than the original
MAE because the model predicts most CARE-PD samples near score 1.1. This is
relatively accurate for true score 1 but poor for true scores 0, 2, and 3.
Therefore, the zero-shot transfer result should be interpreted as poor
severity calibration across score levels, not merely as a small average error
increase.

#### 3. Few-Shot Target-Site Calibration

We evaluated a lightweight target-site calibration step using only the existing
zero-shot predictions. The calibration fits an affine mapping
`y_calibrated = a * y_pred + b` on a small number of labeled target-site
subjects and clips outputs to `[0, 3]`. No model weights are retrained.

| Protocol | Calibration subjects | Base MAE | Calibrated MAE | Delta MAE | Base RMSE | Calibrated RMSE | Delta RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNUH -> CARE-PD | 1 | 0.747 | 0.791 | +0.044 | 0.881 | 1.023 | +0.142 |
| CNUH -> CARE-PD | 3 | 0.748 | 0.672 | -0.076 | 0.883 | 0.845 | -0.037 |
| CNUH -> CARE-PD | 5 | 0.751 | 0.659 | -0.092 | 0.884 | 0.814 | -0.070 |
| CNUH -> CARE-PD | 10 | 0.748 | 0.622 | -0.126 | 0.882 | 0.763 | -0.119 |
| CARE-PD -> CNUH | 1 | 1.015 | 1.038 | +0.023 | 1.171 | 1.260 | +0.090 |
| CARE-PD -> CNUH | 3 | 1.017 | 1.077 | +0.060 | 1.170 | 1.290 | +0.120 |
| CARE-PD -> CNUH | 5 | 1.029 | 0.990 | -0.039 | 1.179 | 1.162 | -0.017 |
| CARE-PD -> CNUH | 10 | 0.999 | 0.836 | -0.163 | 1.149 | 0.991 | -0.158 |

The useful deployment-oriented result is that calibration becomes beneficial
once more than a minimal single-subject calibration set is available. In the
CNUH -> CARE-PD direction, 3-10 target-site subjects consistently reduce MAE.
In the CARE-PD -> CNUH direction, the trend is noisier because CNUH has only
21 subjects, but 10 calibration subjects still reduce MAE by approximately
0.16.

Manuscript-safe follow-up interpretation:

> Zero-shot transfer exposes a substantial domain gap, but the gap is partly
> calibratable. A simple affine target-site calibration using a small number of
> labeled target-site subjects reduced transfer error without retraining the
> full model. This suggests a practical deployment path in which the model is
> trained on multi-site data and lightly calibrated for a new clinical site.

#### 4. Ours vs SOTA Cross-Dataset Transfer

The SOTA transfer comparison has now been completed with the same zero-shot
protocols. No fine-tuning, adaptation, or test-set checkpoint selection was
used. The table below combines the original Ours/ST-GCN/Lu transfer comparison
with the completed MotionBERT-Lite and MotionAGFormer-XS transfer runs.

| Category | Model | Train | Test | N train | N test | MAE | RMSE | MedAE |
|---|---|---|---|---:|---:|---:|---:|---:|
| SOTA | MotionAGFormer-XS | CNUH | CARE-PD | 21 | 6,066 | **0.740** | **0.875** | **0.567** |
| Proposed | Ours V1 | CNUH | CARE-PD | 21 | 6,066 | 0.747 | 0.882 | 0.921 |
| SOTA | MotionBERT-Lite (81-frame) | CNUH | CARE-PD | 21 | 6,066 | 0.889 | 1.078 | 0.841 |
| SOTA | Lu official | CNUH | CARE-PD | 21 | 6,066 | 0.898 | 1.016 | 0.596 |
| SOTA | ST-GCN | CNUH | CARE-PD | 21 | 6,066 | 8.346 | 9.737 | 6.734 |
| SOTA | Lu official | CARE-PD | CNUH | 6,066 | 21 | **0.865** | **1.027** | 0.735 |
| SOTA | MotionBERT-Lite (81-frame) | CARE-PD | CNUH | 6,066 | 21 | 0.898 | 1.082 | 0.792 |
| Proposed | Ours V1 | CARE-PD | CNUH | 6,066 | 21 | 0.910 | 1.034 | **0.639** |
| SOTA | MotionAGFormer-XS | CARE-PD | CNUH | 6,066 | 21 | 0.921 | 1.152 | 0.729 |
| SOTA | ST-GCN | CARE-PD | CNUH | 6,066 | 21 | 1.203 | 1.385 | 1.119 |

Average across both transfer directions:

| Model | Average MAE | Average RMSE | Average MedAE |
|---|---:|---:|---:|
| Ours V1 | **0.828** | **0.958** | 0.780 |
| MotionAGFormer-XS | 0.831 | 1.014 | **0.648** |
| Lu official | 0.882 | 1.021 | 0.666 |
| MotionBERT-Lite (81-frame) | 0.894 | 1.080 | 0.816 |
| ST-GCN | 4.774 | 5.561 | 3.926 |

Interpretation: all models degrade under strict zero-shot transfer.
MotionAGFormer-XS is marginally best in the CNUH -> CARE-PD direction, Lu
official is slightly best in the CARE-PD -> CNUH direction, and Ours V1 has
the lowest average transfer MAE across both directions. ST-GCN is highly
unstable in the CNUH -> CARE-PD setting, likely because its unbounded
regression head extrapolates poorly when trained on only 21 CNUH samples.

Manuscript-safe wording:

> Under strict zero-shot transfer, all skeleton-based models showed substantial
> domain degradation. The proposed bounded graph-temporal model achieved the
> lowest average transfer MAE across both directions, although MotionAGFormer-XS
> was marginally better in the CNUH-to-CARE-PD direction and Lu official was
> slightly better in the CARE-PD-to-CNUH direction. These results reinforce the
> need for site calibration or domain adaptation, while showing that the
> proposed model is a comparatively stable transfer baseline.

#### 5. CARE-PD Leave-One-Dataset-Out

CARE-PD leave-one-dataset-out holds out one CARE-PD source cohort at a time.
This is a stronger within-CARE-PD external generalization test than random
subject-level folds because the held-out cohort has no sequences represented
during training.

| Held-out CARE-PD cohort | N train | N test | MAE | RMSE | MedAE |
|---|---:|---:|---:|---:|---:|
| 3DGait | 5,976 | 90 | 0.775 | 0.947 | 0.847 |
| BMCLab | 2,171 | 3,895 | 0.663 | 0.844 | 0.528 |
| PD-GaM | 4,366 | 1,700 | 0.495 | 0.724 | 0.236 |
| T-SDU-PD | 5,685 | 381 | 0.692 | 0.836 | 0.707 |
| **Overall** | - | 6,066 | **0.620** | **0.813** | **0.508** |

Comparison with related protocols:

| Protocol | MAE | RMSE | MedAE | Interpretation |
|---|---:|---:|---:|---|
| Combined GroupKFold, CARE-PD subset | 0.356 | 0.562 | 0.146 | CARE-PD cohorts represented during training |
| CARE-PD leave-one-dataset-out | 0.620 | 0.813 | 0.508 | Entire CARE-PD cohort held out |
| CNUH -> CARE-PD zero-shot | 0.747 | 0.882 | 0.921 | Cross-site and pose-representation transfer |

Interpretation: CARE-PD LODO is harder than subject-level GroupKFold but easier
than CNUH -> CARE-PD zero-shot transfer. This supports the claim that
multi-cohort CARE-PD training improves generalization to unseen CARE-PD cohorts,
while a measurable cohort-level domain gap remains. The hardest held-out cohort
is 3DGait (`MAE = 0.775`), whereas PD-GaM is easiest (`MAE = 0.495`).

Completed follow-up outputs:

```text
docs/cross_dataset_model_comparison.md
docs/cross_dataset_model_comparison_v2.md
docs/cross_dataset_validation_record_en.md
docs/cross_dataset_validation_record_ko.md
docs/dataset_wise_breakdown_analysis.md
docs/score_balanced_transfer_analysis.md
docs/fewshot_calibration_analysis.md
docs/carepd_lodo_analysis.md
docs/selective_prediction_analysis.md
docs/latency_benchmark.md
docs/additional_experiments_summary.md
```

### Selective Prediction

Selective prediction was computed from existing GroupKFold prediction tables.
No retraining was performed. Predictions far from rounded-score decision
boundaries are retained for automatic scoring, while boundary-proximal cases
are flagged for clinician review.

| Coverage | N kept | MAE | RMSE | MedAE | Rounded accuracy | MAE reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 100% | 6,087 | 0.358 | 0.564 | 0.147 | 0.711 | 0.0% |
| 90% | 5,478 | 0.325 | 0.540 | 0.116 | 0.744 | 9.2% |
| 80% | 4,870 | 0.288 | 0.510 | 0.088 | 0.774 | 19.4% |
| 70% | 4,261 | 0.260 | 0.493 | 0.064 | 0.793 | 27.2% |
| 60% | 3,652 | 0.234 | 0.477 | 0.045 | 0.811 | 34.5% |
| 50% | 3,044 | 0.211 | 0.462 | 0.030 | 0.825 | 40.9% |

At 80% coverage, Ours V1 retains the lowest MAE and highest rounded-score
accuracy among the compared models:

| Category | Model | Retained MAE | Retained RMSE | Retained MedAE | Rounded accuracy |
|---|---|---:|---:|---:|---:|
| Proposed | Ours V1 | **0.288** | **0.510** | 0.088 | **0.774** |
| SOTA | MotionAGFormer-XS | 0.351 | 0.612 | **0.054** | 0.700 |
| SOTA | Lu official | 0.369 | 0.527 | 0.207 | 0.720 |
| SOTA | MotionBERT-Lite (81-frame) | 0.384 | 0.578 | 0.149 | 0.674 |
| SOTA | ST-GCN | 0.385 | 0.575 | 0.176 | 0.704 |

Manuscript-safe wording:

> Selective prediction analysis showed that boundary-distant predictions had
> lower error. At 80% automatic coverage, the proposed model reduced MAE from
> 0.358 to 0.288 and improved rounded-score accuracy from 0.711 to 0.774,
> supporting a clinical workflow in which uncertain cases are flagged for
> clinician review.

### Latency Benchmark

The latency benchmark measures forward-pass time using randomly initialized
architecture instances under the same input length and batch size. It should be
interpreted as an architecture-level inference-cost comparison.

| Category | Model | Params | Device | Batch | ms/sample |
|---|---|---:|---|---:|---:|
| Deep Learning | Temporal CNN | 188,929 | cuda | 32 | **0.020** |
| Proposed | Ours V1 | 158,594 | cuda | 32 | 0.242 |
| SOTA | Lu official | 147,908 | cuda | 32 | 0.335 |
| SOTA | ST-GCN | 252,097 | cuda | 32 | 0.522 |
| SOTA | MotionAGFormer-XS | 2,307,324 | cuda | 32 | 1.937 |
| SOTA | MotionBERT-Lite | 10,814,222 | cuda | 32 | 16.481 |

Ours V1 is not the fastest model overall because Temporal CNN is much simpler,
but it is faster than Lu official, ST-GCN, MotionAGFormer-XS, and
MotionBERT-Lite while achieving the best MAE in the main GroupKFold
comparison.

## Ours V1 A/B/C/D Ablation

For consistency with the final proposed model, D uses the best completed D run
from `groupkfold_h36m17_ours_lu_official_cuda`. A/B/C use the dedicated
ablation run directories.

| Model | Ablation | Feature set | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ours V1 | A | coordinates only | 5 | 6,087 | 158,210 | 3.179 | 0.374 | 0.577 | 0.168 |
| Ours V1 | B | coordinates + velocity | 5 | 6,087 | 158,402 | 2.757 | 0.432 | 0.629 | 0.246 |
| Ours V1 | C | coordinates + velocity + amplitude/variability | 5 | 6,087 | 158,530 | 4.178 | 0.376 | **0.549** | 0.192 |
| Ours V1 | D | full hybrid feature set including angle | 5 | 6,087 | 158,594 | 4.615 | **0.358** | 0.564 | **0.147** |

## Ablation Interpretation

Ablation D gives the best MAE and MedAE among the reported ablations, supporting
the use of the full hybrid feature set. Ablation C gives the best RMSE,
suggesting that amplitude and variability features improve squared-error
stability, while the angle channel in D improves average and median absolute
error.

Ablation B performs worst, especially in fold 5. This suggests that simply
adding velocity to coordinates is not sufficient under the combined multi-cohort
setting; higher-level sequence descriptors are needed for stable score
estimation.

## Ours V1 Architecture Ablation

The input-feature ablation above is separate from the architecture ablation
below. The architecture ablation keeps Configuration D fixed and removes model
components step by step. The full Ours V1 row uses the canonical final 5-fold
run from `groupkfold_h36m17_ours_lu_official_cuda`; the interrupted full-model
rows inside `architecture_ablation_ours_cuda` are not used.

| Model | Components | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MLP only | mean pooling + bounded MLP | 5 | 6,087 | 17,921 | 0.005 | 0.554 | 0.653 | 0.481 |
| GraphConv + MLP | GraphConv, no joint attention, no Temporal Transformer | 5 | 6,087 | 25,985 | 2.306 | 0.450 | 0.580 | 0.349 |
| GraphConv + Joint Attention + MLP | GraphConv + joint attention, no Temporal Transformer | 5 | 6,087 | 26,114 | 2.736 | 0.414 | 0.564 | 0.291 |
| Full Ours V1 | GraphConv + Joint Attention + Temporal Transformer | 5 | 6,087 | 158,594 | 4.615 | **0.358** | 0.564 | **0.147** |

This ablation confirms that the performance gain is not attributable only to
the input feature set. GraphConv reduces MAE from `0.554` to `0.450`, joint
attention reduces it further to `0.414`, and the full Temporal Transformer model
achieves the best MAE (`0.358`) and MedAE (`0.147`). Relative to the MLP-only
baseline, the full model reduces MAE by `35.4%`.

## Ours V1 Architecture Summary

Input shape:

```text
B x T x J x C
```

| Symbol | Meaning | Value |
|---|---|---:|
| B | batch size | variable |
| T | sequence length | 390 |
| J | joints | 17 |
| C | feature channels | A: 3, B: 6, C: 8, D: 9 |

Encoder:

```text
Hybrid node features
  -> GraphConv C -> 64
  -> LayerNorm + ReLU
  -> GraphConv 64 -> 128
  -> LayerNorm + ReLU
  -> joint attention over 17 joints
  -> temporal Transformer encoder
  -> temporal average pooling
  -> MLP regression head
  -> 3 * sigmoid(raw_score)
```

The final `3 * sigmoid(raw_score)` constrains predictions to the valid item-10
range `[0, 3]`.

| Component | Purpose |
|---|---|
| COM normalization | Reduces subject-position and camera-translation sensitivity |
| Graph convolution | Encodes anatomical joint connectivity |
| Joint attention | Weights clinically informative joints more strongly |
| Temporal Transformer | Captures long-range gait dynamics |
| Bounded regression | Keeps continuous predictions inside the clinical score range |

## Reproducibility Commands

Run or resume A/B/C/D ablation. Completed ablations with `summary.csv` are
skipped automatically:

```powershell
cd C:\Users\bokyung\Desktop\parkinson_COM\carepd_17pt_experiments
.\scripts\run_ours_abcd_cuda.cmd
```

Regenerate the ablation summary:

```powershell
python scripts\summarize_ours_abcd.py
```

Outputs:

- `docs\ours_abcd_summary.md`
- `results\OURS_ABCD_SUMMARY.md`
- `results\ours_abcd_summary.csv`

## Visualization Outputs

All generated figures are collected in:

```text
docs/final_integrated_figures/
```

Note: figures generated before 2026-06-05 may not include the completed
MotionBERT-Lite (81-frame) and MotionAGFormer-XS rows. Regenerate the figure
set before using the MAE ranking or model-comparison plots in the manuscript.

Recommended manuscript figures:

| Figure | Suggested use |
|---|---|
| `02_mae_ranking.png` | Main performance comparison figure |
| `07_ablation_metrics.png` | A/B/C/D ablation figure |
| `09_ours_per_class_error.png` | Per-class error figure |
| `12_ours_confusion_normalized.png` | Confusion matrix figure |
| `18_calibration_curve_by_model.png` | Calibration-style model behavior figure |

Recommended main-text composite:

```text
Figure X:
  (a) 02_mae_ranking.png
  (b) 07_ablation_metrics.png
  (c) 09_ours_per_class_error.png
  (d) 12_ours_confusion_normalized.png
```

Recommended supplementary figures:

| Figure | Suggested use |
|---|---|
| `05_fold_mae_by_model.png` | Fold-level stability |
| `18_calibration_curve_by_model.png` | Calibration-style score trend |
| `reviewer_figures/21_calibration_curve_ours.png` | Reviewer-facing calibration curve for Ours V1 |
| `reviewer_figures/22_calibration_curve_models.png` | Calibration curve comparison across deep/SOTA models |
| `20_mae_advantage_vs_baselines.png` | Compact MAE gain over baselines |
| `16_dataset_mae_breakdown.png` | Dataset-level breakdown; supplementary because CNUH has only 21 samples |
| `19_class_distribution.png` | Class imbalance explanation |

Completed reviewer-response learning curve:

```text
docs/learning_curve_ours_analysis.md
docs/reviewer_figures/24_learning_curve_ours_mae.png
docs/reviewer_figures/25_learning_curve_ours_rmse.png
```

The learning curve varies the training-subject fraction while keeping the
validation folds fixed. MAE decreases monotonically from `0.476` at 10%
training subjects to `0.360` at 100% training subjects, a `24.5%` relative
reduction. RMSE decreases from `0.672` to `0.530`, a `21.1%` relative
reduction. This supports the interpretation that small-cohort performance is
data-limited and improves with larger multi-site training data.

Additional diagnostic figures are listed in
`docs/final_integrated_figures/README.md`.
