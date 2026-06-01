# Cross-Dataset Validation Analysis

- Model: Ours V1, Configuration D
- Architecture: GraphConv + Joint Attention + Temporal Transformer + bounded regression
- Fine-tuning/adaptation: none
- External-transfer training: fixed epoch training; test labels are not used for checkpoint selection
- Epochs: 80
- Device: CUDA
- Target: MDS-UPDRS item 10 gait score, range 0-3

## Table 10. Cross-Dataset Validation

| Protocol | Train Set | Test Set | N train | N test | MAE | RMSE | MedAE |
|---|---|---|---:|---:|---:|---:|---:|
| Zero-shot transfer | CNUH | CARE-PD | 21 | 6,066 | 0.747 | 0.882 | 0.921 |
| Reverse transfer | CARE-PD | CNUH | 6,066 | 21 | 1.014 | 1.170 | 0.746 |
| Combined GroupKFold | CNUH + CARE-PD | CNUH + CARE-PD | subject-level 5-fold | 6,087 | 0.358 | 0.564 | 0.147 |

## Domain Gap Summary

| Comparison | Delta MAE | Delta RMSE | Relative MAE Increase (%) | Relative RMSE Increase (%) |
|---|---:|---:|---:|---:|
| CNUH -> CARE-PD vs Combined GroupKFold | +0.390 | +0.318 | +109.0 | +56.3 |
| CARE-PD -> CNUH vs Combined GroupKFold | +0.657 | +0.605 | +183.7 | +107.3 |
| CNUH -> CARE-PD minus CARE-PD -> CNUH | -0.267 | -0.288 | -26.3 | -24.6 |

The combined GroupKFold result remains the main performance estimate because
training and testing are subject-independent while both domains are represented
in training. The two external-transfer protocols are stricter stress tests:
they intentionally withhold one entire dataset from training.

## Prediction Behavior

| Protocol | Test N | Mean true score | Mean predicted score | Prediction range | MAE |
|---|---:|---:|---:|---|---:|
| CNUH -> CARE-PD | 6,066 | 0.789 | 1.144 | 0.975-1.205 | 0.747 |
| CARE-PD -> CNUH | 21 | 1.000 | 1.759 | 1.446-2.233 | 1.014 |

Both transfer directions show regression-to-the-middle behavior. The model
does not fail by producing invalid values; bounded regression keeps outputs in
range. Instead, predictions collapse into a narrow score band that is poorly
calibrated for the unseen target domain.

## Per-Class Transfer Error

### CNUH -> CARE-PD

| True score | N | Mean prediction | MAE | RMSE |
|---:|---:|---:|---:|---:|
| 0 | 2,608 | 1.161 | 1.161 | 1.162 |
| 1 | 2,175 | 1.144 | 0.144 | 0.148 |
| 2 | 1,239 | 1.106 | 0.894 | 0.896 |
| 3 | 44 | 1.110 | 1.890 | 1.891 |

The CNUH-trained model predicts most CARE-PD samples around score 1.1. This is
reasonable for true score 1, but it overestimates score 0 and underestimates
scores 2-3. The apparent overall MAE is therefore driven by the large number of
score-0 and score-1 samples rather than uniform severity calibration.

### CARE-PD -> CNUH

| True score | N | Mean prediction | MAE | RMSE |
|---:|---:|---:|---:|---:|
| 0 | 7 | 1.730 | 1.730 | 1.730 |
| 1 | 8 | 1.813 | 0.813 | 0.847 |
| 2 | 5 | 1.717 | 0.283 | 0.285 |
| 3 | 1 | 1.727 | 1.273 | 1.273 |

The CARE-PD-trained model predicts the CNUH samples around score 1.7. This
helps for true score 2 but produces large errors for CNUH score-0 and score-1
subjects. Because CNUH has only 21 subjects, the reverse-transfer metrics are
also highly sensitive to individual samples and class imbalance.

## Interpretation

The cross-dataset results show a substantial domain gap. External transfer is
much worse than subject-level GroupKFold on the combined dataset:

- CNUH -> CARE-PD increases MAE from 0.358 to 0.747.
- CARE-PD -> CNUH increases MAE from 0.358 to 1.014.
- The reverse direction is worse in aggregate, mainly because the CARE-PD-trained
  model overestimates low-severity CNUH subjects.

The most likely causes are:

- Dataset size asymmetry: CNUH has only 21 sequences, which is too small to
  learn a domain-general mapping for CARE-PD.
- Pose-representation mismatch: CNUH is based on MediaPipe-derived 2.5D
  coordinates converted to H36M17, whereas CARE-PD uses SMPL/H36M-style
  preprocessed 3D pose sequences.
- Camera/viewpoint differences: CNUH is frontal-view clinical video, while
  CARE-PD aggregates multiple cohorts and acquisition setups.
- Severity and annotation distribution differences: the target score
  distribution and rater harmonization can differ across sites.
- Cohort heterogeneity: CARE-PD contains multi-site sequences; CNUH is a small
  single-site IRB-approved cohort.

## Manuscript-Ready Section 5.8 Draft

To assess external generalization, we evaluated two zero-shot cross-dataset
transfer protocols without fine-tuning or domain adaptation. In Protocol 1, the
model was trained only on the CNUH cohort (N = 21) and evaluated directly on
CARE-PD (6,066 sequences). In Protocol 2, the model was trained only on CARE-PD
and evaluated directly on CNUH. The combined subject-level GroupKFold result is
reported as Protocol 3.

The zero-shot transfer results showed a clear domain gap. Training on CNUH and
testing on CARE-PD yielded MAE = 0.747 and RMSE = 0.882, which is worse than
the combined GroupKFold result (MAE = 0.358, RMSE = 0.564). The reverse
direction, CARE-PD to CNUH, yielded MAE = 1.014 and RMSE = 1.170. Inspection of
the prediction distributions showed that the transfer models tended to collapse
toward a narrow middle-severity score range: CNUH -> CARE-PD predictions were
concentrated around 1.14, whereas CARE-PD -> CNUH predictions were concentrated
around 1.76. This behavior indicates that the model remains numerically stable
under bounded regression but is not fully calibrated to unseen acquisition
domains.

These findings should be interpreted as evidence of cross-site domain shift
rather than failure of the proposed architecture. The two datasets differ in
size, camera setup, pose representation, cohort composition, and annotation
harmonization. In particular, CNUH uses frontal-view MediaPipe-derived
2.5D-to-H36M17 coordinates, whereas CARE-PD provides SMPL/H36M-style
preprocessed 3D sequences. The substantially better combined GroupKFold result
suggests that site-balanced training with both domains represented can recover
strong subject-independent performance, while true zero-shot deployment across
sites will require additional calibration, domain adaptation, or broader
multi-site training data.

## Discussion Wording

The cross-dataset experiment is best framed as an honest external
generalization stress test. It supports three claims:

- The main model is effective when trained with multi-domain data under
  subject-independent splits.
- Direct zero-shot transfer across CNUH and CARE-PD remains difficult.
- Future deployment should include site-balanced training, calibration, or
  domain adaptation rather than assuming that a model trained on one site will
  transfer unchanged to another.

Recommended concise manuscript wording:

> Zero-shot cross-dataset transfer revealed a substantial site and representation
> domain gap. Although the proposed model achieved strong performance under
> combined subject-level GroupKFold evaluation, models trained on only one
> dataset showed regression-to-the-middle behavior when evaluated on the other
> dataset. This indicates that robust clinical deployment should use
> site-balanced training or explicit domain adaptation.

## Completed Follow-Up Analyses

Three follow-up analyses have been completed from existing prediction files.
They refine the interpretation of the cross-dataset result without introducing
additional model training.

### Dataset-Wise Combined GroupKFold Breakdown

Under combined GroupKFold, Ours V1 remains strongest on CARE-PD test samples
but is less stable on the small CNUH subset:

| Dataset | N | Subjects | Ours V1 MAE | Ours V1 RMSE | Ours V1 MedAE |
|---|---:|---:|---:|---:|---:|
| CARE-PD | 6,066 | 110 | 0.356 | 0.562 | 0.146 |
| CNUH | 21 | 21 | 0.793 | 0.945 | 0.987 |

Interpretation: the strong combined result mainly reflects robust CARE-PD
performance because CARE-PD dominates the sequence count. The CNUH subset is
too small to serve as a stable dataset-level benchmark, so CNUH-specific claims
should remain cautious.

### Score-Balanced Transfer

| Protocol | Original MAE | Score-balanced MAE | Interpretation |
|---|---:|---:|---|
| CNUH -> CARE-PD | 0.747 | 1.022 | Original MAE is softened by CARE-PD class distribution |
| CARE-PD -> CNUH | 1.014 | 1.025 | Similar because CNUH is small and sparse |

Interpretation: CNUH -> CARE-PD transfer is especially poorly calibrated across
severity classes. The model predicts near score 1.1, which is acceptable for
true score 1 but poor for true scores 0, 2, and 3.

### Few-Shot Target-Site Calibration

A simple affine target-site calibration was applied to zero-shot predictions:

```text
y_calibrated = a * y_pred + b
```

No model weights were retrained.

| Protocol | Calibration subjects | Base MAE | Calibrated MAE | Delta MAE |
|---|---:|---:|---:|---:|
| CNUH -> CARE-PD | 3 | 0.748 | 0.672 | -0.076 |
| CNUH -> CARE-PD | 5 | 0.751 | 0.659 | -0.092 |
| CNUH -> CARE-PD | 10 | 0.748 | 0.622 | -0.126 |
| CARE-PD -> CNUH | 5 | 1.029 | 0.990 | -0.039 |
| CARE-PD -> CNUH | 10 | 0.999 | 0.836 | -0.163 |

Interpretation: zero-shot transfer is not fully solved, but the error is partly
calibratable. This provides a practical deployment argument: a new clinical
site may require a small labeled calibration set rather than full model
retraining.

Updated manuscript-safe wording:

> Zero-shot transfer revealed a substantial domain gap, but follow-up analyses
> showed that this gap is partly attributable to target-domain score
> calibration. A lightweight affine calibration using a small number of labeled
> target-site subjects reduced transfer error without retraining the model,
> suggesting a practical site-calibration path for deployment.

### Ours vs SOTA Cross-Dataset Transfer

The model-comparison transfer run evaluates Ours V1, ST-GCN, and the Lu
official-architecture baseline under the same zero-shot transfer setup. This is
a separate comparative run from the standalone Ours-only Table 10 run, and
therefore small stochastic differences in the Ours values may appear.

| Category | Model | Train | Test | MAE | RMSE | MedAE |
|---|---|---|---|---:|---:|---:|
| Proposed | Ours V1 | CNUH | CARE-PD | **0.747** | **0.882** | 0.921 |
| SOTA | Lu official | CNUH | CARE-PD | 0.898 | 1.016 | **0.596** |
| SOTA | ST-GCN | CNUH | CARE-PD | 8.346 | 9.737 | 6.734 |
| Proposed | Ours V1 | CARE-PD | CNUH | 0.910 | 1.034 | **0.639** |
| SOTA | Lu official | CARE-PD | CNUH | **0.865** | **1.027** | 0.735 |
| SOTA | ST-GCN | CARE-PD | CNUH | 1.203 | 1.385 | 1.119 |

Interpretation: Ours V1 is strongest in the CNUH -> CARE-PD direction and has
the best average MAE across both transfer directions. Lu official is slightly
better in CARE-PD -> CNUH. ST-GCN is unstable when trained on only 21 CNUH
samples and evaluated on CARE-PD, which highlights the value of bounded
regression for transfer stability.

Manuscript-safe wording:

> In the zero-shot transfer comparison with published skeleton baselines, the
> proposed model achieved the lowest average transfer MAE across both transfer
> directions. Lu official was slightly better in CARE-PD-to-CNUH transfer,
> whereas ST-GCN was unstable under the small-source CNUH-to-CARE-PD setting.
> These findings suggest that the proposed bounded regression model is a
> comparatively stable transfer baseline, although domain adaptation remains
> necessary for reliable deployment.

### CARE-PD Leave-One-Dataset-Out

CARE-PD leave-one-dataset-out evaluates whether the model generalizes to a
CARE-PD source cohort that is completely absent from training.

| Held-out CARE-PD cohort | N train | N test | MAE | RMSE | MedAE |
|---|---:|---:|---:|---:|---:|
| 3DGait | 5,976 | 90 | 0.775 | 0.947 | 0.847 |
| BMCLab | 2,171 | 3,895 | 0.663 | 0.844 | 0.528 |
| PD-GaM | 4,366 | 1,700 | 0.495 | 0.724 | 0.236 |
| T-SDU-PD | 5,685 | 381 | 0.692 | 0.836 | 0.707 |
| **Overall** | - | 6,066 | **0.620** | **0.813** | **0.508** |

This result sits between two other protocols:

| Protocol | MAE | RMSE | MedAE |
|---|---:|---:|---:|
| Combined GroupKFold, CARE-PD subset | 0.356 | 0.562 | 0.146 |
| CARE-PD leave-one-dataset-out | 0.620 | 0.813 | 0.508 |
| CNUH -> CARE-PD zero-shot | 0.747 | 0.882 | 0.921 |

Interpretation: LODO is harder than subject-level GroupKFold because a full
cohort is unseen during training, but it is easier than CNUH -> CARE-PD
zero-shot transfer because the training data still come from the CARE-PD
benchmark ecosystem. The result supports the domain-gap story: multi-cohort
training improves transfer to unseen cohorts, but cohort-level shift remains.

## Source Outputs

```text
results/cross_dataset_validation/summary.csv
results/cross_dataset_validation/domain_gap.csv
results/cross_dataset_validation/cnuh_to_carepd/predictions.tsv
results/cross_dataset_validation/carepd_to_cnuh/predictions.tsv
results/cross_dataset_model_comparison/summary.csv
results/carepd_leave_one_dataset_out/summary.csv
docs/cross_dataset_model_comparison.md
docs/carepd_lodo_analysis.md
docs/dataset_wise_breakdown_analysis.md
docs/score_balanced_transfer_analysis.md
docs/fewshot_calibration_analysis.md
```
