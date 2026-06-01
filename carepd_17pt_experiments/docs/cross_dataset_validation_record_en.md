# Cross-Dataset Validation Result Record and Analysis

- Last updated: 2026-06-01
- Model: Ours V1, Configuration D
- Architecture: GraphConv + Joint Attention + Temporal Transformer + bounded regression
- Target: MDS-UPDRS Part III item 3.10 gait score, range 0-3
- Fine-tuning/adaptation: none
- Test-set checkpoint selection: none
- External-transfer training: fixed 80 epochs
- Device: CUDA

## Experiment Purpose

This experiment evaluates external generalization. Unlike the main combined
GroupKFold result, each zero-shot transfer protocol withholds one entire
dataset from training.

The purpose is not to show the best possible performance after adaptation, but
to measure how much performance drops when the model is transferred directly
between acquisition domains.

## Protocols

| Protocol | Train Set | Test Set | Description |
|---|---|---|---|
| Zero-shot transfer | CNUH | CARE-PD | Train only on the small CNUH clinical cohort and evaluate directly on CARE-PD |
| Reverse transfer | CARE-PD | CNUH | Train only on CARE-PD and evaluate directly on the CNUH cohort |
| Combined GroupKFold | CNUH + CARE-PD | CNUH + CARE-PD | Main subject-level GroupKFold result with both domains represented in training |

## Main Results

| Protocol | Train Set | Test Set | N train | N test | MAE | RMSE | MedAE |
|---|---|---|---:|---:|---:|---:|---:|
| Zero-shot transfer | CNUH | CARE-PD | 21 | 6,066 | 0.747 | 0.882 | 0.921 |
| Reverse transfer | CARE-PD | CNUH | 6,066 | 21 | 1.014 | 1.170 | 0.746 |
| Combined GroupKFold | CNUH + CARE-PD | CNUH + CARE-PD | subject-level 5-fold | 6,087 | 0.358 | 0.564 | 0.147 |

## Domain-Gap Quantification

| Comparison | Delta MAE | Delta RMSE | Relative MAE increase | Relative RMSE increase |
|---|---:|---:|---:|---:|
| CNUH -> CARE-PD vs Combined | +0.390 | +0.318 | +109.0% | +56.3% |
| CARE-PD -> CNUH vs Combined | +0.657 | +0.605 | +183.7% | +107.3% |

Interpretation: direct zero-shot transfer is substantially worse than the main
combined GroupKFold setting. The result should be interpreted as a real
cross-site and pose-representation domain gap.

## Prediction Behavior

| Protocol | Test N | Mean true score | Mean predicted score | Prediction range | MAE |
|---|---:|---:|---:|---|---:|
| CNUH -> CARE-PD | 6,066 | 0.789 | 1.144 | 0.975-1.205 | 0.747 |
| CARE-PD -> CNUH | 21 | 1.000 | 1.759 | 1.446-2.233 | 1.014 |

Both transfer directions show regression-to-the-middle behavior. The bounded
regression output remains numerically valid, but the model collapses into a
narrow score range that is poorly calibrated for the unseen target domain.

## Per-Class Transfer Error

### CNUH -> CARE-PD

| True score | N | Mean prediction | MAE | RMSE |
|---:|---:|---:|---:|---:|
| 0 | 2,608 | 1.161 | 1.161 | 1.162 |
| 1 | 2,175 | 1.144 | 0.144 | 0.148 |
| 2 | 1,239 | 1.106 | 0.894 | 0.896 |
| 3 | 44 | 1.110 | 1.890 | 1.891 |

The CNUH-trained model predicts most CARE-PD samples near score 1.1. This is
acceptable for true score 1 but overestimates score 0 and underestimates scores
2 and 3.

### CARE-PD -> CNUH

| True score | N | Mean prediction | MAE | RMSE |
|---:|---:|---:|---:|---:|
| 0 | 7 | 1.730 | 1.730 | 1.730 |
| 1 | 8 | 1.813 | 0.813 | 0.847 |
| 2 | 5 | 1.717 | 0.283 | 0.285 |
| 3 | 1 | 1.727 | 1.273 | 1.273 |

The CARE-PD-trained model predicts CNUH subjects near score 1.7. This helps for
true score 2 but causes large errors for score 0 and 1. Because CNUH has only
21 subjects, this direction is highly sensitive to individual samples.

## Completed Follow-Up Analyses

### 1. Dataset-Wise Combined GroupKFold Breakdown

| Dataset | N | Subjects | Ours V1 MAE | Ours V1 RMSE | Ours V1 MedAE |
|---|---:|---:|---:|---:|---:|
| CARE-PD | 6,066 | 110 | 0.356 | 0.562 | 0.146 |
| CNUH | 21 | 21 | 0.793 | 0.945 | 0.987 |

Analysis: the main combined result is dominated by CARE-PD sequence count.
Ours V1 remains strong on CARE-PD, but CNUH-specific dataset-level conclusions
should be cautious because the CNUH subset has only 21 samples.

### 2. Score-Balanced Transfer

| Protocol | Original MAE | Original RMSE | Score-balanced MAE | Score-balanced RMSE | Balanced - Original MAE |
|---|---:|---:|---:|---:|---:|
| CNUH -> CARE-PD | 0.747 | 0.882 | 1.022 | 1.024 | +0.275 |
| CARE-PD -> CNUH | 1.014 | 1.170 | 1.025 | 1.034 | +0.010 |

Analysis: CNUH -> CARE-PD becomes worse after score balancing, indicating that
the original MAE is softened by the CARE-PD class distribution. The transfer
model is not uniformly calibrated across severity classes.

### 3. Few-Shot Target-Site Calibration

Calibration method:

```text
y_calibrated = a * y_pred + b
```

The calibrated score is clipped to `[0, 3]`. No model weights are retrained.

| Protocol | Calibration subjects | Base MAE | Calibrated MAE | Delta MAE | Base RMSE | Calibrated RMSE | Delta RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNUH -> CARE-PD | 3 | 0.748 | 0.672 | -0.076 | 0.883 | 0.845 | -0.037 |
| CNUH -> CARE-PD | 5 | 0.751 | 0.659 | -0.092 | 0.884 | 0.814 | -0.070 |
| CNUH -> CARE-PD | 10 | 0.748 | 0.622 | -0.126 | 0.882 | 0.763 | -0.119 |
| CARE-PD -> CNUH | 5 | 1.029 | 0.990 | -0.039 | 1.179 | 1.162 | -0.017 |
| CARE-PD -> CNUH | 10 | 0.999 | 0.836 | -0.163 | 1.149 | 0.991 | -0.158 |

Analysis: zero-shot transfer is difficult, but the error is partly calibratable.
This provides a stronger deployment story: a model trained on multi-site data
may still need lightweight target-site calibration when introduced to a new
clinical site.

### 4. Ours vs SOTA Zero-Shot Transfer Comparison

This comparison was run with:

```text
scripts/run_cross_dataset_model_comparison_cuda.cmd
```

It is a separate comparative rerun from the standalone Ours-only transfer
table, so small stochastic differences in Ours values can occur. Use this table
for model-to-model transfer comparison.

| Category | Model | Train | Test | MAE | RMSE | MedAE |
|---|---|---|---|---:|---:|---:|
| Proposed | Ours V1 | CNUH | CARE-PD | **0.747** | **0.882** | 0.921 |
| SOTA | Lu official | CNUH | CARE-PD | 0.898 | 1.016 | **0.596** |
| SOTA | ST-GCN | CNUH | CARE-PD | 8.346 | 9.737 | 6.734 |
| Proposed | Ours V1 | CARE-PD | CNUH | 0.910 | 1.034 | **0.639** |
| SOTA | Lu official | CARE-PD | CNUH | **0.865** | **1.027** | 0.735 |
| SOTA | ST-GCN | CARE-PD | CNUH | 1.203 | 1.385 | 1.119 |

Analysis:

- Ours V1 is best in CNUH -> CARE-PD by MAE and RMSE.
- Lu official is slightly best in CARE-PD -> CNUH by MAE and RMSE.
- Ours V1 has the best average MAE across the two transfer directions
  (`0.829`) compared with Lu official (`0.882`) and ST-GCN (`4.774`).
- ST-GCN is unstable in the tiny-source CNUH -> CARE-PD setting, likely because
  the unbounded regression head extrapolates poorly from only 21 training
  samples.

### 5. CARE-PD Leave-One-Dataset-Out

This validation was run with:

```text
scripts/run_carepd_lodo_cuda.cmd
```

| Held-out CARE-PD cohort | N train | N test | MAE | RMSE | MedAE |
|---|---:|---:|---:|---:|---:|
| 3DGait | 5,976 | 90 | 0.775 | 0.947 | 0.847 |
| BMCLab | 2,171 | 3,895 | 0.663 | 0.844 | 0.528 |
| PD-GaM | 4,366 | 1,700 | 0.495 | 0.724 | 0.236 |
| T-SDU-PD | 5,685 | 381 | 0.692 | 0.836 | 0.707 |
| **Overall** | - | 6,066 | **0.620** | **0.813** | **0.508** |

Analysis: CARE-PD LODO is harder than combined subject-level GroupKFold but
easier than CNUH -> CARE-PD zero-shot transfer. This means that CARE-PD itself
contains measurable cohort-level shift, but multi-cohort training still gives a
better starting point than training only on the small CNUH source set.

## Final Interpretation

The cross-dataset result should not be presented as successful zero-shot
generalization. The safer and more accurate interpretation is:

> Zero-shot cross-dataset transfer revealed a substantial site and
> representation domain gap. However, combined multi-domain training achieved
> strong subject-independent performance, CARE-PD leave-one-dataset-out showed
> that multi-cohort training improves generalization to unseen CARE-PD cohorts,
> and follow-up calibration analysis showed that part of the transfer error can
> be reduced with a small labeled target-site calibration set.

## Manuscript-Safe Claims

Safe:

- The proposed model performs strongly under combined subject-level GroupKFold.
- Direct zero-shot transfer across CNUH and CARE-PD remains difficult.
- The transfer failure appears to be calibration-related, not an invalid-output
  failure.
- A small target-site calibration set can reduce the transfer error without
  retraining the full model.
- CARE-PD leave-one-dataset-out provides stronger within-benchmark
  generalization evidence than ordinary subject-level folds.

Avoid:

- Do not claim complete cross-site zero-shot generalization.
- Do not claim that the model is fully domain-invariant.
- Do not overinterpret the CNUH dataset-wise result because N=21.

## Source Outputs

```text
results/cross_dataset_validation/summary.csv
results/cross_dataset_validation/domain_gap.csv
results/cross_dataset_validation/cnuh_to_carepd/predictions.tsv
results/cross_dataset_validation/carepd_to_cnuh/predictions.tsv
results/carepd_leave_one_dataset_out/summary.csv
docs/cross_dataset_validation_analysis.md
docs/cross_dataset_model_comparison.md
docs/carepd_lodo_analysis.md
docs/dataset_wise_breakdown_analysis.md
docs/score_balanced_transfer_analysis.md
docs/fewshot_calibration_analysis.md
```
