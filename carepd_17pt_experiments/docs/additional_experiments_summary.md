# Additional Experiment Summary

- Last updated: 2026-06-05
- Dataset: CNUH + CARE-PD, H36M-compatible 17-joint gait sequences
- Main target: MDS-UPDRS Part III item 3.10 gait score, range 0-3
- Primary model: Ours V1, Configuration D, bounded regression

## Executive Summary

The additional experiments support keeping Ours V1 as the final proposed
model. The safest manuscript claim is:

> Ours V1 achieves the best MAE under identical subject-level GroupKFold
> evaluation and the lowest average zero-shot transfer MAE across the two
> transfer directions. Lu official achieves the best RMSE in the combined
> GroupKFold comparison, and MotionAGFormer-XS achieves the best MedAE.

Do not claim that Ours V1 is best on every metric. The defensible emphasis is
average absolute clinical-score error, transfer stability, selective prediction,
and lightweight inference cost.

## 1. Encoder Comparison

All models below use the same combined CNUH+CARE-PD 17-joint GroupKFold
5-fold protocol.

| Category | Model | Params | Infer ms/sample | MAE | RMSE | MedAE |
|---|---|---:|---:|---:|---:|---:|
| Proposed | Ours V1 | 158,594 | 4.615 | **0.358** | 0.564 | 0.147 |
| SOTA | Lu official | 147,908 | 0.445 | 0.404 | **0.543** | 0.307 |
| SOTA | MotionAGFormer-XS | 2,307,324 | 6.150 | 0.405 | 0.638 | **0.095** |
| Deep Learning | Temporal CNN | 188,929 | 0.294 | 0.425 | 0.594 | 0.287 |
| SOTA | MotionBERT-Lite (81-frame) | 10,814,222 | 5.243 | 0.442 | 0.625 | 0.247 |
| SOTA | ST-GCN | 252,097 | 22.523 | 0.443 | 0.623 | 0.274 |

Interpretation:

- Ours V1 has the best MAE, which is the primary clinical-score error metric.
- Lu official has the best RMSE, suggesting lower squared-error aggregation.
- MotionAGFormer-XS has the best MedAE, but its MAE/RMSE are weaker than Ours
  V1.
- MotionBERT-Lite (81-frame) is computationally tractable but performs close to
  ST-GCN and does not improve over Ours V1.

## 2. Feature Ablation

| Config | Feature set | Params | Infer ms/sample | MAE | RMSE | MedAE |
|---|---|---:|---:|---:|---:|---:|
| A | coordinates only | 158,210 | 3.179 | 0.374 | 0.577 | 0.168 |
| B | coordinates + velocity | 158,402 | 2.757 | 0.432 | 0.629 | 0.246 |
| C | coordinates + velocity + amplitude/variability | 158,530 | 4.178 | 0.376 | **0.549** | 0.192 |
| D | full hybrid feature set | 158,594 | 4.615 | **0.358** | 0.564 | **0.147** |

Configuration D remains the final operating point because it gives the best MAE
and MedAE. Configuration C gives the best RMSE.

## 3. Architecture Ablation

All rows use Configuration D features. The full Ours V1 row is the canonical
final 5-fold run.

| Model | Components | Params | Infer ms/sample | MAE | RMSE | MedAE |
|---|---|---:|---:|---:|---:|---:|
| MLP only | mean pooling + bounded MLP | 17,921 | 0.005 | 0.554 | 0.653 | 0.481 |
| GraphConv + MLP | GraphConv, no joint attention, no Temporal Transformer | 25,985 | 2.306 | 0.450 | 0.580 | 0.349 |
| GraphConv + Joint Attention + MLP | GraphConv + joint attention, no Temporal Transformer | 26,114 | 2.736 | 0.414 | 0.564 | 0.291 |
| Full Ours V1 | GraphConv + Joint Attention + Temporal Transformer | 158,594 | 4.615 | **0.358** | 0.564 | **0.147** |

Interpretation: each architectural component improves MAE. The full model
reduces MAE by 35.4% relative to the MLP-only baseline.

## 4. Zero-Shot Cross-Dataset Transfer

No fine-tuning, adaptation, or test-set checkpoint selection was used.

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

Interpretation:

- All models degrade under strict cross-dataset transfer.
- MotionAGFormer-XS is marginally best for CNUH -> CARE-PD.
- Lu official is best for CARE-PD -> CNUH.
- Ours V1 has the lowest average transfer MAE across both directions.
- ST-GCN is unstable when trained only on the 21-sample CNUH source set.

Manuscript-safe wording:

> Strict zero-shot transfer revealed a substantial domain gap across clinical
> sites and pose representations. The proposed model achieved the lowest
> average transfer MAE across both directions, although MotionAGFormer-XS was
> marginally better in the CNUH-to-CARE-PD direction and Lu official was
> slightly better in the CARE-PD-to-CNUH direction. These results support the
> need for site calibration while showing that the proposed bounded model is a
> comparatively stable transfer baseline.

## 5. Selective Prediction

Selective prediction keeps predictions farthest from rounded-score decision
boundaries and flags boundary-proximal cases for clinician review.

Ours V1 coverage curve:

| Coverage | N kept | MAE | RMSE | MedAE | Rounded accuracy | MAE reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 100% | 6,087 | 0.358 | 0.564 | 0.147 | 0.711 | 0.0% |
| 90% | 5,478 | 0.325 | 0.540 | 0.116 | 0.744 | 9.2% |
| 80% | 4,870 | 0.288 | 0.510 | 0.088 | 0.774 | 19.4% |
| 70% | 4,261 | 0.260 | 0.493 | 0.064 | 0.793 | 27.2% |
| 60% | 3,652 | 0.234 | 0.477 | 0.045 | 0.811 | 34.5% |
| 50% | 3,044 | 0.211 | 0.462 | 0.030 | 0.825 | 40.9% |

Model comparison at 80% coverage:

| Category | Model | Retained MAE | Retained RMSE | Retained MedAE | Rounded accuracy |
|---|---|---:|---:|---:|---:|
| Proposed | Ours V1 | **0.288** | **0.510** | 0.088 | **0.774** |
| SOTA | MotionAGFormer-XS | 0.351 | 0.612 | **0.054** | 0.700 |
| SOTA | Lu official | 0.369 | 0.527 | 0.207 | 0.720 |
| SOTA | MotionBERT-Lite (81-frame) | 0.384 | 0.578 | 0.149 | 0.674 |
| SOTA | ST-GCN | 0.385 | 0.575 | 0.176 | 0.704 |

Interpretation: selective prediction strengthens the clinical decision-support
story. It shows that the model can auto-score lower-risk cases more accurately
and defer boundary-proximal cases to clinicians.

## 6. Latency Benchmark

This benchmark measures architecture-level forward-pass latency with randomly
initialized models under identical input length and batch size. It is useful
for deployment-cost comparison, not for trained-weight accuracy.

| Category | Model | Params | Device | Batch | ms/sample |
|---|---|---:|---|---:|---:|
| Deep Learning | Temporal CNN | 188,929 | cuda | 32 | **0.020** |
| Proposed | Ours V1 | 158,594 | cuda | 32 | 0.242 |
| SOTA | Lu official | 147,908 | cuda | 32 | 0.335 |
| SOTA | ST-GCN | 252,097 | cuda | 32 | 0.522 |
| SOTA | MotionAGFormer-XS | 2,307,324 | cuda | 32 | 1.937 |
| SOTA | MotionBERT-Lite | 10,814,222 | cuda | 32 | 16.481 |

Interpretation: Temporal CNN is fastest but less accurate than Ours V1. Ours V1
is faster than Lu, ST-GCN, MotionAGFormer-XS, and MotionBERT-Lite in this
forward-pass benchmark while also achieving the best MAE in the main
GroupKFold comparison.

## Recommended Manuscript Placement

| Result | Where to use |
|---|---|
| Encoder comparison | Main results table |
| Feature ablation | Ablation section |
| Architecture ablation | Reviewer-response/main ablation table |
| Zero-shot transfer | Cross-dataset validation section |
| Selective prediction | Clinical decision-support / uncertainty workflow section |
| Latency benchmark | Deployment and efficiency section |

## Source Files

```text
docs/final_integrated_results.md
docs/final_integrated_results_ko.md
docs/motionbert_lite81_groupkfold_result.md
docs/motionagformer_xs_groupkfold_result.md
docs/cross_dataset_model_comparison.md
docs/cross_dataset_model_comparison_v2.md
docs/selective_prediction_analysis.md
docs/latency_benchmark.md
results/cross_dataset_model_comparison/summary.csv
results/cross_dataset_model_comparison_v2/summary.csv
results/selective_prediction/summary.csv
results/latency_benchmark/summary.csv
```
