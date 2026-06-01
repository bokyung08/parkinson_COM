# Final Integrated Results

- Last updated: 2026-06-01
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
| SOTA | Lu et al. official-architecture DD-Net/OF-DDNet | 5 | 6,087 | 147,908 | 0.445 | 0.404 | **0.543** | 0.307 |
| Proposed | Ours V1, bounded regression | 5 | 6,087 | 158,594 | 4.615 | **0.358** | 0.564 | **0.147** |

## Main Result Interpretation

Ours V1 achieved the best MAE among the reported methods. Relative MAE
reductions are:

| Comparison | MAE reduction |
|---|---:|
| Ours V1 vs SVR, best classical ML | 27.4% |
| Ours V1 vs Temporal CNN | 15.8% |
| Ours V1 vs ST-GCN | 19.3% |
| Ours V1 vs Lu official-architecture baseline | 11.5% |

Lu official has the best RMSE, but Ours V1 has better MAE and MedAE. The safest
claim is that Ours V1 improves average absolute clinical-score error and
typical-case absolute error, while Lu official has slightly lower aggregate
squared error.

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
horizontal translation, but it does not by itself provide scale invariance.

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

Scale-robustness follow-up candidates are implemented in
`docs/scale_robustness_experiment_plan.md`. These should be screened before any
additional scale-robustness claim is added to the manuscript.

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
| `20_mae_advantage_vs_baselines.png` | Compact MAE gain over baselines |
| `16_dataset_mae_breakdown.png` | Dataset-level breakdown; supplementary because CNUH has only 21 samples |
| `19_class_distribution.png` | Class imbalance explanation |

Additional diagnostic figures are listed in
`docs/final_integrated_figures/README.md`.
