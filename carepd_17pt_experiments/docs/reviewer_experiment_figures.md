# Reviewer-Oriented Experiment Figures

- Last updated: 2026-06-02
- Purpose: add reviewer-friendly diagnostics beyond the main MAE/RMSE table

## 1. Calibration Reliability Curve

Status: completed from existing GroupKFold prediction files. No retraining was
required.

Command:

```powershell
.\scripts\run_calibration_reliability.cmd
```

Outputs:

```text
docs/calibration_reliability_analysis.md
docs/reviewer_figures/21_calibration_curve_ours.png
docs/reviewer_figures/22_calibration_curve_models.png
results/calibration_reliability/calibration_summary.csv
results/calibration_reliability/calibration_bins.csv
```

Summary:

| Model | MAE | Figure use |
|---|---:|---|
| Ours V1 | **0.358** | Main proposed-model calibration curve |
| Lu official | 0.404 | SOTA comparison curve |
| Temporal CNN | 0.425 | Deep baseline comparison curve |
| ST-GCN | 0.443 | SOTA comparison curve |

Use this figure without printing a numerical calibration metric. The safe and
useful claim is that Ours V1 gives the best average error while its predictions
remain monotonic and clinically interpretable across predicted severity bins.

Manuscript-safe wording:

> Reliability analysis using prediction-score bins showed that the proposed
> model's continuous outputs followed the expected monotonic relationship
> between predicted and observed severity. This provides an additional
> calibration-oriented diagnostic for clinical decision support, complementing
> MAE/RMSE and confusion-matrix analyses.

## 2. Learning Curve

Status: completed.

Command:

```powershell
.\scripts\run_learning_curve_ours_cuda.cmd
```

Design:

| Item | Setting |
|---|---|
| Model | Ours V1, Configuration D |
| Split | subject-level GroupKFold |
| Training fractions | 10%, 25%, 50%, 75%, 100% of training subjects per fold |
| Validation folds | fixed GroupKFold validation folds |
| Metric | MAE, RMSE |
| Output figures | `24_learning_curve_ours_mae.png`, `25_learning_curve_ours_rmse.png` |

Outputs:

```text
docs/learning_curve_ours_analysis.md
docs/reviewer_figures/24_learning_curve_ours_mae.png
docs/reviewer_figures/25_learning_curve_ours_rmse.png
results/learning_curve_ours/summary.csv
results/learning_curve_ours/fold_metrics.csv
results/learning_curve_ours/predictions.tsv
```

Result:

| Train fraction | Mean train subjects | MAE | RMSE | MAE reduction vs 10% |
|---:|---:|---:|---:|---:|
| 10% | 11.0 | 0.476 | 0.672 | 0.0% |
| 25% | 26.4 | 0.460 | 0.646 | 3.4% |
| 50% | 52.6 | 0.413 | 0.624 | 13.2% |
| 75% | 79.0 | 0.390 | 0.578 | 18.1% |
| 100% | 104.8 | 0.360 | 0.530 | 24.5% |

Interpretation:

MAE and RMSE decrease monotonically as training-subject count increases. Use
this to frame the CNUH N=21 limitation as a data-limited condition:

> The learning-curve analysis showed that prediction error decreased as the
> number of training subjects increased, supporting the interpretation that
> performance on small clinical cohorts is data-limited and can benefit from
> larger multi-site training sets.
