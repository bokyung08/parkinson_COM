# MotionAGFormer-XS GroupKFold Result

- Last updated: 2026-06-05
- Source run: `results/groupkfold_h36m17_motionagformer_xs_pretrained_cuda`
- Split: subject-level GroupKFold, 5 folds
- Dataset: CNUH + CARE-PD, H36M-compatible 17-joint sequences
- Target: MDS-UPDRS item 3.10 gait score, range 0-3

## Completed Metrics

| Model | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| MotionAGFormer-XS | 5 | 6,087 | 2,307,324 | 6.150 | 0.405 | 0.638 | 0.095 |

F1 values were computed post-hoc by rounding continuous predictions to the
nearest integer and clipping to `[0, 3]`.

| Model | F1 0-3 | F1 0-2 |
|---|---:|---:|
| MotionAGFormer-XS | 0.561 | 0.636 |

`F1 0-2` excludes true score-3 samples from the evaluation set, matching the
definition used in the manuscript draft.

## Interpretation

MotionAGFormer-XS is now a completed SOTA encoder baseline. Its MAE is close to
the Lu official baseline (`0.405` vs. `0.404`) but remains higher than Ours V1
(`0.358`). Its RMSE (`0.638`) is weaker than both Ours V1 (`0.564`) and Lu
official (`0.543`).

The notable result is MedAE: MotionAGFormer-XS achieves the lowest median
absolute error (`0.095`). Therefore, manuscript wording should separate the
primary MAE claim from the median-error claim:

> Ours V1 achieved the best MAE, Lu et al. achieved the best RMSE, and
> MotionAGFormer-XS achieved the best MedAE among completed baselines.

## Files

- `summary.csv`: final aggregate metrics
- `fold_metrics.csv`: fold-level metrics
- `predictions.tsv`: sample-level predictions used for F1 calculation
- `README.md`: run progress and completion summary
