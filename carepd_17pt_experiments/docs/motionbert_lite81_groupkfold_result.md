# MotionBERT-Lite 81-Frame GroupKFold Result

- Last updated: 2026-06-05
- Source run: `results/groupkfold_h36m17_motionbert_lite81_cuda`
- Split: subject-level GroupKFold, 5 folds
- Dataset: CNUH + CARE-PD, H36M-compatible 17-joint sequences
- Target: MDS-UPDRS item 3.10 gait score, range 0-3
- Temporal adapter: 81 uniformly sampled frames

## Completed Metrics

| Model | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| MotionBERT-Lite (81-frame) | 5 | 6,087 | 10,814,222 | 5.243 | 0.442 | 0.625 | 0.247 |

F1 values were computed post-hoc by rounding continuous predictions to the
nearest integer and clipping to `[0, 3]`.

| Model | F1 0-3 | F1 0-2 |
|---|---:|---:|
| MotionBERT-Lite (81-frame) | 0.457 | 0.613 |

`F1 0-2` excludes true score-3 samples from the evaluation set, matching the
definition used in the manuscript draft.

## Interpretation

MotionBERT-Lite (81-frame) is now a completed SOTA Transformer encoder
baseline. Its MAE (`0.442`) and RMSE (`0.625`) are close to ST-GCN
(`0.443` and `0.623`) but remain worse than Ours V1 (`0.358` and `0.564`) and
Lu official (`0.404` and `0.543`).

The result supports the manuscript claim that the proposed lightweight
graph-attention-temporal model achieves better average absolute error than
larger generic Transformer-style skeleton encoders under the same 17-joint
GroupKFold evaluation.

## Files

- `summary.csv`: final aggregate metrics
- `fold_metrics.csv`: fold-level metrics
- `predictions.tsv`: sample-level predictions used for F1 calculation
- `README.md`: run progress and completion summary
