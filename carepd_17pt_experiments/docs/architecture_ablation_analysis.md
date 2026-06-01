# Architecture Ablation Analysis

- Split: subject-level GroupKFold, 5 folds
- Dataset: CNUH + CARE-PD H36M17
- Input feature configuration: D
- Full Ours V1 row: canonical final run from `groupkfold_h36m17_ours_lu_official_cuda`
- Note: the interrupted full `ours` rows inside `architecture_ablation_ours_cuda` are not used.

## Table

| Model | Components | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MLP only | mean pooling + bounded MLP | 5 | 6087 | 17921 | 0.005 | 0.554 | 0.653 | 0.481 |
| GraphConv + MLP | GraphConv, no joint attention, no Temporal Transformer | 5 | 6087 | 25985 | 2.306 | 0.450 | 0.580 | 0.349 |
| GraphConv + Joint Attention + MLP | GraphConv + joint attention, no Temporal Transformer | 5 | 6087 | 26114 | 2.736 | 0.414 | 0.564 | 0.291 |
| Full Ours V1 | GraphConv + Joint Attention + Temporal Transformer | 5 | 6087 | 158594 | 4.615 | 0.358 | 0.564 | 0.147 |

## Interpretation

The full model reduces MAE from `0.554` in the MLP-only baseline to `0.358`, a relative reduction of `35.4%`. Adding GraphConv substantially improves over MLP-only, adding joint attention further improves MAE, and the full Temporal Transformer model gives the best MAE and MedAE.

Compared with GraphConv + Joint Attention + MLP, adding the Temporal Transformer reduces MAE from `0.414` to `0.358` (`13.7%` relative reduction).

Manuscript-safe wording:

> Architecture ablation confirmed that the performance gain is not attributable solely to the input feature set. GraphConv improved over a mean-pooled MLP baseline, joint attention further reduced error, and the full GraphConv + Joint Attention + Temporal Transformer encoder achieved the lowest MAE and MedAE.
