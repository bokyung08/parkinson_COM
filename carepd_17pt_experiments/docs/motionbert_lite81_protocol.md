# MotionBERT-Lite 81-Frame Protocol

- Last updated: 2026-06-05
- Model key: `motionbert_lite_pretrained`
- Reporting label: `MotionBERT-Lite (81-frame)`
- Output directory: `results/groupkfold_h36m17_motionbert_lite81_cuda`
- Status: completed, 5/5 folds

## Rationale

The original MotionBERT temporal window (`T=243`) was computationally
impractical in the current 5-fold setting. The run saturated GPU memory and did
not finish fold 1 after many hours.

The revised protocol keeps the evaluation design intact and reduces only the
MotionBERT temporal adapter:

| Item | Setting |
|---|---|
| Split | subject-level GroupKFold, 5 folds |
| Dataset | CNUH + CARE-PD |
| Input skeleton | H36M-compatible 17 joints |
| Ablation features | Configuration D |
| Epochs | 80 |
| Batch size | 8 |
| MotionBERT temporal window | 81 frames |

This is preferable to reducing folds or subsampling the dataset because the
evaluation protocol remains directly comparable. The only model-specific change
is the temporal window used by the MotionBERT input adapter.

## Manuscript-Safe Wording

> MotionBERT-Lite was evaluated using an 81-frame uniformly sampled input
> adapter to reduce the quadratic cost of temporal self-attention, while
> preserving the same subject-level GroupKFold split, number of training epochs,
> and 17-joint representation.

## Command

```powershell
cd C:\Users\bokyung\Desktop\parkinson_COM\carepd_17pt_experiments
.\scripts\run_motionbert_lite_pretrained_cuda.cmd
```

If the old 243-frame run is still active, stop that terminal first before
starting this run; otherwise both jobs will compete for the same GPU.

## Completed Result

| Model | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| MotionBERT-Lite (81-frame) | 5 | 6,087 | 10,814,222 | 5.243 | 0.442 | 0.625 | 0.247 |

Post-hoc rounded-score F1 was computed from `predictions.tsv`.

| Model | F1 0-3 | F1 0-2 |
|---|---:|---:|
| MotionBERT-Lite (81-frame) | 0.457 | 0.613 |

`F1 0-2` excludes true score-3 samples from the evaluation set, matching the
definition used in the manuscript draft.

Interpretation: MotionBERT-Lite (81-frame) is computationally tractable and
performs similarly to ST-GCN in MAE/RMSE, but it does not improve over Ours V1.
