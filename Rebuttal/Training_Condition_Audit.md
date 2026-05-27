# Training Condition Audit

## Bottom Line

The three deep models were not trained under fully identical conditions.

`run_all_ablation.py` passes the same `epochs` and `batch_size` arguments to all three ablation runners, but the internal training/evaluation protocol differs by model.

## Condition Comparison

| Item | Proposed Main TF | Fusion TF Baseline | Hybrid Torch Baseline |
|---|---|---|---|
| Runner | `src/run_main_ablation.py` | `src/run_fusion_ablation.py` | `src/run_hybrid_ablation.py` |
| Model builder | `src/model_builder.py` | `src/train_hybrid_fusion.py` | `src/hybrid_gcn.py` |
| Split | 90/10 hold-out | 80/20 hold-out | 80/20 hold-out |
| Split seed | `random_state=42` | `random_state=42` | `random_state=42` |
| LOSO-CV | No | No | No |
| `folds` argument used? | No | No practical CV use | No practical CV use |
| Epochs | passed from CLI | passed from CLI | passed from CLI |
| Batch size | passed from CLI | passed from CLI | passed from CLI |
| Optimizer | Adam | Adam | Adam |
| Learning rate | `1e-4` | `1e-3` | `1e-3` |
| Loss | MSE | MSE | L1 / MAE |
| LR scheduler | ReduceLROnPlateau | ReduceLROnPlateau | ReduceLROnPlateau |
| Gradient clipping | `clipnorm=1.0` in optimizer | not explicit | `clip_grad_norm_=1.0` |
| Sequence length | 390 frames | 390 frames | 390 frames |
| Truncation behavior | Keras `pad_sequences`, default pre-truncation, keeps last 390 frames | `pad_or_clip`, keeps first 390 frames | dataset clip, keeps first 390 frames |
| Target loader | `load_labels(..., target="item10")` default | same `load_labels` default | same `load_labels` default |

## Key Implications

1. The models share the same broad dataset, target loader, ablation definitions, epoch argument, and batch-size argument.
2. They do not share the same validation split size.
3. They do not share the same learning rate.
4. They do not share the same loss function.
5. They do not process the temporal window in exactly the same way.
6. None of the three ablation runners applies LOSO-CV.

## Reviewer-Risk Assessment

Because these conditions are not fully matched, a large performance gap between the proposed model and the two deep baselines could invite reviewer concerns:

- whether the baseline models were under-optimized,
- whether the comparison protocol was inconsistent,
- whether the result is split-dependent in a small dataset,
- whether LOSO-CV should be used for a stronger patient-level evaluation.

## Recommended Rebuttal Position

Do not claim that all deep models were trained under strictly identical optimization settings.

Safer wording:

> We used the same processed pose dataset, target definition, feature configurations, random seed, and epoch/batch-size budget across the ablation runs. However, because each architecture follows its original training implementation, optimizer details such as learning rate, loss function, and temporal truncation strategy differ. To address the reviewer's concern about baseline strength, we additionally included classical machine-learning baselines under a unified Configuration D feature protocol.

## Recommended Next Experiment

For the cleanest rebuttal, run a fresh unified comparison under `Rebuttal/results/` with:

| Condition | Recommended value |
|---|---|
| Split | same 80/20 hold-out or LOSO-CV for all models |
| Target | item10 or gait sum, fixed explicitly |
| Max length | same 390 frames |
| Truncation | same first or last window for all models |
| Epochs | same |
| Batch size | same |
| Learning-rate search | at least `{1e-4, 5e-4, 1e-3}` for each deep model |
| Metrics | MAE, RMSE, MedAE only |

## Important Note About the User-Cited 1.x MAE Values

The currently inspected `results/model_comparison_summary.csv` does not show Hybrid Fusion or Efficient Hybrid MAE values around 1.2-1.9 for the item10 deep-model ablation. It shows approximately:

| Model | MAE range in inspected summary |
|---|---:|
| Fusion TF | 0.401-0.537 |
| Hybrid Torch | 0.444-0.476 |

If the 1.x MAE values come from another run, target definition, LOSO-CV experiment, or gait-sum target, that exact result folder should be audited separately before using those numbers in the manuscript.
