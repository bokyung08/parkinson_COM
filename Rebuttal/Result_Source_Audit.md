# Result Source Audit

This file checks where each number in the rebuttal-ready summaries came from.

## Short Answer

1. The classical baseline models are expected to run very fast here.
   - Only 21 samples are used.
   - Each sequence is converted to pooled tabular statistics.
   - Final baseline input size is 21 x 1782.
   - These are sklearn models, not deep sequence models trained for epochs.

2. The proposed model and ablation numbers were not newly trained by the rebuttal script.
   - They were read from existing result files under `results/`.
   - The newly generated results are only the classical baselines under `Rebuttal/results/`.

## Classical Baseline Runtime Check

Source: `Rebuttal/results/item10_D_80_20/baseline_summary.csv`
   
Hold-out setting:

| Model | n_train | n_test | train_seconds | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Ridge | 16 | 5 | 0.024 | 0.435 | 0.567 |
| Random Forest | 16 | 5 | 0.483 | 0.468 | 0.556 |
| SVR_RBF | 16 | 5 | 0.004 | 0.451 | 0.571 |
| MLP | 16 | 5 | 0.070 | 1.845 | 2.320 |

5-fold CV total training time:

| Model | train_seconds_total | MAE | RMSE |
|---|---:|---:|---:|
| Ridge | 0.025 | 0.839 | 1.052 |
| Random Forest | 2.218 | 0.735 | 0.905 |
| SVR_RBF | 0.021 | 0.783 | 1.010 |
| MLP | 0.356 | 3.146 | 4.570 |

Conclusion: these runtimes are plausible because the dataset is tiny and the models are trained on pooled statistics, not raw video or full neural sequence tensors.

## Baseline Feature Construction

Source files:

- Script: `Rebuttal/run_baseline_comparison.py`
- Config: `Rebuttal/results/item10_D_80_20/run_config.json`
- Manifest: `Rebuttal/results/item10_D_80_20/dataset_manifest.tsv`

Run configuration:

| Item | Value |
|---|---|
| target | item10 |
| ablation | D |
| max_len | 390 |
| samples | 21 |
| holdout_test_size | 0.2 |
| feature_summary | stats |

Each sample starts as `390 x 33 x 9` Configuration D features and is summarized into pooled tabular statistics. The script uses mean, std, min, max, 25th percentile, and 75th percentile over the frame axis, giving 33 x 9 x 6 = 1782 features per sample.

## Proposed Main Model Numbers

The paper-ready documents currently use:

| Config | MAE | RMSE | Source used |
|---|---:|---:|---|
| A | 0.385 | 0.560 | `results/model_comparison_summary.csv` and `results/main_ablation/20260119_194927/ablation_summary.json` |
| B | 0.471 | 0.517 | same as above |
| C | 0.438 | 0.472 | same as above |
| D | 0.407 | 0.460 | same as above |

Important note:

`results/main_ablation/20260119_194927/ablation_summary.csv` is not fully consistent with the JSON/metrics files. In particular, it reports A MAE as 0.485, while `ablation_summary.json`, `abl_A/plots/metrics.json`, and `model_comparison_summary.csv` report A MAE as 0.385. The rebuttal-ready documents use the JSON/model-comparison line, not that CSV line.

Recommended source for manuscript consistency:

- Use `results/model_comparison_summary.csv` for the compact table.
- Cross-check with `results/main_ablation/20260119_194927/ablation_summary.json`.
- Avoid citing `results/main_ablation/20260119_194927/ablation_summary.csv` unless it is regenerated or reconciled.

## Fusion TF Ablation Numbers

The paper-ready documents currently use:

| Config | MAE | RMSE | Source used |
|---|---:|---:|---|
| A | 0.452 | 0.657 | `results/model_comparison_summary.csv`, `results/fusion_tf_ablation/20260119_222635/ablation_summary.json`, and `abl_A/metrics.json` |
| B | 0.537 | 0.665 | same JSON/model-comparison source |
| C | 0.491 | 0.693 | same JSON/model-comparison source |
| D | 0.401 | 0.631 | same JSON/model-comparison source |

Important note:

`results/fusion_tf_ablation/20260119_222635/ablation_summary.csv` has a discrepancy for D MAE, showing 0.421, while `ablation_summary.json`, `abl_D/metrics.json`, and `model_comparison_summary.csv` show 0.401. The paper-ready documents use 0.401.

## Hybrid Torch Ablation Numbers

The paper-ready documents currently use:

| Config | MAE | RMSE | MedAE | Source used |
|---|---:|---:|---:|---|
| A | 0.447 | 0.534 | 0.214 | `results/model_comparison_summary.csv` and `results/hybrid_ablation/20260119_230604/ablation_summary.json` |
| B | 0.476 | 0.493 | 0.372 | same as above |
| C | 0.446 | 0.548 | 0.195 | same as above |
| D | 0.444 | 0.588 | 0.147 | same as above |

The MAE/RMSE/MedAE values are consistent between CSV and JSON for this run. Some correlation/sign metrics differ between CSV and JSON, but those metrics are not used in the current rebuttal-ready text.

## What Was Newly Run

Newly run during this rebuttal work:

| Output | Meaning |
|---|---|
| `Rebuttal/results/item10_D_80_20/` | New classical baseline run for item10 |
| `Rebuttal/results/gait_D_80_20/` | Optional classical baseline run for gait sum target |

Not newly run during this rebuttal work:

| Output | Meaning |
|---|---|
| `results/model_comparison_summary.csv` | Existing model comparison summary |
| `results/main_ablation/20260119_194927/` | Existing proposed model ablation |
| `results/fusion_tf_ablation/20260119_222635/` | Existing Fusion TF ablation |
| `results/hybrid_ablation/20260119_230604/` | Existing Hybrid Torch ablation |

## Practical Recommendation

For rebuttal/manuscript:

1. Use MAE/RMSE/MedAE only.
2. Avoid R2, MAPE, and correlation metrics in the main response because several are unstable or inconsistent across small hold-out files.
3. State that the added classical baselines were run on pooled Configuration D statistics.
4. State that the proposed model results are from the existing Configuration D sequence-model run.
5. If time allows, regenerate all A/B/C/D ablations into one fresh `Rebuttal/results/` folder to remove ambiguity from old CSV/JSON inconsistencies.

## Unified ABCD DL Baseline Runner

New runner:

```powershell
python Rebuttal\run_unified_abcd_dl_baselines.py `
  --processed_dir HospitalData\processed_pose_data `
  --label_dir HospitalData\JSON `
  --target item10 `
  --test_size 0.1 `
  --epochs 20 `
  --batch_size 4 `
  --learning_rate 1e-4 `
  --device cpu
```

Default behavior:

- Runs FusionTF and HybridTorch for ablations A, B, C, and D.
- Writes per-ablation outputs to `Rebuttal/results/unified_dl_baselines/{A,B,C,D}_item10_90_10/`.
- Reuses a completed model output when its `row.json` already exists. Use `--overwrite` to rerun existing D outputs as well.
- Writes an aggregate ABCD table to `Rebuttal/results/unified_dl_baselines/unified_abcd_item10_90_10_RESULTS.md`.

## LOSO-CV Three-Model Runner

New LOSO-CV runner:

```powershell
python Rebuttal\run_loso_three_dl_models.py `
  --processed_dir HospitalData\processed_pose_data `
  --label_dir HospitalData\JSON `
  --target item10 `
  --ablations D `
  --models all `
  --epochs 20 `
  --batch_size 4 `
  --learning_rate 1e-4 `
  --device cuda `
  --out_root Rebuttal\results\loso_dl_models
```

Default LOSO behavior:

- Leaves out one `patient_id` at a time.
- Runs three sequence models when `--models all` is used: `main_tf`, `fusion_tf`, and `hybrid_torch`.
- Writes fold outputs to `Rebuttal/results/loso_dl_models/D_item10_loso/fold_*/{main_tf,fusion_tf,hybrid_torch}/`.
- Reuses completed fold/model outputs when `row.json` already exists. Use `--overwrite` to rerun completed folds.
- Writes the aggregate LOSO table to `Rebuttal/results/loso_dl_models/loso_item10_D_RESULTS.md`.

To run all feature configurations under LOSO-CV, replace `--ablations D` with:

```powershell
--ablations A B C D
```

To rebuild only the summary files after partial runs:

```powershell
python Rebuttal\run_loso_three_dl_models.py `
  --target item10 `
  --ablations D `
  --models all `
  --out_root Rebuttal\results\loso_dl_models `
  --collect_only
```
