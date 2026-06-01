# CARE-PD / External 17-Joint Gait Experiments

This folder is independent from the existing `Rebuttal/` experiments. It is for
the dataset-added version of the paper experiments while preserving all previous
results.

All datasets are converted to one H36M-compatible 17-joint gait format before
training or evaluation. See `docs/data_format.md` for the exact manifest and
joint schema. The experiment set includes:

- Classical ML: Ridge, SVR, Random Forest, shallow MLP.
- Deep learning: temporal CNN.
- Ours: COM-anchored GCN + temporal Transformer on hybrid node features.
- SOTA skeleton baseline: ST-GCN.
- Parkinson gait baseline: Lu et al. MICCAI 2020 official-architecture DD-Net/OF-DDNet port.

Use `lu_ofddnet_official` for the manuscript SOTA comparison. It ports the
official Keras architecture structure: joint collection distances, slow/fast
motion branches, temporal 1D convolutions, dense classifier, and
ordinal-focal classification. The earlier simplified Lu-style baseline is
excluded from the active runner and should not be reported as the official
comparison.

## Latest Result Snapshot

The current manuscript-facing reports are:

- `docs/final_integrated_results.md`
- `docs/final_integrated_results_ko.md`
- `docs/ours_abcd_summary.md`
- `docs/per_class_confusion_analysis.md`
- `docs/com_robustness_status.md`
- `docs/com_robustness_final_analysis.md`
- `docs/cross_dataset_validation_analysis.md`
- `docs/cross_dataset_validation_record_en.md`
- `docs/cross_dataset_validation_record_ko.md`
- `docs/carepd_lodo_analysis.md`
- `docs/calibration_reliability_analysis.md`
- `docs/learning_curve_ours_analysis.md`
- `docs/architecture_ablation_analysis.md`
- `docs/reviewer_revision_action_plan.md`
- `docs/carepd_cohort_split_table.md`
- `docs/lu_reimplementation_fairness.md`
- `docs/reviewer_experiment_figures.md`
- `docs/domain_gap_followup_experiments.md`
- `docs/scale_robustness_full_summary.md`
- `docs/final_integrated_figures/`

Recommended main-text figures:

| Figure | Use |
|---|---|
| `docs/final_integrated_figures/02_mae_ranking.png` | Main model comparison |
| `docs/final_integrated_figures/07_ablation_metrics.png` | Feature ablation |
| `docs/final_integrated_figures/09_ours_per_class_error.png` | Per-class error |
| `docs/final_integrated_figures/12_ours_confusion_normalized.png` | Rounded-score confusion matrix |

Recommended supplementary figures:

| Figure | Use |
|---|---|
| `docs/final_integrated_figures/05_fold_mae_by_model.png` | Fold-level stability |
| `docs/final_integrated_figures/18_calibration_curve_by_model.png` | Calibration-style score trend |
| `docs/final_integrated_figures/20_mae_advantage_vs_baselines.png` | MAE gain over selected baselines |
| `docs/final_integrated_figures/16_dataset_mae_breakdown.png` | Dataset-level breakdown |
| `docs/final_integrated_figures/19_class_distribution.png` | Class imbalance explanation |
| `docs/reviewer_figures/21_calibration_curve_ours.png` | Ours V1 calibration curve |
| `docs/reviewer_figures/22_calibration_curve_models.png` | Calibration curve comparison |

Reviewer-response learning curve is complete. MAE decreases from `0.476` with
10% training subjects to `0.360` with the full training set, a `24.5%` relative
reduction. See `docs/learning_curve_ours_analysis.md` and:

| Figure | Use |
|---|---|
| `docs/reviewer_figures/24_learning_curve_ours_mae.png` | Ours V1 MAE learning curve |
| `docs/reviewer_figures/25_learning_curve_ours_rmse.png` | Ours V1 RMSE learning curve |

Reviewer-requested architecture ablation runner:

```powershell
.\scripts\run_architecture_ablation_cuda.cmd
```

This runs `ours_mlp`, `ours_gcn_mlp`, `ours_gcn_attn_mlp`, and the full `ours`
model under the same subject-level GroupKFold setup.

Final main-table decision:

| Category | Model | MAE | RMSE | MedAE | Decision |
|---|---|---:|---:|---:|---|
| SOTA | ST-GCN | 0.443 | 0.623 | 0.274 | baseline |
| SOTA | Lu et al. official-architecture DD-Net/OF-DDNet | 0.404 | **0.543** | 0.307 | baseline |
| Proposed | Ours V1, bounded regression | **0.358** | 0.564 | **0.147** | final proposed model |

OursV2 is excluded from the main manuscript table because it did not improve the
primary MAE/RMSE metrics over Ours V1.

Current Ours V1 A/B/C/D ablation:

| Ablation | Feature set | MAE | RMSE | MedAE |
|---|---|---:|---:|---:|
| A | coordinates only | 0.374 | 0.577 | 0.168 |
| B | coordinates + velocity | 0.432 | 0.629 | 0.246 |
| C | coordinates + velocity + amplitude/variability | 0.376 | **0.549** | 0.192 |
| D | full hybrid feature set including angle | **0.358** | 0.564 | **0.147** |

The D row is unified to the best completed D run from
`results\groupkfold_h36m17_ours_lu_official_cuda`. A/B/C use the dedicated
ablation runs under `results\groupkfold_h36m17_ours_ablation_{A,B,C}_cuda`.

Current per-class Ours V1 result:

| Score | N | MAE | RMSE |
|---|---:|---:|---:|
| 0 | 2,615 | 0.225 | 0.395 |
| 1 | 2,183 | 0.342 | 0.482 |
| 2 | 1,244 | 0.649 | 0.889 |
| 3 | 45 | 0.738 | 0.930 |

COM robustness has now been evaluated with checkpointed fold inference. COM
centering was effectively invariant to horizontal translation, but scale
perturbations still increased error. See `docs/com_robustness_status.md` and
`docs/com_robustness/`.

Scale-robustness follow-up is complete. The recommended robust operating point
uses the same Ours V1 architecture with median-bone scale normalization and
moderate train-time scale augmentation. It preserves accuracy
(`MAE 0.366`, `RMSE 0.567`) and shows no measurable degradation under the tested
scale and translation perturbations. See
`docs/scale_robustness_full_summary.md` and `docs/scale_robustness_full/`.

Cross-dataset follow-up analyses are complete for the current Ours-focused
scope. The completed analyses
show that combined GroupKFold performance is strongest on CARE-PD
(`Ours V1 MAE 0.356`), score-balanced transfer exposes poor zero-shot severity
calibration, small target-site affine calibration can reduce transfer error
without retraining, and Ours V1 has the best average zero-shot transfer MAE
against ST-GCN and Lu official. CARE-PD leave-one-dataset-out is complete for
Ours V1 (`MAE 0.620`, `RMSE 0.813`) and shows moderate cohort-level shift inside
CARE-PD. See `docs/cross_dataset_validation_analysis.md`,
`docs/cross_dataset_model_comparison.md`, `docs/carepd_lodo_analysis.md`, and
`docs/domain_gap_followup_experiments.md`.

## Environment

```powershell
cd carepd_17pt_experiments
conda env create -f environment.yml
conda activate gait17_external
pip install -r requirements.txt
```

## Convert Existing CNUH MediaPipe Data

Run from this folder:

```powershell
python scripts\prepare_17pt_dataset.py `
  --input_dir ..\HospitalData\processed_pose_data `
  --label_json_dir ..\HospitalData\JSON `
  --dataset_name CNUH `
  --source_format mediapipe33 `
  --target item10 `
  --pattern "*_2_pose.npy"
```

This writes `data/processed/manifest.csv` and per-sample `.npz` files under
`data/processed/CNUH/`.

## Add CARE-PD or Another External Dataset

Only use samples that can be converted to `(T, 17, 3)` gait sequences and have
MDS-UPDRS gait labels. Do not paste commands containing `<...>`; those are
documentation placeholders, not valid PowerShell paths. If a dataset already
exposes H36M17 joints, replace the paths below with the actual local folder and
CSV names:

```powershell
python scripts\prepare_17pt_dataset.py `
  --input_dir data\raw\ExternalH36M17\poses `
  --label_csv data\raw\ExternalH36M17\labels.csv `
  --dataset_name EXTERNAL `
  --source_format h36m17 `
  --target_col target `
  --pattern "*.npy"
```

CARE-PD access/download is intentionally separate. The Hugging Face download
contains SMPL pickles and fold files. For this 17-joint experiment, use the
official CARE-PD h36m preprocessed files from Dataverse, or generate them with
the CARE-PD official preprocessing script (`scripts/preprocess_smpl2h36m.sh` in
the TaatiTeam/CARE-PD repository). The SMPL pickles alone are not valid input to
our 17-joint runner because they contain SMPL pose parameters, not joint
coordinates.

```powershell
python scripts\download_carepd.py --local_dir data\raw\CARE-PD
```

Inspect the UPDRS-labeled walks:

```powershell
python scripts\inspect_carepd_pickles.py --carepd_root data\raw\CARE-PD
```

After obtaining the official h36m `.npz` files:

```powershell
python scripts\prepare_carepd_h36m_dataset.py `
  --carepd_root data\raw\CARE-PD `
  --h36m_root data\raw\CARE-PD\h36m `
  --output_dir data\processed `
  --replace_dataset
```

If the converter cannot infer the `.npz` keys, inspect first:

```powershell
python scripts\prepare_carepd_h36m_dataset.py `
  --h36m_root data\raw\CARE-PD\h36m `
  --inspect_only
```

### Generate CARE-PD H36M From `CARE-PD-master`

If the upstream CARE-PD repository is checked out next to this folder as
`..\CARE-PD-master`, use the same downloaded CARE-PD pickles and generate only
the UPDRS-labeled cohorts needed for this experiment:

```powershell
cd ..\CARE-PD-master
pip install smplx==0.1.28 chumpy

New-Item -ItemType Directory -Force -Path assets\datasets
Copy-Item ..\carepd_17pt_experiments\data\raw\CARE-PD\3DGait.pkl assets\datasets\
Copy-Item ..\carepd_17pt_experiments\data\raw\CARE-PD\BMCLab.pkl assets\datasets\
Copy-Item ..\carepd_17pt_experiments\data\raw\CARE-PD\PD-GaM.pkl assets\datasets\
Copy-Item ..\carepd_17pt_experiments\data\raw\CARE-PD\T-SDU-PD.pkl assets\datasets\

python data\preprocessing\smpl2h36m.py -db 3DGait
python data\preprocessing\smpl2h36m.py -db BMCLab
python data\preprocessing\smpl2h36m.py -db PD-GaM

$env:KMP_DUPLICATE_LIB_OK='TRUE'
python data\preprocessing\smpl2h36m.py -db T-SDU-PD
```

Then return to this folder and merge CARE-PD with the existing CNUH manifest:

```powershell
cd ..\carepd_17pt_experiments
python scripts\prepare_carepd_h36m_dataset.py `
  --carepd_root data\raw\CARE-PD `
  --h36m_root ..\CARE-PD-master\assets\datasets\h36m `
  --output_dir data\processed `
  --replace_dataset
```

The current local conversion produces 6,066 CARE-PD H36M17 sequences from the
four UPDRS-labeled cohorts: 3DGait, BMCLab, PD-GaM, and T-SDU-PD.

## Run Subject-Level GroupKFold Experiments

Use subject-level splits for the dataset-added experiment. Do not use random
sample splits: CARE-PD contains multiple walks and downsampled sequences from
the same participants, so sample-level splitting would leak subject identity and
near-duplicate gait signals across train/test.

```powershell
python scripts\run_loso_experiments.py `
  --manifest data\processed\manifest.csv `
  --out_dir results\groupkfold_h36m17_all `
  --models ridge svr rf mlp_shallow temporal_cnn ours stgcn lu_ofddnet_official `
  --split_strategy groupkfold `
  --n_splits 5 `
  --ablation D `
  --target item10 `
  --epochs 80 `
  --batch_size 8
```

For the original CNUH-only comparison, keep LOSO so it remains directly
comparable with the existing paper results:

```powershell
python scripts\run_loso_experiments.py `
  --manifest data\processed\manifest.csv `
  --out_dir results\loso_h36m17_cnuh_only `
  --models ridge svr rf mlp_shallow temporal_cnn ours stgcn lu_ofddnet_official `
  --datasets CNUH `
  --split_strategy loso `
  --ablation D `
  --target item10 `
  --epochs 80 `
  --batch_size 8
```

To rerun only the official Lu et al. baseline on CUDA without touching existing
results:

```powershell
.\scripts\run_lu_official_cuda.cmd
```

This writes to `results\groupkfold_h36m17_lu_official_cuda`.

To rerun ST-GCN plus the official Lu et al. port in a clean SOTA folder:

```powershell
.\scripts\run_sota_official_cuda.cmd
```

This writes to `results\groupkfold_h36m17_sota_official_cuda`.

To rerun only the proposed model and official Lu et al. port on CUDA:

```powershell
.\scripts\run_ours_lu_official_cuda.cmd
```

This writes to `results\groupkfold_h36m17_ours_lu_official_cuda`. The proposed
model output is bounded to the clinical item-10 range by returning
`3 * sigmoid(raw_score)`.

To run the proposed V1 model across A/B/C/D ablations on CUDA:

```powershell
.\scripts\run_ours_abcd_cuda.cmd
```

This writes to `results\groupkfold_h36m17_ours_ablation_A_cuda`,
`results\groupkfold_h36m17_ours_ablation_B_cuda`,
`results\groupkfold_h36m17_ours_ablation_C_cuda`, and
`results\groupkfold_h36m17_ours_ablation_D_cuda`. Completed ablations are
skipped automatically when their `summary.csv` already exists.

After the ablation runs finish, generate the combined ablation table:

```powershell
python scripts\summarize_ours_abcd.py
```

This writes `docs\ours_abcd_summary.md`, mirrors the same markdown to
`results\OURS_ABCD_SUMMARY.md`, and writes `results\ours_abcd_summary.csv`.

To reproduce COM robustness:

```powershell
.\scripts\run_ours_d_checkpointed_cuda.cmd
.\scripts\run_com_robustness_cuda.cmd
```

The first command recreates the D fold checkpoints once. The second command
performs scale, translation, and combined perturbation inference without
additional training. The completed analysis is stored in `docs\com_robustness`.

To screen scale-robustness candidates:

```powershell
.\scripts\run_scale_robustness_screen_cuda.cmd
```

After screening, run only selected candidates at full 5-fold scale:

```powershell
.\scripts\run_scale_robustness_full_selected_cuda.cmd
```

Outputs:

- `README.md` with live progress while the run is still executing.
- `progress.csv`
- `fold_metrics.csv`
- `predictions.tsv`
- `summary.csv`
- `summary.json`
- `RESULTS.md`

If you stop a run with `Ctrl+C`, all completed folds remain saved. The fold that
was running at the interruption point is marked as `interrupted` in
`progress.csv` and the result-folder `README.md`.

## Sources

- Yan et al., ST-GCN, AAAI 2018.
- Lu et al., Vision-based Estimation of MDS-UPDRS Gait Scores, MICCAI 2020.
- CARE-PD dataset, `vida-adl/CARE-PD`, non-commercial research release.
