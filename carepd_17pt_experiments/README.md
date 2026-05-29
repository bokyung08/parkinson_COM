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
- `docs/ours_abcd_summary.md`

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
