# Parkinson COM-Centered Gait Severity Assessment

Research code for an RGB-only vision-based decision-support system that
estimates Parkinson's disease gait severity from pose sequences. The system
uses COM-centered coordinate normalization, multi-channel gait features, graph
convolution, joint attention, temporal Transformer encoding, and bounded
regression for MDS-UPDRS Part III item 3.10 gait scoring.

This repository is organized for manuscript reproduction while keeping private
clinical data, downloaded datasets, model checkpoints, and generated result
folders out of Git.

## Manuscript

**Title**
An intelligent vision-based decision-support system for Parkinson's gait
severity assessment using COM-centered spatio-temporal modeling from a single
RGB camera

**Authors**
Bokyung Kim, Hieyong Jeong, Md Ilias Bappi, Kyungbaek Kim
Chonnam National University, Gwangju, Republic of Korea

**Target journal**
Artificial Intelligence in Medicine

## Key Idea

The pipeline estimates a continuous MDS-UPDRS gait score from a single RGB
video without wearable sensors or specialized motion-capture hardware.

1. Extract body pose from RGB video.
2. Convert poses to a common 17-joint H36M-compatible skeleton.
3. Normalize joint coordinates around a hip-midpoint COM proxy.
4. Build multi-channel gait features:
   - COM-centered joint coordinates
   - joint velocity
   - motion amplitude
   - motion variability
   - joint angles
5. Train a graph-temporal model with joint attention and temporal Transformer
   encoding.
6. Use bounded regression, `3 * sigmoid(raw_score)`, to keep predictions in
   the clinical score range `[0, 3]`.

## Current Results

Combined dataset:

| Dataset | Subjects | Sequences | Notes |
|---|---:|---:|---|
| CNUH | 21 | 21 | IRB-approved private clinical cohort |
| CARE-PD | 110 | 6,066 | Public multi-site PD gait benchmark |
| Total | 131 | 6,087 | Subject-level GroupKFold, 5 folds |

Main comparison on the combined CNUH + CARE-PD 17-joint dataset:

| Category | Model | Params | MAE | RMSE | MedAE |
|---|---|---:|---:|---:|---:|
| Classical ML | Ridge | 0 | 0.570 | 0.759 | 0.446 |
| Classical ML | SVR | 0 | 0.492 | 0.639 | 0.386 |
| Classical ML | Random Forest | 0 | 0.510 | 0.659 | 0.417 |
| Classical ML | Shallow MLP | 0 | 0.544 | 0.708 | 0.423 |
| Deep Learning | Temporal CNN | 188,929 | 0.425 | 0.594 | 0.287 |
| SOTA | ST-GCN | 252,097 | 0.443 | 0.623 | 0.274 |
| SOTA | Lu et al. official-architecture DD-Net/OF-DDNet | 147,908 | 0.404 | **0.543** | 0.307 |
| Proposed | Ours V1, bounded regression | 158,594 | **0.358** | 0.564 | **0.147** |

Paired statistical tests on sample-level absolute errors:

| Comparison | MAE difference | Bootstrap 95% CI | Wilcoxon p-value |
|---|---:|---|---:|
| Ours V1 vs Lu et al. official architecture | +0.047 | [0.038, 0.055] | 4.12e-61 |
| Ours V1 vs ST-GCN | +0.085 | [0.078, 0.094] | 5.16e-143 |

Positive MAE difference means the baseline has higher MAE than Ours V1.

## Ours V1 Ablation

| Ablation | Feature set | MAE | RMSE | MedAE |
|---|---|---:|---:|---:|
| A | coordinates only | 0.374 | 0.577 | 0.168 |
| B | coordinates + velocity | 0.432 | 0.629 | 0.246 |
| C | coordinates + velocity + amplitude/variability | 0.376 | **0.549** | 0.192 |
| D | full hybrid feature set including angle | **0.358** | 0.564 | **0.147** |

Configuration D is used as the final proposed model because it gives the best
MAE and MedAE.

## Per-Class Behavior

| Score | N | MAE | RMSE |
|---|---:|---:|---:|
| 0 | 2,615 | 0.225 | 0.395 |
| 1 | 2,183 | 0.342 | 0.482 |
| 2 | 1,244 | 0.649 | 0.889 |
| 3 | 45 | 0.738 | 0.930 |

The main limitation is the severe class imbalance for score 3. The model's
errors mostly occur between adjacent clinical scores.

## Recommended Figures

Generated figures are stored under:

```text
carepd_17pt_experiments/docs/final_integrated_figures/
```

Recommended main-text figures:

| Figure | Suggested use |
|---|---|
| `02_mae_ranking.png` | Main performance comparison |
| `07_ablation_metrics.png` | Feature ablation |
| `09_ours_per_class_error.png` | Per-class error |
| `12_ours_confusion_normalized.png` | Rounded-score confusion matrix |

Recommended supplementary figures:

| Figure | Suggested use |
|---|---|
| `05_fold_mae_by_model.png` | Fold-level stability |
| `18_calibration_curve_by_model.png` | Calibration-style score trend |
| `20_mae_advantage_vs_baselines.png` | MAE gain over selected baselines |
| `16_dataset_mae_breakdown.png` | Dataset-level breakdown |
| `19_class_distribution.png` | Class imbalance explanation |

## Repository Layout

```text
carepd_17pt_experiments/
  gait17/                         Core data, feature, model, and training code
  scripts/                        Dataset conversion, experiment, and analysis scripts
  docs/                           Manuscript-facing summaries and figures
  configs/                        Experiment configuration files
  data/                           Local data only, ignored by Git
  results/                        Local generated outputs only, ignored by Git

Rebuttal/                         Earlier reviewer-response experiments
src/                              Original single-site model code
unused/                           Local scratch or archived code, ignored by Git
```

The current manuscript-facing experiment package is
`carepd_17pt_experiments/`.

## Installation

Create an environment for the current 17-joint experiments:

```powershell
cd carepd_17pt_experiments
conda env create -f environment.yml
conda activate gait17_external
pip install -r requirements.txt
```

For CUDA runs, install a PyTorch build compatible with your GPU. The local
experiments were run with PyTorch 2.6.0 and CUDA 11.8 on an NVIDIA RTX 3080.

## Data

Private CNUH videos, labels, pose arrays, downloaded CARE-PD files, generated
results, and model checkpoints are not included in this repository.

Expected local paths:

| Path | Contents | Git status |
|---|---|---|
| `HospitalData/` | Private CNUH clinical data | ignored |
| `carepd_17pt_experiments/data/` | Converted local datasets | ignored |
| `carepd_17pt_experiments/results/` | Experiment outputs | ignored |
| `CARE-PD-master/` | External upstream CARE-PD repository | ignored |

See `carepd_17pt_experiments/docs/data_format.md` for the converted data
format.

## Reproducing the Main Experiments

Run final Ours V1 and Lu official-architecture comparison:

```powershell
cd carepd_17pt_experiments
.\scripts\run_ours_lu_official_cuda.cmd
```

Run ST-GCN and Lu official-architecture SOTA baselines:

```powershell
.\scripts\run_sota_cuda.cmd
```

Run or resume A/B/C/D ablations. Completed ablations are skipped when
`summary.csv` already exists:

```powershell
.\scripts\run_ours_abcd_cuda.cmd
```

Regenerate manuscript-facing tables:

```powershell
python scripts\summarize_ours_abcd.py
python scripts\analyze_per_class_confusion.py
```

Regenerate final figures:

```powershell
python scripts\make_final_figures.py
```

## Manuscript-Facing Documents

| Document | Purpose |
|---|---|
| `carepd_17pt_experiments/docs/final_integrated_results.md` | Final ML/DL/SOTA/Ours results and tables |
| `carepd_17pt_experiments/docs/ours_abcd_summary.md` | A/B/C/D ablation summary |
| `carepd_17pt_experiments/docs/per_class_confusion_analysis.md` | Per-class MAE/RMSE and confusion matrix |
| `carepd_17pt_experiments/docs/com_robustness_status.md` | COM robustness result and interpretation |
| `carepd_17pt_experiments/docs/scale_robustness_experiment_plan.md` | Follow-up scale robustness candidates |
| `carepd_17pt_experiments/docs/final_integrated_figures/README.md` | Figure index and recommended usage |

## COM Robustness

COM robustness was evaluated with checkpointed fold inference. The result
supports horizontal translation robustness but not full camera-distance
robustness: translation offsets changed MAE/RMSE by approximately 0%, whereas
scale factors increased MAE by 12.7% to 46.5%.

To reproduce it:

```powershell
cd carepd_17pt_experiments
.\scripts\run_ours_d_checkpointed_cuda.cmd
.\scripts\run_com_robustness_cuda.cmd
```

The first command recreates the D fold checkpoints once. The second command
does perturbation inference without additional training. The detailed result is
documented in:

```text
carepd_17pt_experiments/docs/com_robustness_status.md
carepd_17pt_experiments/docs/com_robustness/
```

Follow-up candidates for scale robustness are implemented under:

```powershell
cd carepd_17pt_experiments
.\scripts\run_scale_robustness_screen_cuda.cmd
.\scripts\run_scale_robustness_full_selected_cuda.cmd
```

Screening and full-run interpretation are documented in:

```text
carepd_17pt_experiments/docs/scale_robustness_experiment_plan.md
```

## Ethics and Data Availability

The CNUH cohort was collected under Institutional Review Board approval
(CNUH-2025-203) with written informed consent. Private clinical data cannot be
redistributed. CARE-PD is a public non-commercial research dataset available
from its original maintainers.

## Citation

If this repository is used before publication, cite it as:

```text
Kim B, Jeong H, Bappi MI, Kim K.
An intelligent vision-based decision-support system for Parkinson's gait
severity assessment using COM-centered spatio-temporal modeling from a single
RGB camera. Manuscript in preparation.
```

## Git Hygiene

Do not commit private data, downloaded datasets, checkpoints, or generated
result folders. The repository ignores these by default through `.gitignore`.
