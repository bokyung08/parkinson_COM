# Domain-Gap Follow-Up Experiments

These scripts are designed to make the cross-dataset story more informative.
Each experiment writes outputs to its own result folder and manuscript-facing
summary document.

## 1. Dataset-Wise Breakdown Under Combined GroupKFold

Purpose: show whether the strong combined GroupKFold result is dominated by one
dataset or remains competitive within each dataset.

```powershell
.\scripts\run_dataset_wise_breakdown.cmd
```

Outputs:

```text
results\dataset_wise_breakdown\
docs\dataset_wise_breakdown_analysis.md
```

Status: completed once. Ours V1 remains best on CARE-PD test samples
(`MAE = 0.356`) but CNUH remains unstable because it contributes only
21 samples (`MAE = 0.793` for Ours V1).

## 2. Ours vs. SOTA Cross-Dataset Transfer

Purpose: determine whether Ours is relatively better than ST-GCN and the Lu
official-architecture baseline under strict zero-shot transfer.

```powershell
.\scripts\run_cross_dataset_model_comparison_cuda.cmd
```

Outputs:

```text
results\cross_dataset_model_comparison\
docs\cross_dataset_model_comparison.md
```

Status: completed once. Ours V1 is best in the CNUH -> CARE-PD direction
(`MAE = 0.747`, `RMSE = 0.882`) and has the best average zero-shot transfer
MAE across both directions. Lu official is slightly better in the CARE-PD ->
CNUH direction (`MAE = 0.865`, `RMSE = 1.027`). ST-GCN is unstable when trained
only on the 21-sample CNUH source set.

## 3. Few-Shot Target-Site Calibration

Purpose: test whether the zero-shot domain gap can be reduced with a small
number of labeled target-site subjects, without retraining the full model.

```powershell
.\scripts\run_fewshot_calibration.cmd
```

Outputs:

```text
results\fewshot_calibration\
docs\fewshot_calibration_analysis.md
```

Status: completed once using affine calibration. The most useful result is that
CNUH -> CARE-PD improves from `MAE = 0.747` to approximately `0.622` with
10 calibration subjects, and CARE-PD -> CNUH improves from approximately
`1.014` to `0.836` with 10 calibration subjects.

## 4. CARE-PD Leave-One-Dataset-Out

Purpose: evaluate generalization within CARE-PD by holding out one CARE-PD
source cohort at a time.

```powershell
.\scripts\run_carepd_lodo_cuda.cmd
```

Outputs:

```text
results\carepd_leave_one_dataset_out\
docs\carepd_lodo_analysis.md
```

Status: completed once for Ours. The overall CARE-PD leave-one-dataset-out
result is `MAE = 0.620`, `RMSE = 0.813`, and `MedAE = 0.508` across four
held-out CARE-PD source cohorts. This is harder than CARE-PD subject-level
GroupKFold (`MAE = 0.356`) but better than strict CNUH -> CARE-PD zero-shot
transfer (`MAE = 0.747`). Add `--models stgcn lu_ofddnet_official` in the
script only if SOTA LODO comparison is needed.

## 5. Score-Balanced Transfer Analysis

Purpose: separate class-imbalance effects from transfer failure by averaging
per-score MAE/RMSE equally across score classes.

```powershell
.\scripts\run_score_balanced_transfer.cmd
```

Outputs:

```text
results\score_balanced_transfer\
docs\score_balanced_transfer_analysis.md
```

Status: completed once. CNUH -> CARE-PD original MAE is `0.747`, but
score-balanced MAE is `1.022`, showing that the original transfer metric is
softened by the target score distribution. CARE-PD -> CNUH has similar original
and balanced MAE because the CNUH set is very small.

## Run All

The combined runner executes the three analysis scripts first and then launches
the two longer training scripts:

```powershell
.\scripts\run_domain_gap_followups.cmd
```

For practical use, run the long scripts individually so the output folder and
terminal log are easier to inspect.
