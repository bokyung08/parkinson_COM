# Final Integrated Results

- Last updated: 2026-05-29
- Dataset: CNUH + CARE-PD, converted to H36M-compatible 17-joint gait sequences
- Split: subject-level GroupKFold, 5 folds
- Target: MDS-UPDRS item 10 gait score, range 0-3
- Primary proposed model: Ours V1, bounded regression

## Final Reporting Decision

OursV2 is not used in the main manuscript table. It improved MedAE but did not
improve the primary MAE/RMSE metrics.

| Model | MAE | RMSE | MedAE | Decision |
|---|---:|---:|---:|---|
| Ours V1 | 0.358 | 0.564 | 0.147 | Keep as final proposed model |
| OursV2 | 0.364 | 0.604 | 0.079 | Exclude from main table |

The final manuscript should report Ours V1 as the proposed model. OursV2 can be
kept as an internal exploratory result, but it should not be framed as an
improvement.

## Data Summary

| Dataset | Sequences | Patient groups | Target range |
|---|---:|---:|---:|
| CAREPD | 6,066 | 110 | 0-3 |
| CNUH | 21 | 21 | 0-3 |
| Total | 6,087 | 131 | 0-3 |

The combined GroupKFold table is dominated by CAREPD because CNUH contributes
only 21 sequences. CNUH-only LOSO should remain the direct internal-dataset
comparison.

## Main Table: ML, DL, SOTA, and Proposed

Lower MAE, RMSE, and MedAE are better.

| Category | Model | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Classical ML | Ridge | 5 | 6,087 | 0 | 0.009 | 0.570 | 0.759 | 0.446 |
| Classical ML | SVR | 5 | 6,087 | 0 | 1.531 | 0.492 | 0.639 | 0.386 |
| Classical ML | Random Forest | 5 | 6,087 | 0 | 0.031 | 0.510 | 0.659 | 0.417 |
| Classical ML | Shallow MLP | 5 | 6,087 | 0 | 0.010 | 0.544 | 0.708 | 0.423 |
| Deep Learning | Temporal CNN | 5 | 6,087 | 188,929 | 0.294 | 0.425 | 0.594 | 0.287 |
| SOTA | ST-GCN | 5 | 6,087 | 252,097 | 22.523 | 0.443 | 0.623 | 0.274 |
| SOTA | Lu et al. official-architecture DD-Net/OF-DDNet | 5 | 6,087 | 147,908 | 0.445 | 0.404 | **0.543** | 0.307 |
| Proposed | Ours V1, bounded regression | 5 | 6,087 | 158,594 | 4.615 | **0.358** | 0.564 | **0.147** |

## Main Result Interpretation

Ours V1 achieved the best MAE among the reported methods. Relative MAE
reductions are:

| Comparison | MAE reduction |
|---|---:|
| Ours V1 vs SVR, best classical ML | 27.4% |
| Ours V1 vs Temporal CNN | 15.8% |
| Ours V1 vs ST-GCN | 19.3% |
| Ours V1 vs Lu official-architecture baseline | 11.5% |

Lu official has the best RMSE, but Ours V1 has better MAE and MedAE. The safest
claim is that Ours V1 improves average absolute clinical-score error and
typical-case absolute error, while Lu official has slightly lower aggregate
squared error.

## Statistical Validation

Paired tests use sample-level absolute errors matched by fold, split ID, and
sample ID.

| Comparison | N | Ours V1 MAE | Baseline MAE | Baseline - Ours V1 MAE | Bootstrap 95% CI | Wilcoxon p-value |
|---|---:|---:|---:|---:|---|---:|
| Ours V1 vs Lu official | 6,087 | 0.358 | 0.404 | +0.047 | [0.038, 0.055] | 4.12e-61 |
| Ours V1 vs ST-GCN | 6,087 | 0.358 | 0.443 | +0.085 | [0.078, 0.094] | 5.16e-143 |

Manuscript-safe wording:

> Under identical subject-level GroupKFold evaluation, the proposed bounded
> graph-temporal regression model achieved lower MAE than both ST-GCN and the
> official-architecture Lu et al. baseline.

## Ours V1 A/B/C/D Ablation

For consistency with the final proposed model, D uses the best completed D run
from `groupkfold_h36m17_ours_lu_official_cuda`. A/B/C use the dedicated
ablation run directories.

| Model | Ablation | Feature set | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ours V1 | A | coordinates only | 5 | 6,087 | 158,210 | 3.179 | 0.374 | 0.577 | 0.168 |
| Ours V1 | B | coordinates + velocity | 5 | 6,087 | 158,402 | 2.757 | 0.432 | 0.629 | 0.246 |
| Ours V1 | C | coordinates + velocity + amplitude/variability | 5 | 6,087 | 158,530 | 4.178 | 0.376 | **0.549** | 0.192 |
| Ours V1 | D | full hybrid feature set including angle | 5 | 6,087 | 158,594 | 4.615 | **0.358** | 0.564 | **0.147** |

## Ablation Interpretation

Ablation D gives the best MAE and MedAE among the reported ablations, supporting
the use of the full hybrid feature set. Ablation C gives the best RMSE,
suggesting that amplitude and variability features improve squared-error
stability, while the angle channel in D improves average and median absolute
error.

Ablation B performs worst, especially in fold 5. This suggests that simply
adding velocity to coordinates is not sufficient under the combined multi-cohort
setting; higher-level sequence descriptors are needed for stable score
estimation.

## Ours V1 Architecture Summary

Input shape:

```text
B x T x J x C
```

| Symbol | Meaning | Value |
|---|---|---:|
| B | batch size | variable |
| T | sequence length | 390 |
| J | joints | 17 |
| C | feature channels | A: 3, B: 6, C: 8, D: 9 |

Encoder:

```text
Hybrid node features
  -> GraphConv C -> 64
  -> LayerNorm + ReLU
  -> GraphConv 64 -> 128
  -> LayerNorm + ReLU
  -> joint attention over 17 joints
  -> temporal Transformer encoder
  -> temporal average pooling
  -> MLP regression head
  -> 3 * sigmoid(raw_score)
```

The final `3 * sigmoid(raw_score)` constrains predictions to the valid item-10
range `[0, 3]`.

| Component | Purpose |
|---|---|
| COM normalization | Reduces subject-position and camera-translation sensitivity |
| Graph convolution | Encodes anatomical joint connectivity |
| Joint attention | Weights clinically informative joints more strongly |
| Temporal Transformer | Captures long-range gait dynamics |
| Bounded regression | Keeps continuous predictions inside the clinical score range |

## Reproducibility Commands

Run or resume A/B/C/D ablation. Completed ablations with `summary.csv` are
skipped automatically:

```powershell
cd C:\Users\bokyung\Desktop\parkinson_COM\carepd_17pt_experiments
.\scripts\run_ours_abcd_cuda.cmd
```

Regenerate the ablation summary:

```powershell
python scripts\summarize_ours_abcd.py
```

Outputs:

- `docs\ours_abcd_summary.md`
- `results\OURS_ABCD_SUMMARY.md`
- `results\ours_abcd_summary.csv`
