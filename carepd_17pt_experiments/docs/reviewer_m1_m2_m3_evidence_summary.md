# Reviewer Evidence Summary: Architecture Ablation, CARE-PD Split, and Lu Baseline Fairness

- Last updated: 2026-06-02
- Purpose: consolidated evidence for reviewer concerns M1, M2, and M3
- Dataset: CNUH + CARE-PD, H36M-compatible 17-joint gait sequences
- Target: MDS-UPDRS Part III item 10 gait score, range 0-3

This document collects the three reviewer-facing items that should be easy to
cite during revision:

1. architecture ablation of the proposed model,
2. CARE-PD cohort composition and split definition,
3. Lu et al. official-architecture baseline fairness notes.

The integrated result documents still contain the final manuscript-facing
tables. This file is a compact reviewer-response evidence sheet.

## M1. Architecture Ablation

Reviewer concern: the manuscript originally ablated only input feature
configurations A-D, but did not independently test the contribution of
GraphConv, Joint Attention, and Temporal Transformer.

### Experimental Setup

| Item | Setting |
|---|---|
| Split | subject-level GroupKFold, 5 folds |
| Dataset | CNUH + CARE-PD H36M17 |
| Input feature configuration | D, full hybrid feature set |
| Target | MDS-UPDRS item 10, 0-3 |
| Full Ours row | canonical final run from `groupkfold_h36m17_ours_lu_official_cuda` |

The full Ours row uses the original completed 5-fold result. The interrupted
full `ours` rows inside `architecture_ablation_ours_cuda` are not used.

### Architecture Ablation Table

| Model | Components | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MLP only | mean pooling + bounded MLP | 5 | 6,087 | 17,921 | 0.005 | 0.554 | 0.653 | 0.481 |
| GraphConv + MLP | GraphConv, no joint attention, no Temporal Transformer | 5 | 6,087 | 25,985 | 2.306 | 0.450 | 0.580 | 0.349 |
| GraphConv + Joint Attention + MLP | GraphConv + joint attention, no Temporal Transformer | 5 | 6,087 | 26,114 | 2.736 | 0.414 | 0.564 | 0.291 |
| Full Ours V1 | GraphConv + Joint Attention + Temporal Transformer | 5 | 6,087 | 158,594 | 4.615 | **0.358** | 0.564 | **0.147** |

### Interpretation

The ablation supports that the proposed architecture gain is not caused only
by the input feature set.

- GraphConv improves MAE from 0.554 to 0.450 compared with MLP only.
- Joint Attention further improves MAE from 0.450 to 0.414.
- Adding the Temporal Transformer yields the best MAE and MedAE, reducing MAE
  from 0.414 to 0.358.
- Relative to MLP only, the full model reduces MAE by 35.4%.
- Relative to GraphConv + Joint Attention + MLP, the full model reduces MAE by
  13.7%.

RMSE is almost unchanged between GraphConv + Joint Attention + MLP and the full
model. Therefore, the safest claim is that the full encoder mainly improves
average absolute error and typical-case error, while RMSE remains comparable.

### Manuscript-Safe Wording

> Architecture ablation confirmed that the performance gain is not attributable
> solely to the input feature set. GraphConv improved over a mean-pooled MLP
> baseline, joint attention further reduced error, and the full GraphConv +
> Joint Attention + Temporal Transformer encoder achieved the lowest MAE and
> MedAE.

## M2. CARE-PD Cohort Composition and Split Definition

Reviewer concern: CARE-PD is described as a nine-cohort benchmark, but the
current manuscript tables use four cohorts. The included and excluded cohorts
must be stated clearly.

### Included CARE-PD Cohorts

The current H36M17 item-10 experiments use the four CARE-PD cohorts present in
the local processed score-prediction manifest with compatible gait labels.

| CARE-PD cohort | Sequences | Subjects | Score 0 | Score 1 | Score 2 | Score 3 | GroupKFold | LODO |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 3DGait | 90 | 43 | 24 | 42 | 14 | 10 | yes | yes |
| BMCLab | 3,895 | 23 | 1,705 | 1,380 | 810 | 0 | yes | yes |
| PD-GaM | 1,700 | 30 | 783 | 635 | 248 | 34 | yes | yes |
| T-SDU-PD | 381 | 14 | 96 | 118 | 167 | 0 | yes | yes |
| **Total** | **6,066** | **110** | **2,608** | **2,175** | **1,239** | **44** | yes | yes |

The combined CNUH+CARE-PD experiment adds 21 CNUH sequences from 21 subjects,
giving 6,087 total sequences from 131 subject groups.

### Excluded CARE-PD Cohorts

The following downloaded CARE-PD files/cohorts are not included in the current
H36M17 item-10 manuscript tables because they are not present in the local
converted score-prediction manifest used for the experiments:

```text
DNE
E-LC
KUL-DT-T
T-LTC
T-SDU
```

These cohorts should not be claimed as part of the current quantitative
experiments unless they are converted into H36M17 sequences with compatible
MDS-UPDRS item-10 gait labels.

### LODO Split Definition

CARE-PD leave-one-dataset-out holds out one of the four included source cohorts
as the test set and trains on the other three cohorts.

| Held-out cohort | Train sequences | Test sequences |
|---|---:|---:|
| 3DGait | 5,976 | 90 |
| BMCLab | 2,171 | 3,895 |
| PD-GaM | 4,366 | 1,700 |
| T-SDU-PD | 5,685 | 381 |

### Manuscript-Safe Wording

> Although CARE-PD contains nine source cohorts in total, our H36M-compatible
> 17-joint MDS-UPDRS gait-score experiments used the four cohorts available in
> the local processed score-prediction manifest with compatible item-10 labels:
> 3DGait, BMCLab, PD-GaM, and T-SDU-PD. All subject-level GroupKFold and
> CARE-PD LODO splits were constructed from these four cohorts.

## M3. Lu et al. Reimplementation Fairness

Reviewer concern: Lu et al. originally used a different 3D pose extraction
pipeline, so the manuscript must explain what was reproduced and what was
adapted.

### What Was Preserved

The local `lu_ofddnet_official` baseline ports the released DD-Net/OF-DDNet
style architecture:

- joint-collection-distance branch,
- slow pose-motion branch,
- fast pose-motion branch,
- temporal 1D convolution blocks,
- global max pooling,
- dense classifier head,
- ordinal focal loss with expected-score decoding.

### Shared Input Adapter

All skeleton models in the current manuscript use the same H36M-compatible
17-joint input adapter:

```text
T x 17 x 3
```

Therefore, the Lu et al. baseline is an architecture-level comparison under
the shared input protocol, not a full replication of the original VIBE/49-joint
input pipeline.

| Item | Original Lu et al. setting | Current shared setting |
|---|---|---|
| Pose source | 3D pose extracted from video pipeline | H36M17 converted CNUH + CARE-PD pose sequences |
| Joint layout | original model-specific skeleton | shared 17-joint H36M-compatible skeleton |
| Coordinate type | 3D pose sequence | CNUH MediaPipe-derived pseudo-depth + CARE-PD H36M-style coordinates |
| Task output | ordinal MDS-UPDRS gait class/score | expected score in range 0-3 |
| Tuning | original paper setting | same split, epochs, learning rate, and batch size as other deep baselines |

### Hyperparameters Used Locally

| Hyperparameter | Value |
|---|---:|
| Epochs | 80 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Loss | ordinal focal loss |
| Score decoding | expected score over 4 ordinal classes |
| Split | subject-level GroupKFold 5-fold |
| Test-set tuning | none |

### Manuscript-Safe Wording

> Lu et al. was implemented as an official-architecture baseline under the same
> H36M-compatible 17-joint input protocol used for all skeleton models. Because
> the original Lu et al. pipeline used a different 3D pose extraction procedure,
> this comparison should be interpreted as an architecture-level comparison
> under a shared input adapter rather than a full replication of the original
> VIBE/49-joint input setting.

## Recommended Placement in the Manuscript

| Reviewer item | Manuscript location | Action |
|---|---|---|
| M1 Architecture ablation | Results, after input feature ablation | Add architecture ablation table and short interpretation |
| M2 CARE-PD split transparency | Experimental setup or supplementary data table | Add cohort composition and LODO split table |
| M3 Lu fairness | Baselines subsection or appendix | Add shared-adapter explanation and hyperparameter table |

## Bottom Line

These additions directly address the three highest-impact methodological
transparency concerns:

- the proposed architecture components are now independently ablated,
- the CARE-PD cohorts used in the reported tables are explicitly identified,
- the Lu et al. baseline is described as a fair shared-input architecture
  comparison, with its remaining difference from the original 3D pipeline
  stated transparently.
