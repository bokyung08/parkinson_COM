# CARE-PD Cohort Composition Used in This Study

- Source: `data/processed/manifest.csv`
- Scope: H36M-compatible 17-joint MDS-UPDRS item-10 gait-score experiments
- Last updated: 2026-06-02

## Included CARE-PD Cohorts

The current local processed manifest contains four CARE-PD source cohorts with
H36M17 gait sequences and usable MDS-UPDRS gait labels.

| CARE-PD cohort | Sequences | Subjects | Score 0 | Score 1 | Score 2 | Score 3 | Included in GroupKFold | Included in LODO |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 3DGait | 90 | 43 | 24 | 42 | 14 | 10 | yes | yes |
| BMCLab | 3,895 | 23 | 1,705 | 1,380 | 810 | 0 | yes | yes |
| PD-GaM | 1,700 | 30 | 783 | 635 | 248 | 34 | yes | yes |
| T-SDU-PD | 381 | 14 | 96 | 118 | 167 | 0 | yes | yes |
| **Total** | **6,066** | **110** | **2,608** | **2,175** | **1,239** | **44** | yes | yes |

The combined CNUH+CARE-PD experiment adds 21 CNUH sequences from 21 subjects,
giving 6,087 total sequences from 131 subject groups.

## Excluded CARE-PD Cohorts

CARE-PD is described as a nine-cohort benchmark. The current H36M17 item-10
experiments use only the four cohorts above because these are the cohorts
present in the local converted score-prediction manifest. The other downloaded
CARE-PD files/cohorts are not included in the current manuscript tables unless
they are converted into H36M17 sequences with compatible item-10 gait labels.

Locally downloaded but not included in the current processed manifest:

```text
DNE
E-LC
KUL-DT-T
T-LTC
T-SDU
```

Manuscript-safe wording:

> Although CARE-PD contains nine source cohorts in total, our H36M17
> MDS-UPDRS gait-score experiments used the four cohorts available in the local
> processed score-prediction manifest with compatible item-10 labels: 3DGait,
> BMCLab, PD-GaM, and T-SDU-PD. All subject-level GroupKFold and CARE-PD LODO
> splits were constructed from these four cohorts.

## LODO Split Definition

CARE-PD leave-one-dataset-out holds out one of the four included source cohorts
as the test set and trains on the other three cohorts:

| Held-out cohort | Train sequences | Test sequences |
|---|---:|---:|
| 3DGait | 5,976 | 90 |
| BMCLab | 2,171 | 3,895 |
| PD-GaM | 4,366 | 1,700 |
| T-SDU-PD | 5,685 | 381 |

