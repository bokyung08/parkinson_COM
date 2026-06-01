# Lu et al. Official-Architecture Baseline: Reimplementation Notes

- Baseline: Lu et al., MICCAI 2020 vision-based MDS-UPDRS gait-score model
- Local model key: `lu_ofddnet_official`
- Code location: `gait17/models.py`, `LuOFDDNetOfficial`
- Last updated: 2026-06-02

## What Was Preserved

The local baseline ports the released DD-Net/OF-DDNet-style architecture:

- joint-collection-distance branch,
- slow pose-motion branch,
- fast pose-motion branch,
- temporal 1D convolution blocks,
- global max pooling,
- dense classifier head,
- ordinal focal loss with expected-score decoding.

## Shared Input Adapter

For fair comparison under the current manuscript protocol, all skeleton models
use the same H36M-compatible 17-joint sequence format:

```text
T x 17 x 3
```

The Lu official baseline is therefore an architecture-level comparison under a
shared input adapter, not a full reproduction of the original Lu et al. 3D
VIBE/49-joint extraction pipeline.

| Item | Original Lu et al. setting | Current shared setting |
|---|---|---|
| Pose source | 3D pose extracted from video pipeline | H36M17 converted CNUH + CARE-PD pose sequences |
| Joint layout | original model-specific skeleton | shared 17-joint H36M-compatible skeleton |
| Coordinate type | 3D pose sequence | CNUH MediaPipe-derived pseudo-depth + CARE-PD H36M-style coordinates |
| Task output | ordinal MDS-UPDRS gait class/score | expected score in range 0-3 |
| Tuning | original paper setting | same train/validation split, epochs, learning rate, and batch size as other deep baselines |

## Hyperparameters Used Locally

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

## Manuscript-Safe Wording

> Lu et al. was implemented as an official-architecture baseline under the same
> H36M-compatible 17-joint input protocol used for all skeleton models. Because
> the original Lu et al. pipeline used a different 3D pose extraction procedure,
> this comparison should be interpreted as an architecture-level comparison
> under a shared input adapter rather than a full replication of the original
> VIBE/49-joint input setting.

