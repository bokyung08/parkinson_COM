# Parkinson COM-Anchored Gait Severity Estimation

Research code for estimating Parkinsonian gait severity from video-derived pose
sequences normalized around the subject center of mass (COM). The repository is
organized for paper reproduction while excluding private patient data and large
generated artifacts.

## What is included

- COM-relative pose preprocessing and hybrid node features.
- TensorFlow and PyTorch regression models for gait severity estimation.
- Ablation, baseline, LOSO, and reviewer-response experiment scripts.
- Evaluation utilities for metrics, prediction plots, and Bland-Altman plots.

## Repository layout

```text
src/
  features/     COM preprocessing and feature engineering
  models/       TensorFlow/PyTorch model definitions
  trainers/     dataset loaders and training loops
  eval/         metrics and plotting utilities
  run_*.py      experiment entry points
Rebuttal/       reviewer-response experiments and summaries
docs/           data and structure notes
unused/         local-only obsolete code, ignored by Git
```

Top-level modules such as `src.train_hybrid_gcn` are compatibility wrappers.
Canonical implementations live in the subpackages above.

## Installation

```bash
conda env create -f environment.yml
conda activate parkinson_pose_env
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA/CPU environment if the default
`torch` wheel is not appropriate for your machine.

## Data

Private videos, labels, pose arrays, model weights, and generated results are
ignored by Git. See `docs/data.md` for the expected layout.

Typical paths used by the scripts:

- `HospitalData/VIDEO`: raw patient videos.
- `HospitalData/JSON`: label JSON files.
- `dataset/prefinal_preprocessed`: precomputed `*_pose.npy` files.
- `results/`: generated metrics, plots, predictions, and weights.

## Quick start

TensorFlow hybrid fusion model:

```bash
python -m src.train_hybrid_fusion \
  --processed_dir dataset/prefinal_preprocessed \
  --label_dir HospitalData/JSON \
  --epochs 80 \
  --batch_size 4
```

PyTorch hybrid GCN:

```bash
python -m src.train_hybrid_gcn \
  --processed_dir dataset/prefinal_preprocessed \
  --label_dir HospitalData/JSON \
  --epochs 100 \
  --batch_size 4
```

All A/B/C/D ablations:

```bash
python -m src.run_all_ablation \
  --processed_dir dataset/prefinal_preprocessed \
  --label_dir HospitalData/JSON \
  --epochs 20 \
  --batch_size 4
```

Reviewer-response unified DL baselines:

```bash
python -m Rebuttal.run_unified_abcd_dl_baselines \
  --processed_dir dataset/prefinal_preprocessed \
  --label_dir HospitalData/JSON
```

## Feature ablations

- `A`: COM-relative coordinates only.
- `B`: coordinates plus relative velocity.
- `C`: coordinates, relative velocity, amplitude, and variability.
- `D`: full hybrid feature set with joint angles.

## Notes for publication

Record the commit hash, preprocessing window (`max_seconds`, `fps`), ablation
setting, model hyperparameters, and random seeds for each reported experiment.
Do not commit patient data, videos, model weights, or generated result folders.
