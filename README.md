# 🧠 Parkinson COM-Anchored Gait Severity Estimation

> Video-derived pose sequences normalized around the subject center of mass (COM) for Parkinsonian gait severity regression.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Research%20Only-lightgrey?style=flat-square)](./LICENSE)
[![IRB](https://img.shields.io/badge/IRB-CNUH%20Approved-green?style=flat-square)]()

---

## 📌 Overview

This repository contains research code for estimating gait severity in Parkinson's disease patients from video-based pose sequences. Pose keypoints are normalized relative to the subject's **center of mass (COM)** to reduce viewpoint and scale variance, enabling more robust severity regression across subjects.

Organized for paper reproduction while excluding private patient data and large generated artifacts.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow 2.x, PyTorch 1.x+ |
| Graph Neural Network | PyTorch Geometric (GCN) |
| Pose Estimation | MediaPipe / OpenPose (preprocessing) |
| Data Processing | NumPy, pandas, scikit-learn |
| Visualization | Matplotlib, seaborn |
| Experiment Tracking | Custom LOSO-CV scripts |

---

## 📂 Repository Layout

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

> Top-level modules such as `src.train_hybrid_gcn` are compatibility wrappers.  
> Canonical implementations live in the subpackages above.

---

## ⚙️ Installation

```bash
conda env create -f environment.yml
conda activate parkinson_pose_env
pip install -r requirements.txt
```

> Install the PyTorch build that matches your CUDA/CPU environment if the default `torch` wheel is not appropriate for your machine.

---

## 📁 Data

Private videos, labels, pose arrays, model weights, and generated results are **excluded from this repository**. See [`docs/data.md`](docs/data.md) for the expected layout.

| Path | Contents |
|---|---|
| `HospitalData/VIDEO` | Raw patient videos |
| `HospitalData/JSON` | Label JSON files |
| `dataset/prefinal_preprocessed` | Precomputed `*_pose.npy` files |
| `results/` | Generated metrics, plots, predictions, weights |

---

## 🚀 Quick Start

### TensorFlow Hybrid Fusion Model

```bash
python -m src.train_hybrid_fusion \
  --processed_dir dataset/prefinal_preprocessed \
  --label_dir HospitalData/JSON \
  --epochs 80 \
  --batch_size 4
```

### PyTorch Hybrid GCN

```bash
python -m src.train_hybrid_gcn \
  --processed_dir dataset/prefinal_preprocessed \
  --label_dir HospitalData/JSON \
  --epochs 100 \
  --batch_size 4
```

### All A/B/C/D Ablations

```bash
python -m src.run_all_ablation \
  --processed_dir dataset/prefinal_preprocessed \
  --label_dir HospitalData/JSON \
  --epochs 20 \
  --batch_size 4
```

### Reviewer-Response Unified DL Baselines

```bash
python -m Rebuttal.run_unified_abcd_dl_baselines \
  --processed_dir dataset/prefinal_preprocessed \
  --label_dir HospitalData/JSON
```

---

## 🔬 Feature Ablations

| Setting | Description |
|---|---|
| **A** | COM-relative coordinates only |
| **B** | Coordinates + relative velocity |
| **C** | Coordinates + relative velocity + amplitude + variability |
| **D** | Full hybrid feature set with joint angles |

---

## 📝 Notes for Publication

When reporting results, please record the following for reproducibility:

- Git commit hash
- Preprocessing window (`max_seconds`, `fps`)
- Ablation setting (A / B / C / D)
- Model hyperparameters
- Random seeds

> ⚠️ Do **not** commit patient data, videos, model weights, or generated result folders.

---

## 🔒 Ethics

This study was conducted under IRB approval from Chonnam National University Hospital (CNUH). All patient data is de-identified and excluded from this public repository.
