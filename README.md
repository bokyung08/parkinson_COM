# 🧠 COM-Anchored Spatio-Temporal Graph Modeling for Gait Severity Estimation in Parkinson’s Disease

---

```
# Parkinson COM-Anchored Gait Severity Estimation

This repository contains implementations for predicting gait severity (gait UPDRS) from video-derived pose sequences that
are normalized relative to the subject's center of mass (COM). It is organized and documented for reproducibility and
paper submission. Both regression and classification variants are provided.

Key contributions
- COM-normalized hybrid node features: coordinates, relative velocity, amplitude, variability, and joint angles (9 channels).
- COM-anchored graph convolution and temporal modules: TensorFlow ST-GCN-style and PyTorch GCN + Transformer implementations.
- Reproducible evaluation pipeline: cross-validation, saved predictions, and visualization utilities including Bland–Altman plots.

Overview
- `src/` – model implementations, training and evaluation scripts
   - `train_hybrid_fusion.py`: TF-based COM-STGCN (regression) with CV
   - `hybrid_gcn.py`: PyTorch COM-GCN regression model
   - `hybrid_gcn_cls.py`, `train_model_cls.py`: classification (`CLS`) variants
   - `feature_engineering.py`: COM-relative node feature builder (9 channels)
   - `model_builder.py`: Transformer-based pose model (regression)
- `dataset/`, `data/` – preprocessed `.npy` pose files (raw patient data excluded)
- `results/` – training artifacts, predictions, and plots

Important notes
- Files or classes with `CLS` or `_cls` are classification-oriented (e.g. `HybridCOMGCNv2Cls`).
- Regression implementations predict continuous gait UPDRS scores (e.g. `train_hybrid_fusion.py`, `hybrid_gcn.py`).

Installation
```bash
conda env create -f environment.yml
conda activate parkinson_com
pip install -r requirements.txt
```

Quick start examples
- TensorFlow regression (COM-STGCN, cross-validation):
```bash
python -m src.train_hybrid_fusion \
   --processed_dir dataset/prefinal_preprocessed \
   --label_dir HospitalData/JSON \
   --epochs 80 --batch_size 4
```
- PyTorch GCN regression (training script example):
```bash
python -m src.train_hybrid_gcn \
   --data_dir dataset/prefinal_preprocessed \
   --labels HospitalData/JSON \
   --epochs 100
```
- Classification example (target: `item10_class`):
```bash
python -m src.train_model_cls \
   --processed_data_path dataset/prefinal_preprocessed \
   --model_save_path results/models/cls_best.weights.h5
```

Data format
- Preprocessed pose files: `.npy` files where each sample uses either `(T, 33*F)` or `(T, 33, F)` format.
- `feature_engineering.py`'s `build_hybrid_node_features` consumes `(T, 33, 3)` COM-normalized coordinates and returns `(T, 33, 9)` node features.

Reproducibility and evaluation
- Cross-validation artifacts and model outputs are stored under `results/fusion_tf_runs/<timestamp>/`.
- Regression metrics: MAE, RMSE, Spearman correlation.
- Visualizations: scatter plots, residual histograms, and Bland–Altman plots (`src/plot_pred_bland_altman.py`) — especially useful for continuous regression analysis.

Paper-ready metadata
- To enable strict reproducibility, record preprocessing parameters (`max_seconds`, `fps`), ablation setting (A/B/C/D), model hyperparameters (learning rate, batch size, epochs), and random seeds.
- Consider adding `docs/experiments.md` to list experimental configurations and results tables for the manuscript.

License & citation
- Add a license of your choice. When citing this repository in publications, include the repository URL and commit hash.

Contact
- Open an issue or contact the authors for implementation or reproducibility questions.

