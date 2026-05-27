# Data layout

Patient videos, derived pose arrays, labels, and model outputs are intentionally
excluded from Git. Keep private data in one of the ignored data folders and pass
the paths explicitly to the training scripts.

Expected inputs:

- Pose arrays: `*_pose.npy`, shaped either `(T, 33 * F)` or `(T, 33, F)`.
- Labels: JSON files under a label directory such as `HospitalData/JSON`.
- Raw videos: optional, under a video directory such as `HospitalData/VIDEO`.

Common commands:

```bash
python -m src.train_hybrid_fusion --processed_dir dataset/prefinal_preprocessed --label_dir HospitalData/JSON
python -m src.train_hybrid_gcn --processed_dir dataset/prefinal_preprocessed --label_dir HospitalData/JSON
python -m src.run_all_ablation --processed_dir dataset/prefinal_preprocessed --label_dir HospitalData/JSON
```
