# Repository structure

This repository separates reusable model code, experiment entry points, rebuttal
experiments, and ignored local artifacts.

## Source package

- `src/features/`: pose preprocessing helpers and COM-anchored node feature builders.
- `src/models/`: TensorFlow and PyTorch model definitions.
- `src/trainers/`: training loops and dataset loaders.
- `src/eval/`: evaluation metrics and plotting utilities.
- `src/run_*.py`: experiment entry points for ablations, summaries, and diagnostics.

Top-level modules such as `src.hybrid_gcn`, `src.train_hybrid_gcn`, and
`src.evaluate_model` are compatibility wrappers. They preserve older commands
while keeping the canonical implementations in the subpackages above.

## Experiments

- `Rebuttal/`: reviewer-response experiments, baseline comparisons, LOSO runs,
  and paper-ready summaries. Generated outputs under `Rebuttal/results/` are
  ignored by Git through the global `results/` ignore rule.

## Local-only folders

- `data/`, `dataset/`, `HospitalData/`, `hospitalwalkingdata/`: private or large
  data folders and therefore ignored.
- `results/`: generated metrics, plots, model weights, and prediction files.
- `unused/`: obsolete code snapshots and scratch files intentionally excluded
  from the public repository.
