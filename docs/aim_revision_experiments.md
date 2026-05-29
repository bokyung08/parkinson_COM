# AIM Revision Experiment Plan

This file tracks the implementation-oriented plan for the Artificial
Intelligence in Medicine revision.

## Week 1

- Table 8: COM perturbation robustness.
  - Script: `Rebuttal/run_com_robustness.py`
  - Inputs: saved Config D weights, processed pose directory, labels, optional `val_samples.txt`.
  - Output: `table8_com_robustness_summary.csv`, plots, and `RESULTS.md`.
- Table 10/11/12: per-class error, rounded confusion matrix, and paired statistics.
  - Script: `Rebuttal/analyze_prediction_tables.py`
  - Inputs: saved `predictions.tsv` files.
  - Output: `table10_per_class.csv`, `table11_confusion_*.csv`, `table12_statistical_validation.csv`, and `RESULTS.md`.

## Week 2-3

- Table 9: SOTA comparison under identical LOSO-CV.
  - Minimum external baselines: ST-GCN and Lu et al. MICCAI 2020.
  - Preserve original architecture and hyperparameters; add only input adapters.
  - Required metrics: MAE, RMSE, parameter count, and inference time per sample.

## Week 3-4

- Table 13: CARE-PD zero-shot validation.
  - Add MediaPipe 33 to H36M 17 mapping adapter.
  - Report viewpoint/representation domain gap explicitly.

## Usage examples

```bash
python -m Rebuttal.analyze_prediction_tables \
  --prediction FusionTF=Rebuttal/results/unified_dl_baselines/D_item10_90_10/fusion_tf/predictions.tsv \
  --prediction HybridTorch=Rebuttal/results/unified_dl_baselines/D_item10_90_10/hybrid_torch/predictions.tsv \
  --compare FusionTF:HybridTorch \
  --out_dir Rebuttal/results/statistical_validation/D_item10_holdout
```

```bash
python -m Rebuttal.run_com_robustness \
  --processed_dir dataset/prefinal_preprocessed \
  --label_dir HospitalData/JSON \
  --val_samples Rebuttal/results/unified_dl_baselines/D_item10_90_10/val_samples.txt \
  --model fusion_tf \
  --com_weights Rebuttal/results/unified_dl_baselines/D_item10_90_10/fusion_tf/best.weights.h5 \
  --out_dir Rebuttal/results/com_robustness/D_item10_fusion_tf
```
