# Reviewer Revision Action Plan

- Target journal: Artificial Intelligence in Medicine
- Review outcome: minor revision
- Last updated: 2026-06-05

## Executive Assessment

The review is favorable. The requested changes are mainly about methodological
transparency and component-level evidence, not about invalidating the main
result. The main manuscript claim should remain:

> Ours V1 achieves the best MAE under identical subject-level GroupKFold
> evaluation, while Lu official has slightly lower RMSE and MotionAGFormer-XS
> has the lowest MedAE. External transfer reveals a domain gap, but LODO,
> calibration, robustness, and learning curve analyses make the clinical
> decision-support story defensible.

## Required Actions

Consolidated evidence document:

```text
docs/reviewer_m1_m2_m3_evidence_summary.md
```

| ID | Reviewer concern | Status | Action |
|---|---|---|---|
| M1 | Architecture ablation missing | Completed | Add table from `docs/reviewer_m1_m2_m3_evidence_summary.md` |
| M2 | CARE-PD split transparency | Documented locally | Add cohort table from `docs/reviewer_m1_m2_m3_evidence_summary.md` |
| M3 | Lu et al. fairness | Documented locally | Add implementation note from `docs/reviewer_m1_m2_m3_evidence_summary.md` |
| M4 | Calibration quantification | Completed | Use bin-level table from `docs/calibration_reliability_analysis.md`; keep figure metric-free |
| m1 | 2D-only vs 2.5D | Pending optional experiment | Add small auxiliary experiment if time permits |
| m2 | Wilcoxon independence | Pending lightweight analysis | Prefer subject-level aggregated Wilcoxon or limitations note |
| m3 | Positional encoding | Code inspected | Current model has no explicit positional encoding; disclose in Table 1 |
| m4 | Score-2 clinical interpretation | Text revision needed | Add false-negative/alert discussion |
| m5 | CNUH single-site MAE relation | Text revision needed | State it is Table 5 Config D result |
| m6 | CARE-PD citation | Verified current source | Cite NeurIPS 2025 proceedings and/or OpenReview/arXiv |

## Architecture Ablation Result

Output:

```text
docs/architecture_ablation_analysis.md
results/architecture_ablation_summary.csv
```

| Model | Components | MAE | RMSE | MedAE |
|---|---|---:|---:|---:|
| MLP only | mean pooling + bounded MLP | 0.554 | 0.653 | 0.481 |
| GraphConv + MLP | GraphConv, no joint attention, no Temporal Transformer | 0.450 | 0.580 | 0.349 |
| GraphConv + Joint Attention + MLP | GraphConv + joint attention, no Temporal Transformer | 0.414 | 0.564 | 0.291 |
| Full Ours V1 | GraphConv + Joint Attention + Temporal Transformer | 0.358 | 0.564 | 0.147 |

The full model row uses the canonical final 5-fold Ours run, not the interrupted
full-model rows inside `architecture_ablation_ours_cuda`.

## Safe Response Framing

- For M1, do not argue that input-feature ablation is enough. Add the
  architecture ablation table.
- For M2, explicitly state that the current H36M17 score-prediction experiment
  uses four UPDRS-labeled CARE-PD cohorts from the local converted manifest.
- For M3, state that Lu official is an architecture-level reimplementation under
  a shared H36M17 input adapter, not a reproduction of the original VIBE/49-joint
  pipeline.
- For M4, include bin-level calibration differences in a table, but keep the
  calibration figure free of ECE labels.
- For m3, table entry should say: `Temporal positional encoding: none explicit`.
