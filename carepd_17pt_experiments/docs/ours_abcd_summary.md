# Ours V1 A/B/C/D Ablation Summary

| Model | Ablation | Feature set | Status | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ours V1 | A | coordinates only | completed | 5 | 6087 | 158210 | 3.179 | 0.374 | 0.577 | 0.168 |
| Ours V1 | B | coordinates + velocity | completed | 5 | 6087 | 158402 | 2.757 | 0.432 | 0.629 | 0.246 |
| Ours V1 | C | coordinates + velocity + amplitude/variability | completed | 5 | 6087 | 158530 | 4.178 | 0.376 | 0.549 | 0.192 |
| Ours V1 | D | full hybrid feature set | completed | 5 | 6087 | 158594 | 4.615 | 0.358 | 0.564 | 0.147 |
