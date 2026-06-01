# Ours V1 Learning Curve

- Model: Ours V1, Configuration D
- Split: subject-level GroupKFold
- Procedure: keep validation folds fixed, then subsample training subjects within each fold
- Purpose: show whether performance improves as training subject count increases
- Status: completed, 25/25 training jobs

## Summary

| Train fraction | Mean train subjects | Mean train sequences | MAE | RMSE | MedAE | MAE reduction vs 10% |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 11.0 | 608.2 | 0.476 | 0.672 | 0.363 | 0.0% |
| 0.25 | 26.4 | 1,439.4 | 0.460 | 0.646 | 0.292 | 3.4% |
| 0.50 | 52.6 | 2,473.4 | 0.413 | 0.624 | 0.203 | 13.2% |
| 0.75 | 79.0 | 3,561.2 | 0.390 | 0.578 | 0.203 | 18.1% |
| 1.00 | 104.8 | 4,869.6 | 0.360 | 0.530 | 0.197 | 24.5% |

## Interpretation

MAE and RMSE decrease monotonically as the number of training subjects increases.
Moving from the smallest training subset to the full training set reduces MAE
from `0.476` to `0.360`, an absolute reduction of `0.117` score units and a
relative reduction of `24.5%`. RMSE decreases from `0.672` to `0.530`, a
relative reduction of `21.1%`.

This supports a data-scale interpretation: performance on small clinical
cohorts is data-limited, and the proposed model benefits from larger
multi-site training sets. This is useful for discussing the CNUH N=21
limitation without framing it as a failure of the architecture.

The run used one seed (`42`), so the table should be interpreted as a
sample-size sensitivity analysis rather than a full uncertainty estimate across
multiple random subsampling seeds.

## Figures

- `docs\reviewer_figures\24_learning_curve_ours_mae.png`
- `docs\reviewer_figures\25_learning_curve_ours_rmse.png`

## Manuscript-Safe Wording

> The learning-curve analysis showed that prediction error decreased as the number of training subjects increased, supporting the interpretation that performance on small clinical cohorts is data-limited and can benefit from larger multi-site training sets.

More concrete wording:

> Increasing the mean number of training subjects per fold from 11.0 to 104.8
> reduced MAE from 0.476 to 0.360 and RMSE from 0.672 to 0.530, indicating that
> additional multi-site training data substantially improves gait-score
> estimation.
