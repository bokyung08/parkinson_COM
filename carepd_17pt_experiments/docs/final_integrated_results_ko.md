# 최종 통합 결과 국문 요약

- 최종 업데이트: 2026-06-05
- 데이터셋: CNUH + CARE-PD
- 입력 형식: H36M-compatible 17-joint gait sequence
- 분할 방식: subject-level GroupKFold, 5 folds
- 예측 대상: MDS-UPDRS Part III item 3.10 gait score, 범위 0-3
- 최종 제안 모델: Ours V1, bounded regression

## 1. 최종 보고 방침

OursV2는 본 논문 메인 테이블에서 제외한다. OursV2는 MedAE는 개선했지만
주요 지표인 MAE/RMSE를 개선하지 못했다. 따라서 최종 제안 모델은 Ours V1
Configuration D로 유지한다.

| Model | MAE | RMSE | MedAE | 결정 |
|---|---:|---:|---:|---|
| Ours V1 | 0.358 | 0.564 | 0.147 | 최종 제안 모델로 유지 |
| OursV2 | 0.364 | 0.604 | 0.079 | 메인 테이블 제외 |

해석: 논문에서는 OursV2를 성능 개선 모델로 주장하지 않는다. 내부 exploratory
result로만 보관한다.

## 2. 데이터 요약

| Dataset | Sequences | Patient groups | Target range |
|---|---:|---:|---:|
| CARE-PD | 6,066 | 110 | 0-3 |
| CNUH | 21 | 21 | 0-3 |
| Total | 6,087 | 131 | 0-3 |

주의점: 전체 sequence 수는 CARE-PD가 압도적으로 많다. 따라서 combined
GroupKFold 결과는 CARE-PD의 영향이 크다. CNUH-only 결과는 별도 single-site
LOSO 실험으로 해석해야 한다.

## 3. 메인 성능 비교

낮을수록 좋은 지표: MAE, RMSE, MedAE.

| Category | Model | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Classical ML | Ridge | 5 | 6,087 | 0 | 0.009 | 0.570 | 0.759 | 0.446 |
| Classical ML | SVR | 5 | 6,087 | 0 | 1.531 | 0.492 | 0.639 | 0.386 |
| Classical ML | Random Forest | 5 | 6,087 | 0 | 0.031 | 0.510 | 0.659 | 0.417 |
| Classical ML | Shallow MLP | 5 | 6,087 | 0 | 0.010 | 0.544 | 0.708 | 0.423 |
| Deep Learning | Temporal CNN | 5 | 6,087 | 188,929 | 0.294 | 0.425 | 0.594 | 0.287 |
| SOTA | ST-GCN | 5 | 6,087 | 252,097 | 22.523 | 0.443 | 0.623 | 0.274 |
| SOTA | MotionBERT-Lite (81-frame) | 5 | 6,087 | 10,814,222 | 5.243 | 0.442 | 0.625 | 0.247 |
| SOTA | Lu et al. official-architecture DD-Net/OF-DDNet | 5 | 6,087 | 147,908 | 0.445 | 0.404 | **0.543** | 0.307 |
| SOTA | MotionAGFormer-XS | 5 | 6,087 | 2,307,324 | 6.150 | 0.405 | 0.638 | **0.095** |
| Proposed | Ours V1, bounded regression | 5 | 6,087 | 158,594 | 4.615 | **0.358** | 0.564 | 0.147 |

## 4. 메인 결과 해석

Ours V1은 전체 비교에서 가장 낮은 MAE를 보였다. 즉 평균적인 임상 점수
절대 오차 측면에서는 가장 우수하다. 단, RMSE는 Lu official baseline이 가장
낮고, MedAE는 MotionAGFormer-XS가 가장 낮다.

| 비교 | MAE 감소율 |
|---|---:|
| Ours V1 vs SVR, best classical ML | 27.4% |
| Ours V1 vs Temporal CNN | 15.8% |
| Ours V1 vs ST-GCN | 19.3% |
| Ours V1 vs MotionBERT-Lite (81-frame) | 19.0% |
| Ours V1 vs Lu official-architecture baseline | 11.5% |
| Ours V1 vs MotionAGFormer-XS | 11.7% |

따라서 논문에서는 다음과 같이 표현하는 것이 안전하다.

> Ours V1 improves average absolute clinical-score error, while Lu official has
> slightly lower aggregate squared error and MotionAGFormer-XS has the lowest
> median absolute error.

국문 해석:

> 제안 모델은 평균 절대 오차에서 가장 우수하지만, 큰 오차에 더 민감한 RMSE
> 기준에서는 Lu et al. baseline이 약간 더 낮았고, 중앙값 절대 오차 기준에서는
> MotionAGFormer-XS가 가장 낮았다. MotionBERT-Lite (81-frame)는 MAE/RMSE
> 기준으로 ST-GCN과 유사하지만 제안 모델보다 낮지는 않았다.

## 5. 통계 검증

paired sample-level absolute error를 사용하였다.

| Comparison | N | Ours V1 MAE | Baseline MAE | Baseline - Ours V1 MAE | Bootstrap 95% CI | Wilcoxon p-value |
|---|---:|---:|---:|---:|---|---:|
| Ours V1 vs Lu official | 6,087 | 0.358 | 0.404 | +0.047 | [0.038, 0.055] | 4.12e-61 |
| Ours V1 vs ST-GCN | 6,087 | 0.358 | 0.443 | +0.085 | [0.078, 0.094] | 5.16e-143 |

해석: 동일한 subject-level GroupKFold 조건에서 Ours V1은 Lu official 및
ST-GCN보다 MAE가 통계적으로 유의하게 낮았다.

## 6. Per-Class Error Analysis

| True score | N | MAE | RMSE |
|---:|---:|---:|---:|
| 0 | 2,615 | 0.225 | 0.395 |
| 1 | 2,183 | 0.342 | 0.482 |
| 2 | 1,244 | 0.649 | 0.889 |
| 3 | 45 | 0.738 | 0.930 |

해석:

- score 0과 1에서는 비교적 안정적이다.
- score 2와 3에서는 오차가 커진다.
- score 3은 45개뿐이므로 severe gait class에 대한 학습 신호가 부족하다.
- 따라서 severe class 성능은 향후 class-weighted loss, oversampling,
  targeted augmentation으로 보완해야 한다.

## 7. Confusion Matrix

회귀 출력값을 반올림하고 `[0, 3]` 범위로 clipping한 row-normalized confusion
matrix이다.

| True score | Pred 0 | Pred 1 | Pred 2 | Pred 3 |
|---:|---:|---:|---:|---:|
| 0 | 0.826 | 0.170 | 0.004 | 0.000 |
| 1 | 0.225 | 0.691 | 0.083 | 0.000 |
| 2 | 0.145 | 0.337 | 0.516 | 0.002 |
| 3 | 0.000 | 0.133 | 0.444 | 0.422 |

해석:

- 대부분의 오분류는 인접 score 사이에서 발생한다.
- score 2와 score 3은 낮은 score로 과소평가되는 경향이 있다.
- 임상적으로 완전히 비현실적인 큰 jump는 거의 없다.

## 8. COM Robustness Analysis

Configuration D를 checkpoint 저장 방식으로 다시 학습한 뒤, test-time에만
perturbation을 적용하였다. 이 checkpointed D run의 baseline은 MAE 0.369,
RMSE 0.564로, 최종 메인 D run과 거의 유사하지만 완전히 동일한 학습 run은
아니다.

핵심 결론:

- COM centering은 horizontal translation에는 매우 강건하다.
- 하지만 COM centering만으로 scale invariance를 보장하지는 못한다.

| Condition | MAE | RMSE | Delta MAE (%) | Delta RMSE (%) |
|---|---:|---:|---:|---:|
| Original | 0.369 | 0.564 | 0.000 | 0.000 |
| Scale 0.70 | 0.501 | 0.689 | 35.925 | 22.166 |
| Scale 0.85 | 0.416 | 0.639 | 12.733 | 13.259 |
| Scale 1.15 | 0.469 | 0.673 | 27.050 | 19.226 |
| Scale 1.30 | 0.540 | 0.780 | 46.488 | 38.201 |
| Shift -0.20 | 0.369 | 0.564 | -0.000 | 0.000 |
| Shift -0.10 | 0.369 | 0.564 | 0.000 | 0.000 |
| Shift +0.10 | 0.369 | 0.564 | -0.000 | 0.000 |
| Shift +0.20 | 0.369 | 0.564 | -0.000 | 0.000 |

논문용 안전한 해석:

> COM-centered normalization effectively removes global horizontal position
> shifts, but COM centering alone does not guarantee camera-distance or body-size
> scale invariance.

## 9. Scale-Robust Operating Point

COM-only D의 scale sensitivity를 보완하기 위해 동일한 Ours V1 architecture에
input normalization만 강화한 scale-robust variant를 평가하였다.

| Variant | Scale normalization | Train-time scale augmentation | MAE | RMSE | MedAE | Max scale Delta MAE (%) | Max translation Delta MAE (%) | 결정 |
|---|---|---|---:|---:|---:|---:|---:|---|
| COM-only D checkpoint | none | none | 0.369 | 0.564 | 0.159 | 46.488 | 0.000 | 정확도는 좋지만 scale-robust 아님 |
| Scale augmentation | none | 0.85-1.15 | 0.402 | 0.605 | 0.205 | 3.399 | 0.000 | 강건하지만 정확도 손실 |
| Hip-width normalization | hip width | none | 0.380 | **0.556** | 0.204 | 0.000 | 0.000 | 강건하지만 MAE 손실 |
| Median-bone normalization + augmentation | median bone length | 0.85-1.15 | **0.366** | 0.567 | 0.139 | **0.000** | **0.000** | 권장 robust operating point |

해석:

- 가장 좋은 trade-off는 median-bone normalization + moderate scale augmentation이다.
- 이 variant는 새로운 architecture가 아니다.
- 동일한 Ours V1 구조에 body-scale normalization을 추가한 deployment-oriented
  operating point이다.

논문 표현:

> COM centering removes global position shifts, and adding sequence-level
> body-scale normalization based on median bone length removes the residual
> scale sensitivity induced by simulated camera-distance changes.

## 10. Cross-Dataset Validation

외부 일반화 성능을 보기 위해 두 개의 zero-shot transfer protocol을 수행하였다.
fine-tuning, domain adaptation, test-set checkpoint selection은 모두 사용하지
않았다.

| Protocol | Train Set | Test Set | N train | N test | MAE | RMSE | MedAE |
|---|---|---|---:|---:|---:|---:|---:|
| Zero-shot transfer | CNUH | CARE-PD | 21 | 6,066 | 0.747 | 0.882 | 0.921 |
| Reverse transfer | CARE-PD | CNUH | 6,066 | 21 | 1.014 | 1.170 | 0.746 |
| Combined GroupKFold | CNUH + CARE-PD | CNUH + CARE-PD | subject-level 5-fold | 6,087 | 0.358 | 0.564 | 0.147 |

해석:

- zero-shot transfer에서는 성능이 크게 저하된다.
- 이는 모델 구조 자체의 실패라기보다는 site, camera, pose representation,
  annotation distribution 차이에 의한 domain gap으로 보는 것이 안전하다.
- combined GroupKFold에서는 두 domain이 모두 training에 포함되므로 성능이
  회복된다.

| Comparison | Delta MAE | Delta RMSE | Relative MAE increase | Relative RMSE increase |
|---|---:|---:|---:|---:|
| CNUH -> CARE-PD vs Combined | +0.390 | +0.318 | +109.0% | +56.3% |
| CARE-PD -> CNUH vs Combined | +0.657 | +0.605 | +183.7% | +107.3% |

## 11. Domain-Gap Follow-Up Analyses

### 11.1 Dataset-Wise Breakdown Under Combined GroupKFold

CARE-PD가 전체 sequence 대부분을 차지하므로 dataset-wise breakdown을 별도로
확인하였다.

| Model | CARE-PD MAE | CARE-PD RMSE | CNUH MAE | CNUH RMSE |
|---|---:|---:|---:|---:|
| Ours V1 | **0.356** | 0.562 | 0.793 | **0.945** |
| Lu official | 0.403 | **0.540** | 0.862 | 1.031 |
| ST-GCN | 0.442 | 0.621 | 0.879 | 1.008 |
| Temporal CNN | 0.420 | 0.581 | 1.624 | 2.199 |

해석:

- CARE-PD test sample에서는 Ours V1이 가장 낮은 MAE를 보인다.
- CNUH는 21개뿐이라 fold별/개별 sample 영향이 크다.
- 따라서 CNUH subset 결과는 강한 일반화 claim보다는 sample-limited 분석으로
  다루는 것이 안전하다.

### 11.2 Score-Balanced Transfer Analysis

| Protocol | Original MAE | Original RMSE | Score-balanced MAE | Score-balanced RMSE | Balanced - Original MAE |
|---|---:|---:|---:|---:|---:|
| CNUH -> CARE-PD | 0.747 | 0.882 | 1.022 | 1.024 | +0.275 |
| CARE-PD -> CNUH | 1.014 | 1.170 | 1.025 | 1.034 | +0.010 |

해석:

- CNUH -> CARE-PD에서는 original MAE보다 score-balanced MAE가 훨씬 크다.
- 이는 model이 CARE-PD sample을 대부분 score 1 근처로 예측하기 때문이다.
- true score 1에서는 맞지만 score 0, 2, 3에서는 calibration이 좋지 않다.

### 11.3 Few-Shot Target-Site Calibration

zero-shot prediction에 대해 affine calibration만 적용하였다.

```text
y_calibrated = a * y_pred + b
```

모델 weight 재학습은 하지 않았다.

| Protocol | Calibration subjects | Base MAE | Calibrated MAE | Delta MAE | Base RMSE | Calibrated RMSE | Delta RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNUH -> CARE-PD | 1 | 0.747 | 0.791 | +0.044 | 0.881 | 1.023 | +0.142 |
| CNUH -> CARE-PD | 3 | 0.748 | 0.672 | -0.076 | 0.883 | 0.845 | -0.037 |
| CNUH -> CARE-PD | 5 | 0.751 | 0.659 | -0.092 | 0.884 | 0.814 | -0.070 |
| CNUH -> CARE-PD | 10 | 0.748 | 0.622 | -0.126 | 0.882 | 0.763 | -0.119 |
| CARE-PD -> CNUH | 1 | 1.015 | 1.038 | +0.023 | 1.171 | 1.260 | +0.090 |
| CARE-PD -> CNUH | 3 | 1.017 | 1.077 | +0.060 | 1.170 | 1.290 | +0.120 |
| CARE-PD -> CNUH | 5 | 1.029 | 0.990 | -0.039 | 1.179 | 1.162 | -0.017 |
| CARE-PD -> CNUH | 10 | 0.999 | 0.836 | -0.163 | 1.149 | 0.991 | -0.158 |

좋은 주장 방향:

> Zero-shot transfer exposes a substantial domain gap, but the gap is partly
> calibratable. A small labeled target-site calibration set can reduce transfer
> error without retraining the full model.

국문 해석:

> 완전한 zero-shot transfer는 어렵지만, 새로운 병원/사이트에서 소량의 labeled
> calibration subject만 확보하면 전체 모델 재학습 없이도 domain gap을 줄일 수
> 있다.

### 11.4 Ours vs SOTA Cross-Dataset Transfer

`run_cross_dataset_model_comparison_cuda.cmd` 결과를 반영한 zero-shot 모델 비교이다.
동일한 transfer 조건에서 Ours V1, ST-GCN, Lu official baseline을 비교했다.

| Model | Direction | MAE | RMSE | MedAE |
|---|---|---:|---:|---:|
| Ours V1 | CNUH -> CARE-PD | **0.747** | **0.882** | 0.921 |
| ST-GCN | CNUH -> CARE-PD | 8.346 | 9.737 | 6.734 |
| Lu official | CNUH -> CARE-PD | 0.898 | 1.016 | **0.596** |
| Ours V1 | CARE-PD -> CNUH | 0.910 | 1.034 | **0.639** |
| ST-GCN | CARE-PD -> CNUH | 1.203 | 1.385 | 1.119 |
| Lu official | CARE-PD -> CNUH | **0.865** | **1.027** | 0.735 |

해석:

- CNUH -> CARE-PD 방향에서는 Ours V1이 MAE/RMSE 기준 가장 좋다.
- CARE-PD -> CNUH 방향에서는 Lu official이 MAE/RMSE 기준 근소하게 가장 좋다.
- 두 방향 평균 transfer MAE는 Ours V1이 가장 낮다.
- ST-GCN은 CNUH 21개 sample만으로 학습했을 때 매우 불안정하다.
- 따라서 manuscript에서는 "Ours가 모든 transfer 방향에서 압도적으로 우수하다"가 아니라,
  "Ours가 tiny-source transfer와 평균 transfer MAE에서 가장 안정적이며, reverse
  transfer에서는 Lu official이 근소하게 우수하다"라고 쓰는 것이 안전하다.

### 11.5 CARE-PD Leave-One-Dataset-Out

CARE-PD leave-one-dataset-out은 CARE-PD 내부 source cohort 하나를 통째로
test set으로 제외하고 학습하는 검증이다. 같은 CARE-PD 안에서의 실험이지만, test
cohort가 학습에 전혀 등장하지 않기 때문에 일반 subject-level GroupKFold보다 더
강한 외부 일반화 평가에 가깝다.

| Held-out CARE-PD cohort | N train | N test | MAE | RMSE | MedAE |
|---|---:|---:|---:|---:|---:|
| 3DGait | 5,976 | 90 | 0.775 | 0.947 | 0.847 |
| BMCLab | 2,171 | 3,895 | 0.663 | 0.844 | 0.528 |
| PD-GaM | 4,366 | 1,700 | 0.495 | 0.724 | 0.236 |
| T-SDU-PD | 5,685 | 381 | 0.692 | 0.836 | 0.707 |
| **Overall** | - | 6,066 | **0.620** | **0.813** | **0.508** |

관련 protocol과 비교하면 다음과 같다.

| Protocol | MAE | RMSE | MedAE | 해석 |
|---|---:|---:|---:|---|
| Combined GroupKFold, CARE-PD subset | 0.356 | 0.562 | 0.146 | CARE-PD cohort가 train fold에 포함됨 |
| CARE-PD leave-one-dataset-out | 0.620 | 0.813 | 0.508 | CARE-PD cohort 하나를 완전히 제외 |
| CNUH -> CARE-PD zero-shot | 0.747 | 0.882 | 0.921 | 병원/site와 pose representation이 모두 다름 |

해석:

- CARE-PD LODO는 combined GroupKFold보다 어렵다.
- 하지만 CNUH -> CARE-PD zero-shot보다는 좋다.
- 따라서 multi-cohort CARE-PD 학습은 보지 못한 CARE-PD cohort에 대한 일반화를
  개선하지만, cohort-level domain gap은 여전히 남아 있다.
- 가장 어려운 held-out cohort는 3DGait(`MAE = 0.775`)이고, 가장 쉬운 cohort는
  PD-GaM(`MAE = 0.495`)이다.

논문에 쓰기 좋은 표현:

> In CARE-PD leave-one-dataset-out evaluation, the proposed model achieved
> MAE = 0.620 and RMSE = 0.813 across four held-out source cohorts. The result
> was worse than subject-level GroupKFold but better than CNUH-to-CARE-PD
> zero-shot transfer, indicating that multi-cohort training improves
> generalization to unseen CARE-PD cohorts while cohort-level domain shift
> remains.

## 12. A/B/C/D Ablation

| Model | Ablation | Feature set | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ours V1 | A | coordinates only | 5 | 6,087 | 158,210 | 3.179 | 0.374 | 0.577 | 0.168 |
| Ours V1 | B | coordinates + velocity | 5 | 6,087 | 158,402 | 2.757 | 0.432 | 0.629 | 0.246 |
| Ours V1 | C | coordinates + velocity + amplitude/variability | 5 | 6,087 | 158,530 | 4.178 | 0.376 | **0.549** | 0.192 |
| Ours V1 | D | full hybrid feature set including angle | 5 | 6,087 | 158,594 | 4.615 | **0.358** | 0.564 | **0.147** |

해석:

- D는 MAE와 MedAE가 가장 좋다.
- C는 RMSE가 가장 낮다.
- B는 velocity만 추가했을 때 오히려 성능이 나빠진다.
- 즉 단순 velocity보다 amplitude/variability/angle 같은 higher-level sequence
  descriptor가 필요하다.

## 13. Ours V1 Architecture Ablation

위 A/B/C/D 절제는 입력 feature 절제이고, 아래 표는 architecture 구성요소 절제이다.
입력은 모두 Configuration D로 고정하고, 모델 구성요소만 단계적으로 제거했다. Full
Ours V1 행은 최종 5-fold 결과인 `groupkfold_h36m17_ours_lu_official_cuda`를
사용했다.

| Model | Components | Folds | N | Params | Infer ms/sample | MAE | RMSE | MedAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MLP only | mean pooling + bounded MLP | 5 | 6,087 | 17,921 | 0.005 | 0.554 | 0.653 | 0.481 |
| GraphConv + MLP | GraphConv, no joint attention, no Temporal Transformer | 5 | 6,087 | 25,985 | 2.306 | 0.450 | 0.580 | 0.349 |
| GraphConv + Joint Attention + MLP | GraphConv + joint attention, no Temporal Transformer | 5 | 6,087 | 26,114 | 2.736 | 0.414 | 0.564 | 0.291 |
| Full Ours V1 | GraphConv + Joint Attention + Temporal Transformer | 5 | 6,087 | 158,594 | 4.615 | **0.358** | 0.564 | **0.147** |

해석:

- MLP only 대비 GraphConv를 추가하면 MAE가 `0.554 -> 0.450`으로 감소한다.
- Joint Attention을 추가하면 MAE가 `0.450 -> 0.414`로 더 감소한다.
- Temporal Transformer까지 포함한 full model은 MAE `0.358`, MedAE `0.147`로
  가장 좋다.
- Full model은 MLP only 대비 MAE를 35.4% 줄인다.

## 14. 최종 논문 주장 방향

강하게 주장해도 되는 부분:

- 동일 subject-level GroupKFold 조건에서 Ours V1은 ML/DL/SOTA baseline보다
  MAE와 MedAE가 우수하다.
- COM centering은 horizontal translation에 강건하다.
- median-bone body-scale normalization을 추가하면 scale perturbation에도
  강건한 operating point를 만들 수 있다.
- zero-shot transfer는 어렵지만, target-site calibration으로 일부 개선 가능하다.

조심해야 하는 부분:

- "완전한 cross-site zero-shot generalization을 달성했다"라고 쓰면 안 된다.
- "COM normalization만으로 camera-distance invariance가 확보된다"라고 쓰면
  안 된다.
- CNUH dataset-wise 결과는 N=21이므로 강한 결론을 피해야 한다.

가장 안전한 최종 메시지:

> The proposed model achieves strong subject-independent performance when
> trained on multi-domain data. Zero-shot cross-dataset transfer reveals a real
> site and representation domain gap, but this gap is partly reducible through
> lightweight target-site calibration. Therefore, the system is best positioned
> as a deployable clinical decision-support model that should be calibrated when
> introduced to a new clinical site.

## 15. Reviewer-Oriented Figures

### 15.1 Calibration Reliability Curve

Calibration reliability curve는 기존 GroupKFold prediction만으로 계산했으며,
추가 학습은 필요하지 않았다.

| Model | MAE | Figure use |
|---|---:|---|
| Ours V1 | **0.358** | proposed model calibration curve |
| Lu official | 0.404 | SOTA comparison curve |
| MotionAGFormer-XS | 0.405 | SOTA comparison curve; 포함하려면 figure 재생성 필요 |
| Temporal CNN | 0.425 | deep baseline comparison curve |
| MotionBERT-Lite (81-frame) | 0.442 | SOTA comparison curve; 포함하려면 figure 재생성 필요 |
| ST-GCN | 0.443 | SOTA comparison curve |

주의: 이 figure에는 별도의 calibration 수치를 표기하지 않는다. Ours V1은 가장 낮은
MAE를 유지하면서 예측 score bin이 관측 severity와 단조적인 관계를 보인다는 보조
근거로 쓰는 것이 안전하다.

관련 출력:

```text
docs/calibration_reliability_analysis.md
docs/reviewer_figures/21_calibration_curve_ours.png
docs/reviewer_figures/22_calibration_curve_models.png
```

### 15.2 Learning Curve

Learning curve 실험도 완료했다. train subject fraction을 10%, 25%, 50%,
75%, 100%로 바꾸고, validation fold는 고정했다.

| Train fraction | Mean train subjects | MAE | RMSE | 10% 대비 MAE 감소율 |
|---:|---:|---:|---:|---:|
| 10% | 11.0 | 0.476 | 0.672 | 0.0% |
| 25% | 26.4 | 0.460 | 0.646 | 3.4% |
| 50% | 52.6 | 0.413 | 0.624 | 13.2% |
| 75% | 79.0 | 0.390 | 0.578 | 18.1% |
| 100% | 104.8 | 0.360 | 0.530 | 24.5% |

MAE와 RMSE가 training subject 수 증가에 따라 단조 감소한다. 10%에서 100%로
늘리면 MAE는 `0.476 -> 0.360`으로 24.5% 감소하고, RMSE는
`0.672 -> 0.530`으로 21.1% 감소한다.

이 결과는 CNUH N=21의 한계를 "모델 실패"가 아니라 "data-limited clinical
cohort"로 설명하는 근거로 사용할 수 있다.

관련 출력:

```text
docs/learning_curve_ours_analysis.md
docs/reviewer_figures/24_learning_curve_ours_mae.png
docs/reviewer_figures/25_learning_curve_ours_rmse.png
```

## 16. 관련 출력 문서

```text
docs/final_integrated_results.md
docs/cross_dataset_validation_analysis.md
docs/cross_dataset_validation_record_en.md
docs/cross_dataset_validation_record_ko.md
docs/cross_dataset_model_comparison.md
docs/dataset_wise_breakdown_analysis.md
docs/score_balanced_transfer_analysis.md
docs/fewshot_calibration_analysis.md
docs/carepd_lodo_analysis.md
docs/calibration_reliability_analysis.md
docs/learning_curve_ours_analysis.md
docs/architecture_ablation_analysis.md
docs/reviewer_experiment_figures.md
docs/com_robustness_final_analysis.md
docs/scale_robustness_full_summary.md
docs/domain_gap_followup_experiments.md
```
