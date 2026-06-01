# Cross-Dataset Validation 결과 기록 및 분석

- 최종 업데이트: 2026-06-01
- 모델: Ours V1, Configuration D
- 구조: GraphConv + Joint Attention + Temporal Transformer + bounded regression
- 예측 대상: MDS-UPDRS Part III item 3.10 gait score, 범위 0-3
- fine-tuning/adaptation: 사용하지 않음
- test-set checkpoint selection: 사용하지 않음
- external-transfer 학습: fixed 80 epochs
- device: CUDA

## 실험 목적

이 실험은 외부 일반화 성능을 확인하기 위한 것이다. 메인 combined GroupKFold
실험과 달리, zero-shot transfer에서는 한 데이터셋 전체를 학습에서 제외하고
다른 데이터셋에 바로 평가한다.

따라서 이 실험의 목적은 adaptation 이후의 최고 성능을 보이는 것이 아니라,
서로 다른 acquisition domain 사이에서 모델을 그대로 이동했을 때 성능이 얼마나
떨어지는지 측정하는 것이다.

## 실험 프로토콜

| Protocol | Train Set | Test Set | 설명 |
|---|---|---|---|
| Zero-shot transfer | CNUH | CARE-PD | 작은 CNUH clinical cohort만 학습하고 CARE-PD에 바로 평가 |
| Reverse transfer | CARE-PD | CNUH | CARE-PD만 학습하고 CNUH에 바로 평가 |
| Combined GroupKFold | CNUH + CARE-PD | CNUH + CARE-PD | 두 domain이 모두 학습에 포함된 subject-level GroupKFold 메인 결과 |

## 주요 결과

| Protocol | Train Set | Test Set | N train | N test | MAE | RMSE | MedAE |
|---|---|---|---:|---:|---:|---:|---:|
| Zero-shot transfer | CNUH | CARE-PD | 21 | 6,066 | 0.747 | 0.882 | 0.921 |
| Reverse transfer | CARE-PD | CNUH | 6,066 | 21 | 1.014 | 1.170 | 0.746 |
| Combined GroupKFold | CNUH + CARE-PD | CNUH + CARE-PD | subject-level 5-fold | 6,087 | 0.358 | 0.564 | 0.147 |

## Domain Gap 정량화

| Comparison | Delta MAE | Delta RMSE | Relative MAE increase | Relative RMSE increase |
|---|---:|---:|---:|---:|
| CNUH -> CARE-PD vs Combined | +0.390 | +0.318 | +109.0% | +56.3% |
| CARE-PD -> CNUH vs Combined | +0.657 | +0.605 | +183.7% | +107.3% |

해석: direct zero-shot transfer는 combined GroupKFold보다 훨씬 어렵다. 이
결과는 모델이 완전히 실패했다기보다, site 차이와 pose representation 차이에서
오는 실제 domain gap으로 해석하는 것이 안전하다.

## 예측 분포 분석

| Protocol | Test N | Mean true score | Mean predicted score | Prediction range | MAE |
|---|---:|---:|---:|---|---:|
| CNUH -> CARE-PD | 6,066 | 0.789 | 1.144 | 0.975-1.205 | 0.747 |
| CARE-PD -> CNUH | 21 | 1.000 | 1.759 | 1.446-2.233 | 1.014 |

두 transfer 방향 모두 regression-to-the-middle 현상을 보인다. bounded
regression 덕분에 출력값은 유효 범위 안에 있지만, unseen target domain에서는
예측값이 좁은 점수 구간으로 몰린다. 즉 문제는 invalid output이 아니라
target-domain calibration 부족이다.

## Per-Class Transfer Error

### CNUH -> CARE-PD

| True score | N | Mean prediction | MAE | RMSE |
|---:|---:|---:|---:|---:|
| 0 | 2,608 | 1.161 | 1.161 | 1.162 |
| 1 | 2,175 | 1.144 | 0.144 | 0.148 |
| 2 | 1,239 | 1.106 | 0.894 | 0.896 |
| 3 | 44 | 1.110 | 1.890 | 1.891 |

해석: CNUH로 학습한 모델은 CARE-PD sample 대부분을 score 1.1 근처로 예측한다.
true score 1에는 비교적 맞지만, score 0은 과대평가하고 score 2-3은
과소평가한다.

### CARE-PD -> CNUH

| True score | N | Mean prediction | MAE | RMSE |
|---:|---:|---:|---:|---:|
| 0 | 7 | 1.730 | 1.730 | 1.730 |
| 1 | 8 | 1.813 | 0.813 | 0.847 |
| 2 | 5 | 1.717 | 0.283 | 0.285 |
| 3 | 1 | 1.727 | 1.273 | 1.273 |

해석: CARE-PD로 학습한 모델은 CNUH sample을 score 1.7 근처로 예측한다.
true score 2에는 도움이 되지만 score 0과 1에서는 큰 오차가 발생한다. CNUH는
21명뿐이므로 이 방향의 결과는 개별 sample과 class imbalance에 매우 민감하다.

## 완료된 Follow-Up 분석

### 1. Combined GroupKFold의 Dataset-Wise Breakdown

| Dataset | N | Subjects | Ours V1 MAE | Ours V1 RMSE | Ours V1 MedAE |
|---|---:|---:|---:|---:|---:|
| CARE-PD | 6,066 | 110 | 0.356 | 0.562 | 0.146 |
| CNUH | 21 | 21 | 0.793 | 0.945 | 0.987 |

해석: combined 결과는 sequence 수 기준으로 CARE-PD의 영향이 크다. Ours V1은
CARE-PD에서는 매우 강하지만, CNUH만 따로 보면 sample 수가 21개뿐이라 안정적
dataset-level 결론을 내리기 어렵다.

### 2. Score-Balanced Transfer

| Protocol | Original MAE | Original RMSE | Score-balanced MAE | Score-balanced RMSE | Balanced - Original MAE |
|---|---:|---:|---:|---:|---:|
| CNUH -> CARE-PD | 0.747 | 0.882 | 1.022 | 1.024 | +0.275 |
| CARE-PD -> CNUH | 1.014 | 1.170 | 1.025 | 1.034 | +0.010 |

해석: CNUH -> CARE-PD에서는 score-balanced MAE가 원래 MAE보다 훨씬 크다.
이는 원래 MAE가 CARE-PD의 class distribution에 의해 완화되어 보였다는 뜻이다.
즉 zero-shot transfer 모델은 severity class 전반에 균일하게 calibration되어
있지 않다.

### 3. Few-Shot Target-Site Calibration

Calibration 방법:

```text
y_calibrated = a * y_pred + b
```

calibrated score는 `[0, 3]`으로 clipping하였다. 모델 weight는 재학습하지 않았다.

| Protocol | Calibration subjects | Base MAE | Calibrated MAE | Delta MAE | Base RMSE | Calibrated RMSE | Delta RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNUH -> CARE-PD | 3 | 0.748 | 0.672 | -0.076 | 0.883 | 0.845 | -0.037 |
| CNUH -> CARE-PD | 5 | 0.751 | 0.659 | -0.092 | 0.884 | 0.814 | -0.070 |
| CNUH -> CARE-PD | 10 | 0.748 | 0.622 | -0.126 | 0.882 | 0.763 | -0.119 |
| CARE-PD -> CNUH | 5 | 1.029 | 0.990 | -0.039 | 1.179 | 1.162 | -0.017 |
| CARE-PD -> CNUH | 10 | 0.999 | 0.836 | -0.163 | 1.149 | 0.991 | -0.158 |

해석: zero-shot transfer 자체는 어렵지만, 오차의 일부는 calibration으로 줄일 수
있다. 이는 실제 배포 관점에서 중요한 장점이다. 새로운 병원/사이트에 모델을
도입할 때 전체 모델을 재학습하지 않더라도, 소량의 labeled target-site calibration
subject를 사용해 transfer error를 줄일 수 있다.

### 4. Ours vs SOTA Zero-Shot Transfer 비교

이 비교는 다음 명령으로 실행된 결과이다.

```text
scripts/run_cross_dataset_model_comparison_cuda.cmd
```

이 결과는 Ours-only standalone transfer 결과와 별도로 수행된 comparative rerun이다.
따라서 Ours 값에 약간의 stochastic 차이가 있을 수 있다. 모델 간 zero-shot transfer
비교에는 아래 표를 사용한다.

| Category | Model | Train | Test | MAE | RMSE | MedAE |
|---|---|---|---|---:|---:|---:|
| Proposed | Ours V1 | CNUH | CARE-PD | **0.747** | **0.882** | 0.921 |
| SOTA | Lu official | CNUH | CARE-PD | 0.898 | 1.016 | **0.596** |
| SOTA | ST-GCN | CNUH | CARE-PD | 8.346 | 9.737 | 6.734 |
| Proposed | Ours V1 | CARE-PD | CNUH | 0.910 | 1.034 | **0.639** |
| SOTA | Lu official | CARE-PD | CNUH | **0.865** | **1.027** | 0.735 |
| SOTA | ST-GCN | CARE-PD | CNUH | 1.203 | 1.385 | 1.119 |

해석:

- CNUH -> CARE-PD 방향에서는 Ours V1이 MAE/RMSE 기준 가장 좋다.
- CARE-PD -> CNUH 방향에서는 Lu official이 MAE/RMSE 기준 근소하게 가장 좋다.
- 두 방향 평균 MAE는 Ours V1이 `0.829`로 가장 낮다. Lu official은 `0.882`,
  ST-GCN은 `4.774`이다.
- ST-GCN은 CNUH 21개만 학습한 뒤 CARE-PD에 평가할 때 매우 불안정하다. 이는
  unbounded regression head가 작은 source dataset에서 외삽에 취약하기 때문일
  가능성이 있다.

### 5. CARE-PD Leave-One-Dataset-Out

이 검증은 다음 명령어로 수행했다.

```text
scripts/run_carepd_lodo_cuda.cmd
```

| Held-out CARE-PD cohort | N train | N test | MAE | RMSE | MedAE |
|---|---:|---:|---:|---:|---:|
| 3DGait | 5,976 | 90 | 0.775 | 0.947 | 0.847 |
| BMCLab | 2,171 | 3,895 | 0.663 | 0.844 | 0.528 |
| PD-GaM | 4,366 | 1,700 | 0.495 | 0.724 | 0.236 |
| T-SDU-PD | 5,685 | 381 | 0.692 | 0.836 | 0.707 |
| **Overall** | - | 6,066 | **0.620** | **0.813** | **0.508** |

해석: CARE-PD LODO는 combined subject-level GroupKFold보다 어렵지만
CNUH -> CARE-PD zero-shot보다는 좋다. 즉 CARE-PD 내부에도 cohort-level
shift가 존재하지만, 여러 CARE-PD cohort를 함께 학습하면 작은 CNUH source만으로
학습하는 것보다 보지 못한 CARE-PD cohort에 더 잘 일반화된다.

## 최종 해석

이 결과는 successful zero-shot generalization으로 제시하면 안 된다. 더 안전하고
정확한 해석은 다음과 같다.

> Zero-shot cross-dataset transfer revealed a substantial site and
> representation domain gap. However, combined multi-domain training achieved
> strong subject-independent performance, CARE-PD leave-one-dataset-out showed
> that multi-cohort training improves generalization to unseen CARE-PD cohorts,
> and follow-up calibration analysis showed that part of the transfer error can
> be reduced with a small labeled target-site calibration set.

국문 정리:

> 완전한 zero-shot 외부 일반화는 아직 어렵다. 그러나 CNUH와 CARE-PD를 함께
> 학습에 포함하면 subject-independent 성능은 강하게 회복되고, 새로운 site에서는
> 소량의 labeled calibration subject만으로도 transfer error를 일부 줄일 수 있다.

## 논문에 사용 가능한 주장

안전하게 주장 가능한 부분:

- 제안 모델은 combined subject-level GroupKFold에서 강한 성능을 보인다.
- CNUH와 CARE-PD 사이의 direct zero-shot transfer는 어렵다.
- transfer 실패는 invalid output 문제가 아니라 target-domain calibration 문제에
  가깝다.
- 소량의 target-site calibration set으로 전체 모델 재학습 없이 transfer error를
  줄일 수 있다.

피해야 할 표현:

- 완전한 cross-site zero-shot generalization을 달성했다고 주장하면 안 된다.
- 모델이 완전히 domain-invariant하다고 주장하면 안 된다.
- CNUH dataset-wise 결과는 N=21이므로 과해석하면 안 된다.

## Source Outputs

```text
results/cross_dataset_validation/summary.csv
results/cross_dataset_validation/domain_gap.csv
results/cross_dataset_validation/cnuh_to_carepd/predictions.tsv
results/cross_dataset_validation/carepd_to_cnuh/predictions.tsv
results/carepd_leave_one_dataset_out/summary.csv
docs/cross_dataset_validation_analysis.md
docs/cross_dataset_model_comparison.md
docs/carepd_lodo_analysis.md
docs/dataset_wise_breakdown_analysis.md
docs/score_balanced_transfer_analysis.md
docs/fewshot_calibration_analysis.md
```
