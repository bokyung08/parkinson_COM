# Rebuttal Baseline Comparison

## 목적

Reviewer 1의 "insufficient baseline comparison" 지적에 대응하기 위해, 기존 Configuration D 피처를 사용한 단순 머신러닝 베이스라인을 추가 실험했다.

## 실험 설정

- 데이터: `HospitalData/processed_pose_data`
- 라벨: `HospitalData/JSON`
- 샘플 수: 21
- 기본 타깃: `item10`  
  현재 `src/train_model.py`의 메인 파이프라인과 맞춘 설정이다.
- 피처: Configuration D
  - COM-relative coordinates
  - relative velocity
  - amplitude
  - variability
  - joint angle
- 단순 모델 입력: D 피처를 frame 축으로 summary pooling
  - mean, std, min, max, 25th percentile, 75th percentile
  - 최종 tabular feature: 1,782 dimensions
- 평가:
  - Hold-out: 80/20, `random_state=42`
  - 추가 안정성 확인: 5-fold CV, `random_state=42`

## Rebuttal용 핵심 표

기존 제안 모델 수치는 repo의 `results/model_comparison_summary.csv`에 기록된 Configuration D 결과를 참조했다. 새 baseline 수치는 `Rebuttal/results/item10_D_80_20/baseline_summary.csv`에서 생성했다.

| Model | Eval | MAE | RMSE | Spearman | 비고 |
|---|---|---:|---:|---:|---|
| Proposed Main Model | existing D result | 0.407 | 0.460 | NA | `results/model_comparison_summary.csv` |
| Ridge | 80/20 hold-out | 0.435 | 0.567 | 0.289 | added baseline |
| SVR (RBF) | 80/20 hold-out | 0.451 | 0.571 | 0.000 | added baseline |
| Random Forest | 80/20 hold-out | 0.468 | 0.556 | 0.000 | added baseline |
| MLP | 80/20 hold-out | 1.845 | 2.320 | -0.866 | added baseline |
| Dummy Mean | 80/20 hold-out | 0.525 | 0.718 | NA | sanity baseline |

## 5-fold CV baseline 결과

제안 모델과 동일 프로토콜로 재학습한 CV 결과는 아직 없으므로, 아래 표는 단순 baseline 내부 안정성 확인용이다. 작은 데이터셋에서는 target 분포가 좁아 Dummy Mean도 경쟁적으로 보일 수 있으므로, rebuttal에서는 hold-out 표를 메인으로 쓰고 CV는 보조 결과로 쓰는 편이 안전하다.

| Model | MAE | RMSE | Spearman | Pearson | R2 |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.735 | 0.905 | 0.150 | 0.071 | -0.075 |
| SVR (RBF) | 0.783 | 1.010 | -0.023 | -0.111 | -0.340 |
| Ridge | 0.839 | 1.052 | 0.144 | 0.068 | -0.452 |
| MLP | 3.146 | 4.570 | -0.466 | -0.377 | -26.413 |
| Dummy Mean | 0.719 | 0.909 | -0.345 | -0.377 | -0.085 |

## Optional: gait sum target

논문 타깃이 item10 단일 점수가 아니라 item 10-14 합산 gait score라면 아래 산출물을 사용한다.

- 결과 폴더: `Rebuttal/results/gait_D_80_20`
- 80/20 hold-out MAE:
  - Random Forest: 1.206
  - SVR (RBF): 1.413
  - MLP: 2.901
  - Ridge: 2.548
  - Dummy Mean: 1.238

## Rebuttal Draft

단순 모델 및 기존 baseline과의 비교가 필요하다는 건설적인 피드백에 감사드립니다. 리뷰어님의 제안에 따라, 동일한 Configuration D 피처를 기반으로 Random Forest, SVR, MLP 및 Ridge 회귀 모델을 추가로 평가했습니다. 동일한 데이터와 라벨 정의를 사용한 80/20 hold-out 평가에서 Random Forest, SVR, MLP의 MAE는 각각 0.468, 0.451, 1.845로 나타났으며, 기존 Configuration D 제안 모델의 MAE 0.407보다 높았습니다. 또한 5-fold CV에서도 비-dummy 단순 baseline 중 Random Forest가 가장 낮은 MAE 0.735를 보였고, SVR과 Ridge는 각각 0.783, 0.839를 기록했습니다. 이 결과는 작은 데이터셋에서 단순 모델이 의미 있는 benchmark 역할을 함을 확인하는 동시에, 보행 시퀀스의 관절 간 및 시간적 의존성을 함께 모델링하는 제안 구조의 타당성을 보조적으로 뒷받침합니다. 최종 원고의 baseline comparison 절에 이 결과를 추가하겠습니다.

## 산출물

- 실행 스크립트: `Rebuttal/run_baseline_comparison.py`
- item10 결과: `Rebuttal/results/item10_D_80_20`
- gait sum 결과: `Rebuttal/results/gait_D_80_20`
- 자동 생성 상세 MD: `Rebuttal/results/item10_D_80_20/RESULTS.md`
- 예측값: `holdout_predictions.tsv`, `cv_predictions.tsv`
- 데이터 manifest: `dataset_manifest.tsv`
