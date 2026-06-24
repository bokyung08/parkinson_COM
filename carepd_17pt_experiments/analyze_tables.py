#!/usr/bin/env python3
"""
Table 10 & 13 Validation Analysis
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

pred_file = "results/groupkfold_h36m17_ours_lu_official_cuda/predictions.tsv"
print("=" * 80)
print("예측 파일 로드 중...")
df = pd.read_csv(pred_file, sep="\t")
print(f"전체 행 수: {len(df)}")
df = df[df['model'] == 'ours'].copy()
print(f"'ours' 필터 후: {len(df)} 행")

def discretize(y):
    return np.clip(np.round(y), 0, 3).astype(int)

def compute_f1_013(y_true, y_pred):
    y_true_d = discretize(y_true)
    y_pred_d = discretize(y_pred)
    return f1_score(y_true_d, y_pred_d, labels=[0,1,2,3], average='macro', zero_division=0)

def compute_f1_012(y_true, y_pred):
    y_true_d = discretize(y_true)
    y_pred_d = discretize(y_pred)
    return f1_score(y_true_d, y_pred_d, labels=[0,1,2], average='macro', zero_division=0)

print("\n" + "=" * 80)
print("검증: 풀링 지표를 논문과 비교")
pooled_mae = np.mean(np.abs(df['y_true'] - df['y_pred']))
pooled_rmse = np.sqrt(np.mean((df['y_true'] - df['y_pred']) ** 2))
pooled_f1_013 = compute_f1_013(df['y_true'].values, df['y_pred'].values)
pooled_f1_012 = compute_f1_012(df['y_true'].values, df['y_pred'].values)

print(f"Pooled MAE:     {pooled_mae:.6f} (논문: 0.358000, 차이: {abs(pooled_mae - 0.358):.6f})")
print(f"Pooled RMSE:    {pooled_rmse:.6f} (논문: 0.564000, 차이: {abs(pooled_rmse - 0.564):.6f})")
print(f"Pooled F1_0-3:  {pooled_f1_013:.6f} (논문: 0.661000, 차이: {abs(pooled_f1_013 - 0.661):.6f})")
print(f"Pooled F1_0-2:  {pooled_f1_012:.6f} (논문: 0.691000, 차이: {abs(pooled_f1_012 - 0.691):.6f})")

mae_ok = abs(pooled_mae - 0.358) <= 0.005
rmse_ok = abs(pooled_rmse - 0.564) <= 0.005
f1_013_ok = abs(pooled_f1_013 - 0.661) <= 0.005
f1_012_ok = abs(pooled_f1_012 - 0.691) <= 0.005

if not (mae_ok and rmse_ok and f1_013_ok and f1_012_ok):
    print("\n[FAIL] 검증 실패")
    exit(1)
print("\n[PASS] 검증 통과! 모두 +/-0.005 이내")

print("\n" + "=" * 80)
print("TABLE 10: Per-fold 계산 검증")
folds = sorted(df['fold'].unique())
for fold in folds:
    df_fold = df[df['fold'] == fold]
    mae = np.mean(np.abs(df_fold['y_true'] - df_fold['y_pred']))
    rmse = np.sqrt(np.mean((df_fold['y_true'] - df_fold['y_pred']) ** 2))
    f1_013 = compute_f1_013(df_fold['y_true'].values, df_fold['y_pred'].values)
    f1_012 = compute_f1_012(df_fold['y_true'].values, df_fold['y_pred'].values)
    print(f"Fold {fold} (n={len(df_fold)}): MAE={mae:.4f}, RMSE={rmse:.4f}, F1_0-3={f1_013:.4f}, F1_0-2={f1_012:.4f}")

print("\n" + "=" * 80)
print("TABLE 13: Deferred set 검증")

def boundary_distance(y_pred):
    boundaries = np.array([0.5, 1.5, 2.5])
    return np.min(np.abs(y_pred[:, None] - boundaries[None, :]), axis=1)

u_values = boundary_distance(df['y_pred'].values)
df['u'] = u_values

# Clinical band
df['defer_clinical'] = (df['y_pred'] >= 1.3) & (df['y_pred'] <= 1.7)
n_defer_clinical = df['defer_clinical'].sum()
pct_clinical = 100.0 * n_defer_clinical / len(df)

df_defer = df[df['defer_clinical']]
y_true_d = discretize(df_defer['y_true'].values)
counts = np.bincount(y_true_d, minlength=4)
mae_defer = np.mean(np.abs(df_defer['y_true'] - df_defer['y_pred']))
mae_retain = np.mean(np.abs(df[~df['defer_clinical']]['y_true'] - df[~df['defer_clinical']]['y_pred']))
u_defer = np.mean(df_defer['u'])
u_retain = np.mean(df[~df['defer_clinical']]['u'])
total_error = np.sum(np.abs(df['y_true'] - df['y_pred']))
defer_error = np.sum(np.abs(df_defer['y_true'] - df_defer['y_pred']))
pct_error = 100.0 * defer_error / total_error

print(f"\n임상 밴드 [1.3, 1.7]:")
print(f"  Deferred: {n_defer_clinical} ({pct_clinical:.2f}%)")
print(f"  True score 0/1/2/3: {counts[0]}/{counts[1]}/{counts[2]}/{counts[3]}")
print(f"  MAE: deferred={mae_defer:.3f}, retained={mae_retain:.3f}")
print(f"  u: deferred={u_defer:.3f}, retained={u_retain:.3f}")
print(f"  전체 절대오차의 {pct_error:.2f}% 차지")

# 70% coverage
coverage_target = 0.70
n_retain = int(len(df) * coverage_target)
tau_70 = np.sort(df['u'].values)[len(df) - n_retain]
df['defer_70'] = df['u'] < tau_70

n_defer_70 = df['defer_70'].sum()
pct_70 = 100.0 * n_defer_70 / len(df)

df_defer_70 = df[df['defer_70']]
y_true_d_70 = discretize(df_defer_70['y_true'].values)
counts_70 = np.bincount(y_true_d_70, minlength=4)
mae_defer_70 = np.mean(np.abs(df_defer_70['y_true'] - df_defer_70['y_pred']))
mae_retain_70 = np.mean(np.abs(df[~df['defer_70']]['y_true'] - df[~df['defer_70']]['y_pred']))
u_defer_70 = np.mean(df_defer_70['u'])
u_retain_70 = np.mean(df[~df['defer_70']]['u'])
defer_error_70 = np.sum(np.abs(df_defer_70['y_true'] - df_defer_70['y_pred']))
pct_error_70 = 100.0 * defer_error_70 / total_error

print(f"\n70% coverage (tau={tau_70:.4f}):")
print(f"  Deferred: {n_defer_70} ({pct_70:.2f}%)")
print(f"  True score 0/1/2/3: {counts_70[0]}/{counts_70[1]}/{counts_70[2]}/{counts_70[3]}")
print(f"  MAE: deferred={mae_defer_70:.3f}, retained={mae_retain_70:.3f}")
print(f"  u: deferred={u_defer_70:.3f}, retained={u_retain_70:.3f}")
print(f"  전체 절대오차의 {pct_error_70:.2f}% 차지")

print("\n" + "=" * 80)
print("[DONE] 모든 검증 완료!")
