# H36M17 Data Format

All datasets in this experiment folder must be converted to the same 17-joint
format before training.

## Processed Files

```text
data/processed/
  manifest.csv
  <dataset>/<sample_id>.npz
```

Each `.npz` file contains:

- `joints`: float32 array shaped `(T, 17, 3)`.

`manifest.csv` columns:

- `dataset`
- `sample_id`
- `patient_id`
- `path`
- `target`
- `frames`

The `path` column is relative to the folder containing `manifest.csv`.

## Joint Order

0. Pelvis
1. R_Hip
2. R_Knee
3. R_Ankle
4. L_Hip
5. L_Knee
6. L_Ankle
7. Spine
8. Thorax
9. Neck/Nose
10. Head
11. L_Shoulder
12. L_Elbow
13. L_Wrist
14. R_Shoulder
15. R_Elbow
16. R_Wrist
