@echo off
cd /d "%~dp0.."

for %%A in (A B C D) do (
  if exist "results\groupkfold_h36m17_ours_ablation_%%A_cuda\summary.csv" (
    echo [INFO] Skipping Ours V1 ablation %%A; summary already exists.
  ) else (
    echo [INFO] Running Ours V1 ablation %%A
    ".venv_cuda\Scripts\python.exe" scripts\run_loso_experiments.py ^
      --manifest data\processed\manifest.csv ^
      --out_dir results\groupkfold_h36m17_ours_ablation_%%A_cuda ^
      --models ours ^
      --split_strategy groupkfold ^
      --n_splits 5 ^
      --ablation %%A ^
      --target item10 ^
      --epochs 80 ^
      --batch_size 8 ^
      --device cuda
    if errorlevel 1 exit /b 1
  )
)
