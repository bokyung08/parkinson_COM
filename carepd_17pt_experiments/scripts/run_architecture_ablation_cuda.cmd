@echo off
setlocal
cd /d "%~dp0\.."

if exist ".venv_cuda\Scripts\python.exe" (
  set PYTHON_EXE=.venv_cuda\Scripts\python.exe
) else (
  set PYTHON_EXE=python
)

"%PYTHON_EXE%" scripts\run_loso_experiments.py ^
  --manifest data\processed\manifest.csv ^
  --out_dir results\architecture_ablation_ours_cuda ^
  --models ours_mlp ours_gcn_mlp ours_gcn_attn_mlp ours ^
  --split_strategy groupkfold ^
  --n_splits 5 ^
  --target item10 ^
  --ablation D ^
  --max_len 390 ^
  --epochs 80 ^
  --batch_size 16 ^
  --device cuda
