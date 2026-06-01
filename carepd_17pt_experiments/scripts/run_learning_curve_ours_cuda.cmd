@echo off
setlocal
cd /d "%~dp0\.."

if exist ".venv_cuda\Scripts\python.exe" (
  set PYTHON_EXE=.venv_cuda\Scripts\python.exe
) else (
  set PYTHON_EXE=python
)

"%PYTHON_EXE%" scripts\run_learning_curve_ours.py ^
  --manifest data\processed\manifest.csv ^
  --out_dir results\learning_curve_ours ^
  --fig_dir docs\reviewer_figures ^
  --doc_path docs\learning_curve_ours_analysis.md ^
  --fractions 0.10 0.25 0.50 0.75 1.00 ^
  --seeds 42 ^
  --n_splits 5 ^
  --epochs 80 ^
  --batch_size 16 ^
  --device cuda
