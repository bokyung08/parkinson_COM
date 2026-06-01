@echo off
setlocal
cd /d "%~dp0\.."

if exist ".venv_cuda\Scripts\python.exe" (
  set PYTHON_EXE=.venv_cuda\Scripts\python.exe
) else (
  set PYTHON_EXE=python
)

"%PYTHON_EXE%" scripts\summarize_architecture_ablation.py ^
  --ablation_dir results\architecture_ablation_ours_cuda ^
  --full_ours_dir results\groupkfold_h36m17_ours_lu_official_cuda ^
  --out_csv results\architecture_ablation_summary.csv ^
  --doc_path docs\architecture_ablation_analysis.md
