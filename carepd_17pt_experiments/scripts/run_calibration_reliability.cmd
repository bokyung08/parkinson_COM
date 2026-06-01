@echo off
setlocal
cd /d "%~dp0\.."

if exist ".venv_cuda\Scripts\python.exe" (
  set PYTHON_EXE=.venv_cuda\Scripts\python.exe
) else (
  set PYTHON_EXE=python
)

"%PYTHON_EXE%" scripts\analyze_calibration_reliability.py ^
  --results_root results ^
  --out_dir results\calibration_reliability ^
  --fig_dir docs\reviewer_figures ^
  --doc_path docs\calibration_reliability_analysis.md ^
  --n_bins 6
