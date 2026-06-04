@echo off
cd /d "%~dp0.."
".venv_cuda\Scripts\python.exe" scripts\analyze_selective_prediction.py ^
  --root . ^
  --out_dir results\selective_prediction ^
  --doc_path docs\selective_prediction_analysis.md
