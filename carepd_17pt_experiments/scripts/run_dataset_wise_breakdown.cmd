@echo off
cd /d "%~dp0.."
".venv_cuda\Scripts\python.exe" scripts\analyze_dataset_wise_breakdown.py ^
  --results_root results ^
  --out_dir results\dataset_wise_breakdown ^
  --doc_path docs\dataset_wise_breakdown_analysis.md
