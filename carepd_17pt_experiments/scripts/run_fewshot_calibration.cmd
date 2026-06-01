@echo off
cd /d "%~dp0.."
".venv_cuda\Scripts\python.exe" scripts\run_fewshot_calibration.py ^
  --manifest data\processed\manifest.csv ^
  --out_dir results\fewshot_calibration ^
  --doc_path docs\fewshot_calibration_analysis.md ^
  --calibration_subjects 1 3 5 10 ^
  --repeats 50
