@echo off
cd /d "%~dp0.."
".venv_cuda\Scripts\python.exe" scripts\run_score_balanced_transfer.py ^
  --out_dir results\score_balanced_transfer ^
  --doc_path docs\score_balanced_transfer_analysis.md
