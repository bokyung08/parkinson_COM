@echo off
cd /d "%~dp0.."
".venv_cuda\Scripts\python.exe" scripts\run_scale_robustness_screen.py
