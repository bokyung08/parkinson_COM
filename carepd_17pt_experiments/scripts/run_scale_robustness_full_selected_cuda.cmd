@echo off
cd /d "%~dp0.."
powershell -ExecutionPolicy Bypass -File scripts\run_scale_robustness_full_selected_cuda.ps1
