@echo off
cd /d "%~dp0.."
call scripts\run_dataset_wise_breakdown.cmd
call scripts\run_score_balanced_transfer.cmd
call scripts\run_fewshot_calibration.cmd
call scripts\run_cross_dataset_model_comparison_cuda.cmd
call scripts\run_carepd_lodo_cuda.cmd
