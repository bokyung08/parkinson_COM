$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

.\.venv_cuda\Scripts\python.exe scripts\run_cross_dataset_validation.py `
  --manifest data\processed\manifest.csv `
  --out_dir results\cross_dataset_validation `
  --protocols cnuh_to_carepd carepd_to_cnuh `
  --combined_summary results\groupkfold_h36m17_ours_lu_official_cuda\summary.csv `
  --ablation D `
  --epochs 80 `
  --batch_size 8 `
  --eval_batch_size 64 `
  --device cuda
