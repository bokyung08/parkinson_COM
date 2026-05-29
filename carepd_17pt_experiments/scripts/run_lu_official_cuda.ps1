$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv_cuda\Scripts\python.exe"

& $python scripts\run_loso_experiments.py `
  --manifest data\processed\manifest.csv `
  --out_dir results\groupkfold_h36m17_lu_official_cuda `
  --models lu_ofddnet_official `
  --split_strategy groupkfold `
  --n_splits 5 `
  --ablation D `
  --target item10 `
  --epochs 80 `
  --batch_size 8 `
  --device cuda

exit $LASTEXITCODE
