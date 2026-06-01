$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv_cuda\Scripts\python.exe"

& $python scripts\run_com_robustness.py `
  --manifest data\processed\manifest.csv `
  --checkpoint_dir results\groupkfold_h36m17_ours_d_checkpointed_cuda\checkpoints `
  --out_dir docs\com_robustness `
  --n_splits 5 `
  --batch_size 32 `
  --device cuda

exit $LASTEXITCODE
