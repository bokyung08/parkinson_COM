$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv_cuda\Scripts\python.exe"

foreach ($ablation in @("A", "B", "C", "D")) {
  $summary = "results\groupkfold_h36m17_ours_ablation_${ablation}_cuda\summary.csv"
  if (Test-Path $summary) {
    Write-Host "[INFO] Skipping Ours V1 ablation $ablation; summary already exists."
    continue
  }

  Write-Host "[INFO] Running Ours V1 ablation $ablation"
  & $python scripts\run_loso_experiments.py `
    --manifest data\processed\manifest.csv `
    --out_dir "results\groupkfold_h36m17_ours_ablation_${ablation}_cuda" `
    --models ours `
    --split_strategy groupkfold `
    --n_splits 5 `
    --ablation $ablation `
    --target item10 `
    --epochs 80 `
    --batch_size 8 `
    --device cuda

  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}
