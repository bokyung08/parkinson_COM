$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = ".venv_cuda\Scripts\python.exe"
$variants = @(
  @{ Name = "baseline_none"; ScaleNorm = "none"; AugMin = "1.00"; AugMax = "1.00" },
  @{ Name = "median_bone"; ScaleNorm = "median_bone"; AugMin = "1.00"; AugMax = "1.00" },
  @{ Name = "torso"; ScaleNorm = "torso"; AugMin = "1.00"; AugMax = "1.00" },
  @{ Name = "hip_width"; ScaleNorm = "hip_width"; AugMin = "1.00"; AugMax = "1.00" },
  @{ Name = "scale_aug_moderate"; ScaleNorm = "none"; AugMin = "0.85"; AugMax = "1.15" },
  @{ Name = "scale_aug_wide"; ScaleNorm = "none"; AugMin = "0.70"; AugMax = "1.30" },
  @{ Name = "median_bone_aug_moderate"; ScaleNorm = "median_bone"; AugMin = "0.85"; AugMax = "1.15" }
)

foreach ($variant in $variants) {
  $name = $variant.Name
  $resultDir = "results\screen_scale_robustness_$name"
  $robustDir = "docs\scale_robustness_screen\$name"

  if (Test-Path "$resultDir\summary.csv") {
    Write-Host "[INFO] Skipping training for $name; summary.csv exists."
  } else {
    Write-Host "[INFO] Training screen candidate $name"
    & $python scripts\run_loso_experiments.py `
      --manifest data\processed\manifest.csv `
      --out_dir $resultDir `
      --models ours `
      --split_strategy groupkfold `
      --n_splits 5 `
      --fold_limit 2 `
      --ablation D `
      --target item10 `
      --epochs 30 `
      --batch_size 8 `
      --device cuda `
      --scale_normalization $variant.ScaleNorm `
      --scale_aug_min $variant.AugMin `
      --scale_aug_max $variant.AugMax `
      --save_checkpoints
  }

  if (Test-Path "$robustDir\summary.csv") {
    Write-Host "[INFO] Skipping robustness for $name; summary.csv exists."
  } else {
    Write-Host "[INFO] Evaluating robustness for $name"
    & $python scripts\run_com_robustness.py `
      --manifest data\processed\manifest.csv `
      --checkpoint_dir "$resultDir\checkpoints" `
      --out_dir $robustDir `
      --n_splits 5 `
      --fold_limit 2 `
      --batch_size 32 `
      --device cuda `
      --scale_values 0.70 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.30 `
      --translation_values -0.20 -0.10 0.00 0.10 0.20 `
      --combined_values "0.70,-0.20" "1.30,0.20" "0.85,0.10" "1.15,-0.10"
  }
}

& $python scripts\summarize_scale_robustness_candidates.py `
  --root docs\scale_robustness_screen `
  --out docs\scale_robustness_candidate_summary.md

& $python scripts\analyze_scale_feature_invariance.py `
  --manifest data\processed\manifest.csv `
  --out_dir docs\scale_feature_invariance `
  --sample_limit 1000
