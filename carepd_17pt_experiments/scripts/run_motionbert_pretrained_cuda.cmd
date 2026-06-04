@echo off
cd /d "%~dp0.."
".venv_cuda\Scripts\python.exe" scripts\run_loso_experiments.py ^
  --manifest data\processed\manifest.csv ^
  --out_dir results\groupkfold_h36m17_motionbert_pretrained_cuda ^
  --models motionbert_pretrained ^
  --split_strategy groupkfold ^
  --n_splits 5 ^
  --ablation D ^
  --target item10 ^
  --max_len 243 ^
  --epochs 80 ^
  --batch_size 4 ^
  --device cuda
