@echo off
cd /d "%~dp0.."
".venv_cuda\Scripts\python.exe" scripts\run_loso_experiments.py ^
  --manifest data\processed\manifest.csv ^
  --out_dir results\groupkfold_h36m17_motionagformer_xs_pretrained_cuda ^
  --models motionagformer_xs_pretrained ^
  --split_strategy groupkfold ^
  --n_splits 5 ^
  --ablation D ^
  --target item10 ^
  --max_len 27 ^
  --epochs 80 ^
  --batch_size 8 ^
  --device cuda
