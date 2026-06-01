@echo off
cd /d "%~dp0.."
".venv_cuda\Scripts\python.exe" scripts\run_carepd_lodo.py ^
  --manifest data\processed\manifest.csv ^
  --out_dir results\carepd_leave_one_dataset_out ^
  --doc_path docs\carepd_lodo_analysis.md ^
  --models ours ^
  --ablation D ^
  --epochs 80 ^
  --batch_size 8 ^
  --eval_batch_size 64 ^
  --device cuda
