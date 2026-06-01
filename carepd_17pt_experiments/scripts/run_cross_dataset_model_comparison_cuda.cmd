@echo off
cd /d "%~dp0.."
".venv_cuda\Scripts\python.exe" scripts\run_cross_dataset_model_comparison.py ^
  --manifest data\processed\manifest.csv ^
  --out_dir results\cross_dataset_model_comparison ^
  --doc_path docs\cross_dataset_model_comparison.md ^
  --models ours stgcn lu_ofddnet_official ^
  --protocols cnuh_to_carepd carepd_to_cnuh ^
  --ablation D ^
  --epochs 80 ^
  --batch_size 8 ^
  --eval_batch_size 64 ^
  --device cuda
