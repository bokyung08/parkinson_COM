@echo off
cd /d "%~dp0.."
".venv_cuda\Scripts\python.exe" scripts\run_cross_dataset_model_comparison.py ^
  --manifest data\processed\manifest.csv ^
  --out_dir results\cross_dataset_model_comparison_v2 ^
  --doc_path docs\cross_dataset_model_comparison_v2.md ^
  --models motionbert_lite_pretrained motionagformer_xs_pretrained ^
  --protocols cnuh_to_carepd carepd_to_cnuh ^
  --ablation D ^
  --max_len 81 ^
  --epochs 80 ^
  --batch_size 8 ^
  --eval_batch_size 16 ^
  --device cuda
