@echo off
cd /d "%~dp0.."
".venv_cuda\Scripts\python.exe" scripts\benchmark_latency.py ^
  --manifest data\processed\manifest.csv ^
  --out_dir results\latency_benchmark ^
  --doc_path docs\latency_benchmark.md ^
  --models ours temporal_cnn stgcn lu_ofddnet_official motionbert_lite_pretrained motionagformer_xs_pretrained ^
  --ablation D ^
  --max_len 390 ^
  --batch_size 32 ^
  --warmup 20 ^
  --iters 100 ^
  --device cuda
