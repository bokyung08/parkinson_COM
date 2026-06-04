from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from gait17.data import load_manifest, materialize_arrays
from gait17.models import make_model
from gait17.training import LU_CLASSIFIERS, parameter_count


MODEL_LABELS = {
    "ours": ("Proposed", "Ours V1"),
    "stgcn": ("SOTA", "ST-GCN"),
    "lu_ofddnet_official": ("SOTA", "Lu official"),
    "motionbert": ("SOTA", "MotionBERT-style"),
    "motionagformer": ("SOTA", "MotionAGFormer-style"),
    "motionbert_pretrained": ("SOTA", "MotionBERT pretrained"),
    "motionbert_lite_pretrained": ("SOTA", "MotionBERT-Lite"),
    "motionagformer_xs_pretrained": ("SOTA", "MotionAGFormer-XS pretrained"),
    "temporal_cnn": ("Deep Learning", "Temporal CNN"),
}


def input_kind(model_name: str) -> str:
    return (
        "coords"
        if model_name
        in {
            "stgcn",
            "motionbert",
            "motionagformer",
            "motionbert_pretrained",
            "motionbert_lite_pretrained",
            "motionagformer_xs_pretrained",
            *LU_CLASSIFIERS,
        }
        else "hybrid"
    )


def benchmark_model(model_name: str, x: np.ndarray, args) -> dict:
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = make_model(model_name, int(x.shape[-1])).to(device)
    model.eval()
    tensor = torch.from_numpy(x[: args.batch_size]).float().to(device)
    if tensor.shape[0] < args.batch_size:
        repeat = int(np.ceil(args.batch_size / max(tensor.shape[0], 1)))
        tensor = tensor.repeat((repeat, 1, 1, 1))[: args.batch_size]
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iters):
            _ = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    category, display = MODEL_LABELS[model_name]
    total_samples = args.batch_size * args.iters
    return {
        "category": category,
        "model": model_name,
        "display_model": display,
        "params": parameter_count(model),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "iters": int(args.iters),
        "latency_ms_per_batch": float(elapsed * 1000.0 / args.iters),
        "latency_ms_per_sample": float(elapsed * 1000.0 / total_samples),
    }


def markdown_table(df: pd.DataFrame) -> list[str]:
    cols = ["category", "display_model", "params", "device", "batch_size", "latency_ms_per_sample", "latency_ms_per_batch"]
    labels = {
        "category": "Category",
        "display_model": "Model",
        "params": "Params",
        "device": "Device",
        "batch_size": "Batch",
        "latency_ms_per_sample": "ms/sample",
        "latency_ms_per_batch": "ms/batch",
    }
    lines = [
        "| " + " | ".join(labels[c] for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in df[cols].to_dict(orient="records"):
        lines.append(
            "| "
            + " | ".join(
                f"{float(row[c]):.3f}" if c.startswith("latency") else f"{int(row[c])}" if c in {"params", "batch_size"} else str(row[c])
                for c in cols
            )
            + " |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark forward latency for skeleton encoder models.")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--out_dir", default="results/latency_benchmark")
    parser.add_argument("--doc_path", default="docs/latency_benchmark.md")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "ours",
            "temporal_cnn",
            "stgcn",
            "lu_ofddnet_official",
            "motionbert_pretrained",
            "motionbert_lite_pretrained",
            "motionagformer_xs_pretrained",
        ],
    )
    parser.add_argument("--ablation", choices=["A", "B", "C", "D"], default="D")
    parser.add_argument("--max_len", type=int, default=390)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_manifest(manifest_path)
    sample_idx = np.arange(min(len(df), max(args.batch_size, 64)))
    arrays = {}
    for kind in {"coords", "hybrid"}:
        x, _, _ = materialize_arrays(manifest_path, sample_idx, args.max_len, args.ablation, kind)
        arrays[kind] = x

    rows = []
    for model_name in args.models:
        print(f"[INFO] Benchmarking {model_name}")
        rows.append(benchmark_model(model_name, arrays[input_kind(model_name)], args))
    result = pd.DataFrame(rows).sort_values("latency_ms_per_sample")
    result.to_csv(out_dir / "summary.csv", index=False)
    lines = [
        "# Latency Benchmark",
        "",
        "- Measurement: forward pass only",
        "- Weights: randomly initialized architecture instances",
        "- Purpose: architecture-level inference cost comparison under identical input length and batch size",
        f"- Batch size: `{args.batch_size}`",
        f"- Warmup iterations: `{args.warmup}`",
        f"- Timed iterations: `{args.iters}`",
        "",
        *markdown_table(result),
    ]
    Path(args.doc_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote {out_dir} and {args.doc_path}")


if __name__ == "__main__":
    main()
