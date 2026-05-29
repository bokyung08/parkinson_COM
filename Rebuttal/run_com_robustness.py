"""Evaluate COM-normalization robustness under coordinate perturbations.

The script loads saved model weights and evaluates score deviations after
artificial scale and x-translation perturbations. It supports one or two
conditions, typically Config D with COM normalization and a raw-coordinate
counterpart trained without COM normalization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Rebuttal.unified_data_utils import (
    apply_ablation,
    build_hybrid_node_features,
    ensure_pose_tensor,
    load_labels,
    pad_features,
    patient_id_from_pose_file,
)


@dataclass(frozen=True)
class Condition:
    name: str
    weights: Path
    apply_com: bool


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def load_val_samples(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def com_normalize(coords: np.ndarray) -> np.ndarray:
    if coords.shape[1] <= 24:
        raise ValueError("COM normalization expects MediaPipe joints including hips 23 and 24.")
    com = (coords[:, 23:24, :] + coords[:, 24:25, :]) / 2.0
    return coords - com


def perturb_coords(coords: np.ndarray, perturb_type: str, level: float) -> np.ndarray:
    out = coords.copy()
    if perturb_type == "scale":
        return out * level
    if perturb_type == "translation_x":
        out[..., 0] = out[..., 0] + level
        return out
    raise ValueError(f"Unsupported perturbation type: {perturb_type}")


def load_coordinate_samples(
    processed_dir: Path,
    label_dir: Path,
    target: str,
    max_len: int,
    val_samples: set[str] | None,
) -> tuple[list[str], np.ndarray, list[np.ndarray]]:
    labels = load_labels(label_dir, target=target)
    ids: list[str] = []
    y_values: list[float] = []
    coords_list: list[np.ndarray] = []

    for pose_path in sorted(processed_dir.rglob("*_2_pose.npy")):
        sample_id = pose_path.name.replace("_pose.npy", "")
        if val_samples is not None and sample_id not in val_samples:
            continue
        patient_id = patient_id_from_pose_file(pose_path)
        if patient_id not in labels:
            continue
        raw = np.load(pose_path)
        coords = ensure_pose_tensor(raw).astype(np.float32)
        if coords.shape[0] > max_len:
            coords = coords[-max_len:]
        ids.append(sample_id)
        y_values.append(float(labels[patient_id]))
        coords_list.append(coords)

    if not coords_list:
        raise ValueError("No samples found for robustness evaluation.")
    return ids, np.asarray(y_values, dtype=np.float32), coords_list


def build_feature_batch(
    coords_list: list[np.ndarray],
    perturb_type: str,
    level: float,
    apply_com: bool,
    ablation: str,
    max_len: int,
) -> np.ndarray:
    x_list = []
    for coords in coords_list:
        perturbed = perturb_coords(coords, perturb_type, level)
        if apply_com:
            perturbed = com_normalize(perturbed)
        features = build_hybrid_node_features(perturbed)
        features = apply_ablation(features, ablation)
        x_list.append(pad_features(features, max_len))
    return np.asarray(x_list, dtype=np.float32)


def build_fusion_model(input_shape: tuple[int, int, int], learning_rate: float):
    from Rebuttal.run_unified_fusion_tf import build_model

    return build_model(input_shape, input_shape[1], learning_rate)


def load_predictor(model_key: str, weights: Path, input_shape: tuple[int, int, int], learning_rate: float, device_name: str):
    if model_key == "fusion_tf":
        model = build_fusion_model(input_shape, learning_rate)
        model.load_weights(str(weights))

        def predict(x: np.ndarray) -> np.ndarray:
            return model.predict(x, verbose=0).reshape(-1)

        params = int(model.count_params())
        return predict, params

    if model_key == "hybrid_torch":
        import torch
        from src.hybrid_gcn import HybridCOMGCNv2

        device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
        model = HybridCOMGCNv2(in_channels=input_shape[-1]).to(device)
        model.load_state_dict(torch.load(weights, map_location=device))
        model.eval()

        def predict(x: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                xb = torch.from_numpy(x).float().to(device)
                return model(xb).cpu().numpy().reshape(-1)

        params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        return predict, params

    raise ValueError(f"Unsupported --model: {model_key}")


def evaluate_condition(
    condition: Condition,
    args: argparse.Namespace,
    ids: list[str],
    y_true: np.ndarray,
    coords_list: list[np.ndarray],
    perturb_type: str,
    levels: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_level = 1.0 if perturb_type == "scale" else 0.0
    x_ref = build_feature_batch(
        coords_list,
        perturb_type,
        reference_level,
        condition.apply_com,
        args.ablation,
        args.max_len,
    )
    predict, params = load_predictor(args.model, condition.weights, x_ref.shape[1:], args.learning_rate, args.device)
    start = time.perf_counter()
    pred_ref = predict(x_ref)
    infer_ms = (time.perf_counter() - start) * 1000.0 / max(len(x_ref), 1)

    summary_rows = []
    prediction_rows = []
    for level in levels:
        x = build_feature_batch(coords_list, perturb_type, level, condition.apply_com, args.ablation, args.max_len)
        pred = predict(x)
        deviations = np.abs(pred - pred_ref)
        residual = pred - y_true
        summary_rows.append(
            {
                "condition": condition.name,
                "model": args.model,
                "params": params,
                "inference_ms_per_sample": infer_ms,
                "perturbation": perturb_type,
                "level": level,
                "n": int(len(pred)),
                "mean_score_deviation": float(np.mean(deviations)),
                "std_score_deviation": float(np.std(deviations)),
                "mae_vs_true": float(np.mean(np.abs(residual))),
                "rmse_vs_true": float(np.sqrt(np.mean(residual ** 2))),
            }
        )
        for sample_id, true_value, ref_value, pred_value, deviation in zip(ids, y_true, pred_ref, pred, deviations):
            prediction_rows.append(
                {
                    "condition": condition.name,
                    "perturbation": perturb_type,
                    "level": level,
                    "sample_id": sample_id,
                    "y_true": float(true_value),
                    "reference_pred": float(ref_value),
                    "perturbed_pred": float(pred_value),
                    "score_deviation": float(deviation),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows)


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    for perturbation, group in summary.groupby("perturbation"):
        plt.figure()
        for condition, cgroup in group.groupby("condition"):
            cgroup = cgroup.sort_values("level")
            plt.plot(cgroup["level"], cgroup["mean_score_deviation"], marker="o", label=condition)
        plt.xlabel("Scale" if perturbation == "scale" else "Translation dx")
        plt.ylabel("|score(perturbed) - score(original)|")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{perturbation}_score_deviation.png", dpi=200)
        plt.close()


def write_markdown(summary: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# COM Normalization Robustness",
        "",
        "## Protocol",
        "",
        f"- Model: `{args.model}`",
        f"- Ablation: `{args.ablation}`",
        f"- Max length: `{args.max_len}`",
        f"- Target: `{args.target}`",
        "",
        "If the input pose arrays are already COM-normalized, the raw-coordinate condition should be interpreted as an approximate translation/scale sensitivity check unless separately trained raw-coordinate weights are supplied.",
        "",
        "## Table 8. Perturbation Robustness",
        "",
        "| Condition | Perturbation | Level | N | Mean abs delta score | SD | MAE vs true | RMSE vs true |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.sort_values(["perturbation", "condition", "level"]).iterrows():
        lines.append(
            "| {condition} | {perturbation} | {level:.3f} | {n} | {mean:.3f} | {std:.3f} | {mae:.3f} | {rmse:.3f} |".format(
                condition=row["condition"],
                perturbation=row["perturbation"],
                level=float(row["level"]),
                n=int(row["n"]),
                mean=float(row["mean_score_deviation"]),
                std=float(row["std_score_deviation"]),
                mae=float(row["mae_vs_true"]),
                rmse=float(row["rmse_vs_true"]),
            )
        )
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Table 8 COM perturbation robustness evaluation.")
    parser.add_argument("--processed_dir", default="HospitalData/processed_pose_data")
    parser.add_argument("--label_dir", default="HospitalData/JSON")
    parser.add_argument("--val_samples", default=None, help="Optional val_samples.txt to reproduce a saved split.")
    parser.add_argument("--out_dir", default="Rebuttal/results/com_robustness")
    parser.add_argument("--model", choices=["fusion_tf", "hybrid_torch"], default="fusion_tf")
    parser.add_argument("--com_weights", default=None, help="Weights for COM-normalized Config D model.")
    parser.add_argument("--raw_weights", default=None, help="Weights for raw-coordinate Config D model.")
    parser.add_argument("--target", choices=["item10", "gait"], default="item10")
    parser.add_argument("--ablation", choices=["A", "B", "C", "D"], default="D")
    parser.add_argument("--max_len", type=int, default=390)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_values", default="0.7,0.85,1.0,1.15,1.3")
    parser.add_argument("--translation_values", default="-0.2,-0.1,0,0.1,0.2")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    conditions = []
    if args.com_weights:
        conditions.append(Condition("ConfigD_COM", Path(args.com_weights), True))
    if args.raw_weights:
        conditions.append(Condition("ConfigD_noCOM", Path(args.raw_weights), False))
    if not conditions:
        raise SystemExit("Provide at least one of --com_weights or --raw_weights.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_samples = load_val_samples(Path(args.val_samples)) if args.val_samples else None
    ids, y_true, coords_list = load_coordinate_samples(
        Path(args.processed_dir),
        Path(args.label_dir),
        args.target,
        args.max_len,
        val_samples,
    )

    summary_frames = []
    prediction_frames = []
    for condition in conditions:
        for perturbation, levels in (
            ("scale", parse_float_list(args.scale_values)),
            ("translation_x", parse_float_list(args.translation_values)),
        ):
            summary, predictions = evaluate_condition(condition, args, ids, y_true, coords_list, perturbation, levels)
            summary_frames.append(summary)
            prediction_frames.append(predictions)

    summary_df = pd.concat(summary_frames, ignore_index=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True)
    summary_df.to_csv(out_dir / "table8_com_robustness_summary.csv", index=False)
    prediction_df.to_csv(out_dir / "table8_com_robustness_predictions.tsv", sep="\t", index=False)
    plot_summary(summary_df, out_dir)
    write_markdown(summary_df, out_dir, args)
    config = vars(args).copy()
    config["n_samples"] = len(ids)
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[INFO] Saved COM robustness outputs to {out_dir}")


if __name__ == "__main__":
    main()
