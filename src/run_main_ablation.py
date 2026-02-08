import argparse
import json
import os
from datetime import datetime

from src.train_model import train_pose_model
from src.evaluate_model import evaluate_and_plot


ABLA_MODES = ["A", "B", "C", "D"]  # A: 좌표만(3ch), B: 좌표+속도(6ch), C: 좌표+속도+amp/var(8ch), D: 풀 9채널(좌표+속도+가속도)


def run_ablation(processed_dir, label_dir, epochs, batch_size):
    summary = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("results", "main_ablation", timestamp)
    os.makedirs(base_dir, exist_ok=True)

    for mode in ABLA_MODES:
        run_id = f"{timestamp}_abl{mode}"
        model_path = os.path.join(base_dir, f"abl_{mode}", "best_pose_model.weights.h5")
        plots_dir = os.path.join(base_dir, f"abl_{mode}", "plots")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        print(f"[INFO] Ablation {mode}: starting training...")
        model, history, (X_val, y_reg_val, ids_val) = train_pose_model(
            processed_dir,
            model_path,
            label_dir=label_dir,
            ablation=mode,
            epochs=epochs,
            batch_size=batch_size,
        )

        # Evaluation + timing
        evaluate_and_plot(model_path, X_val, y_reg_val, ids_val, plots_dir)

        # Load metrics
        metrics_path = os.path.join(plots_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            m = {k: float(v) for k, v in m.items()}
            m.update({
                "ablation": mode,
            })
            summary.append(m)

    # Save summary CSV/JSON
    if summary:
        import pandas as pd

        df = pd.DataFrame(summary)
        df.to_csv(os.path.join(base_dir, "ablation_summary.csv"), index=False)
        with open(os.path.join(base_dir, "ablation_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Ablation summary saved to {base_dir}")
    else:
        print("[WARN] No metrics collected; check runs.")


def main():
    parser = argparse.ArgumentParser(description="Ablation study for main TF model (feature channel slicing)")
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # train_pose_model already uses ReduceLROnPlateau/ModelCheckpoint internally
    run_ablation(args.processed_dir, args.label_dir, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
