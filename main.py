import argparse
import os
from datetime import datetime

import numpy as np

from src.data_preprocessing import process_video_for_pose
from src.evaluate_model import evaluate_and_plot
from src.train_model import train_pose_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract COM-normalized pose features, train the baseline model, and evaluate it."
    )
    parser.add_argument("--video_dir", default="HospitalData/VIDEO")
    parser.add_argument("--processed_dir", default="HospitalData/processed_pose_data")
    parser.add_argument("--label_dir", default="HospitalData/JSON")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=390)
    parser.add_argument("--ablation", choices=["A", "B", "C", "D"], default="D")
    parser.add_argument(
        "--video_suffix",
        default="_2",
        help="Only process videos whose stem ends with this suffix. Use an empty string to process all videos.",
    )
    return parser.parse_args()


def save_training_history(history, output_dir):
    if history is None:
        return

    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    hist = history.history

    plt.figure()
    plt.plot(hist.get("loss", []), label="train_loss")
    plt.plot(hist.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "history_loss.png"))
    plt.close()

    if "mae" in hist:
        plt.figure()
        plt.plot(hist.get("mae", []), label="train_mae")
        plt.plot(hist.get("val_mae", []), label="val_mae")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("Training/Validation MAE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "history_mae.png"))
        plt.close()


def extract_pose_files(video_dir, processed_dir, output_video_dir, video_suffix):
    for patient_dir in os.listdir(video_dir):
        full_patient_dir = os.path.join(video_dir, patient_dir)
        if not os.path.isdir(full_patient_dir):
            continue

        output_dir = os.path.join(processed_dir, patient_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"[STEP] Processing patient folder: {patient_dir}")

        for file_name in os.listdir(full_patient_dir):
            if not file_name.lower().endswith((".mp4", ".avi", ".mov")):
                continue

            stem = os.path.splitext(file_name)[0]
            if video_suffix and not stem.endswith(video_suffix):
                print(f"[SKIP] Video suffix does not match {video_suffix}: {file_name}")
                continue

            video_path = os.path.join(full_patient_dir, file_name)
            save_path = os.path.join(output_dir, f"{stem}_pose.npy")
            if os.path.exists(save_path):
                print(f"[SKIP] Pose file already exists: {save_path}")
                continue

            pose_data = process_video_for_pose(video_path, output_video_dir)
            if pose_data is None:
                continue

            np.save(save_path, pose_data)
            print(f"[DONE] Saved pose features: {save_path}")


def main():
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_dir = os.path.join("results", "video_outputs_pose_only", run_id)
    model_save_path = os.path.join("results", "models", run_id, "best_pose_model.weights.h5")
    plots_output_dir = os.path.join("results", "plots", run_id)

    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(plots_output_dir, exist_ok=True)

    extract_pose_files(args.video_dir, args.processed_dir, output_video_dir, args.video_suffix)

    model, history, validation_data = train_pose_model(
        args.processed_dir,
        model_save_path,
        label_dir=args.label_dir,
        ablation=args.ablation,
        max_len=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    del model

    save_training_history(history, plots_output_dir)
    X_val, y_reg_val, ids_val = validation_data
    evaluate_and_plot(model_save_path, X_val, y_reg_val, ids_val, plots_output_dir)


if __name__ == "__main__":
    main()
