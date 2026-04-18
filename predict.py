"""
predict.py — Run Stage 1 inference on new patient folders.

Usage:
    python predict.py --patient_dir path/to/patient_folder
    python predict.py --batch_dir path/to/folder_of_patients

Output:
    patient_id | probability | prediction | attention_weights_per_image
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from config import TRAIN, DEVICE, DATA
from dataset import build_patient_records, RetinalBagDataset
from model import Stage1Model
from torch.utils.data import DataLoader
from train import collate_bags


PHASE_NAMES = {0: "Early", 1: "Mid", 2: "Late", 3: "Periph", 4: "Unknown"}


@torch.no_grad()
def predict_patient(model, images, phases, threshold):
    model.eval()
    images = images.to(DEVICE)
    phases = phases.to(DEVICE)

    logit, alpha = model(images, phases)
    prob = torch.sigmoid(logit).item()
    pred = 1 if prob >= threshold else 0
    alpha_np = alpha.cpu().numpy()

    return prob, pred, alpha_np


def run_inference(model_path: str, patient_records: list, threshold: float = None):
    if threshold is None:
        threshold = TRAIN["decision_threshold"]

    # Load model
    model = Stage1Model().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded from: {model_path}\n")

    dataset = RetinalBagDataset(patient_records, training=False)

    results = []
    for i in range(len(dataset)):
        images, phases, label, meta = dataset[i]
        prob, pred, alpha = predict_patient(model, images, phases, threshold)

        print(f"Patient: {meta['patient_id']}")
        print(f"  Prediction:  {'RV PRESENT' if pred == 1 else 'NORMAL'}")
        print(f"  Probability: {prob:.3f}  (threshold: {threshold})")
        print(f"  Attention weights per image:")
        for j, (path, weight) in enumerate(zip(meta["paths"], alpha)):
            phase_idx = phases[j].item()
            fname = Path(path).name
            print(f"    [{PHASE_NAMES[phase_idx]:8s}] α={weight:.3f}  ← {fname}")
        if label.item() >= 0:
            print(f"  True label:  {int(label.item())} "
                  f"({'RV' if label.item() == 1 else 'Normal'})")
        print()

        results.append({
            "patient_id":  meta["patient_id"],
            "probability": prob,
            "prediction":  pred,
            "attention":   alpha.tolist(),
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 inference")
    parser.add_argument("--model",    default="outputs/fold_1/best_model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--rv_dir",   default=DATA["rv_dir"])
    parser.add_argument("--norm_dir", default=DATA["normal_dir"])
    parser.add_argument("--threshold", type=float, default=TRAIN["decision_threshold"])
    args = parser.parse_args()

    records = build_patient_records(args.rv_dir, args.norm_dir)
    run_inference(args.model, records, threshold=args.threshold)
