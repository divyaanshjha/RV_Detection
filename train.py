"""
train.py — Training loop, evaluation, and 5-fold cross-validation.

Run:
    python train.py

What this does:
  1. Loads all patient records from your two folders
  2. Splits into 5 stratified folds (patients never split across folds)
  3. For each fold:
       Phase A: frozen backbone, train attention + classifier
       Phase B: unfreeze top blocks, fine-tune end-to-end
  4. Evaluates on the held-out fold
  5. Saves per-fold results + best model checkpoint
  6. Prints final averaged metrics across all folds
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, recall_score
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import DATA, TRAIN, DEVICE, MODEL
from dataset import build_patient_records, RetinalBagDataset, phase_summary
from model import Stage1Model, freeze_backbone, unfreeze_top_blocks, compute_class_weights, get_loss_fn


# ──────────────────────────────────────────────────────────────
# COLLATE FUNCTION
# (DataLoader needs this because bags have variable N)
# ──────────────────────────────────────────────────────────────

def collate_bags(batch):
    """
    MIL bags can't be stacked into a single tensor because each patient
    has a different number of images. We return a list of bags instead.
    batch: list of (images, phases, label, meta) tuples
    """
    images = [item[0] for item in batch]   # list of (N_i, 3, H, W)
    phases = [item[1] for item in batch]   # list of (N_i,)
    labels = torch.stack([item[2] for item in batch])  # (B,) — stackable
    metas  = [item[3] for item in batch]
    return images, phases, labels, metas


# ──────────────────────────────────────────────────────────────
# SINGLE EPOCH — TRAINING
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for images_list, phases_list, labels, _ in loader:
        labels = labels.to(DEVICE)

        batch_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)

        for images, phases, label in zip(images_list, phases_list, labels):
            images = images.to(DEVICE)
            phases = phases.to(DEVICE)
            label  = label.unsqueeze(0)   # (1,) for BCEWithLogitsLoss

            logit, _ = model(images, phases)
            logit = logit.unsqueeze(0)    # (1,)
            loss  = loss_fn(logit, label)

            # Accumulate loss across patients in the batch
            batch_loss = batch_loss + loss

            prob = torch.sigmoid(logit).item()
            all_probs.append(prob)
            all_labels.append(int(label.item()))

        # Average loss over batch
        batch_loss = batch_loss / len(images_list)
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss, all_labels, all_probs


# ──────────────────────────────────────────────────────────────
# SINGLE EPOCH — EVALUATION
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, loss_fn, threshold=None):
    """
    Run evaluation on a loader.
    Returns: dict of metrics + raw probs/labels for further analysis.
    """
    if threshold is None:
        threshold = TRAIN["decision_threshold"]

    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for images_list, phases_list, labels, _ in loader:
        labels = labels.to(DEVICE)

        for images, phases, label in zip(images_list, phases_list, labels):
            images = images.to(DEVICE)
            phases = phases.to(DEVICE)
            label  = label.unsqueeze(0)

            logit, _ = model(images, phases)
            logit = logit.unsqueeze(0)
            loss  = loss_fn(logit, label)

            total_loss += loss.item()
            prob = torch.sigmoid(logit).item()
            all_probs.append(prob)
            all_labels.append(int(label.item()))

    # Convert to predictions using the chosen threshold
    preds = [1 if p >= threshold else 0 for p in all_probs]

    avg_loss = total_loss / max(len(all_probs), 1)

    # Compute metrics
    metrics = {
        "loss":        avg_loss,
        "sensitivity": recall_score(all_labels, preds, pos_label=1, zero_division=0),
        "specificity": recall_score(all_labels, preds, pos_label=0, zero_division=0),
        "f1":          f1_score(all_labels, preds, zero_division=0),
        "auc_roc":     roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
    }

    return metrics, all_labels, all_probs, preds


# ──────────────────────────────────────────────────────────────
# TRAIN ONE FOLD
# ──────────────────────────────────────────────────────────────

def train_fold(fold_idx, train_records, val_records, pos_weight):
    """
    Train and evaluate a single cross-validation fold.
    Phase A: frozen backbone.
    Phase B: top blocks unfrozen.
    Returns: best validation metrics for this fold.
    """
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx + 1} / {TRAIN['n_folds']}")
    print(f"  Train: {len(train_records)} patients | Val: {len(val_records)} patients")
    print(f"{'='*60}")

    # Datasets
    train_ds = RetinalBagDataset(train_records, training=True)
    val_ds   = RetinalBagDataset(val_records,   training=False)

    train_loader = DataLoader(train_ds, batch_size=TRAIN["batch_size"],
                              shuffle=True,  collate_fn=collate_bags)
    val_loader   = DataLoader(val_ds,   batch_size=TRAIN["batch_size"],
                              shuffle=False, collate_fn=collate_bags)

    # Model + loss
    model   = Stage1Model().to(DEVICE)
    loss_fn = get_loss_fn(pos_weight[1].unsqueeze(0).to(DEVICE))  # upweight RV class

    # Output directory for this fold
    fold_dir = Path(DATA["output_dir"]) / f"fold_{fold_idx + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    best_sensitivity = 0.0
    best_metrics     = {}
    best_epoch       = 0

    # ── PHASE A: frozen backbone ──────────────────────────────
    print(f"\n--- Phase A: Frozen backbone ({TRAIN['phaseA_epochs']} epochs) ---")
    freeze_backbone(model.backbone)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=TRAIN["phaseA_lr"],
        weight_decay=TRAIN["weight_decay"],
    )

    for epoch in range(TRAIN["phaseA_epochs"]):
        t0 = time.time()
        train_loss, _, _ = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_metrics, _, _, _ = evaluate(model, val_loader, loss_fn)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:02d}/{TRAIN['phaseA_epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Sensitivity: {val_metrics['sensitivity']:.3f} | "
              f"Specificity: {val_metrics['specificity']:.3f} | "
              f"AUC: {val_metrics['auc_roc']:.3f} | "
              f"F1: {val_metrics['f1']:.3f} | "
              f"{elapsed:.1f}s")

        if val_metrics["sensitivity"] >= best_sensitivity:
            best_sensitivity = val_metrics["sensitivity"]
            best_metrics     = val_metrics
            best_epoch       = epoch + 1
            torch.save(model.state_dict(), fold_dir / "best_model.pth")

    # ── PHASE B: unfreeze top blocks ─────────────────────────
    print(f"\n--- Phase B: Fine-tuning top {TRAIN['unfreeze_top_n_blocks']} blocks "
          f"({TRAIN['phaseB_epochs']} epochs) ---")
    unfreeze_top_blocks(model.backbone, TRAIN["unfreeze_top_n_blocks"])

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=TRAIN["phaseB_lr"],
        weight_decay=TRAIN["weight_decay"],
    )

    for epoch in range(TRAIN["phaseB_epochs"]):
        t0 = time.time()
        train_loss, _, _ = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_metrics, _, _, _ = evaluate(model, val_loader, loss_fn)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:02d}/{TRAIN['phaseB_epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Sensitivity: {val_metrics['sensitivity']:.3f} | "
              f"Specificity: {val_metrics['specificity']:.3f} | "
              f"AUC: {val_metrics['auc_roc']:.3f} | "
              f"F1: {val_metrics['f1']:.3f} | "
              f"{elapsed:.1f}s")

        if val_metrics["sensitivity"] >= best_sensitivity:
            best_sensitivity = val_metrics["sensitivity"]
            best_metrics     = val_metrics
            best_epoch       = epoch + TRAIN["phaseA_epochs"] + 1
            torch.save(model.state_dict(), fold_dir / "best_model.pth")

    print(f"\n  Best sensitivity: {best_sensitivity:.3f} at epoch {best_epoch}")
    print(f"  Best metrics: {best_metrics}")

    # Save fold results
    with open(fold_dir / "results.json", "w") as f:
        json.dump({"best_epoch": best_epoch, "metrics": best_metrics}, f, indent=2)

    return best_metrics


# ──────────────────────────────────────────────────────────────
# MAIN — 5-FOLD CROSS-VALIDATION
# ──────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(TRAIN["seed"])
    np.random.seed(TRAIN["seed"])

    print(f"\nDevice: {DEVICE}")
    print(f"Backbone: {MODEL['backbone']}")

    # ── Build dataset ─────────────────────────────────────────
    records = build_patient_records(DATA["rv_dir"], DATA["normal_dir"])
    phase_summary(records)

    labels = [r["label"] for r in records]

    # ── Compute class weights ─────────────────────────────────
    pos_weight = compute_class_weights(labels)

    # ── 5-fold stratified split ───────────────────────────────
    skf = StratifiedKFold(
        n_splits=TRAIN["n_folds"],
        shuffle=True,
        random_state=TRAIN["seed"],
    )

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(records, labels)):
        train_records = [records[i] for i in train_idx]
        val_records   = [records[i] for i in val_idx]

        metrics = train_fold(fold_idx, train_records, val_records, pos_weight)
        fold_metrics.append(metrics)

    # ── Aggregate across folds ────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL RESULTS — AVERAGED ACROSS ALL FOLDS")
    print(f"{'='*60}")

    metric_keys = ["sensitivity", "specificity", "f1", "auc_roc"]
    summary = {}
    for key in metric_keys:
        vals = [m[key] for m in fold_metrics]
        mean = np.mean(vals)
        std  = np.std(vals)
        summary[key] = {"mean": mean, "std": std, "per_fold": vals}
        print(f"  {key:15s}: {mean:.3f} ± {std:.3f}  |  per fold: {[f'{v:.3f}' for v in vals]}")

    # Save summary
    out_dir = Path(DATA["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {DATA['output_dir']}/")
    print("Done.")


if __name__ == "__main__":
    main()