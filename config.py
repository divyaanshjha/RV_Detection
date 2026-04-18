"""
config.py — All hyperparameters and paths in one place.
Change things here, not scattered across files.
"""

import os

# ──────────────────────────────────────────────
# PATHS  (edit these to match your folder layout)
# ──────────────────────────────────────────────
DATA = {
    # Parent folder that contains two sub-folders:
    #   RV_PATIENTS/
    #       C001/
    #           C001_EARLY_PHASE_OS_.png
    #           C001_MID_PHASE_OS_.png
    #           ...
    #   NORMAL_PATIENTS/
    #       N001/
    #           N001_LATE_PHASE_OS_.png
    #           ...
    "rv_dir":     "data/RV_PATIENTS",      # folder of patients WITH vasculitis
    "normal_dir": "data/NORMAL_PATIENTS",  # folder of patients WITHOUT vasculitis
    "output_dir": "outputs",               # where checkpoints + results are saved
}

# ──────────────────────────────────────────────
# PHASE DETECTION
# Keywords to look for in filenames (case-insensitive).
# Order matters for the embedding index.
# ──────────────────────────────────────────────
PHASES = {
    "EARLY":      0,
    "MID":        1,
    "LATE":       2,
    "PERIPH":     3,   # catches PERIPHERY, PERIPHERAL, PERIPH
    "UNKNOWN":    4,   # fallback if no keyword matched
}

# ──────────────────────────────────────────────
# IMAGE PREPROCESSING
# ──────────────────────────────────────────────
IMAGE = {
    "size":        224,          # ViT expects 224×224 (RETFound and vanilla ViT both)
    "mean":        (0.485, 0.456, 0.406),   # ImageNet mean (RETFound compatible)
    "std":         (0.229, 0.224, 0.225),   # ImageNet std
}

# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
MODEL = {
    "backbone":         "retfound",   # "retfound" | "vit_base" | "efficientnet_b3"
    "feature_dim":      1024,         # RETFound ViT-Large output dim
    "phase_embed_dim":  32,           # learned phase embedding size
    "combined_dim":     1056,         # feature_dim + phase_embed_dim
    "num_phases":       5,            # early/mid/late/periph/unknown
    "attn_hidden_dim":  128,          # attention network hidden size
    "classifier_hidden": 256,         # FC layer before output
    "dropout":          0.3,
    "num_classes":      1,            # binary output (sigmoid, not softmax)
}

# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
TRAIN = {
    "n_folds":          5,
    "seed":             42,

    # Phase A — backbone frozen
    "phaseA_epochs":    10,
    "phaseA_lr":        1e-3,

    # Phase B — top blocks unfrozen
    "phaseB_epochs":    30,
    "phaseB_lr":        1e-5,

    "weight_decay":     1e-4,
    "batch_size":       1,      # 1 patient = 1 bag (standard for MIL)

    # Stage 1 specific: lower threshold to maximise recall
    "decision_threshold": 0.3,

    # How many top transformer blocks to unfreeze in Phase B
    # RETFound ViT-Large has 24 blocks total
    "unfreeze_top_n_blocks": 4,
}

# ──────────────────────────────────────────────
# AUGMENTATION (training only)
# ──────────────────────────────────────────────
AUGMENT = {
    "hflip_prob":       0.5,
    "vflip_prob":       0.3,
    "rotation_deg":     15,
    "brightness":       0.2,
    "contrast":         0.2,
    "blur_prob":        0.2,
    "blur_kernel":      5,
}

# ──────────────────────────────────────────────
# DEVICE  (auto-detected, no change needed)
# ──────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"