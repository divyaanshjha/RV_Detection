"""
dataset.py — Parses patient folders, tags images by phase, builds MIL bags.

A "bag" is one patient's complete set of images.
Each image becomes one instance in the bag.
The bag has a single binary label: 1 = RV present, 0 = normal.
"""

import os
import re
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from config import DATA, PHASES, IMAGE, AUGMENT


# ──────────────────────────────────────────────────────────────
# PHASE DETECTION
# ──────────────────────────────────────────────────────────────

def detect_phase(filename: str) -> int:
    """
    Infer the FA phase from the image filename.
    Returns an integer index defined in config.PHASES.

    Examples:
        "C167_EARLY_PHASE_OS_.png"  → 0  (EARLY)
        "C167_MID_PHASE_OS_.png"    → 1  (MID)
        "C167_LATE_PHASE_OS_.png"   → 2  (LATE)
        "C167_PERIPH_OS_.png"       → 3  (PERIPH)
        "C167_IMG_01.png"           → 4  (UNKNOWN)
    """
    name_upper = filename.upper()
    for keyword, idx in PHASES.items():
        if keyword == "UNKNOWN":
            continue
        if keyword in name_upper:
            return idx
    return PHASES["UNKNOWN"]


# ──────────────────────────────────────────────────────────────
# TRANSFORMS
# ──────────────────────────────────────────────────────────────

def get_transforms(training: bool) -> T.Compose:
    """
    Returns the image transform pipeline.
    Training: includes augmentation.
    Validation/test: only resize + normalize.
    """
    base = [
        T.Resize((IMAGE["size"], IMAGE["size"])),
        T.ToTensor(),
        T.Normalize(mean=IMAGE["mean"], std=IMAGE["std"]),
    ]

    if not training:
        return T.Compose(base)

    augment = [
        T.Resize((IMAGE["size"], IMAGE["size"])),
        T.RandomHorizontalFlip(p=AUGMENT["hflip_prob"]),
        T.RandomVerticalFlip(p=AUGMENT["vflip_prob"]),
        T.RandomRotation(degrees=AUGMENT["rotation_deg"]),
        T.ColorJitter(
            brightness=AUGMENT["brightness"],
            contrast=AUGMENT["contrast"],
        ),
        T.RandomApply(
            [T.GaussianBlur(kernel_size=AUGMENT["blur_kernel"])],
            p=AUGMENT["blur_prob"],
        ),
        T.ToTensor(),
        T.Normalize(mean=IMAGE["mean"], std=IMAGE["std"]),
    ]
    return T.Compose(augment)


# ──────────────────────────────────────────────────────────────
# PATIENT RECORD BUILDER
# ──────────────────────────────────────────────────────────────

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def build_patient_records(rv_dir: str, normal_dir: str) -> list[dict]:
    """
    Walk both directories, build one record per patient.

    Returns a list of dicts:
        {
            "patient_id": "C167",
            "label":      1,          # 1 = RV, 0 = Normal
            "images": [
                {"path": "/path/to/img.png", "phase": 2},
                ...
            ]
        }
    """
    records = []

    for label, folder in [(1, rv_dir), (0, normal_dir)]:
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(
                f"Folder not found: {folder}\n"
                f"Please update DATA['rv_dir'] and DATA['normal_dir'] in config.py"
            )

        # Each sub-folder = one patient
        patient_dirs = sorted([d for d in folder_path.iterdir() if d.is_dir()])

        if not patient_dirs:
            print(f"Warning: No sub-folders found in {folder}. "
                  f"Expected one folder per patient.")

        for patient_dir in patient_dirs:
            images = []

            # Use rglob to find images recursively — handles both flat and
            # nested folder structures inside each patient folder:
            #   PatientID/PatientID_EARLY.png          (flat — original)
            #   PatientID/subfolder/img.png            (nested — this fixes the issue)
            for f in sorted(patient_dir.rglob("*")):
                if f.is_dir():
                    continue
                if f.suffix.lower() not in VALID_EXTENSIONS:
                    continue
                phase_idx = detect_phase(f.name)
                images.append({
                    "path":  str(f),
                    "phase": phase_idx,
                })

            if not images:
                print(f"Warning: No images found for patient {patient_dir.name}, skipping.")
                continue

            records.append({
                "patient_id": patient_dir.name,
                "label":      label,
                "images":     images,
            })

    print(f"\nDataset built:")
    print(f"  RV patients:     {sum(1 for r in records if r['label'] == 1)}")
    print(f"  Normal patients: {sum(1 for r in records if r['label'] == 0)}")
    print(f"  Total patients:  {len(records)}")
    avg_imgs = sum(len(r['images']) for r in records) / len(records)
    print(f"  Avg images/patient: {avg_imgs:.1f}\n")

    return records


# ──────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ──────────────────────────────────────────────────────────────

class RetinalBagDataset(Dataset):
    """
    MIL bag dataset. Each __getitem__ returns one patient's bag.

    Returns:
        images  — Tensor of shape (N, 3, H, W) where N = number of images
        phases  — LongTensor of shape (N,) — phase index per image
        label   — scalar float tensor (0.0 or 1.0)
        meta    — dict with patient_id and per-image paths (for debugging)
    """

    def __init__(self, records: list[dict], training: bool = False):
        self.records   = records
        self.transform = get_transforms(training)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        img_tensors = []
        phase_idxs  = []

        for img_info in record["images"]:
            try:
                img = Image.open(img_info["path"]).convert("RGB")
                tensor = self.transform(img)
                img_tensors.append(tensor)
                phase_idxs.append(img_info["phase"])
            except Exception as e:
                print(f"Warning: could not load {img_info['path']}: {e}")
                continue

        if not img_tensors:
            raise RuntimeError(
                f"Patient {record['patient_id']} has no loadable images."
            )

        images = torch.stack(img_tensors)           # (N, 3, H, W)
        phases = torch.tensor(phase_idxs, dtype=torch.long)  # (N,)
        label  = torch.tensor(record["label"], dtype=torch.float32)

        meta = {
            "patient_id": record["patient_id"],
            "paths":      [i["path"] for i in record["images"]],
        }

        return images, phases, label, meta


# ──────────────────────────────────────────────────────────────
# UTILITY: summary of phase distribution
# ──────────────────────────────────────────────────────────────

def phase_summary(records: list[dict]):
    """Print how many images of each phase exist across the dataset."""
    phase_names = {v: k for k, v in PHASES.items()}
    counts = {k: 0 for k in PHASES.values()}
    for r in records:
        for img in r["images"]:
            counts[img["phase"]] += 1
    print("Phase distribution:")
    for idx, count in sorted(counts.items()):
        print(f"  {phase_names[idx]:10s}: {count} images")


# ──────────────────────────────────────────────────────────────
# DIAGNOSTIC: inspect folder structure before training
# ──────────────────────────────────────────────────────────────

def diagnose_folder(folder: str, max_patients: int = 5):
    """
    Print the first few patient folders and their contents.
    Run this if you're getting 'No images found' warnings to understand
    what structure your data is actually in.

    Usage:
        from dataset import diagnose_folder
        diagnose_folder("path/to/RV_PATIENTS")
    """
    folder_path = Path(folder)
    print(f"\n── Diagnosing: {folder} ──")

    if not folder_path.exists():
        print(f"  ERROR: folder does not exist.")
        return

    # List top-level contents
    top_items = sorted(folder_path.iterdir())
    dirs  = [x for x in top_items if x.is_dir()]
    files = [x for x in top_items if x.is_file()]

    print(f"  Top-level directories: {len(dirs)}")
    print(f"  Top-level files: {len(files)}")

    if files:
        print(f"  Sample top-level files (first 3):")
        for f in files[:3]:
            print(f"    {f.name}")

    print(f"\n  First {min(max_patients, len(dirs))} patient folders:")
    for patient_dir in dirs[:max_patients]:
        # List everything inside (up to 2 levels deep)
        all_items = list(patient_dir.rglob("*"))
        img_files = [f for f in all_items
                     if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
        non_img   = [f for f in all_items
                     if f.is_file() and f.suffix.lower() not in VALID_EXTENSIONS]
        subdirs   = [f for f in all_items if f.is_dir()]

        print(f"\n  [{patient_dir.name}]")
        print(f"    Sub-folders:   {len(subdirs)}")
        print(f"    Image files:   {len(img_files)}")
        print(f"    Other files:   {len(non_img)}")
        if img_files:
            print(f"    Sample images:")
            for img in img_files[:4]:
                rel = img.relative_to(patient_dir)
                print(f"      {rel}")
        if non_img:
            print(f"    Non-image files (first 3):")
            for f in non_img[:3]:
                print(f"      {f.name}  ({f.suffix})")