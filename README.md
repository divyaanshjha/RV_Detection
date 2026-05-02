# Stage 1 — Retinal Vasculitis Detection

Detects whether retinal vasculitis is **present or absent** from
fluorescein angiography images using RETFound + Attention MIL.

---

## Folder Structure Expected

```
data/
  RV_PATIENTS/
      C001/
          C001_EARLY_PHASE_OS_.png
          C001_MID_PHASE_OS_.png
          C001_LATE_PHASE_OS_.png
      C002/
          C002_LATE_PHASE_OS_.png   ← missing phases are fine
      ...
  NORMAL_PATIENTS/
      N001/
          N001_EARLY_PHASE_OS_.png
          N001_MID_PHASE_OS_.png
      ...
```

---

## Setup

```bash
source myenv/bin/activate
pip install -r requirements.txt
```

### Optional: Download RETFound weights (strongly recommended)

```bash
mkdir retfound_weights
# Download RETFound_cfp_weights.pth from:
# https://github.com/rmaphoh/RETFound_MAE
# Place it in retfound_weights/
```

If weights are not downloaded, the code falls back to ImageNet ViT-Large
and prints a warning. This is fine for testing the pipeline locally.

---

## Configuration

Edit `config.py`:

```python
DATA = {
    "rv_dir":     "data/RV_PATIENTS",    # ← change to your actual path
    "normal_dir": "data/NORMAL_PATIENTS", # ← change to your actual path
    "output_dir": "outputs",
}
```

---

## Training

```bash
python train.py
```

This runs 5-fold cross-validation.
On CPU (local): use this for testing the pipeline runs end-to-end
with a small subset. Full training needs a GPU (Colab/Kaggle).

**Tip for testing locally (no GPU):**
Temporarily set in config.py:

```python
TRAIN["phaseA_epochs"] = 1
TRAIN["phaseB_epochs"] = 1
```

And copy 4-5 patients per class into your data folders to verify
the pipeline runs without errors before moving to Colab.

---

## Inference on New Patients

```bash
python predict.py --model outputs/fold_1/best_model.pth
```

Output shows per-patient prediction + attention weights per image,
telling you which FA phase the model found most diagnostic.

---

## Files

| File               | Purpose                                        |
| ------------------ | ---------------------------------------------- |
| `config.py`        | All hyperparameters — edit here, not elsewhere |
| `dataset.py`       | Parses folders, tags phases, builds MIL bags   |
| `model.py`         | RETFound backbone + attention MIL + classifier |
| `train.py`         | 5-fold CV training loop with Phase A + Phase B |
| `predict.py`       | Inference on new/existing patients             |
| `requirements.txt` | Python dependencies                            |

---

## Outputs

```
outputs/
  fold_1/
      best_model.pth    ← saved when sensitivity improves
      results.json      ← best metrics for this fold
  fold_2/
      ...
  cv_summary.json       ← averaged metrics across all 5 folds
```

---

## Moving to Colab / Kaggle

1. Upload this entire `stage1/` folder
2. Upload your `data/` folder
3. Install requirements: `!pip install -r requirements.txt`
4. Run: `!python train.py`

On a T4 GPU (free Colab), expect ~2-5 minutes per epoch for 300 patients.
