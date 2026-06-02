# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Run PanEcho on Raw DICOM Echo Clips
# MAGIC
# MAGIC Loads PanEcho (Holste et al., JAMA 2025) and runs inference on raw DICOM
# MAGIC cine loops from the PMBB echo dataset. Outputs a CSV with PanEcho's 39
# MAGIC clinical predictions for each clip.
# MAGIC
# MAGIC **No data is modified — DICOMs are read only.**

# COMMAND ----------

# MAGIC %pip install torch torchvision pydicom opencv-python-headless tqdm

# COMMAND ----------

# ── Config — edit these paths before running ──────────────────────────────────

DICOM_DIR  = "/Volumes/biobank_analytics/verma_lab/pmbb_echo/"
OUTPUT_DIR = "/Workspace/VermaLab/Sahil_EchoCV/panecho_results/"
N_SAMPLES  = 20       # set to None to run on all clips
CLIP_LEN   = 16       # number of frames sampled per clip

# COMMAND ----------

import os
import random
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
import warnings
from pathlib import Path
from tqdm import tqdm

random.seed(42)

# Auto-detect best device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Load PanEcho model

# COMMAND ----------

print("Loading PanEcho (downloads weights on first run ~300MB)...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = torch.hub.load("CarDS-Yale/PanEcho", "PanEcho", force_reload=False, clip_len=CLIP_LEN)

model.eval()
model = model.to(device)
print("PanEcho loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Find and sample DICOMs

# COMMAND ----------

dicom_dir = Path(DICOM_DIR)
all_dicoms = list(dicom_dir.rglob("*.dcm"))
print(f"Total DICOMs found: {len(all_dicoms)}")

sample = random.sample(all_dicoms, min(N_SAMPLES, len(all_dicoms))) if N_SAMPLES else all_dicoms
print(f"Running PanEcho on {len(sample)} clips")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Load and preprocess DICOM frames

# COMMAND ----------

# ImageNet normalization — matches PanEcho training
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None, None]
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None, None]

def load_dicom_clip(path, clip_len=16, size=224):
    """
    Read a DICOM cine loop, uniformly sample clip_len frames, resize to
    size x size, normalize, and return tensor of shape (1, 3, T, H, W).
    Returns None if the file cannot be read.
    """
    try:
        ds = pydicom.dcmread(str(path), force=True)
        pixels = ds.pixel_array.astype(np.uint8)
    except Exception as e:
        return None

    # Handle single frame vs multi-frame
    if pixels.ndim == 2:
        pixels = pixels[np.newaxis]
    elif pixels.ndim == 4:
        pixels = pixels[:, :, :, 0]

    n_total = pixels.shape[0]
    if n_total == 0:
        return None

    # Uniformly sample frames
    indices = np.linspace(0, n_total - 1, clip_len, dtype=int)
    frames = []
    for i in indices:
        frame = pixels[i] if i < n_total else pixels[-1]
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)   # echo is grayscale → convert to 3ch
        frames.append(frame)

    # Stack to (C, T, H, W), normalize
    video = np.stack(frames).transpose(3, 0, 1, 2).astype(np.float32) / 255.0
    video = (video - _MEAN) / _STD
    return torch.from_numpy(video).unsqueeze(0)   # (1, C, T, H, W)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Run PanEcho inference

# COMMAND ----------

def flatten_preds(preds):
    """Convert PanEcho output dict to flat {column: scalar} dict."""
    flat = {}
    for task, val in preds.items():
        if isinstance(val, torch.Tensor):
            v = val.detach().cpu().float()
            if v.numel() == 1:
                flat[task] = round(v.item(), 4)
            else:
                for i, scalar in enumerate(v.flatten().tolist()):
                    flat[f"{task}_cls{i}"] = round(scalar, 4)
        else:
            flat[task] = val
    return flat


os.makedirs(OUTPUT_DIR, exist_ok=True)
records = []
failed = []

for dcm_path in tqdm(sample, desc="PanEcho"):
    tensor = load_dicom_clip(str(dcm_path), clip_len=CLIP_LEN)
    if tensor is None:
        failed.append(dcm_path.name)
        continue

    try:
        with torch.no_grad():
            preds = model(tensor.to(device))
    except RuntimeError as e:
        failed.append(dcm_path.name)
        continue

    row = {"file": dcm_path.name, "path": str(dcm_path)}
    row.update(flatten_preds(preds))
    records.append(row)

print(f"\nProcessed: {len(records)} | Failed: {len(failed)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Save and review results

# COMMAND ----------

df = pd.DataFrame(records)
out_csv = os.path.join(OUTPUT_DIR, "panecho_results.csv")
df.to_csv(out_csv, index=False)

print(f"Results saved to: {out_csv}")
print(f"\nColumns: {list(df.columns)}")

display(df)
