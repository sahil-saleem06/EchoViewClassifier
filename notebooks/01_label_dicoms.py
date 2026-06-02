# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Stage 1 — Label DICOMs with UCSF View Classifier
# MAGIC
# MAGIC Runs 20 random DICOM echo clips through the UCSF/Yale pretrained view classifier.
# MAGIC Outputs a CSV of predicted views and saves a sample frame from each clip
# MAGIC so a clinician can verify the labels.
# MAGIC
# MAGIC **Before running:**
# MAGIC - Upload the 3 checkpoint files to `dbfs:/FileStore/view_classifier/`
# MAGIC - Upload `view_classifier.py` from CarDS-Yale/echo-severe-AS to the same folder

# COMMAND ----------

# MAGIC %pip install tensorflow pydicom opencv-python-headless scikit-image tqdm

# COMMAND ----------

# ── Config — edit these paths before running ──────────────────────────────────

DICOM_DIR   = "/Volumes/biobank_analytics/verma_lab/pmbb_echo/"
CHECKPOINT  = "/Workspace/VermaLab/Sahil_EchoCV/view_classifier/view_23_e5_class_11-Mar-2018"
OUTPUT_DIR  = "/Workspace/VermaLab/Sahil_EchoCV/view_labels/"
N_SAMPLES   = 20

# COMMAND ----------

import os
import sys
import random
import numpy as np
import pandas as pd
import pydicom
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from pathlib import Path
from tqdm import tqdm

# Add the view_classifier.py location to path so we can import it
sys.path.append("/Workspace/VermaLab/Sahil_EchoCV/view_classifier/")
from view_classifier import Network

random.seed(42)
print("Imports OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Find and sample DICOMs

# COMMAND ----------

dicom_dir = Path(DICOM_DIR)
all_dicoms = list(dicom_dir.rglob("*.dcm"))
print(f"Total DICOMs found: {len(all_dicoms)}")

sample = random.sample(all_dicoms, min(N_SAMPLES, len(all_dicoms)))
print(f"Sampled {len(sample)} clips for labeling")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Load the view classifier

# COMMAND ----------

# Load the 23 view class names
classes_file = Path(CHECKPOINT).parent / "viewclasses_view_23_e5_class_11-Mar-2018.txt"
CLASSES = pd.read_csv(classes_file, header=None).iloc[:, 0].tolist()
print(f"View classes: {CLASSES}")

# Load model from checkpoint
tf.reset_default_graph()
sess = tf.Session()
model = Network(0.0, 0.0, 1, 23, False)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, CHECKPOINT)
print("Model loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Map 23 subclasses → 4 standard views

# COMMAND ----------

VIEW_MAP = {
    "plax_far": "PLAX", "plax_plax": "PLAX", "plax_laz": "PLAX", "plax_lac": "PLAX",
    "psax_az":  "PSAX", "psax_mv":   "PSAX", "psax_pap": "PSAX", "psax_avz": "PSAX", "psax_apex": "PSAX",
    "a4c": "A4C", "a4c_lvocc_s": "A4C", "a4c_laocc": "A4C",
    "a2c": "A2C", "a2c_lvocc_s": "A2C", "a2c_laocc": "A2C",
    "a3c": "OTHER", "a3c_lvocc_s": "OTHER", "a3c_laocc": "OTHER",
    "a5c": "OTHER", "rvinf": "OTHER", "suprasternal": "OTHER", "subcostal": "OTHER", "other": "OTHER",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Run classifier on sampled clips

# COMMAND ----------

def load_dicom_frames(path, n_frames=10, size=224):
    """Read a DICOM, uniformly sample n_frames, resize to size x size (grayscale)."""
    try:
        ds = pydicom.dcmread(str(path), force=True)
        pixels = ds.pixel_array.astype(np.uint8)
    except Exception as e:
        return None, None

    if pixels.ndim == 2:
        pixels = pixels[np.newaxis]
    elif pixels.ndim == 4:
        pixels = pixels[:, :, :, 0]

    n_total = pixels.shape[0]
    if n_total == 0:
        return None, None

    indices = np.linspace(0, n_total - 1, min(n_frames, n_total), dtype=int)
    frames = []
    for i in indices:
        frame = cv2.resize(pixels[i], (size, size), interpolation=cv2.INTER_AREA)
        frames.append(frame[..., np.newaxis])

    preview = pixels[n_total // 2]   # middle frame for visual review
    return np.stack(frames), preview


# Make output folders
output_dir = Path(OUTPUT_DIR)
frames_dir = output_dir / "sample_frames"
output_dir.mkdir(parents=True, exist_ok=True)
frames_dir.mkdir(parents=True, exist_ok=True)

records = []

for dcm_path in tqdm(sample, desc="Classifying"):
    frames, preview = load_dicom_frames(dcm_path)
    if frames is None:
        print(f"  Skipping {dcm_path.name} — could not load")
        continue

    # Run through model — average probabilities across frames
    probs = model.probabilities(sess, frames).mean(axis=0)
    pred_idx   = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    pred_view  = VIEW_MAP.get(pred_class, "OTHER")

    records.append({
        "file":               dcm_path.name,
        "predicted_view":     pred_view,
        "predicted_subclass": pred_class,
        "confidence":         round(float(probs[pred_idx]), 4),
        "probs_plax":  round(float(sum(probs[i] for i, c in enumerate(CLASSES) if VIEW_MAP.get(c) == "PLAX")), 4),
        "probs_psax":  round(float(sum(probs[i] for i, c in enumerate(CLASSES) if VIEW_MAP.get(c) == "PSAX")), 4),
        "probs_a4c":   round(float(sum(probs[i] for i, c in enumerate(CLASSES) if VIEW_MAP.get(c) == "A4C")),  4),
        "probs_a2c":   round(float(sum(probs[i] for i, c in enumerate(CLASSES) if VIEW_MAP.get(c) == "A2C")),  4),
        "probs_other": round(float(sum(probs[i] for i, c in enumerate(CLASSES) if VIEW_MAP.get(c) == "OTHER")), 4),
        "path":               str(dcm_path),
    })

    # Save sample frame for clinician review
    if preview is not None:
        cv2.imwrite(str(frames_dir / f"{dcm_path.stem}.png"), preview)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Save and review results

# COMMAND ----------

df = pd.DataFrame(records)[["file", "predicted_view", "predicted_subclass",
                              "confidence", "probs_plax", "probs_psax",
                              "probs_a4c", "probs_a2c", "probs_other", "path"]]

df.to_csv(output_dir / "view_labels.csv", index=False)

print(f"Labeled {len(df)} clips")
print(f"\nView distribution:")
print(df["predicted_view"].value_counts().to_string())

display(df)
