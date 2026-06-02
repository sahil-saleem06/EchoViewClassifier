#!/usr/bin/env python3
"""
Run the UCSF/Yale view classifier on a random sample of DICOM echo clips.

Reads DICOMs directly from a shared volume (read-only — originals are never
modified). Writes a CSV of view predictions and saves one sample frame per
clip as a PNG so a clinician can visually verify the labels.

Usage (Databricks notebook):
    %run /path/to/label_dicoms.py

Or as a script:
    python label_dicoms.py \
        --dicom_dir /Volumes/biobank_analytics/verma_lab/pmbb_echo/ \
        --checkpoint /dbfs/FileStore/view_classifier/view_23_e5_class_11-Mar-2018 \
        --output_dir /dbfs/FileStore/view_labels/ \
        --n_samples 20
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
import pydicom
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from pathlib import Path
from tqdm import tqdm

# ── View class mapping ────────────────────────────────────────────────────────
# Maps the 23 UCSF classes down to 4 standard TTE views (+ OTHER)

VIEW_MAP = {
    "plax_far":     "PLAX",
    "plax_plax":    "PLAX",
    "plax_laz":     "PLAX",
    "plax_lac":     "PLAX",
    "psax_az":      "PSAX",
    "psax_mv":      "PSAX",
    "psax_pap":     "PSAX",
    "psax_avz":     "PSAX",
    "psax_apex":    "PSAX",
    "a4c":          "A4C",
    "a4c_lvocc_s":  "A4C",
    "a4c_laocc":    "A4C",
    "a2c":          "A2C",
    "a2c_lvocc_s":  "A2C",
    "a2c_laocc":    "A2C",
    "a3c":          "OTHER",
    "a3c_lvocc_s":  "OTHER",
    "a3c_laocc":    "OTHER",
    "a5c":          "OTHER",
    "rvinf":        "OTHER",
    "suprasternal": "OTHER",
    "subcostal":    "OTHER",
    "other":        "OTHER",
}


# ── DICOM loading ─────────────────────────────────────────────────────────────

def load_dicom_frames(path: str, n_frames: int = 10, size: int = 224):
    """
    Read a DICOM cine loop, uniformly sample n_frames, resize to size x size.
    Returns array of shape (n_frames, size, size, 1) — grayscale, uint8.
    Returns None if the file cannot be read or has no pixel data.
    """
    try:
        ds = pydicom.dcmread(path, force=True)
        pixels = ds.pixel_array.astype(np.uint8)
    except Exception as e:
        print(f"  Could not read {path}: {e}")
        return None, None

    # Handle single frame vs multi-frame DICOMs
    if pixels.ndim == 2:
        pixels = pixels[np.newaxis]          # (1, H, W)
    elif pixels.ndim == 4:
        pixels = pixels[:, :, :, 0]          # (F, H, W, C) → (F, H, W)

    n_total = pixels.shape[0]
    if n_total == 0:
        return None, None

    # Uniformly sample frames
    indices = np.linspace(0, n_total - 1, min(n_frames, n_total), dtype=int)
    frames = []
    for i in indices:
        frame = cv2.resize(pixels[i], (size, size), interpolation=cv2.INTER_AREA)
        frames.append(frame[..., np.newaxis])   # (H, W, 1)

    return np.stack(frames), pixels[len(pixels) // 2]   # frames + middle frame for preview


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, n_classes: int = 23):
    """
    Load the UCSF VGG-16 view classifier from a TF1 checkpoint.
    Returns (sess, model) ready for inference.
    """
    from view_classifier import Network   # from echo-severe-AS/preprocessing/

    tf.reset_default_graph()
    sess = tf.Session()
    model = Network(0.0, 0.0, 1, n_classes, False)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    print(f"Loaded view classifier from {checkpoint_path}")
    return sess, model


# ── Inference ─────────────────────────────────────────────────────────────────

def classify_clip(sess, model, frames: np.ndarray, classes: list) -> dict:
    """
    Run one clip through the model.
    frames: (n_frames, 224, 224, 1)
    Returns dict with predicted view, confidence, and full probability breakdown.
    """
    probs = model.probabilities(sess, frames)           # (n_frames, 23)
    mean_probs = probs.mean(axis=0)                     # average across frames
    pred_idx = int(np.argmax(mean_probs))
    pred_class = classes[pred_idx]
    pred_view = VIEW_MAP.get(pred_class, "OTHER")

    return {
        "predicted_subclass": pred_class,
        "predicted_view":     pred_view,
        "confidence":         round(float(mean_probs[pred_idx]), 4),
        "probs_plax":  round(float(sum(mean_probs[i] for i, c in enumerate(classes) if VIEW_MAP.get(c) == "PLAX")), 4),
        "probs_psax":  round(float(sum(mean_probs[i] for i, c in enumerate(classes) if VIEW_MAP.get(c) == "PSAX")), 4),
        "probs_a4c":   round(float(sum(mean_probs[i] for i, c in enumerate(classes) if VIEW_MAP.get(c) == "A4C")),  4),
        "probs_a2c":   round(float(sum(mean_probs[i] for i, c in enumerate(classes) if VIEW_MAP.get(c) == "A2C")),  4),
        "probs_other": round(float(sum(mean_probs[i] for i, c in enumerate(classes) if VIEW_MAP.get(c) == "OTHER")), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(42)

    dicom_dir  = Path(args.dicom_dir)
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "sample_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── Find and sample DICOMs ────────────────────────────────────────────────
    all_dicoms = list(dicom_dir.rglob("*.dcm"))
    if not all_dicoms:
        raise FileNotFoundError(f"No .dcm files found under {dicom_dir}")

    sample = random.sample(all_dicoms, min(args.n_samples, len(all_dicoms)))
    print(f"Found {len(all_dicoms)} DICOMs — sampling {len(sample)}")

    # ── Load view classes ─────────────────────────────────────────────────────
    classes_file = Path(args.checkpoint).parent / "viewclasses_view_23_e5_class_11-Mar-2018.txt"
    classes = pd.read_csv(classes_file, header=None).iloc[:, 0].tolist()

    # ── Load model ────────────────────────────────────────────────────────────
    sess, model = load_model(args.checkpoint)

    # ── Run classifier ────────────────────────────────────────────────────────
    records = []
    for dcm_path in tqdm(sample, desc="Classifying"):
        frames, preview_frame = load_dicom_frames(str(dcm_path), n_frames=10)
        if frames is None:
            print(f"  Skipping {dcm_path.name} — could not load frames")
            continue

        result = classify_clip(sess, model, frames, classes)
        result["file"] = dcm_path.name
        result["path"] = str(dcm_path)
        records.append(result)

        # Save a sample frame as PNG for clinician review
        if preview_frame is not None:
            preview_path = frames_dir / f"{dcm_path.stem}.png"
            cv2.imwrite(str(preview_path), preview_frame)

    # ── Save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(records)[["file", "predicted_view", "predicted_subclass",
                                 "confidence", "probs_plax", "probs_psax",
                                 "probs_a4c", "probs_a2c", "probs_other", "path"]]
    out_csv = output_dir / "view_labels.csv"
    df.to_csv(out_csv, index=False)

    print(f"\nLabeled {len(df)} clips")
    print(f"Results CSV:    {out_csv}")
    print(f"Sample frames:  {frames_dir}")
    print(f"\nView distribution:\n{df['predicted_view'].value_counts().to_string()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label DICOM echo clips with view classifier")
    parser.add_argument("--dicom_dir",   required=True,
                        help="Path to folder containing DICOM files")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to view_23_e5_class_11-Mar-2018 checkpoint (no file extension)")
    parser.add_argument("--output_dir",  default="view_labels",
                        help="Where to write the CSV and sample frames")
    parser.add_argument("--n_samples",   type=int, default=20,
                        help="Number of clips to sample")
    args = parser.parse_args()
    main(args)
