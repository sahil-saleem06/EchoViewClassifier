#!/usr/bin/env python3
"""
Sanity check — run the UCSF view classifier on EchoNet-Dynamic AVI files locally.

EchoNet-Dynamic is all A4C, so every prediction should come back as 'a4c'.
If it does, the classifier and weights are working correctly.

Setup (one time):
    conda create -n view_classifier python=3.9 -y
    conda activate view_classifier
    pip install tensorflow-macos torch opencv-python-headless numpy pandas tqdm scikit-image

Also requires:
    - view_classifier.py  (from CarDS-Yale/echo-severe-AS/preprocessing/)
    - viewclasses_view_23_e5_class_11-Mar-2018.txt  (same folder)
    - view_23_e5_class_11-Mar-2018.*  (3 checkpoint files from Dropbox)

    Place all of the above in the same directory as this script.

TF2 patch — open view_classifier.py and change line 1 from:
    import tensorflow as tf
to:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

Usage:
    python local_tests/label_avis.py \
        --video_dir ~/PanEchoTest/data/EchoNet-Dynamic/Videos \
        --checkpoint local_tests/view_23_e5_class_11-Mar-2018 \
        --n_samples 20
"""

import argparse
import sys
import random
import numpy as np
import pandas as pd
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from pathlib import Path
from tqdm import tqdm

# Add this script's directory to path so view_classifier.py can be imported
sys.path.append(str(Path(__file__).parent))
from view_classifier import Network


# ── View class mapping ────────────────────────────────────────────────────────

VIEW_MAP = {
    "plax_far": "PLAX", "plax_plax": "PLAX", "plax_laz": "PLAX", "plax_lac": "PLAX",
    "psax_az":  "PSAX", "psax_mv":   "PSAX", "psax_pap": "PSAX", "psax_avz": "PSAX", "psax_apex": "PSAX",
    "a4c": "A4C", "a4c_lvocc_s": "A4C", "a4c_laocc": "A4C",
    "a2c": "A2C", "a2c_lvocc_s": "A2C", "a2c_laocc": "A2C",
    "a3c": "OTHER", "a3c_lvocc_s": "OTHER", "a3c_laocc": "OTHER",
    "a5c": "OTHER", "rvinf": "OTHER", "suprasternal": "OTHER", "subcostal": "OTHER", "other": "OTHER",
}


# ── Video loading ─────────────────────────────────────────────────────────────

def load_avi_frames(path, n_frames=10, size=224):
    """
    Read an AVI, uniformly sample n_frames, resize to size x size (grayscale).
    Returns array of shape (n_frames, size, size, 1), or None if unreadable.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(frame[..., np.newaxis])   # (H, W, 1)

    cap.release()
    return np.stack(frames) if frames else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(42)

    video_dir = Path(args.video_dir)
    checkpoint = args.checkpoint

    # ── Find and sample AVIs ──────────────────────────────────────────────────
    all_avis = list(video_dir.glob("*.avi"))
    if not all_avis:
        raise FileNotFoundError(f"No AVI files found in {video_dir}")

    sample = random.sample(all_avis, min(args.n_samples, len(all_avis)))
    print(f"Found {len(all_avis)} videos — testing on {len(sample)}")

    # ── Load view classes ─────────────────────────────────────────────────────
    classes_file = Path(__file__).parent / "viewclasses_view_23_e5_class_11-Mar-2018.txt"
    classes = pd.read_csv(classes_file, header=None).iloc[:, 0].tolist()

    # ── Load model ────────────────────────────────────────────────────────────
    tf.reset_default_graph()
    sess = tf.Session()
    model = Network(0.0, 0.0, 1, 23, False)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)
    print("Model loaded\n")

    # ── Run classifier ────────────────────────────────────────────────────────
    records = []
    for avi_path in tqdm(sample, desc="Classifying"):
        frames = load_avi_frames(str(avi_path))
        if frames is None:
            print(f"  Skipping {avi_path.name} — could not load")
            continue

        probs = model.probabilities(sess, frames).mean(axis=0)
        pred_idx   = int(np.argmax(probs))
        pred_class = classes[pred_idx]
        pred_view  = VIEW_MAP.get(pred_class, "OTHER")

        records.append({
            "file":               avi_path.name,
            "predicted_view":     pred_view,
            "predicted_subclass": pred_class,
            "confidence":         round(float(probs[pred_idx]), 4),
        })

    # ── Results ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    print("\n── Results ──────────────────────────────")
    print(df.to_string(index=False))
    print(f"\nView distribution (should be mostly A4C):")
    print(df["predicted_view"].value_counts().to_string())

    # Save to local_tests/
    out_path = Path(__file__).parent / "avi_labels.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check UCSF classifier on EchoNet AVIs")
    parser.add_argument("--video_dir",  required=True, help="Path to EchoNet-Dynamic Videos folder")
    parser.add_argument("--checkpoint", required=True, help="Path to view_23_e5_class_11-Mar-2018 checkpoint")
    parser.add_argument("--n_samples",  type=int, default=20, help="Number of videos to test")
    args = parser.parse_args()
    main(args)
