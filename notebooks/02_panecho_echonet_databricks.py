# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Run PanEcho on EchoNet-Dynamic (Databricks)
# MAGIC
# MAGIC Runs PanEcho inference on EchoNet-Dynamic AVI files stored on DBFS.
# MAGIC Outputs a CSV of all 39 clinical predictions per video.
# MAGIC
# MAGIC **No data is modified — videos are read only.**

# COMMAND ----------

# MAGIC %pip install torch torchvision opencv-python-headless tqdm

# COMMAND ----------

# ── Config ────────────────────────────────────────────────────────────────────

VIDEO_DIR  = "/dbfs/FileStore/tables/PanEcho/Videos/"
OUTPUT_DIR = "/Workspace/VermaLab/Sahil_EchoCV/panecho_echonet/"
CLIP_LEN   = 16      # frames sampled per video
MAX_VIDEOS = None    # set to a number (e.g. 100) to test on a subset first

# COMMAND ----------

import os
import warnings
import numpy as np
import pandas as pd
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

# Auto-detect device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Load PanEcho

# COMMAND ----------

print("Loading PanEcho (downloads weights on first run ~300MB)...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = torch.hub.load("CarDS-Yale/PanEcho", "PanEcho", force_reload=False, clip_len=CLIP_LEN)
model.eval().to(device)
print("PanEcho loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Find AVI files

# COMMAND ----------

all_avis = sorted(Path(VIDEO_DIR).glob("*.avi"))
if MAX_VIDEOS:
    all_avis = all_avis[:MAX_VIDEOS]
print(f"Videos to process: {len(all_avis)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Load and preprocess video frames

# COMMAND ----------

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None, None]
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None, None]

def load_video(path, clip_len=16, size=224):
    """Read AVI, uniformly sample clip_len frames, normalize, return (1, 3, T, H, W) tensor."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    if not frames:
        return None
    # Uniformly sample frames
    idx = np.linspace(0, len(frames) - 1, clip_len, dtype=int)
    frames = [frames[i] for i in idx]
    video = np.stack(frames).transpose(3, 0, 1, 2).astype(np.float32) / 255.0
    video = (video - _MEAN) / _STD
    return torch.from_numpy(video).unsqueeze(0)   # (1, C, T, H, W)

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Run inference

# COMMAND ----------

out_csv = os.path.join(OUTPUT_DIR, "panecho_echonet_results.csv")
write_header = not os.path.exists(out_csv)
failed = []

with open(out_csv, "a", buffering=1) as fout:
    for avi_path in tqdm(all_avis, desc="PanEcho"):
        tensor = load_video(str(avi_path), clip_len=CLIP_LEN)
        if tensor is None:
            failed.append(avi_path.name)
            continue
        try:
            with torch.no_grad():
                preds = model(tensor.to(device))
        except Exception as e:
            failed.append(avi_path.name)
            continue

        row = {"file": avi_path.name}
        row.update(flatten_preds(preds))

        row_df = pd.DataFrame([row])
        row_df.to_csv(fout, index=False, header=write_header)
        write_header = False

print(f"\nProcessed: {len(all_avis) - len(failed)} | Failed: {len(failed)}")
print(f"Results saved to: {out_csv}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Preview results

# COMMAND ----------

df = pd.read_csv(out_csv)
print(f"Shape: {df.shape}")
display(df.head(10))
