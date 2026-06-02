# EchoViewClassifier

Automated echocardiogram view classification for TTE data.
Classifies echo clips into 4 standard views: **PLAX, PSAX, A4C, A2C**.

---

## Overview

This repo has two stages:

**Stage 1 — Label your data** using the established UCSF/Yale view classifier (VGG-16, trained 2018) to generate ground truth view labels from raw DICOM clips.

**Stage 2 — Train a modern classifier** (EfficientNet-B0, PyTorch) on those labels to produce a fast, open-source, reproducible view classifier.

---

## Stage 1 · Label DICOMs with the UCSF/Yale Classifier

> Run this on Databricks against your raw TTE data.

### Prerequisites

1. Download the pretrained UCSF weights from [Dropbox](https://www.dropbox.com/sh/0tkcf7e0ljgs0b8/AACBnNiXZ7PetYeCcvb-Z9MSa?dl=0):
   - `view_23_e5_class_11-Mar-2018.data-00000-of-00001`
   - `view_23_e5_class_11-Mar-2018.index`
   - `view_23_e5_class_11-Mar-2018.meta`
2. Upload them to your Databricks DBFS (e.g. `dbfs:/FileStore/view_classifier/`)
3. Also upload `view_classifier.py` from [CarDS-Yale/echo-severe-AS](https://github.com/CarDS-Yale/echo-severe-AS/blob/main/preprocessing/view_classifier.py)

### Install dependencies
```bash
%pip install tensorflow pydicom opencv-python-headless scikit-image
```

### Run
```bash
python scripts/label_dicoms.py \
    --dicom_dir /Volumes/biobank_analytics/verma_lab/pmbb_echo/ \
    --checkpoint /dbfs/FileStore/view_classifier/view_23_e5_class_11-Mar-2018 \
    --output_dir /dbfs/FileStore/view_labels/ \
    --n_samples 20
```

### Output
```
view_labels/
├── view_labels.csv       ← predicted view + confidence per clip
└── sample_frames/        ← one PNG frame per clip for clinician review
```

---

## Stage 2 · Train the Modern EfficientNet Classifier

> Run after you have validated labels from Stage 1.

### Data layout
```
data/
  train/
    A2C/   A4C/   PLAX/   PSAX/
  val/
    A2C/   A4C/   PLAX/   PSAX/
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train
```bash
python train.py --data data/ --epochs 30 --batch-size 32
```

### Predict
```bash
python predict.py --checkpoint checkpoints/best.pt --input image.png
python predict.py --checkpoint checkpoints/best.pt --input frames/
```

### Output
```json
{
  "file": "frame_001.png",
  "predicted_view": "A4C",
  "confidence": 0.9821,
  "probabilities": {"A2C": 0.005, "A4C": 0.982, "PLAX": 0.008, "PSAX": 0.005}
}
```

---

## Repository Structure

```
EchoViewClassifier/
│
├── classifier/               # Model package
│   ├── model.py              # EfficientNet-B0 architecture
│   ├── dataset.py            # Data loading
│   └── transforms.py         # Image preprocessing
│
├── scripts/
│   ├── label_dicoms.py       # Stage 1: label raw DICOMs with UCSF classifier
│   └── extract_frames.py     # Utility: extract frames from AVI files
│
├── train.py                  # Stage 2: train EfficientNet classifier
├── predict.py                # Stage 2: run inference on new images
└── requirements.txt
```

---

## Citation

If you use the UCSF pretrained weights, please cite the original echocv work:

> Ghorbani A, Ouyang D, Abid A, et al. Deep learning interpretation of echocardiograms. *NPJ Digital Medicine*, 2020.