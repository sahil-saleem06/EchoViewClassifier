# EchoViewClassifier

A two-stage pipeline for echocardiogram view classification on TTE data.

**Stage 1** uses the established UCSF/Yale pretrained classifier (VGG-16, 2018) to label raw DICOM clips at scale — without training anything from scratch.

**Stage 2** trains a modern EfficientNet-B0 classifier (PyTorch) on those labels, producing a fast, open-source, reproducible view classifier that predicts one of 4 standard TTE views: **PLAX · PSAX · A4C · A2C**.

The old model provides the reliable labels. The new model provides a modern, deployable system built on top of them.

---

## Repository Structure

```
EchoViewClassifier/
│
├── local_tests/                          # OLD classifier — local sanity checks
│   ├── label_avis.py                     # Test UCSF classifier on EchoNet AVI files
│   ├── view_classifier.py                # UCSF VGG-16 architecture (TF2 patched)
│   ├── viewclasses_view_23_e5_class_11-Mar-2018.txt  # 23 view class names
│   ├── evaluate_sanity_check.R           # Precision/recall evaluation (yardstick)
│   ├── avi_labels.csv                    # Sanity check results (20 EchoNet clips)
│   └── sanity_check_results.md           # Notes on sanity check findings
│   * Checkpoint weights not included — download from Dropbox (see Stage 1 setup)
│
├── notebooks/                            # OLD classifier — Databricks deployment
│   └── 01_label_dicoms.py               # Run UCSF classifier on raw TTE DICOMs
│
├── scripts/                              # Shared utilities
│   ├── label_dicoms.py                   # Label DICOMs with UCSF classifier (script version)
│   └── extract_frames.py                 # Extract frames from AVI files
│
├── classifier/                           # NEW classifier — EfficientNet model package
│   ├── model.py                          # EfficientNet-B0 architecture
│   ├── dataset.py                        # Data loading
│   └── transforms.py                     # Image preprocessing
│
├── train.py                              # NEW classifier — training loop
├── predict.py                            # NEW classifier — inference
└── requirements.txt
```

---

## Stage 1 · Label DICOMs with the UCSF/Yale Classifier

The UCSF classifier was trained on a large dataset of echocardiograms and classifies clips into 23 granular subviews, which this pipeline maps down to 4 standard views. It runs on Databricks against raw DICOM data and outputs a CSV of labels plus sample frames for clinician review.

### Setup

1. Download the pretrained UCSF weights from [Dropbox](https://www.dropbox.com/sh/0tkcf7e0ljgs0b8/AACBnNiXZ7PetYeCcvb-Z9MSa?dl=0):
   - `view_23_e5_class_11-Mar-2018.data-00000-of-00001`
   - `view_23_e5_class_11-Mar-2018.index`
   - `view_23_e5_class_11-Mar-2018.meta`
2. Also download from [CarDS-Yale/echo-severe-AS](https://github.com/CarDS-Yale/echo-severe-AS/blob/main/preprocessing/):
   - `view_classifier.py`
   - `viewclasses_view_23_e5_class_11-Mar-2018.txt`
3. Upload all 5 files to your Databricks workspace under `Sahil_EchoCV/view_classifier/`

### Run (Databricks)

Open `notebooks/01_label_dicoms.py`, attach a cluster, and run all cells. Key config at the top of the notebook:

```python
DICOM_DIR  = "/Volumes/biobank_analytics/verma_lab/pmbb_echo/"
CHECKPOINT = "/Workspace/VermaLab/Sahil_EchoCV/view_classifier/view_23_e5_class_11-Mar-2018"
OUTPUT_DIR = "/Workspace/VermaLab/Sahil_EchoCV/view_labels/"
N_SAMPLES  = 20
```

### Output

```
view_labels/
├── view_labels.csv       ← predicted view + confidence for each clip
└── sample_frames/        ← one PNG frame per clip for clinician review
```

---

## Stage 2 · Train the Modern EfficientNet Classifier

Run after Stage 1 labels have been validated by a clinician.

### Data layout

Organize validated clips into folders by view:

```
data/
  train/
    A2C/    A4C/    PLAX/    PSAX/
  val/
    A2C/    A4C/    PLAX/    PSAX/
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
# Single image
python predict.py --checkpoint checkpoints/best.pt --input frame.png

# Directory of images
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

## Citation

If you use the UCSF pretrained weights, please cite:

> Ghorbani A, Ouyang D, Abid A, et al. Deep learning interpretation of echocardiograms. *NPJ Digital Medicine*, 2020.
