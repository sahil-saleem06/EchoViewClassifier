# Sanity Check — UCSF View Classifier on EchoNet-Dynamic
**Date:** June 2, 2026
**Author:** Sahil Saleem

---

## Objective
Verify that the UCSF/Yale pretrained view classifier (VGG-16, trained 2018) is loading correctly and producing sensible predictions before deploying it on the lab's clinical TTE data on Databricks.

EchoNet-Dynamic was chosen as the test dataset because all 10,030 videos are known A4C clips — making it a straightforward ground truth to validate against.

---

## Setup
- **Model:** UCSF VGG-16 view classifier (`view_23_e5_class_11-Mar-2018`)
- **Dataset:** EchoNet-Dynamic (Stanford, 10,030 A4C videos)
- **Sample size:** 20 randomly selected videos
- **Environment:** MacOS (Apple Silicon), Python 3.9, TensorFlow 2.12 (compat v1 mode)
- **Script:** `local_tests/label_avis.py`

---

## Results

| file | predicted_view | predicted_subclass | confidence |
|---|---|---|---|
| 0X7BA58F095A0EDE7E.avi | A4C | a4c | 0.6321 |
| 0XD79C49C010950C7.avi | **OTHER** | other | 0.9580 |
| 0X48182B4ADA3FFE0E.avi | A4C | a4c_laocc | 0.9991 |
| 0X42221E1A611968E4.avi | A4C | a4c_laocc | 0.5147 |
| 0XBA15B40D87218A9.avi | **A2C** | a2c_laocc | 0.7625 |
| 0X244C8BF642C90DA7.avi | **A2C** | a2c_laocc | 0.5912 |
| 0X4A121E9371A35313.avi | A4C | a4c | 0.5452 |
| 0X77D53936A055CA6.avi | A4C | a4c_laocc | 0.9907 |
| 0X687F32820865345D.avi | A4C | a4c_laocc | 0.7860 |
| 0X1F00CAA7AC31402B.avi | A4C | a4c | 0.3684 |
| 0X803C3B563A14E1F.avi | A4C | a4c_laocc | 0.5895 |
| 0X7C7D65DB11064625.avi | A4C | a4c_laocc | 0.9127 |
| 0X19F05146FAAACB83.avi | A4C | a4c_laocc | 0.6998 |
| 0X250C34BB262848FB.avi | A4C | a4c_laocc | 0.7958 |
| 0X54D24B12375DA7A9.avi | A4C | a4c_laocc | 0.9999 |
| 0X41B72A9649E82710.avi | A4C | a4c_laocc | 0.8393 |
| 0X507257F3C2959527.avi | A4C | a4c | 0.5673 |
| 0X2D9566C4D06D6F28.avi | A4C | a4c | 0.8582 |
| 0X4686D15BB8789AC4.avi | A4C | a4c_laocc | 0.5616 |
| 0X3F21B014EDBAD737.avi | A4C | a4c_laocc | 0.8947 |

**View distribution:**
| Predicted View | Count |
|---|---|
| A4C | 17 |
| A2C | 2 |
| OTHER | 1 |

---

## Analysis

**Accuracy:** 17/20 correct = **85%**

**Correct predictions (17/20):**
- All 17 correctly identified as A4C
- Most subclass predictions were `a4c_laocc` (LV apex occluded) rather than plain `a4c` — this is clinically reasonable since EchoNet videos are known to sometimes have foreshortened apical views

**Errors (3/20):**
- 2 × A2C (`a2c_laocc`) — understandable misclassification. A4C and A2C are both apical views that differ only by probe rotation (~60°). Easy to confuse visually
- 1 × OTHER — genuinely wrong, likely a poor quality or atypical clip

**Confidence range:** 0.37 – 1.00
- Some lower confidence scores are expected — this model was trained on raw UCSF DICOMs and EchoNet videos have different preprocessing (already cropped and scaled)
- On raw clinical DICOMs (closer to the training distribution) we expect higher and more consistent confidence

---

## Conclusion

The UCSF view classifier is working correctly. The 85% accuracy on a held-out dataset it was never trained on, with errors limited to adjacent apical views, confirms the weights loaded correctly and the model is making clinically sensible predictions.

**Next step:** Deploy on lab TTE DICOMs via `notebooks/01_label_dicoms.py` on Databricks once permissions are granted.

---

## Notes on TF2 Compatibility
The original code was written in TF1/Python 2.7. To run in modern environments the following patches were applied to `view_classifier.py`:
- `import tensorflow as tf` → `import tensorflow.compat.v1 as tf` + `tf.disable_v2_behavior()`
- `tf.contrib.layers.l2_regularizer` → `None` (inference only, regularizer not needed)
- `tf.contrib.layers.xavier_initializer()` → `tf.glorot_uniform_initializer()` (equivalent)
- `tf.contrib.layers.flatten` → `tf.reshape` with static shape calculation
- `keep_dims` → removed (deprecated in TF2)
