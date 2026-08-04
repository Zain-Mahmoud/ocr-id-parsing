# ID Extraction Pipeline — Plan Comparison

## Project outline

We are building a OCR vision detection model to extract field information from Egyptian National IDs to be used for eKYC at the Egyptian Stock Exchange, so we are looking for close to SOTA performance. 

## Quick comparison

| | Plan A | Plan B | Plan C |
|---|---|---|---|
| **Approach** | End-to-end VLM | yolo26 Detection + VLM | yolo26 Detection + OCR |
| **Accuracy** | Lower | Highest | Lowest |
| **GPU usage** | Medium | High | Lowest |
| **Complexity** | Simplest | Moderate | Most moving parts |

---

## Plan A — One-Shot VLM

**Pipeline:** Fine-tuned VLM handles detection and extraction end-to-end, in a single pass.

**Models:** Fine-tuned `Qwen3-VL-2B`

| Pros | Cons |
|---|---|
| Simplest pipeline — one model, one step | Prone to hallucination / inaccuracy |
| Medium GPU consumption | Requires clean, precise input images to work reliably |

---

## Plan B — Detection + VLM

**Pipeline:** YOLO segmentation (bounding boxes) → OpenCV orientation/resizing → VLM extraction

**Models:** Post-trained `YOLO26` (Roboflow dataset) + fine-tuned `Qwen3-VL-2B`

| Pros | Cons |
|---|---|
| Most accurate of the three | GPU-intensive, especially at inference |
| Can handle imperfect input framing (YOLO + OpenCV normalize it first) | Possibly overkill depending on real-world image quality |

---

## Plan C — Detection + OCR

**Pipeline:** YOLO segmentation (bounding boxes + field locations) → OpenCV → OCR line extraction

**Models:** Post-trained `YOLO26` for segmentation, `YOLO26` for field detection (localization only, not extraction), fine-tuned Tesseract Arabic model (`ara.traineddata` base or custom `ioy_mi_400_nn_fn_nns.traineddata`) or `arabic_PP-OCRv5_mobile_rec`

| Pros | Cons |
|---|---|
| Lowest GPU consumption | Least accurate and most moving parts|

---

### Data
For `yolo` training: we have ~6500 roboflow samples ready for training using the `ultralytics` api for segmentation (`yolo26n-seg`) and we also have access to more Egyptian national IDs that have field bounding box mappings through Roboflow. 

For VLM training (and OCR fine-tuning if needed): we have access to about 5000 labelled samples with field transcriptions. We also have a synthetic data generation script that generates fake IDs following the same structure as Egyptian cards, with a good level of randomization (names, religions, gender, jobs, etc). It can also augment the images by applying rotations, grey-scale images, etc. This script also produces line text images for the OCR models (can also be modified to generate augmented line images).

---

