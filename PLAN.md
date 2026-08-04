# ID Extraction Pipeline — Plan Comparison

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
| Lowest GPU consumption | Least accurate|

---
