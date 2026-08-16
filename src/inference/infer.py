"""
Inference pipeline combining YOLO26 OBB detection, a dedicated digit
detector for the national ID number, EasyOCR for other text fields,
and a Qwen3-VL-2B fallback, gated by confidence thresholds and validated
against validate.py's structural checks.
"""

from __future__ import annotations

import json
import logging
import math

import cv2
import numpy as np
from easyocr import Reader
from PIL import Image, ImageEnhance
from tqdm.auto import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results
from unsloth import FastVisionModel
from arabic_reshaper import reshape
import bidi

import validate

logger = logging.getLogger(__name__)

OCR_CONF_THRESHOLD = 0.7
DETECTION_CONF_THRESHOLD = 0.7
MAX_VLM_RETRIES = 2

DIGIT_FIELD_FORMATS = {
    "national_id": {"digit_count": 14, "separator_positions": ()},
    "issue_date": {"digit_count": 6, "separator_positions": (4,)},
    "expiration_date": {"digit_count": 8, "separator_positions": (4, 6)},
}

FIELD_STRUCTURE = {
    "side": "string, 'front' or 'back'",
    "first_name": "string (arabic)",
    "last_name": "string (arabic)",
    "national_id": "string, 14 digits",
    "address": "string (arabic)",
    "address2": "string (arabic)",
    "issue_date": "string, formatted date (arabic)",
    "expiration_date": "string, formatted date (arabic)",
    "job_title": "string (arabic)",
    "gender": "string, 'male' or 'female' (arabic)",
    "religion": "string, 'muslim' or 'christian' (arabic)",
    "marital_status": "string, 'single', 'married' or 'widow' (arabic)",
}

SYSTEM_PROMPT = """
    You are a Vision Language Model tasked with extracting field values from an Egyptian national identity
    document. You must extract the fields without making any changes to the fields and return them
    as they are in Arabic script. If there is a field that you are unable to extract, do not attempt
    to infer or guess the value of that field based on other information.
"""

USER_PROMPT = f"""
    You are given one side of an Egyptian National ID, either front or back.
    First determine which side you are looking at, then extract only the fields that belong
    to that side (front: first_name, last_name, address, address2, national_id, side.
    back: issue_date, expiration_date, job_title, gender, religion, marital_status, national_id, side).
    Field reference: {FIELD_STRUCTURE}.
    Always include "side" set to exactly "front" or "back", and always include "national_id".
    Do not include fields that don't belong to the detected side.
    Return all fields as they appear, with no changes or corrections. Return valid JSON only,
    with no surrounding text or markdown fences.
"""

def digit_translate(text: str):
    arabic = "٠١٢٣٤٥٦٧٨٩"
    english = "0123456789"
    transl_map = str.maketrans(english, arabic)
    return text.translate(transl_map)

def load_yolo(path: str) -> YOLO:
    return YOLO(path, task="obb")


def load_digit_model(path: str) -> YOLO:
    """Plain detect-task model, one class per digit 0-9 — not OBB."""
    return YOLO(path)


def load_ocr_reader() -> Reader:
    return Reader(["ar"])

def load_vlm(path: str):
    generation_model, tokenizer = FastVisionModel.from_pretrained(path, load_in_4bit=True)
    FastVisionModel.for_inference(generation_model)
    return generation_model, tokenizer


def load(
    yolo_path: str = "./models/yolo/best.onnx",
    vlm_path: str = "./models/qwen3-vlm",
    digit_model_path: str = "./models/digit_detector/detect_id.pt",
) -> dict:
    return {
        "yolo": load_yolo(yolo_path),
        "vlm": load_vlm(vlm_path),
        "digit_model": load_digit_model(digit_model_path),
        "ocr_reader": load_ocr_reader(),
    }

def preprocess_image(image: Image.Image) -> Image.Image:
    """Desaturates for contrast/lighting robustness, kept 3-channel to
    match the detector's training distribution (including its own
    grayscale-augmented samples, which were always 3-channel)."""
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  # was mislabeled BGR2GRAY
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    pil_image = Image.fromarray(gray_3ch)

    return pil_image


def get_line_crop(image: np.ndarray, box, padding: float = 0.2) -> np.ndarray | None:
    """Crop + straighten a single OBB detection out of the full scene image.
    Returns None for degenerate boxes — callers must check for this."""
    xc, yc, w, h, r = [float(v) for v in box]
    angle_deg = math.degrees(r)

    img_h, img_w = image.shape[:2]
    w_pad, h_pad = w * (1 + padding), h * (1 + (padding/2))

    diag = int(math.ceil(math.hypot(w_pad, h_pad) / 2)) + 2
    patch_left = int(max(xc - diag, 0))
    patch_top = int(max(yc - diag, 0))
    patch_right = int(min(xc + diag, img_w))
    patch_bottom = int(min(yc + diag, img_h))
    patch = image[patch_top:patch_bottom, patch_left:patch_right]

    if patch.size == 0:
        return None

    patch_cx, patch_cy = xc - patch_left, yc - patch_top
    M = cv2.getRotationMatrix2D(center=(patch_cx, patch_cy), angle=angle_deg, scale=1.0)
    rotated = cv2.warpAffine(patch, M, (patch.shape[1], patch.shape[0]), flags=cv2.INTER_CUBIC)

    left = int(max(patch_cx - w_pad / 2 - 10, 0))
    right = int(min(patch_cx + w_pad / 2 + 10, rotated.shape[1]))
    top = int(max(patch_cy - h_pad / 2 - 10, 0))
    bottom = int(min(patch_cy + h_pad / 2 + 10, rotated.shape[0]))

    if right <= left or bottom <= top:
        return None

    return rotated[top:bottom, left:right]


def extract(result: Results) -> dict:
    """Turn a YOLO Results object into {class_name: {conf, line_crop}}"""
    classes = result.names
    obb = result.obb
    orig_image = result.orig_img
    extracts: dict = {}

    if obb is None:
        return extracts

    for i in range(len(obb.data)):
        curr_class_idx = int(obb.cls[i].item())
        curr_class_label = classes[curr_class_idx]
        curr_conf = obb.conf[i].item()
        curr_xywhr = obb.xywhr[i]

        line_crop = get_line_crop(orig_image, curr_xywhr, padding=0.8)
        if line_crop is None:
            logger.warning("degenerate crop for class %r, skipping", curr_class_label)
            continue

        existing = extracts.get(curr_class_label)
        if existing is not None and existing["conf"] >= curr_conf:
            continue

        extracts[curr_class_label] = {"conf": curr_conf, "line_crop": line_crop}

    return extracts


def infer_side(detected_classes: set[str]) -> str | None:
    front_hits = len(detected_classes & validate.FRONT_ONLY_FIELDS)
    back_hits = len(detected_classes & validate.BACK_ONLY_FIELDS)
    if front_hits == 0 and back_hits == 0:
        return None
    return "front" if front_hits >= back_hits else "back"


def find_unsure_classes(detections: dict, side: str | None) -> set[str]:
    unsure = {cls for cls, data in detections.items() if data["conf"] < DETECTION_CONF_THRESHOLD}

    if side is not None:
        expected = validate.expected_keys(side) - {"side"}
        missing = expected - set(detections.keys())
        unsure |= missing

    return unsure


def detect(model: YOLO, image) -> tuple[dict, set[str], str | None]:
    results = model(image, device="cpu")[0]
    detections = extract(results)
    side = infer_side(set(detections.keys()))
    unsure = find_unsure_classes(detections, side)
    return detections, unsure, side

def read_digits_via_digit_model(
    digit_model: YOLO,
    line_crop: np.ndarray,
    expected_digit_count: int | None = None,
    separator_positions: tuple[int, ...] = (),
) -> tuple[str, float]:
    """Reads any fixed-format digit field by detecting individual digit
    glyphs and sorting by x-position"""
    results = digit_model.predict(line_crop, device="cpu")[0]
    if results.boxes is None or len(results.boxes) == 0:
        return "", 0.0
    
    detected = []
    for box in results.boxes:
        cls = str(int(box.cls.item()))
        cls = digit_translate(cls)
        conf = box.conf.item()
        x1 = box.xyxy[0][0].item()
        detected.append((cls, x1, conf))

    detected.sort(key=lambda d: d[1])  # left-to-right by x-position
    digits = [str(cls) for cls, _, _ in detected]
    confs = [conf for _, _, conf in detected]

    chars = []
    for i, digit in enumerate(digits):
        chars.append(digit)
        if (i + 1) in separator_positions:
            chars.append("/")
    text = "".join(chars)

    if expected_digit_count is not None and len(digits) != expected_digit_count:
        logger.warning(
            "digit count mismatch: got %d, expected %d (%r)",
            len(digits), expected_digit_count, text,
        )
        return text, 0.0  # force a fallback rather than trust a malformed count

    conf = min(confs) if confs else 0.0  # weakest digit, not an average
    return text, conf


def ocr_inference(digit_model: YOLO, ocr_reader: Reader, detections: dict) -> dict:
    """Recognizes text for every detected field crop except id_card."""
    conversions = {}

    for cls, data in detections.items():
        if cls == "id_card":
            continue

        line_crop = data["line_crop"]

        if cls in DIGIT_FIELD_FORMATS:
            fmt = DIGIT_FIELD_FORMATS[cls]
            text, conf = read_digits_via_digit_model(
                digit_model, line_crop,
                expected_digit_count=fmt["digit_count"],
                separator_positions=fmt["separator_positions"],
            )
        else:
            results = ocr_reader.readtext(line_crop, detail=1, paragraph=True)
            text = bidi.get_display(reshape(" ".join(r[1] for r in results).strip())).strip()
            conf = 0.8

        conversions[cls] = {"text": text, "conf": conf}

    return conversions


def find_unsure_recognitions(recognitions: dict) -> set[str]:
    return {cls for cls, data in recognitions.items() if data["conf"] < OCR_CONF_THRESHOLD}


def vlm_inference(model, tokenizer, image: Image.Image) -> dict | None:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]},
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=256,
        use_cache=True,
        do_sample=False,
    )

    input_length = inputs["input_ids"].shape[1]
    raw = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("VLM returned invalid JSON: %r", raw[:200])
        return None


def vlm_fallback_with_retry(model, tokenizer, image: Image.Image) -> tuple[dict | None, int]:
    parsed = None
    code = -1

    for attempt in range(MAX_VLM_RETRIES + 1):
        parsed = vlm_inference(model, tokenizer, image)
        code = validate.validate_dict(parsed) if parsed is not None else -1

        if code == 0:
            return parsed, code

        logger.info("VLM attempt %d failed validation: %s", attempt + 1, validate.describe(code))

    return parsed, code


def predict(
    detection_model: YOLO,
    generation_model,
    tokenizer,
    digit_model: YOLO,
    ocr_reader: Reader,
    image: Image.Image,
) -> dict:
    preprocessed_image = preprocess_image(image)
    np_image = np.array(preprocessed_image)

    detections, unsure_detections, side = detect(detection_model, np_image)

    if "id_card" in unsure_detections or side is None:
        parsed, code = vlm_fallback_with_retry(generation_model, tokenizer, preprocessed_image)
        return _finalize(parsed, code)

    if unsure_detections:
        card_crop_np = detections["id_card"]["line_crop"]
        card_crop = Image.fromarray(card_crop_np)
        parsed, code = vlm_fallback_with_retry(generation_model, tokenizer, card_crop)
        return _finalize(parsed, code)

    recognitions = ocr_inference(digit_model, ocr_reader, detections)
    unsure_recognitions = find_unsure_recognitions(recognitions)

    if unsure_recognitions:
        card_crop_np = detections["id_card"]["line_crop"]
        card_crop = Image.fromarray(card_crop_np)
        parsed, code = vlm_fallback_with_retry(generation_model, tokenizer, card_crop)
        return _finalize(parsed, code)

    results = {cls: data["text"] for cls, data in recognitions.items()}
    results["side"] = side
    code = validate.validate_dict(results)
    return _finalize(results, code)


def _finalize(parsed: dict | None, code: int) -> dict:
    if code == 0:
        return {"status": "ok", "fields": parsed}
    return {
        "status": "failed",
        "reason": validate.describe(code),
        "severity": validate.severity_of(code).name,
        "fields": parsed,
    }


def batch_infer(detection_model, generation_model, tokenizer, digit_model, ocr_reader, samples: list) -> list[dict]:
    predictions = []
    for i, sample in enumerate(tqdm(samples)):
        try:
            prediction = predict(detection_model, generation_model, tokenizer, digit_model, ocr_reader, sample)
        except Exception:
            logger.exception("sample %d failed with an unhandled exception", i)
            prediction = {"status": "failed", "reason": "unhandled_exception", "fields": None}
        predictions.append(prediction)

    return predictions
