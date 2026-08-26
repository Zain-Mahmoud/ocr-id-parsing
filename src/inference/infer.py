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
from dataclasses import dataclass

import cv2
import numpy as np
from easyocr import Reader
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results
from unsloth import FastVisionModel
from arabic_reshaper import reshape
from bidi.algorithm import get_display

import validate

logger = logging.getLogger(__name__)

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


def digit_translate(text: str) -> str:
    western = "0123456789"
    eastern = "٠١٢٣٤٥٦٧٨٩"
    return text.translate(str.maketrans(western, eastern))


@dataclass
class PipelineConfig:
    """All the tunable knobs in one place, instead of scattered module
    globals and inline magic numbers"""

    detection_conf_threshold: float = 0.7
    ocr_conf_threshold: float = 0.7
    max_vlm_retries: int = 2

    digit_conf_threshold: float = 0.15
    digit_iou_threshold: float = 0.3

    min_card_crop_dim: int = 500

    field_padding: float = 0.4
    digit_field_padding: float = 0.8


class IDExtractionPipeline:
    def __init__(
        self,
        yolo_path: str = "./models/yolo/best.onnx",
        vlm_path: str = "./models/qwen3-vlm",
        digit_model_path: str = "./models/digit_detector/detect_digit.onnx",
        ocr_languages: tuple[str, ...] = ("ar",),
        config: PipelineConfig | None = None,
    ):
        self.config = config if config else PipelineConfig()

        self.yolo = self._load_yolo(yolo_path)
        self.digit_model = self._load_digit_model(digit_model_path)
        self.ocr_reader = self._load_ocr_reader(ocr_languages)
        self.vlm_model, self.vlm_tokenizer = self._load_vlm(vlm_path)

    @staticmethod
    def _load_yolo(path: str) -> YOLO:
        return YOLO(path, task="obb")

    @staticmethod
    def _load_digit_model(path: str) -> YOLO:
        return YOLO(path, task="detect")

    @staticmethod
    def _load_ocr_reader(languages: tuple[str, ...]) -> Reader:
        return Reader(list(languages))

    @staticmethod
    def _load_vlm(path: str):
        generation_model, tokenizer = FastVisionModel.from_pretrained(path, load_in_4bit=True)
        FastVisionModel.for_inference(generation_model)
        return generation_model, tokenizer

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Desaturates for contrast/lighting robustness, kept 3-channel to
        match the detector's training distribution."""
        rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        pil_image = Image.fromarray(gray_3ch)
        return pil_image


    @staticmethod
    def _get_line_crop(image: np.ndarray, box, padding: float, cls: str) -> np.ndarray | None:
        """Crop + straighten a single OBB detection out of the full scene
        image. padding is applied symmetrically to width AND height via
        true division — returns None for degenerate boxes."""
        xc, yc, w, h, r = [float(v) for v in box]
        angle_deg = math.degrees(r)
        if cls == "national_id":
            padding = 1.1
            addr = 500
            addl = 60
        else:
            padding = 0.2
            addr = 0
            addl = 0
        img_h, img_w = image.shape[:2]
        w_pad, h_pad = w * (1 + padding), h * (1 + padding)

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

        left = int(max((patch_cx - w_pad / 2) + addl, 0))
        right = int(min((patch_cx + w_pad / 2) + addr, rotated.shape[1]))
        top = int(max(patch_cy - h_pad / 2, 0))
        bottom = int(min(patch_cy + h_pad / 2, rotated.shape[0]))

        if right <= left or bottom <= top:
            return None

        return rotated[top:bottom, left:right]

    def _extract(self, result: Results) -> dict:
        """Turn a YOLO Results object into {class_name: {conf, line_crop}}.
        Duplicate classes keep the higher-confidence instance; degenerate
        crops are dropped."""
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

            padding = (
                self.config.digit_field_padding
                if curr_class_label in DIGIT_FIELD_FORMATS
                else self.config.field_padding
            )
            line_crop = self._get_line_crop(orig_image, curr_xywhr, padding, cls=curr_class_label)
            if line_crop is None:
                logger.warning("degenerate crop for class %r, skipping", curr_class_label)
                continue

            existing = extracts.get(curr_class_label)
            if existing is not None and existing["conf"] >= curr_conf:
                continue
            extracts[curr_class_label] = {"conf": curr_conf, "line_crop": line_crop}

        return extracts

    @staticmethod
    def _infer_side(detected_classes: set[str]) -> str | None:
        front_hits = len(detected_classes & validate.FRONT_ONLY_FIELDS)
        back_hits = len(detected_classes & validate.BACK_ONLY_FIELDS)
        if front_hits == 0 and back_hits == 0:
            return None
        return "front" if front_hits >= back_hits else "back"

    def _find_unsure_classes(self, detections: dict, side: str | None) -> set[str]:
        unsure = {
            cls for cls, data in detections.items()
            if data["conf"] < self.config.detection_conf_threshold
        }

        if side is not None:
            expected = validate.expected_keys(side) - {"side"}
            missing = expected - set(detections.keys())
            unsure |= missing

        return unsure

    def detect(self, image) -> tuple[dict, set[str], str | None]:
        """Two-pass detection: pass 1 locates the card in the full scene,
        pass 2 re-runs on the straightened card crop for sharper field
        detections"""
        first_pass = self.yolo(image, device="cpu", verbose=False)[0]
        first_detections = self._extract(first_pass)

        if "id_card" not in first_detections:
            side = self._infer_side(set(first_detections.keys()))
            unsure = self._find_unsure_classes(first_detections, side)
            return first_detections, unsure, side

        card_crop = first_detections["id_card"]["line_crop"]

        second_pass = self.yolo(card_crop, device="cpu", verbose=False)[0]
        second_detections = self._extract(second_pass)
        second_detections["id_card"] = first_detections["id_card"]  # see docstring

        side = self._infer_side(set(second_detections.keys()))
        unsure = self._find_unsure_classes(second_detections, side)
        return second_detections, unsure, side


    def _read_digits(
        self,
        line_crop: np.ndarray,
    ) -> tuple[str, float]:
        """Reads a fixed-format digit field by detecting individual digit
        glyphs and sorting by x-position."""

        result = self.ocr_reader.recognize(line_crop)[0]
        conf = result[2]
        digits = result[1].split(' ')
        final_digits = ""
        for block in digits[::-1]:
            for char in block:
                final_digits += char
        
        return final_digits, conf

    def ocr_inference(self, detections: dict) -> dict:
        """Recognizes text for every detected field crop except id_card."""
        conversions = {}

        for cls, data in detections.items():
            if cls == "id_card":
                continue

            line_crop = data["line_crop"]
            Image.fromarray(line_crop).show()
            if cls == "national_id":
                text, conf = self._read_digits(
                    line_crop,
                )
            else:
                results = self.ocr_reader.recognize(line_crop)[0]
                if not results:
                    text, conf = "", 0.0
                else:
                    raw_text = results[1]
                    text = reshape(raw_text).strip() 
                    conf = float(results[2])

            conversions[cls] = {"text": text, "conf": conf}

        return conversions

    def _find_unsure_recognitions(self, recognitions: dict) -> set[str]:
        return {
            cls for cls, data in recognitions.items()
            if data["conf"] < self.config.ocr_conf_threshold
        }


    def _vlm_inference(self, image: Image.Image) -> dict | None:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]},
        ]

        input_text = self.vlm_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.vlm_tokenizer(
            image, input_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.vlm_model.device)

        output = self.vlm_model.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=True,
            do_sample=False,
        )

        input_length = inputs["input_ids"].shape[1]
        raw = self.vlm_tokenizer.decode(output[0][input_length:], skip_special_tokens=True)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("VLM returned invalid JSON: %r", raw[:200])
            return None

    def _vlm_fallback_with_retry(self, image: Image.Image) -> tuple[dict | None, int]:
        parsed = None
        code = -1

        for attempt in range(self.config.max_vlm_retries + 1):
            parsed = self._vlm_inference(image)
            code = validate.validate_dict(parsed) if parsed is not None else -1

            if code == 0:
                return parsed, code

            logger.info("VLM attempt %d failed validation: %s", attempt + 1, validate.describe(code))

        return parsed, code

    def predict(self, image: Image.Image) -> dict:
        preprocessed_image = self.preprocess_image(image)
        np_image = np.array(preprocessed_image)

        detections, unsure_detections, side = self.detect(np_image)

        if "id_card" in unsure_detections or side is None:
            parsed, code = self._vlm_fallback_with_retry(preprocessed_image)
            return self._finalize(parsed, code)

        if unsure_detections:
            card_crop = Image.fromarray(detections["id_card"]["line_crop"])
            parsed, code = self._vlm_fallback_with_retry(card_crop)
            return self._finalize(parsed, code)

        recognitions = self.ocr_inference(detections)
        unsure_recognitions = self._find_unsure_recognitions(recognitions)

        if unsure_recognitions:
            card_crop = Image.fromarray(detections["id_card"]["line_crop"])
            parsed, code = self._vlm_fallback_with_retry(card_crop)
            return self._finalize(parsed, code)
        results = {cls: data["text"] for cls, data in recognitions.items()}
        results["side"] = side
        code = validate.validate_dict(results)
        return self._finalize(results, code)

    @staticmethod
    def _finalize(parsed: dict | None, code: int) -> dict:

        if code == 0:
            return {"status": "ok", "fields": parsed}
        return {
            "status": "failed",
            "reason": validate.describe(code),
            "severity": validate.severity_of(code).name,
            "fields": parsed,
        }

    def batch_predict(self, samples: list) -> list[dict]:
        predictions = []
        for i, sample in enumerate(tqdm(samples)):
            try:
                prediction = self.predict(sample)
            except Exception:
                logger.exception("sample %d failed with an unhandled exception", i)
                prediction = {"status": "failed", "reason": "unhandled_exception", "fields": None}
            predictions.append(prediction)

        return predictions

if __name__ == "__main__":
    pipeline = IDExtractionPipeline()
    path = input("enter path").strip()
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    print(pipeline.predict())
