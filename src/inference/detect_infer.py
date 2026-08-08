"""
Inference pipeline combining YOLO26 segmentation, OpenCV orientations
and Qwen3-2b-VL inference
"""

from unsloth import FastVisionModel
from PIL import Image, ImageEnhance
from tqdm.auto import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np
import cv2
import json

field_structure = {
    "side": "string 'front' or 'back'",
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
    "marital_status": "string, 'single', 'married' or 'widow' (arabic)"
}

SYSTEM_PROMPT = f'''
    You are a Vision Language Model tasked with extracted field values from an Egyptian national identity
    document. You must extract the fields without making any changes to the fields and return them
    as they are in Arabic script. If there is something you cannot extract, do not attempt to infer it based
    on other information.
'''
USER_PROMPT = f'''
    You are given one side of an Egyptian National ID, either front or back.
    Extract all the fields that are on this side out of the ID and report which side it was
    following this format: {field_structure}.  The key order does not matter. Return all 
    fields that you detect as they appear and do not make any changes or updates to any of the fields. Return in json format.
'''


def preprocess_image(image: Image.Image):
    grey_image = image.convert('L')
    enhancer = ImageEnhance.Contrast(grey_image)
    enhanced_image = enhancer.enhance(1.5)

    return enhanced_image

def load_yolo():
    model = YOLO("./models/yolo/best.onnx")
    return model

def load_vlm():
    generation_model, tokenizer = FastVisionModel.from_pretrained("./models/model_name", load_in_4bit=True)
    FastVisionModel.for_inference(generation_model)
    return generation_model, tokenizer

def load():
    yolo_model = load_yolo()
    vlm_model, vlm_tokenizer = load_vlm()
    return {"yolo": yolo_model, "vlm": (vlm_model, vlm_tokenizer)}


def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)

def orient(segmented_image):

    if segmented_image.masks is None:
        return None

    original = segmented_image.orig_img
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    boxes = segmented_image.boxes
    best_idx = int(boxes.conf.argmax())

    mask = segmented_image.masks.data[best_idx]
    mask = mask.cpu().numpy()

    mask = cv2.resize(
        mask,
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    binary = (mask > 0.5).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contour = max(contours, key=cv2.contourArea)

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True,
    )

    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(contour)
        pts = cv2.boxPoints(rect)

    pts = order_points(pts)

    (top_left, top_right, bottom_right, bottom_left) = pts

    widthA = np.linalg.norm(bottom_right - bottom_left)
    widthB = np.linalg.norm(top_right - top_left)
    maxWidth = int(round(max(widthA, widthB)))

    heightA = np.linalg.norm(top_right - bottom_right)
    heightB = np.linalg.norm(top_left - bottom_left)
    maxHeight = int(round(max(heightA, heightB)))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)

    warped = cv2.warpPerspective(
        original,
        M,
        (maxWidth, maxHeight),
        flags=cv2.INTER_CUBIC,
    )

    final_image = Image.fromarray(warped)
    return final_image

def get_line_crop(image: Image.Image, box):
    xc, yc, w, h, r = box
    center = (float(xc), float(yc))

    M = cv2.getRotationMatrix2D(center=center, angle=r, scale=1.0) 
    rotated = cv2.warpAffine(image, M, image.size[::-1], flags=cv2.INTER_LINEAR)

    left = min(xc-w / 2, 0)
    right = max(xc+w / 2, image.width)
    top = min(yc-h / 2, 0)
    bottom = max(yc+h / 2, image.height)

    line_crop = rotated[top:bottom, left:right]
    return line_crop

def extract(result: Results) -> dict:
    classes = result.names

    obb = result.obb
    orig_image = result.orig_image
    extracts = {}

    for i in range(len(obb.data)):
        curr_class_idx = obb.cls[i].item()
        curr_class_label = classes[curr_class_idx]
        curr_conf = obb.conf[i].item()
        curr_xyhwr = obb.xywhr[i]

        extracts[curr_class_label] = {"conf": curr_conf, "line_crop": get_line_crop(orig_image, curr_xyhwr)}


def detect(model: YOLO, image):

    results = model(image)[0]
    extracted_results = extract(results)

    return 



def vlm_inference(model, tokenizer, image):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT}
            ]
        }
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=256,
        use_cache=True,
        do_sample=False,
    )

    input_length = inputs["input_ids"].shape[1]
    inference = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
    inference = json.loads(inference)

    return inference


def infer(detection_model: YOLO, generation_model, tokenizer, image):

    preprocessed_image = preprocess_image(image)

    detected = detect(detection_model, preprocessed_image)

    if detected is None:
        raise ValueError("No ID card detected in this image — YOLO segmentation returned no mask.")


    inference = vlm_inference(generation_model, tokenizer, preprocessed_image)

    return inference


def batch_infer(detection_model, generation_model, tokenizer, samples):

    predictions = []

    for sample in tqdm(samples):
        prediction = infer(detection_model, generation_model, tokenizer, sample)
        predictions.append(prediction)

    return predictions



if __name__ == "__main__":
    models = load()
    yolo = models['yolo']
    vlm_generation, vlm_tokenizer = models['vlm']
