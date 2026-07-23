from unsloth import FastVisionModel
import json 
from PIL import Image, ImageEnhance
from tqdm.auto import tqdm
from ultralytics import YOLO
import numpy as np
import cv2

field_structure = {
    "first_name": "string (arabic)",
    "last_name": "string (arabic)",
    "national_id": "string, 14 digits",
    "address": "string (arabic)",
    "address2": "string (arabic)",
    "birthdate": "string, formatted date (arabic)",
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
    You are given two sides of an Egyptian National ID, front and back.
    Extract all the fields, regardless of the side they are found on, out of the ID 
    following this format: {field_structure}.  The key order does not matter. Return all 
    fields as they appear and do not make any changes or updates to any of the fields. Return in json format.
'''


def preprocess_image(image: Image):

    grey_image = image.convert('L')
    enhancer = ImageEnhance.Contrast(grey_image)
    enhanced_image = enhancer.enhance(1.5)

    return enhanced_image

def load_yolo():
    model = YOLO("/path/to/trained/model.onnx")

    return model

def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)

def detect(model: YOLO, image):

    results = model(image)
    result = results[0]

    if result.masks is None:
        return None

    original = result.orig_img

    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    boxes = result.boxes

    best_idx = int(boxes.conf.argmax())

    mask = result.masks.data[best_idx]

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

def generate(model, tokenizer, image_front, image_back):

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
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT}
            ]
        }
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    inputs = tokenizer(
        [image_front, image_back],  
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
    return inference


def infer(detection_model: YOLO, generation_model, tokenizer, front, back):

    detected_front, detected_back = detect(detection_model, front), detect(detection_model, back)
    while detected_front is None or detected_back is None:  # TODO might be too expensive to keep running
        detected_front, detected_back = detected_back(detection_model, generation_model, tokenizer, front, back)
    preprocessed_front, preprocessed_back = preprocess_image(detected_front), preprocess_image(detected_back)
    inference = generate(generation_model, tokenizer, preprocessed_front, preprocessed_back)

    return inference


def batch_infer(model, tokenizer, samples):
    
    predictions = []

    for sample in tqdm(samples):
        prediction = infer(model, tokenizer, sample[0], sample[1])
        predictions.append(prediction)

    return predictions


model, tokenizer = FastVisionModel.from_pretrained("./models/model_name", load_in_4bit=True)
FastVisionModel.for_inference(model)