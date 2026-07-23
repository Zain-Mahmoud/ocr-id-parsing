from unsloth import FastVisionModel
import json 
from PIL import Image, ImageEnhance
from tqdm.auto import tqdm

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


def infer(model, tokenizer, sample: Image):
    FastVisionModel.for_inference(model)

    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image'
                }, 
                {
                    'type': 'text',
                    'text': USER_PROMPT
                }
            ]
        }
    ]
    preprocessed_image = preprocess_image(sample)
    input_text = tokenizer.apply_chat_templates(messages, add_generation_prompt=True)
    inputs = tokenizer(preprocessed_image, input_text, add_special_tokens=False, return_tensors='pt').to(model.device)

    inference = model.generate(**inputs, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1)

    return inference


def batch_infer(model, tokenizer, samples):
    FastVisionModel.for_inference(model)
    predictions = []

    for sample in tqdm(samples):
        prediction = infer(model, tokenizer, sample)
        predictions.append(prediction)

    return predictions


model, tokenzier = FastVisionModel.from_pretrained("./models/model_name", load_in_4bit=True)
