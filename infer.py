from unsloth import FastVisionModel
import json
from PIL import Image, ImageEnhance
from tqdm.auto import tqdm

ERRORS = {
    -1: "invalid_json",
    -2: "invalid_keys",
    -3: "invalid_national_id_length",
    -4: "invalid_national_id_characters",
    -5: "invalid_gender",
    -6: "invalid_national_id_checksum"
}

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

def convert_digits(text, to_eastern=True):
    western = "0123456789"
    eastern = "٠١٢٣٤٥٦٧٨٩"
    if to_eastern:
        table = str.maketrans(western, eastern)
    else:
        table = str.maketrans(eastern, western)

    return text.translate(table)

def preprocess_image(image: Image.Image):

    grey_image = image.convert('L')
    enhancer = ImageEnhance.Contrast(grey_image)
    enhanced_image = enhancer.enhance(1.5)

    return enhanced_image


def infer(model, tokenizer, sample):

    preprocessed_image = preprocess_image(sample)

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
        preprocessed_image,
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

    input_length = inputs["input_ids"].shape[1]  # calculate input token length to skip in output
    inference = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)

    return inference


def batch_infer(model, tokenizer, samples):
    predictions = []

    for sample in tqdm(samples):
        prediction = infer(model, tokenizer, sample)
        predictions.append(prediction)

    return predictions


def validate(response):

    try:
        parsed_response = json.loads(response)
    except:
        return -1
    
    if set(parsed_response.keys()) != set(field_structure.keys()):
        return -2
    
    national_id_english = convert_digits(parsed_response['national_id'])

    if len(national_id_english) != 14:
        return -3

    national_id_english = convert_digits(parsed_response['national_id'], to_eastern=False)

    if len(national_id_english) != 14:
        return -3

    def __validate_checksum(n_id: str) -> bool:
        w = (2, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2)
        t = sum(int(d) * w[i] for i, d in enumerate(n_id[:13]))
        k = 11 - t % 11
        k = 0 if k == 10 else (1 if k == 11 else k)
        return k == int(n_id[-1])

    if not __validate_checksum(national_id_english):
        return -6
    
    try:
        int(parsed_response['national_id'])
    except:
        return -4

    if "front" in parsed_response:
        ...
    else:
        if parsed_response['gender'] not in {'ذكر', 'أنثي'}:
            return -5
    
    return 0

    

model, tokenizer = FastVisionModel.from_pretrained("./models/model_name", load_in_4bit=True)
FastVisionModel.for_inference(model)