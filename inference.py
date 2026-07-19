from unsloth import FastVisionModel
import json 

field_structure = {
    "full_name": "string (arabic)",
    "national_id": "string, 14 digits",
    "address": "string (arabic)",
}

USER_PROMPT = f'''
    Extract all the fields out of this Egyptian national ID following this format: {field_structure}.
    Return all fields as they appear and do not make any changes or updates to any of the fields. Return in
    in json format
'''


model, tokenzier = FastVisionModel.from_pretrained("./models/model_name", load_in_4bit=False)


def infer(model, tokenizer, sample):
    
    FastVisionModel.for_infer(model)

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

    input_text = tokenizer.apply_chat_templates(messages, add_generation_prompt=True)
    inputs = tokenizer(sample, input_text, add_special_tokens=False, return_tensors='pt').to(model.device)

    inference = model.generate(**inputs, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1)

    return inference


def batch_infer(model, tokenizer, samples):
    FastVisionModel.for_infer(model)
    predictions = []
    for sample in samples:
        try:
            prediction = json.loads(infer(model, tokenizer, sample['image']))
        except json.JSONDecodeError:
            prediction = None
        predictions.append(prediction)
    return predictions
