from unsloth import FastVisionModel

field_structure = {
    "full_name": "string (arabic)",
    "national_id": "string, 14 digits",
    "address": "string (arabic)",
}

USER_PROMPT = f'''
    Extract all the fields out of this Egyptian national ID following this format: {field_structure}.
    Return all fields as they appear and do not make any changes or updates to any of the fields.
'''


model, tokenzier = FastVisionModel.from_pretrained("./models/model_name", load_in_4bit=False)
FastVisionModel.for_inference(model)


def infer(model, tokenizer, sample):
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
    inputs = tokenizer(sample, input_text, add_special_tokens=False, return_tensors='pt').to('cuda')

    inference = model.generate(**inputs, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1)

    return inference


