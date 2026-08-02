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

def convert_digits(text, to_eastern=True):
    western = "0123456789"
    eastern = "٠١٢٣٤٥٦٧٨٩"
    if to_eastern:
        table = str.maketrans(western, eastern)
    else:
        table = str.maketrans(eastern, western)

    return text.translate(table)

ERRORS = {
    -1: "invalid_json",
    -2: "invalid_keys",
    -3: "invalid_national_id_length",
    -4: "invalid_national_id_characters",
    -5: "invalid_gender",
    -6: "invalid_national_id_checksum"
}

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
