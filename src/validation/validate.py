"""
Validation module to perform structural checks on the output of inference
models (either the OCR cascade or the VLM fallback).
"""

from __future__ import annotations

import json
from enum import IntEnum

COMMON_FIELDS = {"side", "national_id"}
FRONT_ONLY_FIELDS = {"first_name", "last_name", "address", "address2"}
BACK_ONLY_FIELDS = {
    "issue_date", "expiration_date", "job_title",
    "gender", "religion", "marital_status",
}

VALID_GENDER_VALUES = {"ذكر", "أنثى"}
VALID_GOVERNORATE_CODES = {f"{i:02d}" for i in range(1, 28)} | {"88"}

_ARABIC_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_digits(text: str) -> str:
    """Normalize Arabic-Indic numerals to Western digits for validation."""
    return text.translate(_ARABIC_TO_WESTERN)


class Severity(IntEnum):
    OK = 0
    RETRY = 1
    REJECT = 2


ERRORS = {
    0: ("ok", Severity.OK),
    -1: ("invalid_json", Severity.RETRY),
    -2: ("missing_side_key", Severity.RETRY),
    -3: ("invalid_side_value", Severity.RETRY),
    -4: ("invalid_keys_for_side", Severity.RETRY),
    -10: ("invalid_national_id_characters", Severity.REJECT),
    -11: ("invalid_national_id_length", Severity.REJECT),
    -20: ("invalid_national_id_century", Severity.REJECT),
    -21: ("invalid_national_id_month", Severity.REJECT),
    -22: ("invalid_national_id_day", Severity.REJECT),
    -23: ("invalid_national_id_governorate", Severity.REJECT),
    -24: ("invalid_national_id_checksum", Severity.REJECT),
    -30: ("invalid_gender", Severity.REJECT),
}


def describe(code: int) -> str:
    return ERRORS.get(code, ("unknown_error", Severity.REJECT))[0]


def severity_of(code: int) -> Severity:
    return ERRORS.get(code, (None, Severity.REJECT))[1]


def requires_retry(code: int) -> bool:
    return severity_of(code) == Severity.RETRY


def expected_keys(side: str) -> set[str]:
    """The exact key set a valid result dict must have for a given side."""
    if side not in ("front", "back"):
        raise ValueError(f"side must be 'front' or 'back', got {side!r}")
    return COMMON_FIELDS | (FRONT_ONLY_FIELDS if side == "front" else BACK_ONLY_FIELDS)


def _validate_national_id(national_id) -> int:
    if not isinstance(national_id, str):
        return -10

    normalized = normalize_digits(national_id)

    if not normalized.isdigit():
        return -10
    if len(normalized) != 14:
        return -11

    century_digit = normalized[0]
    mm = normalized[3:5]
    dd = normalized[5:7]
    gov_code = normalized[7:9]

    if century_digit not in ("2", "3"):
        return -20
    if not (1 <= int(mm) <= 12):
        return -21
    if not (1 <= int(dd) <= 31):
        return -22
    if gov_code not in VALID_GOVERNORATE_CODES:
        return -23
    if not _checksum_valid(normalized):
        return -24

    return 0


def _checksum_valid(n_id: str) -> bool:
    weights = (2, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2)
    total = sum(int(d) * w for d, w in zip(n_id[:13], weights))
    check = 11 - (total % 11)
    check = 0 if check == 10 else (1 if check == 11 else check)
    return check == int(n_id[-1])


def validate_dict(parsed: dict) -> int:
    """Same checks as validate(), but on an already-parsed dict — avoids a
    pointless json.dumps/json.loads round trip when the caller already has
    the result in memory (e.g. the OCR-cascade path in detect_infer.py)."""
    if not isinstance(parsed, dict):
        return -1

    if "side" not in parsed:
        return -2

    side = parsed["side"]
    if side not in ("front", "back"):
        return -3

    if set(parsed.keys()) != expected_keys(side):
        return -4

    id_code = _validate_national_id(parsed["national_id"])
    if id_code != 0:
        return id_code

    if side == "back" and parsed.get("gender") not in VALID_GENDER_VALUES:
        return -30

    return 0


def validate(response: str) -> int:
    try:
        parsed = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return -1
    return validate_dict(parsed)