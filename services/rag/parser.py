"""
services/rag/parser.py

Extract the minimum SoC buffer fraction from a policy text chunk.

The value is expected to follow the phrase
"minimum state of charge buffer of <value><unit>" but this module also
tolerates free-form occurrences of "<value> percent" so the agent layer
can call it on any text that contains a percentage.
"""
from __future__ import annotations

import re
from typing import Optional

_BUFFER_PHRASE = re.compile(
    r"minimum\s+state\s+of\s+charge\s+buffer\s+of\s+"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?|\.[0-9]+|[a-z\-]+)"
    r"\s*(?P<unit>%|percent)?",
    re.IGNORECASE,
)

_FREE_PERCENT = re.compile(
    r"(?P<value>[0-9]+(?:\.[0-9]+)?|\.[0-9]+|[a-z\-]+)"
    r"\s*(?P<unit>%|percent)\b",
    re.IGNORECASE,
)

_WORD_TO_NUMBER: dict[str, float] = {
    "five": 5.0,
    "ten": 10.0,
    "twenty": 20.0,
    "thirty": 30.0,
    "thirty-five": 35.0,
    "thirtyfive": 35.0,
}


def _to_float(token: str) -> Optional[float]:
    """Parse a numeric or word-form token into a float.

    Returns None if the token cannot be interpreted as one of the
    supported buffer values.
    """
    token = token.strip().lower().replace("_", "-")
    if not token:
        return None
    try:
        value = float(token)
    except ValueError:
        value = _WORD_TO_NUMBER.get(token)
    if value is None:
        return None
    return value


def _normalize(value: float, unit: Optional[str]) -> float:
    """Convert the raw value to a fraction in [0.0, 1.0]."""
    unit = (unit or "").lower()
    if unit in ("%", "percent"):
        return value / 100.0
    if value > 1.0:
        return value / 100.0
    return value


def _extract(text: str, pattern: re.Pattern[str]) -> Optional[float]:
    match = pattern.search(text)
    if not match:
        return None
    raw_value = match.group("value")
    unit = match.groupdict().get("unit")
    value = _to_float(raw_value)
    if value is None:
        return None
    return _normalize(value, unit)


def extract_buffer_constraint(text: str) -> Optional[float]:
    """Return the minimum SoC buffer fraction encoded in `text`.

    Searches for the canonical phrase "minimum state of charge buffer of"
    first; if that does not match, falls back to any "<n> percent" or
    "<n>%" mention. Returns None if no parseable value is found.
    """
    parsed = _extract(text, _BUFFER_PHRASE)
    if parsed is not None:
        return parsed
    return _extract(text, _FREE_PERCENT)
