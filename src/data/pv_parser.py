"""Parse raw property-value strings while retaining multi-valued attributes."""

import re
import unicodedata


def normalize_text(value):
    """Normalize width and whitespace; never invent missing specifications."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def parse_pv(raw):
    """Return (attributes, issues); preserve distinct values in input order."""
    attributes, issues = {}, []
    if not isinstance(raw, str):
        return {}, ["pv_not_string"]
    if not raw.strip():
        return {}, ["pv_empty"]
    for part in raw.split("#;#"):
        if "#:#" not in part:
            issues.append("missing_key_value_separator")
            continue
        key, value = (normalize_text(x) for x in part.split("#:#", 1))
        if not key or not value:
            issues.append("empty_key_or_value")
            continue
        values = attributes.setdefault(key, [])
        if value not in values:
            values.append(value)
    return attributes, issues
