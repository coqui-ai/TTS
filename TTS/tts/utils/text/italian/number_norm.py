import re

# 1.234.567 -> 1234567  (Italian thousands separator)
_THOUSANDS_DOT_RE = re.compile(r"\b(\d{1,3})(\.\d{3})+\b")

# decimal comma: 3,14 -> 3 virgola 14
_DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")

# decimal dot (only when it doesn't look like thousands groups): 3.14 -> 3 virgola 14
_DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")


def normalize_numbers(text: str) -> str:
    # Remove thousands separators like 1.234.567
    def _rm_thousands(m: re.Match) -> str:
        return m.group(0).replace(".", "")

    text = _THOUSANDS_DOT_RE.sub(_rm_thousands, text)

    # Percent / currency / degree signs (simple, robust)
    text = re.sub(r"(\d)\s*%", r"\1 percento", text)
    text = re.sub(r"€\s*(\d+(?:[.,]\d+)?)", r"\1 euro", text)
    text = re.sub(r"(\d+(?:[.,]\d+)?)\s*€", r"\1 euro", text)
    text = re.sub(r"(\d+(?:[.,]\d+)?)\s*°C\b", r"\1 gradi celsius", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+(?:[.,]\d+)?)\s*°\b", r"\1 gradi", text)

    # Convert decimal separators between digits to spoken form.
    # Order matters: after removing thousands dots, remaining dot between digits is likely decimal.
    text = _DECIMAL_COMMA_RE.sub(" virgola ", text)
    text = _DECIMAL_DOT_RE.sub(" virgola ", text)

    return text
