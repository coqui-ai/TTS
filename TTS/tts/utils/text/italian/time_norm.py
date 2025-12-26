import re

_TIME_RE = re.compile(r"\b([01]?\d|2[0-3])[:\.]([0-5]\d)\b")

# 0..59 (sufficient for minutes)
_IT_0_19 = [
    "zero",
    "uno",
    "due",
    "tre",
    "quattro",
    "cinque",
    "sei",
    "sette",
    "otto",
    "nove",
    "dieci",
    "undici",
    "dodici",
    "tredici",
    "quattordici",
    "quindici",
    "sedici",
    "diciassette",
    "diciotto",
    "diciannove",
]
_IT_TENS = {
    20: "venti",
    30: "trenta",
    40: "quaranta",
    50: "cinquanta",
}


def _it_number_0_59(n: int) -> str:
    if 0 <= n < 20:
        return _IT_0_19[n]
    tens = (n // 10) * 10
    unit = n % 10
    if tens not in _IT_TENS:
        return str(n)
    if unit == 0:
        return _IT_TENS[tens]
    # Keep it simple (no elision rules like "ventuno/ventotto"); XTTS usually handles both.
    return f"{_IT_TENS[tens]} { _IT_0_19[unit] }"


def expand_time_italian(text: str) -> str:
    """
    09:00 -> 9
    09:05 -> 9 e cinque
    14.30 -> 14 e trenta
    """

    def repl(m: re.Match) -> str:
        hh = int(m.group(1))
        mm = int(m.group(2))
        if mm == 0:
            return f"{hh}"
        return f"{hh} e {_it_number_0_59(mm)}"

    return _TIME_RE.sub(repl, text)
