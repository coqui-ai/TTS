import re

import inflect

_inflect = inflect.engine()

_time_re = re.compile(
    r"""\b
                          ((0?[0-9])|(1[0-1])|(1[2-9])|(2[0-3]))  # hours
                          :
                          ([0-5][0-9])                            # minutes
                          \s*(a\\.m\\.|am|pm|p\\.m\\.|a\\.m|p\\.m)? # am/pm
                          \b""",
    re.IGNORECASE | re.X,
)


def _expand_num(n: int) -> str:
    return _inflect.number_to_words(n)


def _expand_time_english(match: "re.Match") -> str:
    hour = int(match.group(1))
    past_noon = hour >= 12
    time = []
    if hour > 12:
        hour -= 12
    elif hour == 0:
        hour = 12
        past_noon = True
    time.append(_expand_num(hour))

    minute = int(match.group(6))
    if minute > 0:
        if minute < 10:
            time.append("oh")
        time.append(_expand_num(minute))
    am_pm = match.group(7)
    if am_pm is None:
        time.append("p m" if past_noon else "a m")
    else:
        time.extend(list(am_pm.replace(".", "")))
    return " ".join(time)


def expand_time_english(text: str) -> str:
    return re.sub(_time_re, _expand_time_english, text)
