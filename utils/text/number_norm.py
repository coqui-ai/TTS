import re

_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'([0-9]+)(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')

_units = [
    '', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
    'seventeen', 'eighteen', 'nineteen'
]

_tens = [
    '',
    'ten',
    'twenty',
    'thirty',
    'forty',
    'fifty',
    'sixty',
    'seventy',
    'eighty',
    'ninety',
]

_digit_groups = [
    '',
    'thousand',
    'million',
    'billion',
    'trillion',
    'quadrillion',
]

_ordinal_suffixes = [
    ('one', 'first'),
    ('two', 'second'),
    ('three', 'third'),
    ('five', 'fifth'),
    ('eight', 'eighth'),
    ('nine', 'ninth'),
    ('twelve', 'twelfth'),
    ('ty', 'tieth'),
]


def _remove_commas(m):
    return m.group(1).replace(',', '')


def _expand_decimal_point(m):
    return m.group(1).replace('.', ' point ')


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    if dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    if cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    return 'zero dollars'


def _standard_number_to_words(n, digit_group):
    parts = []
    if n >= 1000:
        # Format next higher digit group.
        parts.append(_standard_number_to_words(n // 1000, digit_group + 1))
        n = n % 1000

    if n >= 100:
        parts.append('%s hundred' % _units[n // 100])
    if n % 100 >= len(_units):
        parts.append(_tens[(n % 100) // 10])
        parts.append(_units[(n % 100) % 10])
    else:
        parts.append(_units[n % 100])
    if n > 0:
        parts.append(_digit_groups[digit_group])
    return ' '.join([x for x in parts if x])


def _number_to_words(n):
    # Handle special cases first, then go to the standard case:
    if n >= 1000000000000000000:
        return str(n)  # Too large, just return the digits
    if n == 0:
        return 'zero'
    if n % 100 == 0 and n % 1000 != 0 and n < 3000:
        return _standard_number_to_words(n // 100, 0) + ' hundred'
    return _standard_number_to_words(n, 0)


def _expand_number(m):
    return _number_to_words(int(m.group(0)))


def _expand_ordinal(m):
    num = _number_to_words(int(m.group(1)))
    for suffix, replacement in _ordinal_suffixes:
        if num.endswith(suffix):
            return num[:-len(suffix)] + replacement
    return num + 'th'


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text
