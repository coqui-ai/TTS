import json
import os
import re

import inflect
import pandas as pd
import pypinyin
import torch
from num2words import num2words
from tokenizers import Tokenizer
from unidecode import unidecode

from TTS.tts.utils.text.cleaners import english_cleaners

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    text = text.replace('"', "")
    return text


def expand_numbers_multilang(text, lang):
    # TODO: Handle text more carefully. Currently, it just converts numbers without any context.
    # Find all numbers in the input string
    numbers = re.findall(r"\d+", text)

    # Transliterate the numbers to text
    for num in numbers:
        transliterated_num = "".join(num2words(num, lang=lang))
        text = text.replace(num, transliterated_num, 1)

    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def multilingual_cleaners(text, lang):
    text = lowercase(text)
    text = expand_numbers_multilang(text, lang)
    text = collapse_whitespace(text)
    text = text.replace('"', "")
    if lang == "tr":
        text = text.replace("İ", "i")
        text = text.replace("Ö", "ö")
        text = text.replace("Ü", "ü")
    return text


def english_cleaners(text):
    """Pipeline for English text, including number and abbreviation expansion."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    text = text.replace('"', "")
    return text


def remove_extraneous_punctuation(word):
    replacement_punctuation = {"{": "(", "}": ")", "[": "(", "]": ")", "`": "'", "—": "-", "—": "-", "`": "'", "ʼ": "'"}
    replace = re.compile(
        "|".join([re.escape(k) for k in sorted(replacement_punctuation, key=len, reverse=True)]), flags=re.DOTALL
    )
    word = replace.sub(lambda x: replacement_punctuation[x.group(0)], word)

    # TODO: some of these are spoken ('@', '%', '+', etc). Integrate them into the cleaners.
    extraneous = re.compile(r"^[@#%_=\$\^&\*\+\\]$")
    word = extraneous.sub("", word)
    return word


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


_whitespace_re = re.compile(r"\s+")


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def arabic_cleaners(text):
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def chinese_cleaners(text):
    text = lowercase(text)
    text = "".join(
        [p[0] for p in pypinyin.pinyin(text, style=pypinyin.Style.TONE3, heteronym=False, neutral_tone_with_five=True)]
    )
    return text


class VoiceBpeTokenizer:
    def __init__(self, vocab_file=None, preprocess=None):
        self.tokenizer = None

        if vocab_file is not None:
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = json.load(f)

            self.language = vocab["model"]["language"] if "language" in vocab["model"] else None

            if preprocess is None:
                self.preprocess = "pre_tokenizer" in vocab and vocab["pre_tokenizer"]
            else:
                self.preprocess = preprocess

            self.tokenizer = Tokenizer.from_file(vocab_file)

    def preprocess_text(self, txt, lang):
        if lang == "ja":
            import pykakasi

            kks = pykakasi.kakasi()
            results = kks.convert(txt)
            txt = " ".join([result["kana"] for result in results])
            txt = basic_cleaners(txt)
        elif lang == "en":
            txt = english_cleaners(txt)
        elif lang == "ar":
            txt = arabic_cleaners(txt)
        elif lang == "zh-cn":
            txt = chinese_cleaners(txt)
        else:
            txt = multilingual_cleaners(txt, lang)
        return txt

    def encode(self, txt, lang):
        if self.preprocess:
            txt = self.preprocess_text(txt, lang)
        txt = txt.replace(" ", "[SPACE]")
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(" ", "")
        txt = txt.replace("[SPACE]", " ")
        txt = txt.replace("[STOP]", "")
        txt = txt.replace("[UNK]", "")
        return txt
