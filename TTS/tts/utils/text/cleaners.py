'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
from unidecode import unidecode
from .number_norm import normalize_numbers
from .abbreviations import abbreviations_en, abbreviations_fr
from .time import expand_time_english
from TTS.tts.utils.chinese_mandarin.numbers import replace_numbers_to_characters_in_text


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')


def expand_abbreviations(text, lang='en'):
    if lang == 'en':
        _abbreviations = abbreviations_en
    elif lang == 'fr':
        _abbreviations = abbreviations_fr
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text).strip()


def convert_to_ascii(text):
    return unidecode(text)


def remove_aux_symbols(text):
    text = re.sub(r'[\<\>\(\)\[\]\"]+', '', text)
    return text

def replace_symbols(text, lang='en'):
    text = text.replace(';', ',')
    text = text.replace('-', ' ')
    text = text.replace(':', ',')
    if lang == 'en':
        text = text.replace('&', ' and ')
    elif lang == 'fr':
        text = text.replace('&', ' et ')
    elif lang == 'pt':
        text = text.replace('&', ' e ')
    return text

def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def basic_german_cleaners(text):
    '''Pipeline for German text'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


# TODO: elaborate it
def basic_turkish_cleaners(text):
    '''Pipeline for Turkish text'''
    text = text.replace("I", "ı")
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_time_english(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = replace_symbols(text)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text


def french_cleaners(text):
    '''Pipeline for French text. There is no need to expand numbers, phonemizer already does that'''
    text = expand_abbreviations(text, lang='fr')
    text = lowercase(text)
    text = replace_symbols(text, lang='fr')
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text


def portuguese_cleaners(text):
    '''Basic pipeline for Portuguese text. There is no need to expand abbreviation and
        numbers, phonemizer already does that'''
    text = lowercase(text)
    text = replace_symbols(text, lang='pt')
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text


def chinese_mandarin_cleaners(text: str) -> str:
    '''Basic pipeline for chinese'''
    text = replace_numbers_to_characters_in_text(text)
    return text


def phoneme_cleaners(text):
    '''Pipeline for phonemes mode, including number and abbreviation expansion.'''
    text = expand_numbers(text)
    text = convert_to_ascii(text)
    text = expand_abbreviations(text)
    text = replace_symbols(text)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text
