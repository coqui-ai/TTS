# -*- coding: utf-8 -*-

import re
import phonemizer
from phonemizer.phonemize import phonemize
from TTS.utils.text import cleaners
from TTS.utils.text.symbols import symbols, phonemes, _phoneme_punctuations, _bos, \
    _eos

# Mappings from symbol to numeric ID and vice versa:
_SYMBOL_TO_ID = {s: i for i, s in enumerate(symbols)}
_ID_TO_SYMBOL = {i: s for i, s in enumerate(symbols)}

_PHONEMES_TO_ID = {s: i for i, s in enumerate(phonemes)}
_ID_TO_PHONEMES = {i: s for i, s in enumerate(phonemes)}

# Regular expression matching text enclosed in curly braces:
_CURLY_RE = re.compile(r'(.*?)\{(.+?)\}(.*)')

# Regular expression matching punctuations, ignoring empty space
PHONEME_PUNCTUATION_PATTERN = r'['+_phoneme_punctuations+']+'


def text2phone(text, language):
    '''
    Convert graphemes to phonemes.
    '''
    seperator = phonemizer.separator.Separator(' |', '', '|')
    #try:
    punctuations = re.findall(PHONEME_PUNCTUATION_PATTERN, text)
    ph = phonemize(text, separator=seperator, strip=False, njobs=1, backend='espeak', language=language)
    ph = ph[:-1].strip() # skip the last empty character
    # Replace \n with matching punctuations.
    if punctuations:
        # if text ends with a punctuation.
        if text[-1] == punctuations[-1]:
            for punct in punctuations[:-1]:
                ph = ph.replace('| |\n', '|'+punct+'| |', 1)
            try:
                ph = ph + punctuations[-1]
            except:
                print(text)
        else:
            for punct in punctuations:
                ph = ph.replace('| |\n', '|'+punct+'| |', 1)
    return ph


def pad_with_eos_bos(phoneme_sequence):
    return [_PHONEMES_TO_ID[_bos]] + list(phoneme_sequence) + [_PHONEMES_TO_ID[_eos]]


def phoneme_to_sequence(text, cleaner_names, language, enable_eos_bos=False):
    sequence = []
    text = text.replace(":", "")
    clean_text = _clean_text(text, cleaner_names)
    to_phonemes = text2phone(clean_text, language)
    if to_phonemes is None:
        print("!! After phoneme conversion the result is None. -- {} ".format(clean_text))
    # iterate by skipping empty strings - NOTE: might be useful to keep it to have a better intonation.
    for phoneme in filter(None, to_phonemes.split('|')):
        sequence += _phoneme_to_sequence(phoneme)
    # Append EOS char
    if enable_eos_bos:
        sequence = pad_with_eos_bos(sequence)
    return sequence


def sequence_to_phoneme(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _ID_TO_PHONEMES:
            s = _ID_TO_PHONEMES[symbol_id]
            result += s
    return result.replace('}{', ' ')


def text_to_sequence(text, cleaner_names):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

      The text can optionally have ARPAbet sequences enclosed in curly braces embedded
      in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

      Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through

      Returns:
        List of integers corresponding to the symbols in the text
    '''
    sequence = []
    # Check for curly braces and treat their contents as ARPAbet:
    while text:
        m = _CURLY_RE.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(
            _clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)
    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _ID_TO_SYMBOL:
            s = _ID_TO_SYMBOL[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s
    return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(syms):
    return [_SYMBOL_TO_ID[s] for s in syms if _should_keep_symbol(s)]


def _phoneme_to_sequence(phons):
    return [_PHONEMES_TO_ID[s] for s in list(phons) if _should_keep_phoneme(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _SYMBOL_TO_ID and s not in ['~', '^', '_']


def _should_keep_phoneme(p):
    return p in _PHONEMES_TO_ID and p not in ['~', '^', '_']
