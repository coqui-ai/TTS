# -*- coding: utf-8 -*-

import re
import phonemizer
from phonemizer.phonemize import phonemize
from utils.text import cleaners
from utils.text.symbols import symbols, phonemes, _phoneme_punctuations

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

_phonemes_to_id = {s: i for i, s in enumerate(phonemes)}
_id_to_phonemes = {i: s for i, s in enumerate(phonemes)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

# Regular expression matchinf punctuations, ignoring empty space
pat = r'['+_phoneme_punctuations[:-1]+']+'


def text2phone(text, language):
    '''
    Convert graphemes to phonemes.
    '''
    seperator = phonemizer.separator.Separator(' |', '', '|')
    #try:
    punctuations = re.findall(pat, text)
    ph = phonemize(text, separator=seperator, strip=False, njobs=1, backend='espeak', language=language)
    # Replace \n with matching punctuations.
    if len(punctuations) > 0:
        for punct in punctuations[:-1]:
             ph = ph.replace('| |\n', '|'+punct+'| |', 1)
        try:
             ph = ph[:-1] + punctuations[-1]
        except:
             print(text)
    return ph


def phoneme_to_sequence(text, cleaner_names, language):
    '''
    TODO: This ignores punctuations
    '''
    sequence = []
    clean_text = _clean_text(text, cleaner_names)
    phonemes = text2phone(clean_text, language)
#     print(phonemes.replace('|', ''))
    if phonemes is None:
        print("!! After phoneme conversion the result is None. -- {} ".format(clean_text))
    for phoneme in phonemes.split('|'):
        # print(word, ' -- ', phonemes_text)
        sequence += _phoneme_to_sequence(phoneme)
    # Append EOS char
    sequence.append(_phonemes_to_id['~'])
    return sequence


def sequence_to_phoneme(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_phonemes:
            s = _id_to_phonemes[symbol_id]
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
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(
            _clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    # Append EOS token
    sequence.append(_symbol_to_id['~'])
    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
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


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _phoneme_to_sequence(phonemes):
    return [_phonemes_to_id[s] for s in list(phonemes) if _should_keep_phoneme(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s is not '_' and s is not '~'


def _should_keep_phoneme(p):
    return p in _phonemes_to_id and p is not '_' and p is not '~'
