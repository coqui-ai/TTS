# -*- coding: utf-8 -*-
'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''
from utils.text import cmudict

_pad = '_'
_eos = '~'
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '
_punctuations = '!\'(),-.:;? '
_phonemes = [
    '$',
    '&',
    'a',
    'b',
    'd',
    'd͡ʒ',
    'e',
    'f',
    'h',
    'i',
    'j',
    'k',
    'l',
    'm',
    'n',
    'o',
    'p',
    's',
    't',
    't͡ʃ',
    'u',
    'v',
    'w',
    'z',
    '£',
    'à',
    'â',
    'æ',
    'è',
    'é',
    'ê',
    'ð',
    'ü',
    'ŋ',
    'ɑ',
    'ɔ',
    'ə',
    'ɛ',
    'ɡ',
    'ɪ',
    'ɹ',
    'ɹ̩',
    'ʃ',
    'ʊ',
    'ʌ',
    'ʒ',
    'θ'
]
_phonemes = sorted(list(set(_phonemes)))

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in _phonemes]

# Export all symbols:
symbols = [_pad, _eos] + list(_characters) + _arpabet
phonemes = [_pad, _eos] + list(_phonemes) + list(_punctuations)

if __name__ == '__main__':
    print(" > TTS symbols ")
    print(symbols)
    print(" > TTS phonemes ")
    print(phonemes)
