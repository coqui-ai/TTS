# -*- coding: utf-8 -*-
'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''


def make_symbols(characters, phonemes=None, punctuations='!\'(),-.:;? ', pad='_', eos='~', bos='^'):# pylint: disable=redefined-outer-name
    ''' Function to create symbols and phonemes '''
    _symbols = [pad, eos, bos] + list(characters)
    _phonemes = None
    if phonemes is not None:
        _phonemes_sorted = sorted(list(set(phonemes)))
        # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
        _arpabet = ['@' + s for s in _phonemes_sorted]
        # Export all symbols:
        _phonemes = [pad, eos, bos] + list(_phonemes_sorted) + list(punctuations)
        _symbols += _arpabet
    return _symbols, _phonemes

_pad = '_'
_eos = '~'
_bos = '^'
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '
_punctuations = '!\'(),-.:;? '

# Phonemes definition (All IPA characters)
_vowels = 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ'
_non_pulmonic_consonants = 'ʘɓǀɗǃʄǂɠǁʛ'
_pulmonic_consonants = 'pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ'
_suprasegmentals = 'ˈˌːˑ'
_other_symbols = 'ʍwɥʜʢʡɕʑɺɧʲ'
_diacrilics = 'ɚ˞ɫ'
_phonemes = _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics

symbols, phonemes = make_symbols(_characters, _phonemes, _punctuations, _pad, _eos, _bos)

# Generate ALIEN language
# from random import shuffle
# shuffle(phonemes)


def parse_symbols():
    return {'pad': _pad,
            'eos': _eos,
            'bos': _bos,
            'characters': _characters,
            'punctuations': _punctuations,
            'phonemes': _phonemes}


if __name__ == '__main__':
    print(" > TTS symbols {}".format(len(symbols)))
    print(symbols)
    print(" > TTS phonemes {}".format(len(phonemes)))
    print(''.join(sorted(phonemes)))
