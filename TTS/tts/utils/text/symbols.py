# -*- coding: utf-8 -*-
'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''
def make_symbols(characters, phonemes, punctuations='!\'(),-.:;? ', pad='_', eos='~', bos='^'):# pylint: disable=redefined-outer-name
    ''' Function to create symbols and phonemes '''
    _phonemes_sorted = sorted(list(phonemes))

    # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
    _arpabet = ['@' + s for s in _phonemes_sorted]

    # Export all symbols:
    _symbols = [pad, eos, bos] + list(characters) + _arpabet
    _phonemes = [pad, eos, bos] + list(_phonemes_sorted) + list(punctuations)

    return _symbols, _phonemes

_pad = '_'
_eos = '~'
_bos = '^'
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '
_punctuations = '!\'(),-.:;? '
_phoneme_punctuations = '.!;:,?'

# Phonemes definition
_vowels = 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ'
_non_pulmonic_consonants = 'ʘɓǀɗǃʄǂɠǁʛ'
_pulmonic_consonants = 'pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ'
_suprasegmentals = 'ˈˌːˑ'
_other_symbols = 'ʍwɥʜʢʡɕʑɺɧ'
_diacrilics = 'ɚ˞ɫ'
_phonemes = _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics

symbols, phonemes = make_symbols(_characters, _phonemes, _punctuations, _pad, _eos, _bos)

# Generate ALIEN language
# from random import shuffle
# shuffle(phonemes)

if __name__ == '__main__':
    print(" > TTS symbols {}".format(len(symbols)))
    print(symbols)
    print(" > TTS phonemes {}".format(len(phonemes)))
    print(phonemes)
