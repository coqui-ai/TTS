# -*- coding: utf-8 -*-
"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from coqpit import Coqpit, check_argument


def make_symbols(
    characters, phonemes=None, punctuations="!'(),-.:;? ", pad="_", eos="~", bos="^", unique=True,
):  # pylint: disable=redefined-outer-name
    """Function to create symbols and phonemes
    TODO: create phonemes_to_id and symbols_to_id dicts here."""
    _symbols = list(characters)
    _symbols = [bos] + _symbols if len(bos) > 0 and bos is not None else _symbols
    _symbols = [eos] + _symbols if len(bos) > 0 and eos is not None else _symbols
    _symbols = [pad] + _symbols if len(bos) > 0 and pad is not None else _symbols
    _phonemes = None
    if phonemes is not None:
        _phonemes_sorted = (
            sorted(list(set(phonemes))) if unique else sorted(list(phonemes))
        )  # this is to keep previous models compatible.
        # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
        _arpabet = ["@" + s for s in _phonemes_sorted]
        # Export all symbols:
        _phonemes = [pad, eos, bos] + list(_phonemes_sorted) + list(punctuations)
        _symbols += _arpabet
    return _symbols, _phonemes


_pad = "_"
_eos = "~"
_bos = "^"
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;? "
_punctuations = "!'(),-.:;? "

# Phonemes definition (All IPA characters)
_vowels = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ"
_non_pulmonic_consonants = "ʘɓǀɗǃʄǂɠǁʛ"
_pulmonic_consonants = "pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ"
_suprasegmentals = "ˈˌːˑ"
_other_symbols = "ʍwɥʜʢʡɕʑɺɧʲ"
_diacrilics = "ɚ˞ɫ"
_phonemes = _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics

symbols, phonemes = make_symbols(_characters, _phonemes, _punctuations, _pad, _eos, _bos)

# Generate ALIEN language
# from random import shuffle
# shuffle(phonemes)


def parse_symbols():
    return {
        "pad": _pad,
        "eos": _eos,
        "bos": _bos,
        "characters": _characters,
        "punctuations": _punctuations,
        "phonemes": _phonemes,
    }


class SymbolEmbedding:
    @dataclass
    class SymbolEmbeddingJSON(Coqpit):
        embeddings: Dict[str, List[int]] = field(default_factory=lambda: {"A": [0.0]})

    def __init__(self, symbol_embedding_filename):
        self.symbol_index_lut = {}
        self.weight_matrix = None
        self.load_symbol_embedding(symbol_embedding_filename)

    def __getitem__(self, x):
        return self.symbol_index_lut[x]

    def symbols(self):
        return self.symbol_index_lut.keys()

    def num_symbols(self):
        return len(self.symbol_index_lut)

    def embedding_size(self):
        return self.weight_matrix.shape[1]

    """
    Fills in symbol embedding object from filepath with to JSON
    """

    def load_symbol_embedding(self, filename):
        symbol_embedding_json = self.SymbolEmbeddingJSON()
        symbol_embedding_json.load_json(filename)

        embeddings = []

        for index, (symbol, embedding) in enumerate(symbol_embedding_json.embeddings.items()):
            self.symbol_index_lut[symbol] = index
            # embedding = symbol_embedding_json.embeddings[symbol]
            embeddings.append(np.array(embedding))

        self.weight_matrix = np.vstack(embeddings)


if __name__ == "__main__":
    print(" > TTS symbols {}".format(len(symbols)))
    print(symbols)
    print(" > TTS phonemes {}".format(len(phonemes)))
    print("".join(sorted(phonemes)))
