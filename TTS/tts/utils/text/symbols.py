# -*- coding: utf-8 -*-
"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
"""


def parse_symbols():
    return {
        "pad": _pad,
        "eos": _eos,
        "bos": _bos,
        "characters": _characters,
        "punctuations": _punctuations,
        "phonemes": _phonemes,
    }


def make_symbols(
    characters,
    phonemes=None,
    punctuations="!'(),-.:;? ",
    pad="<PAD>",
    eos="<EOS>",
    bos="<BOS>",
    blank="<BLNK>",
    unique=True,
):  # pylint: disable=redefined-outer-name
    """Function to create default characters and phonemes"""
    _symbols = list(characters)
    _symbols = [bos] + _symbols if len(bos) > 0 and bos is not None else _symbols
    _symbols = [eos] + _symbols if len(bos) > 0 and eos is not None else _symbols
    _symbols = [pad] + _symbols if len(bos) > 0 and pad is not None else _symbols
    _symbols = [blank] + _symbols if len(bos) > 0 and blank is not None else _symbols
    _phonemes = None
    if phonemes is not None:
        _phonemes_sorted = (
            sorted(list(set(phonemes))) if unique else sorted(list(phonemes))
        )  # this is to keep previous models compatible.
        # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
        # _arpabet = ["@" + s for s in _phonemes_sorted]
        # Export all symbols:
        _phonemes = [pad, eos, bos] + list(_phonemes_sorted) + list(punctuations)
        # _symbols += _arpabet
    return _symbols, _phonemes


_pad = "<PAD>"
_eos = "<EOS>"
_bos = "<BOS>"
_blank = "<BLNK>"  # TODO: check if we need this alongside with PAD
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


class BaseCharacters:
    """🐸BaseCharacters class

    Every vocabulary class should inherit from this class.

    Args:
        characters (str):
            Main set of characters to be used in the vocabulary.

        punctuations (str):
            Characters to be treated as punctuation.

        pad (str):
            Special padding character that would be ignored by the model.

        eos (str):
            End of the sentence character.

        bos (str):
            Beginning of the sentence character.

        blank (str):
            Optional character used between characters by some models for better prosody.

        is_unique (bool):
            Remove duplicates from the provided characters. Defaults to True.

        is_sorted (bool):
            Sort the characters in alphabetical order. Defaults to True.
    """

    def __init__(
        self,
        characters: str,
        punctuations: str,
        pad: str,
        eos: str,
        bos: str,
        blank: str,
        is_unique: bool = True,
        is_sorted: bool = True,
    ) -> None:
        self._characters = characters
        self._punctuations = punctuations
        self._pad = pad
        self._eos = eos
        self._bos = bos
        self._blank = blank
        self.is_unique = is_unique
        self.is_sorted = is_sorted
        self._create_vocab()

    @property
    def characters(self):
        return self._characters

    @characters.setter
    def characters(self, characters):
        self._characters = characters
        self._vocab = self.create_vocab()

    @property
    def punctuations(self):
        return self._punctuations

    @punctuations.setter
    def punctuations(self, punctuations):
        self._punctuations = punctuations
        self._vocab = self.create_vocab()

    @property
    def pad(self):
        return self._pad

    @pad.setter
    def pad(self, pad):
        self._pad = pad
        self._vocab = self.create_vocab()

    @property
    def eos(self):
        return self._eos

    @eos.setter
    def eos(self, eos):
        self._eos = eos
        self._vocab = self.create_vocab()

    @property
    def bos(self):
        return self._bos

    @bos.setter
    def bos(self, bos):
        self._bos = bos
        self._vocab = self.create_vocab()

    @property
    def blank(self):
        return self._bos

    @bos.setter
    def blank(self, bos):
        self._bos = bos
        self._vocab = self.create_vocab()

    @property
    def vocab(self):
        return self._vocab

    @property
    def num_chars(self):
        return len(self._vocab)

    def _create_vocab(self):
        _vocab = self.characters
        if self.is_unique:
            _vocab = list(set(_vocab))
        if self.is_sorted:
            _vocab = sorted(_vocab)
        _vocab = list(_vocab)
        _vocab = [self.bos] + _vocab if len(self.bos) > 0 and self.bos is not None else _vocab
        _vocab = [self.eos] + _vocab if len(self.bos) > 0 and self.eos is not None else _vocab
        _vocab = [self.pad] + _vocab if len(self.bos) > 0 and self.pad is not None else _vocab
        self._vocab = _vocab + list(self._punctuations)
        self._char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self._id_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        assert len(self.vocab) == len(self._char_to_id) == len(self._id_to_char)

    def char_to_id(self, char: str) -> int:
        return self._char_to_id[char]

    def id_to_char(self, idx: int) -> str:
        return self._id_to_char[idx]

    @staticmethod
    def init_from_config(config: "Coqpit"):
        return BaseCharacters(
            **config.characters if config.characters is not None else {},
        )


class IPAPhonemes(BaseCharacters):
    """🐸IPAPhonemes class to manage `TTS.tts` model vocabulary

    Intended to be used with models using IPAPhonemes as input.
    It uses system defaults for the undefined class arguments.

    Args:
        characters (str):
            Main set of case-sensitive characters to be used in the vocabulary. Defaults to `_phonemes`.

        punctuations (str):
            Characters to be treated as punctuation. Defaults to `_punctuations`.

        pad (str):
            Special padding character that would be ignored by the model. Defaults to `_pad`.

        eos (str):
            End of the sentence character. Defaults to `_eos`.

        bos (str):
            Beginning of the sentence character. Defaults to `_bos`.

        is_unique (bool):
            Remove duplicates from the provided characters. Defaults to True.

        is_sorted (bool):
            Sort the characters in alphabetical order. Defaults to True.
    """

    def __init__(
        self,
        characters: str = _phonemes,
        punctuations: str = _punctuations,
        pad: str = _pad,
        eos: str = _eos,
        bos: str = _bos,
        is_unique: bool = True,
        is_sorted: bool = True,
    ) -> None:
        super().__init__(characters, punctuations, pad, eos, bos, is_unique, is_sorted)

    @staticmethod
    def init_from_config(config: "Coqpit"):
        return IPAPhonemes(
            **config.characters if config.characters is not None else {},
        )


class Graphemes(BaseCharacters):
    """🐸Graphemes class to manage `TTS.tts` model vocabulary

    Intended to be used with models using graphemes as input.
    It uses system defaults for the undefined class arguments.

    Args:
        characters (str):
            Main set of case-sensitive characters to be used in the vocabulary. Defaults to `_characters`.

        punctuations (str):
            Characters to be treated as punctuation. Defaults to `_punctuations`.

        pad (str):
            Special padding character that would be ignored by the model. Defaults to `_pad`.

        eos (str):
            End of the sentence character. Defaults to `_eos`.

        bos (str):
            Beginning of the sentence character. Defaults to `_bos`.

        is_unique (bool):
            Remove duplicates from the provided characters. Defaults to True.

        is_sorted (bool):
            Sort the characters in alphabetical order. Defaults to True.
    """

    def __init__(
        self,
        characters: str = _characters,
        punctuations: str = _punctuations,
        pad: str = _pad,
        eos: str = _eos,
        bos: str = _bos,
        is_unique: bool = True,
        is_sorted: bool = True,
    ) -> None:
        super().__init__(characters, punctuations, pad, eos, bos, is_unique, is_sorted)

    @staticmethod
    def init_from_config(config: "Coqpit"):
        return Graphemes(
            **config.characters if config.characters is not None else {},
        )



if __name__ == "__main__":
    gr = Graphemes()
    ph = IPAPhonemes()

    print(gr.vocab)
    print(ph.vocab)

    print(gr.num_chars)
    assert "a" == gr.id_to_char(gr.char_to_id("a"))
