def parse_symbols():
    return {
        "pad": _pad,
        "eos": _eos,
        "bos": _bos,
        "characters": _characters,
        "punctuations": _punctuations,
        "phonemes": _phonemes,
    }


# DEFAULT SET OF GRAPHEMES
_pad = "<PAD>"
_eos = "<EOS>"
_bos = "<BOS>"
_blank = "<BLNK>"  # TODO: check if we need this alongside with PAD
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_punctuations = "!'(),-.:;? "


# DEFAULT SET OF IPA PHONEMES
# Phonemes definition (All IPA characters)
_vowels = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ"
_non_pulmonic_consonants = "ʘɓǀɗǃʄǂɠǁʛ"
_pulmonic_consonants = "pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ"
_suprasegmentals = "ˈˌːˑ"
_other_symbols = "ʍwɥʜʢʡɕʑɺɧʲ"
_diacrilics = "ɚ˞ɫ"
_phonemes = _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics


def create_graphemes(
    characters=_characters,
    punctuations=_punctuations,
    pad=_pad,
    eos=_eos,
    bos=_bos,
    blank=_blank,
    unique=True,
):  # pylint: disable=redefined-outer-name
    """Function to create default characters and phonemes"""
    # create graphemes
    _graphemes = list(characters)
    _graphemes = [bos] + _graphemes if len(bos) > 0 and bos is not None else _graphemes
    _graphemes = [eos] + _graphemes if len(bos) > 0 and eos is not None else _graphemes
    _graphemes = [pad] + _graphemes if len(bos) > 0 and pad is not None else _graphemes
    _graphemes = [blank] + _graphemes if len(bos) > 0 and blank is not None else _graphemes
    _graphemes = _graphemes + list(punctuations)
    return _graphemes, _phonemes


def create_phonemes(
    phonemes=_phonemes, punctuations=_punctuations, pad=_pad, eos=_eos, bos=_bos, blank=_blank, unique=True
):
    # create phonemes
    _phonemes = None
    _phonemes_sorted = (
        sorted(list(set(phonemes))) if unique else sorted(list(phonemes))
    )  # this is to keep previous models compatible.
    _phonemes = list(_phonemes_sorted)
    _phonemes = [bos] + _phonemes if len(bos) > 0 and bos is not None else _phonemes
    _phonemes = [eos] + _phonemes if len(bos) > 0 and eos is not None else _phonemes
    _phonemes = [pad] + _phonemes if len(bos) > 0 and pad is not None else _phonemes
    _phonemes = [blank] + _phonemes if len(bos) > 0 and blank is not None else _phonemes
    _phonemes = _phonemes + list(punctuations)
    _phonemes = [pad, eos, bos] + list(_phonemes_sorted) + list(punctuations)
    return _phonemes


graphemes = create_graphemes(_characters, _phonemes, _punctuations, _pad, _eos, _bos)
phonemes = create_phonemes(_phonemes, _punctuations, _pad, _eos, _bos, _blank)


class BaseCharacters:
    """🐸BaseCharacters class

        Every new character class should inherit from this.

        Characters are oredered as follows ```[PAD, EOS, BOS, BLANK, CHARACTERS, PUNCTUATIONS]```.

        If you need a custom order, you need to define inherit from this class and override the ```_create_vocab``` method.

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
    el
            is_sorted (bool):
                Sort the characters in alphabetical order. Only applies to `self.characters`. Defaults to True.
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
        self._create_vocab()

    @property
    def punctuations(self):
        return self._punctuations

    @punctuations.setter
    def punctuations(self, punctuations):
        self._punctuations = punctuations
        self._create_vocab()

    @property
    def pad(self):
        return self._pad

    @pad.setter
    def pad(self, pad):
        self._pad = pad
        self._create_vocab()

    @property
    def eos(self):
        return self._eos

    @eos.setter
    def eos(self, eos):
        self._eos = eos
        self._create_vocab()

    @property
    def bos(self):
        return self._bos

    @bos.setter
    def bos(self, bos):
        self._bos = bos
        self._create_vocab()

    @property
    def blank(self):
        return self._blank

    @blank.setter
    def blank(self, blank):
        self._blank = blank
        self._create_vocab()

    @property
    def vocab(self):
        return self._vocab

    @property
    def num_chars(self):
        return len(self._vocab)

    def _create_vocab(self):
        _vocab = self._characters
        if self.is_unique:
            _vocab = list(set(_vocab))
        if self.is_sorted:
            _vocab = sorted(_vocab)
        _vocab = list(_vocab)
        _vocab = [self._blank] + _vocab if self._blank is not None and len(self._blank) > 0 else _vocab
        _vocab = [self._bos] + _vocab if self._bos is not None and len(self._bos) > 0 else _vocab
        _vocab = [self._eos] + _vocab if self._eos is not None and len(self._eos) > 0 else _vocab
        _vocab = [self._pad] + _vocab if self._pad is not None and len(self._pad) > 0 else _vocab
        self._vocab = _vocab + list(self._punctuations)
        self._char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self._id_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        if self.is_unique:
            assert (
                len(self.vocab) == len(self._char_to_id) == len(self._id_to_char)
            ), f" [!] There are duplicate characters in the character set. {set([x for x in self.vocab if self.vocab.count(x) > 1])}"

    def char_to_id(self, char: str) -> int:
        return self._char_to_id[char]

    def id_to_char(self, idx: int) -> str:
        return self._id_to_char[idx]

    def print_log(self, level: int = 0):
        """
        Prints the vocabulary in a nice format.
        """
        indent = "\t" * level
        print(f"{indent}| > Characters: {self._characters}")
        print(f"{indent}| > Punctuations: {self._punctuations}")
        print(f"{indent}| > Pad: {self._pad}")
        print(f"{indent}| > EOS: {self._eos}")
        print(f"{indent}| > BOS: {self._bos}")
        print(f"{indent}| > Blank: {self._blank}")
        print(f"{indent}| > Vocab: {self.vocab}")
        print(f"{indent}| > Num chars: {self.num_chars}")

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

        blank (str):
            Optional character used between characters by some models for better prosody. Defaults to `_blank`.

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
        blank: str = _blank,
        is_unique: bool = True,
        is_sorted: bool = True,
    ) -> None:
        super().__init__(characters, punctuations, pad, eos, bos, blank, is_unique, is_sorted)

    @staticmethod
    def init_from_config(config: "Coqpit"):
        # band-aid for compatibility with old models
        characters = None
        if "characters" in config:
            if "phonemes" in config.characters:
                config.characters["characters"] = config.characters["phonemes"]
                # delattr(config.characters, "phonemes")

            return IPAPhonemes(
                characters=config.characters["characters"],
                punctuations=config.characters["punctuations"],
                pad=config.characters["pad"],
                eos=config.characters["eos"],
                bos=config.characters["bos"],
                blank=config.characters["blank"],
                is_unique=config.characters["is_unique"],
                is_sorted=config.characters["is_sorted"],
            )
        return characters


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
        blank: str = _blank,
        is_unique: bool = True,
        is_sorted: bool = True,
    ) -> None:
        super().__init__(characters, punctuations, pad, eos, bos, blank, is_unique, is_sorted)

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
