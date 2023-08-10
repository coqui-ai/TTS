from typing import Dict

from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.belarusian.phonemizer import belarusian_text_to_phonemes

_DEF_BE_PUNCS = ",!."  # TODO


class BEL_Phonemizer(BasePhonemizer):
    """🐸TTS be phonemizer using functions in `TTS.tts.utils.text.belarusian.phonemizer`

    Args:
        punctuations (str):
            Set of characters to be treated as punctuation. Defaults to `_DEF_BE_PUNCS`.

        keep_puncs (bool):
            If True, keep the punctuations after phonemization. Defaults to False.
    """

    language = "be"

    def __init__(self, punctuations=_DEF_BE_PUNCS, keep_puncs=True, **kwargs):  # pylint: disable=unused-argument
        super().__init__(self.language, punctuations=punctuations, keep_puncs=keep_puncs)

    @staticmethod
    def name():
        return "be_phonemizer"

    @staticmethod
    def phonemize_be(text: str, separator: str = "|") -> str:  # pylint: disable=unused-argument
        return belarusian_text_to_phonemes(text)

    def _phonemize(self, text, separator):
        return self.phonemize_be(text, separator)

    @staticmethod
    def supported_languages() -> Dict:
        return {"be": "Belarusian"}

    def version(self) -> str:
        return "0.0.1"

    def is_available(self) -> bool:
        return True


if __name__ == "__main__":
    txt = "тэст"
    e = BEL_Phonemizer()
    print(e.supported_languages())
    print(e.version())
    print(e.language)
    print(e.name())
    print(e.is_available())
    print("`" + e.phonemize(txt) + "`")
