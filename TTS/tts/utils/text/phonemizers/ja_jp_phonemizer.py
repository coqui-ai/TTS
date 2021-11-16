from typing import Dict

from TTS.tts.utils.text.japanese.phonemizer import japanese_text_to_phonemes
from TTS.tts.utils.text.phonemizers.base import BasePhonemizer

_DEF_JA_PUNCS = "、.,[]()?!〽~『』「」【】"


class JA_JP_Phonemizer(BasePhonemizer):
    """🐸TTS Ja-Jp phonemizer using functions in `TTS.tts.utils.text.japanese.phonemizer`

    TODO: someone with JA knowledge should check this implementation
    """

    language = "ja-jp"

    def __init__(self, punctuations=_DEF_JA_PUNCS, keep_puncs=False, **kwargs):
        super().__init__(self.language, punctuations=punctuations, keep_puncs=keep_puncs)

    @staticmethod
    def name():
        return "ja_jp_phonemizer"

    def phonemize_jajp(self, text: str, separator: str = "|") -> str:
        ph = japanese_text_to_phonemes(text)
        if separator is not None or separator != "":
            return separator.join(ph)
        return ph

    def _phonemize(self, text, separator):
        return self.phonemize_jajp(text, separator)

    @staticmethod
    def supported_languages() -> Dict:
        return {"ja-jp": "Japanese (Japan)"}

    def version(self) -> str:
        return "0.0.1"

    def is_available(self) -> bool:
        return True


if __name__ == "__main__":
    text = "これは、電話をかけるための私の日本語の例のテキストです。"
    e = JA_JP_Phonemizer()
    print(e.supported_languages())
    print(e.version())
    print(e.language)
    print(e.name())
    print(e.is_available())
    print("`" + e.phonemize(text) + "`")
