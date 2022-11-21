from typing import Dict

import pyopenjtalk

from TTS.tts.utils.text.phonemizers.base import BasePhonemizer


class PyOpenJTalk_Phonemizer(BasePhonemizer):
    """🐸TTS Ja-Jp phonemizer using PyOpenJTalk
    https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
    """

    language = "ja-jp"

    def __init__(self, **kwargs):
        super().__init__(self.language)

    @staticmethod
    def name():
        return "pyopenjtalk_phonemizer"

    def _phonemize(self, text: str, separator: str = "|") -> str:
        # phones is a str object separated by space
        ph = pyopenjtalk.g2p(text, kana=False)
        ph = ph.split(" ")
        if separator is not None:
            return separator.join(ph)
        return ph

    def phonemize(self, text: str, separator="|") -> str:
        """Custom phonemize for JP_JA

        Skip pre-post processing steps used by the other phonemizers.
        """
        return self._phonemize(text, separator)

    @staticmethod
    def supported_languages() -> Dict:
        return {"ja-jp": "Japanese (Japan)"}

    def version(self) -> str:
        return "0.0.1"

    def is_available(self) -> bool:
        return True
