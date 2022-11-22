import re
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
        if separator != "":
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


class PyOpenJTalk_Accent_Phonemizer(BasePhonemizer):
    """🐸TTS Ja-Jp phonemizer using PyOpenJTalk
    https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
    """

    language = "ja-jp"

    def __init__(self, **kwargs):
        super().__init__(self.language)

    @staticmethod
    def name():
        return "pyopenjtalk_accent_phonemizer"

    def _phonemize(self, text: str, separator: str = "|") -> str:
        ph = []
        for labels in pyopenjtalk.run_frontend(text)[1]:
            p = re.findall(r"\-(.*?)\+.*?\/A:([0-9\-]+).*?\/F:.*?_([0-9]+)", labels)
            if len(p) == 1:
                ph += [p[0][0], p[0][2], p[0][1]]
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
