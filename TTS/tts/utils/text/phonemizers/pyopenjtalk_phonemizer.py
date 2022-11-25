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


class PyOpenJTalk_Prosody_Phonemizer(BasePhonemizer):
    """🐸TTS Ja-Jp phonemizer using PyOpenJTalk
    https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py

    Intonation Symbols
    ^: BOS
    $: EOS (Declarative)
    ?: EOS (interogative)
    _: Pause
    #: Accent phrase border
    ]: Pitch falling
    [: Pitch rising
    """

    language = "ja-jp"

    def __init__(self, **kwargs):
        super().__init__(self.language)

    @staticmethod
    def name():
        return "pyopenjtalk_prosody_phonemizer"

    def _phonemize(self, text: str, separator: str = "|") -> str:
        ph = []

        drop_unvoiced_vowels = True
        labels = pyopenjtalk.extract_fullcontext(text)
        N = len(labels)

        for n in range(N):
            lab_curr = labels[n]

            # current phoneme
            p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)

            # deal unvoiced vowels as normal vowels
            if drop_unvoiced_vowels and p3 in "AEIOU":
                p3 = p3.lower()

            # deal with sil at the beginning and the end of text
            if p3 == "sil":
                assert n == 0 or n == N - 1
                if n == 0:
                    ph.append("^")
                elif n == N - 1:
                    # check question form or not
                    e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                    if e3 == 0:
                        ph.append("$")
                    elif e3 == 1:
                        ph.append("?")
                continue
            elif p3 == "pau":
                ph.append("_")
                continue
            else:
                ph.append(p3)

            # accent type and position info (forward or backward)
            a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
            a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
            a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

            # number of mora in accent phrase
            f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

            a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
            # accent phrase border
            if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
                ph.append("#")
            # pitch falling
            elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
                ph.append("]")
            # pitch rising
            elif a2 == 1 and a2_next == 2:
                ph.append("[")

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


def _numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))
