import logging
import subprocess
import tempfile
from typing import Dict, List

from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.punctuation import Punctuation


def is_tool(name):
    from shutil import which

    return which(name) is not None


if is_tool("espeak-ng"):
    _DEF_ESPEAK_LIB = "espeak-ng"
elif is_tool("espeak"):
    _DEF_ESPEAK_LIB = "espeak"
else:
    _DEF_ESPEAK_LIB = None


def _espeak_exe(espeak_lib: str, args: List, sync=False) -> List[str]:
    cmd = [
        espeak_lib,
        "-b",
        "1",  # UTF8 text encoding
    ]
    cmd.extend(args)
    logging.debug("espeakng: executing %s" % repr(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res = iter(p.stdout.readline, b"")
    if not sync:
        p.stdout.close()
        if p.stderr:
            p.stderr.close()
        if p.stdin:
            p.stdin.close()
        return res
    res2 = []
    for line in res:
        res2.append(line)
    p.stdout.close()
    if p.stderr:
        p.stderr.close()
    if p.stdin:
        p.stdin.close()
    p.wait()
    return res2


class ESpeak(BasePhonemizer):
    """ESpeak wrapper calling `espeak` or `espeak-ng` from the command-line the perform G2P

    Args:
        language (str):
            Valid language code for the used backend.

        backend (str):
            Name of the backend library to use. `espeak` or `espeak-ng`. If None, set automatically
            prefering `espeak-ng` over `espeak`. Defaults to None.

        punctuations (str):
            Characters to be treated as punctuation. Defaults to Punctuation.default_puncs().

        keep_puncs (bool):
            If True, keep the punctuations after phonemization. Defaults to True.
    """

    _ESPEAK_LIB = _DEF_ESPEAK_LIB

    def __init__(self, language: str, backend=None, punctuations=Punctuation.default_puncs(), keep_puncs=True):
        if self._ESPEAK_LIB is None:
            raise Exception("Unknown backend: %s" % backend)
        super().__init__(language, punctuations=punctuations, keep_puncs=keep_puncs)

    def auto_set_espeak_lib(self) -> None:
        if is_tool("espeak-ng"):
            self._ESPEAK_LIB = "espeak-ng"
        elif is_tool("espeak"):
            self._ESPEAK_LIB = "espeak"
        else:
            raise Exception("Cannot set backend automatically. espeak-ng or espeak not found")

    @staticmethod
    def name():
        return "espeak"

    def phonemize_espeak(self, text: str, separator: str = "|", tie=False) -> str:
        """Convert input text to phonemes.

        Args:
            text (str):
                Text to be converted to phonemes.

            tie (bool, optional) : When True use a '͡' character between
                consecutive characters of a single phoneme. Else separate phoneme
                with '_'. This option requires espeak>=1.49. Default to False.
        """
        # set arguments
        args = ["-q", "-v", f"{self._language}"]
        if tie:
            args.append("--ipa=1")  # use '͡' between phonemes
        else:
            args.append("--ipa=3")  # split with '_'
        if tie:
            args.append("--tie=%s" % tie)
        args.append(text)
        # compute phonemes
        phonemes = ""
        for line in _espeak_exe(self._ESPEAK_LIB, args, sync=True):
            logging.debug("line: %s" % repr(line))
            phonemes += line.decode("utf8").strip()
        return phonemes.replace("_", separator)

    def _phonemize(self, text, separator=None):
        return self.phonemize_espeak(text, separator, tie=False)

    @staticmethod
    def supported_languages() -> Dict:
        """Get a dictionary of supported languages.

        Returns:
            Dict: Dictionary of language codes.
        """
        if _DEF_ESPEAK_LIB is None:
            raise {}
        args = ["--voices"]
        langs = {}
        count = 0
        for line in _espeak_exe(_DEF_ESPEAK_LIB, args, sync=True):
            line = line.decode("utf8").strip()
            if count > 0:
                cols = line.split()
                lang_code = cols[1]
                lang_name = cols[3]
                langs[lang_code] = lang_name
            logging.debug("line: %s" % repr(line))
            count += 1
        return langs

    def version(self):
        """Get the version of the used backend.

        Returns:
            str: Version of the used backend.
        """
        args = ["--version"]
        for line in self._espeak_exe(args, sync=True):
            version = line.decode("utf8").strip().split()[2]
            logging.debug("line: %s" % repr(line))
            return version

    @classmethod
    def is_available(cls):
        """Return true if ESpeak is available else false"""
        return is_tool("espeak") or is_tool("espeak-ng")


if __name__ == "__main__":
    e = ESpeak(language="en-us")
    print(e.supported_languages())
    print(e.version())
    print(e.language)
    print(e.name())
    print(e.is_available())

    e = ESpeak(language="en-us", keep_puncs=False)
    print("`" + e.phonemize("hello how are you today?") + "`")

    e = ESpeak(language="en-us", keep_puncs=True)
    print("`" + e.phonemize("hello how are you today?") + "`")
