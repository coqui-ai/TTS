import os
import unittest
import warnings

from TTS.tts.utils.text.belarusian.phonemizer import belarusian_text_to_phonemes

_TEST_CASES = """
Фанетычны канвертар/fanʲɛˈtɨt͡ʂnɨ kanˈvʲɛrtar
Гэтак мы працавалі/ˈɣɛtak ˈmɨ prat͡saˈvalʲi
"""


class TestText(unittest.TestCase):
    def test_belarusian_text_to_phonemes(self):
        try:
            os.environ["BEL_FANETYKA_JAR"]
        except KeyError:
            warnings.warn(
                "You need to define 'BEL_FANETYKA_JAR' environment variable as path to the fanetyka.jar file to test Belarusian phonemizer",
                Warning,
            )
            return

        for line in _TEST_CASES.strip().split("\n"):
            text, phonemes = line.split("/")
            self.assertEqual(belarusian_text_to_phonemes(text), phonemes)


if __name__ == "__main__":
    unittest.main()
