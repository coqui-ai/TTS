import unittest

from TTS.tts.utils.text.korean.phonemizer import korean_text_to_phonemes

_TEST_CASES = """
안녕하세요./안녕하세요.
"""


class TestText(unittest.TestCase):
    def test_korean_text_to_phonemes(self):
        for line in _TEST_CASES.strip().split("\n"):
            text, phone = line.split("/")
            self.assertEqual(korean_text_to_phonemes(text), phone)


if __name__ == "__main__":
    unittest.main()