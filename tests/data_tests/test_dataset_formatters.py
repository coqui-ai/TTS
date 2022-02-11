import os
import unittest

from tests import get_tests_input_path
from TTS.tts.datasets.formatters import common_voice


class TestTTSFormatters(unittest.TestCase):
    def test_common_voice_preprocessor(self):  # pylint: disable=no-self-use
        root_path = get_tests_input_path()
        meta_file = "common_voice.tsv"
        items = common_voice(root_path, meta_file)
        assert items[0]["text"] == "The applicants are invited for coffee and visa is given immediately."
        assert items[0]["audio_file"] == os.path.join(get_tests_input_path(), "clips", "common_voice_en_20005954.wav")

        assert items[-1]["text"] == "Competition for limited resources has also resulted in some local conflicts."
        assert items[-1]["audio_file"] == os.path.join(get_tests_input_path(), "clips", "common_voice_en_19737074.wav")
