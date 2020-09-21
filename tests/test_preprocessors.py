import unittest
import os
from tests import get_tests_input_path

from TTS.tts.datasets.preprocess import common_voice


class TestPreprocessors(unittest.TestCase):

    def test_common_voice_preprocessor(self):  #pylint: disable=no-self-use
        root_path = get_tests_input_path()
        meta_file = "common_voice.tsv"
        items = common_voice(root_path, meta_file)
        assert items[0][0] == 'The applicants are invited for coffee and visa is given immediately.'
        assert items[0][1] == os.path.join(get_tests_input_path(), "clips", "common_voice_en_20005954.wav")

        assert items[-1][0] == "Competition for limited resources has also resulted in some local conflicts."
        assert items[-1][1] == os.path.join(get_tests_input_path(), "clips", "common_voice_en_19737074.wav")
