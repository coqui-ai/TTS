import unittest
import os
from TTS.tests import get_tests_input_path

from TTS.datasets.preprocess import common_voice


class TestPreprocessors(unittest.TestCase):

    def test_common_voice_preprocessor(self):
        root_path = get_tests_input_path()
        meta_file = "common_voice.tsv"
        items = common_voice(root_path, meta_file)
        assert items[0][0] == "Man sollte den Länderfinanzausgleich durch " \
                              "einen Bundesliga-Soli ersetzen."
        assert items[0][1] == os.path.join(get_tests_input_path(), "clips",
                                           "21fce545b24d9a5af0403b949e95e8dd3"
                                           "c10c4ff3e371f14e4d5b4ebf588670b7c"
                                           "9e618285fc872d94a89ed7f0217d9019f"
                                           "e5de33f1577b49dcd518eacf63c4b.wav")

        assert items[-1][0] == "Warum werden da keine strafrechtlichen " \
                               "Konsequenzen gezogen?"
        assert items[-1][1] == os.path.join(get_tests_input_path(), "clips",
                                            "ad2f69e053b0e20e01c82b9821fe5787f1"
                                            "cc8e4b0b97f0e4cab1e9a652c577169c82"
                                            "44fb222281a60ee3081854014113e04c4c"
                                            "a43643100b7c01dab0fac11974.wav")
