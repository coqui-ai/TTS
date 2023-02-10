import os
import unittest

from tests import get_tests_data_path, get_tests_output_path
from TTS.api import TTS

OUTPUT_PATH = os.path.join(get_tests_output_path(), "test_python_api.wav")
cloning_test_wav_path = os.path.join(get_tests_data_path(), "ljspeech/wavs/LJ001-0028.wav")


class TTSTest(unittest.TestCase):
    def test_single_speaker_model(self):
        tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False, gpu=False)

        error_raised = False
        try:
            tts.tts_to_file(text="Ich bin eine Testnachricht.", speaker="Thorsten", language="de")
        except ValueError:
            error_raised = True

        tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path=OUTPUT_PATH)

        self.assertTrue(error_raised)
        self.assertFalse(tts.is_multi_speaker)
        self.assertFalse(tts.is_multi_lingual)
        self.assertIsNone(tts.speakers)
        self.assertIsNone(tts.languages)

    def test_multi_speaker_multi_lingual_model(self):
        tts = TTS()
        tts.load_model_by_name(tts.models[0])  # YourTTS
        tts.tts_to_file(text="Hello world!", speaker=tts.speakers[0], language=tts.languages[0], file_path=OUTPUT_PATH)

        self.assertTrue(tts.is_multi_speaker)
        self.assertTrue(tts.is_multi_lingual)
        self.assertGreater(len(tts.speakers), 1)
        self.assertGreater(len(tts.languages), 1)

    def test_voice_cloning(self):  # pylint: disable=no-self-use
        tts = TTS()
        tts.load_model_by_name("tts_models/multilingual/multi-dataset/your_tts")
        tts.tts_to_file("Hello world!", speaker_wav=cloning_test_wav_path, language="en", file_path=OUTPUT_PATH)
