import os
import unittest

from tests import get_tests_data_path, get_tests_output_path
from TTS.api import CS_API, TTS

OUTPUT_PATH = os.path.join(get_tests_output_path(), "test_python_api.wav")
cloning_test_wav_path = os.path.join(get_tests_data_path(), "ljspeech/wavs/LJ001-0028.wav")


is_coqui_available = os.environ.get("COQUI_STUDIO_TOKEN")


if is_coqui_available:

    class CS_APITest(unittest.TestCase):
        def test_speakers(self):
            tts = CS_API()
            self.assertGreater(len(tts.speakers), 1)

        def test_emotions(self):
            tts = CS_API()
            self.assertGreater(len(tts.emotions), 1)

        def test_list_calls(self):
            tts = CS_API()
            self.assertGreater(len(tts.list_voices()), 1)
            self.assertGreater(len(tts.list_speakers()), 1)
            self.assertGreater(len(tts.list_all_speakers()), 1)
            self.assertGreater(len(tts.list_speakers_as_tts_models()), 1)

        def test_name_to_speaker(self):
            tts = CS_API()
            speaker_name = tts.list_speakers_as_tts_models()[0].split("/")[2]
            speaker = tts.name_to_speaker(speaker_name)
            self.assertEqual(speaker.name, speaker_name)

        def test_tts(self):
            tts = CS_API()
            wav, sr = tts.tts(text="This is a test.", speaker_name=tts.list_speakers()[0].name)
            self.assertEqual(sr, 44100)
            self.assertGreater(len(wav), 1)

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

        def test_studio_model(self):
            tts = TTS(model_name="coqui_studio/en/Zacharie Aimilios/coqui_studio")
            tts.tts_to_file(text="This is a test.")

            # check speed > 2.0 raises error
            raised_error = False
            try:
                _ = tts.tts(text="This is a test.", speed=4.0, emotion="Sad")  # should raise error with speed > 2.0
            except ValueError:
                raised_error = True
            self.assertTrue(raised_error)

            # check emotion is invalid
            raised_error = False
            try:
                _ = tts.tts(text="This is a test.", speed=2.0, emotion="No Emo")  # should raise error with speed > 2.0
            except ValueError:
                raised_error = True
            self.assertTrue(raised_error)

            # check valid call
            wav = tts.tts(text="This is a test.", speed=2.0, emotion="Sad")
            self.assertGreater(len(wav), 0)

        def test_fairseq_model(self):  # pylint: disable=no-self-use
            tts = TTS(model_name="tts_models/eng/fairseq/vits")
            tts.tts_to_file(text="This is a test.")

        def test_multi_speaker_multi_lingual_model(self):
            tts = TTS()
            tts.load_tts_model_by_name(tts.models[0])  # YourTTS
            tts.tts_to_file(
                text="Hello world!", speaker=tts.speakers[0], language=tts.languages[0], file_path=OUTPUT_PATH
            )

            self.assertTrue(tts.is_multi_speaker)
            self.assertTrue(tts.is_multi_lingual)
            self.assertGreater(len(tts.speakers), 1)
            self.assertGreater(len(tts.languages), 1)

        def test_voice_cloning(self):  # pylint: disable=no-self-use
            tts = TTS()
            tts.load_tts_model_by_name("tts_models/multilingual/multi-dataset/your_tts")
            tts.tts_to_file("Hello world!", speaker_wav=cloning_test_wav_path, language="en", file_path=OUTPUT_PATH)

        def test_voice_conversion(self):  # pylint: disable=no-self-use
            tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False, gpu=False)
            tts.voice_conversion_to_file(
                source_wav=cloning_test_wav_path,
                target_wav=cloning_test_wav_path,
                file_path=OUTPUT_PATH,
            )
