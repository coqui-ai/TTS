import os
import unittest

import numpy as np

from tests import get_tests_input_path
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config

encoder_config_path = os.path.join(get_tests_input_path(), "test_speaker_encoder_config.json")
encoder_model_path = os.path.join(get_tests_input_path(), "dummy_speaker_encoder.pth.tar")
sample_wav_path = os.path.join(get_tests_input_path(), "../data/ljspeech/wavs/LJ001-0001.wav")
x_vectors_file_path = os.path.join(get_tests_input_path(), "../data/dummy_speakers.json")


class SpeakerManagerTest(unittest.TestCase):
    """Test SpeakerManager for loading embedding files and computing x_vectors from waveforms"""
    @staticmethod
    def test_speaker_embedding():
        # load config
        config = load_config(encoder_config_path)
        config["audio"]["resample"] = True

        # load audio processor and speaker encoder
        ap = AudioProcessor(**config.audio)
        manager = SpeakerManager(encoder_model_path=encoder_model_path, encoder_config_path=encoder_config_path)

        # load a sample audio and compute embedding
        waveform = ap.load_wav(sample_wav_path)
        mel = ap.melspectrogram(waveform)
        x_vector = manager.compute_x_vector(mel.T)
        assert x_vector.shape[1] == 256

    @staticmethod
    def test_speakers_file_processing():
        manager = SpeakerManager(x_vectors_file_path=x_vectors_file_path)
        print(manager.num_speakers)
        print(manager.x_vector_dim)
        print(manager.clip_ids)
        x_vector = manager.get_x_vector_by_clip(manager.clip_ids[0])
        assert len(x_vector) == 256
        x_vectors = manager.get_x_vectors_by_speaker(manager.speaker_ids[0])
        assert len(x_vectors[0]) == 256
        x_vector1 = manager.get_mean_x_vector(manager.speaker_ids[0], num_samples=2, randomize=True)
        assert len(x_vector1) == 256
        x_vector2 = manager.get_mean_x_vector(manager.speaker_ids[0], num_samples=2, randomize=False)
        assert len(x_vector2) == 256
        assert np.sum(np.array(x_vector1) - np.array(x_vector2)) != 0
