import os
import unittest

import numpy as np
import torch

from tests import get_tests_input_path
from TTS.config import load_config
from TTS.speaker_encoder.utils.generic_utils import setup_model
from TTS.speaker_encoder.utils.io import save_checkpoint
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor

encoder_config_path = os.path.join(get_tests_input_path(), "test_speaker_encoder_config.json")
encoder_model_path = os.path.join(get_tests_input_path(), "checkpoint_0.pth.tar")
sample_wav_path = os.path.join(get_tests_input_path(), "../data/ljspeech/wavs/LJ001-0001.wav")
sample_wav_path2 = os.path.join(get_tests_input_path(), "../data/ljspeech/wavs/LJ001-0002.wav")
d_vectors_file_path = os.path.join(get_tests_input_path(), "../data/dummy_speakers.json")


class SpeakerManagerTest(unittest.TestCase):
    """Test SpeakerManager for loading embedding files and computing d_vectors from waveforms"""

    @staticmethod
    def test_speaker_embedding():
        # load config
        config = load_config(encoder_config_path)
        config.audio.resample = True

        # create a dummy speaker encoder
        model = setup_model(config)
        save_checkpoint(model, None, None, get_tests_input_path(), 0)

        # load audio processor and speaker encoder
        ap = AudioProcessor(**config.audio)
        manager = SpeakerManager(encoder_model_path=encoder_model_path, encoder_config_path=encoder_config_path)

        # load a sample audio and compute embedding
        waveform = ap.load_wav(sample_wav_path)
        mel = ap.melspectrogram(waveform)
        d_vector = manager.compute_d_vector(mel.T)
        assert d_vector.shape[1] == 256

        # compute d_vector directly from an input file
        d_vector = manager.compute_d_vector_from_clip(sample_wav_path)
        d_vector2 = manager.compute_d_vector_from_clip(sample_wav_path)
        d_vector = torch.FloatTensor(d_vector)
        d_vector2 = torch.FloatTensor(d_vector2)
        assert d_vector.shape[0] == 256
        assert (d_vector - d_vector2).sum() == 0.0

        # compute d_vector from a list of wav files.
        d_vector3 = manager.compute_d_vector_from_clip([sample_wav_path, sample_wav_path2])
        d_vector3 = torch.FloatTensor(d_vector3)
        assert d_vector3.shape[0] == 256
        assert (d_vector - d_vector3).sum() != 0.0

        # remove dummy model
        os.remove(encoder_model_path)

    @staticmethod
    def test_speakers_file_processing():
        manager = SpeakerManager(d_vectors_file_path=d_vectors_file_path)
        print(manager.num_speakers)
        print(manager.d_vector_dim)
        print(manager.clip_ids)
        d_vector = manager.get_d_vector_by_clip(manager.clip_ids[0])
        assert len(d_vector) == 256
        d_vectors = manager.get_d_vectors_by_speaker(manager.speaker_names[0])
        assert len(d_vectors[0]) == 256
        d_vector1 = manager.get_mean_d_vector(manager.speaker_names[0], num_samples=2, randomize=True)
        assert len(d_vector1) == 256
        d_vector2 = manager.get_mean_d_vector(manager.speaker_names[0], num_samples=2, randomize=False)
        assert len(d_vector2) == 256
        assert np.sum(np.array(d_vector1) - np.array(d_vector2)) != 0
