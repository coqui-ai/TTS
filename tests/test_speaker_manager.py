import os
import unittest

import numpy as np
import torch

from tests import get_tests_input_path
from TTS.speaker_encoder.model import SpeakerEncoder
from TTS.speaker_encoder.utils.generic_utils import save_checkpoint
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config

encoder_config_path = os.path.join(get_tests_input_path(), "test_speaker_encoder_config.json")
encoder_model_path = os.path.join(get_tests_input_path(), "checkpoint_0.pth.tar")
sample_wav_path = os.path.join(get_tests_input_path(), "../data/ljspeech/wavs/LJ001-0001.wav")
sample_wav_path2 = os.path.join(get_tests_input_path(), "../data/ljspeech/wavs/LJ001-0002.wav")
x_vectors_file_path = os.path.join(get_tests_input_path(), "../data/dummy_speakers.json")


class SpeakerManagerTest(unittest.TestCase):
    """Test SpeakerManager for loading embedding files and computing x_vectors from waveforms"""

    @staticmethod
    def test_speaker_embedding():
        # load config
        config = load_config(encoder_config_path)
        config["audio"]["resample"] = True

        # create a dummy speaker encoder
        model = SpeakerEncoder(**config.model)
        save_checkpoint(model, None, None, get_tests_input_path(), 0, 0)

        # load audio processor and speaker encoder
        ap = AudioProcessor(**config.audio)
        manager = SpeakerManager(encoder_model_path=encoder_model_path, encoder_config_path=encoder_config_path)

        # load a sample audio and compute embedding
        waveform = ap.load_wav(sample_wav_path)
        mel = ap.melspectrogram(waveform)
        x_vector = manager.compute_x_vector(mel.T)
        assert x_vector.shape[1] == 256

        # compute x_vector directly from an input file
        x_vector = manager.compute_x_vector_from_clip(sample_wav_path)
        x_vector2 = manager.compute_x_vector_from_clip(sample_wav_path)
        x_vector = torch.FloatTensor(x_vector)
        x_vector2 = torch.FloatTensor(x_vector2)
        assert x_vector.shape[0] == 256
        assert (x_vector - x_vector2).sum() == 0.0

        # compute x_vector from a list of wav files.
        x_vector3 = manager.compute_x_vector_from_clip([sample_wav_path, sample_wav_path2])
        x_vector3 = torch.FloatTensor(x_vector3)
        assert x_vector3.shape[0] == 256
        assert (x_vector - x_vector3).sum() != 0.0

        # remove dummy model
        os.remove(encoder_model_path)

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
