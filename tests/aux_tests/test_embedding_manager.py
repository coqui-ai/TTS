import os
import unittest

import numpy as np
import torch
from trainer.io import save_checkpoint

from tests import get_tests_input_path
from TTS.config import load_config
from TTS.encoder.utils.generic_utils import setup_encoder_model
from TTS.tts.utils.managers import EmbeddingManager
from TTS.utils.audio import AudioProcessor

encoder_config_path = os.path.join(get_tests_input_path(), "test_speaker_encoder_config.json")
encoder_model_path = os.path.join(get_tests_input_path(), "checkpoint_0.pth")
sample_wav_path = os.path.join(get_tests_input_path(), "../data/ljspeech/wavs/LJ001-0001.wav")
sample_wav_path2 = os.path.join(get_tests_input_path(), "../data/ljspeech/wavs/LJ001-0002.wav")
embedding_file_path = os.path.join(get_tests_input_path(), "../data/dummy_speakers.json")
embeddings_file_path2 = os.path.join(get_tests_input_path(), "../data/dummy_speakers2.json")
embeddings_file_pth_path = os.path.join(get_tests_input_path(), "../data/dummy_speakers.pth")


class EmbeddingManagerTest(unittest.TestCase):
    """Test emEeddingManager for loading embedding files and computing embeddings from waveforms"""

    @staticmethod
    def test_speaker_embedding():
        # load config
        config = load_config(encoder_config_path)
        config.audio.resample = True

        # create a dummy speaker encoder
        model = setup_encoder_model(config)
        save_checkpoint(config, model, None, None, 0, 0, get_tests_input_path())

        # load audio processor and speaker encoder
        manager = EmbeddingManager(encoder_model_path=encoder_model_path, encoder_config_path=encoder_config_path)

        # load a sample audio and compute embedding
        ap = AudioProcessor(**config.audio)
        waveform = ap.load_wav(sample_wav_path)
        mel = ap.melspectrogram(waveform)
        embedding = manager.compute_embeddings(mel)
        assert embedding.shape[1] == 256

        # compute embedding directly from an input file
        embedding = manager.compute_embedding_from_clip(sample_wav_path)
        embedding2 = manager.compute_embedding_from_clip(sample_wav_path)
        embedding = torch.FloatTensor(embedding)
        embedding2 = torch.FloatTensor(embedding2)
        assert embedding.shape[0] == 256
        assert (embedding - embedding2).sum() == 0.0

        # compute embedding from a list of wav files.
        embedding3 = manager.compute_embedding_from_clip([sample_wav_path, sample_wav_path2])
        embedding3 = torch.FloatTensor(embedding3)
        assert embedding3.shape[0] == 256
        assert (embedding - embedding3).sum() != 0.0

        # remove dummy model
        os.remove(encoder_model_path)

    def test_embedding_file_processing(self):  # pylint: disable=no-self-use
        manager = EmbeddingManager(embedding_file_path=embeddings_file_pth_path)
        # test embedding querying
        embedding = manager.get_embedding_by_clip(manager.clip_ids[0])
        assert len(embedding) == 256
        embeddings = manager.get_embeddings_by_name(manager.embedding_names[0])
        assert len(embeddings[0]) == 256
        embedding1 = manager.get_mean_embedding(manager.embedding_names[0], num_samples=2, randomize=True)
        assert len(embedding1) == 256
        embedding2 = manager.get_mean_embedding(manager.embedding_names[0], num_samples=2, randomize=False)
        assert len(embedding2) == 256
        assert np.sum(np.array(embedding1) - np.array(embedding2)) != 0

    def test_embedding_file_loading(self):
        # test loading a json file
        manager = EmbeddingManager(embedding_file_path=embedding_file_path)
        self.assertEqual(manager.num_embeddings, 384)
        self.assertEqual(manager.embedding_dim, 256)
        # test loading a pth file
        manager = EmbeddingManager(embedding_file_path=embeddings_file_pth_path)
        self.assertEqual(manager.num_embeddings, 384)
        self.assertEqual(manager.embedding_dim, 256)
        # test loading a pth files with duplicate embedding keys
        with self.assertRaises(Exception) as context:
            manager = EmbeddingManager(embedding_file_path=[embeddings_file_pth_path, embeddings_file_pth_path])
        self.assertTrue("Duplicate embedding names" in str(context.exception))
        # test loading embedding files with different embedding keys
        manager = EmbeddingManager(embedding_file_path=[embeddings_file_pth_path, embeddings_file_path2])
        self.assertEqual(manager.embedding_dim, 256)
        self.assertEqual(manager.num_embeddings, 384 * 2)
