import os
import unittest

import torch

from tests import get_tests_input_path

from tests import get_tests_output_path, run_cli

from TTS.tts.utils.generic_utils import setup_model

from TTS.config import load_config
from TTS.tts.utils.text.symbols import phonemes, symbols

torch.manual_seed(1)

# pylint: disable=protected-access
class TestExtractTTSSpectrograms(unittest.TestCase):
    @staticmethod
    def test_GlowTTS():
        # set paths
        config_path = os.path.join(get_tests_input_path(), "test_glow_tts.json")
        checkpoint_path = os.path.join(get_tests_output_path(), 'checkpoint_test.pth.tar')
        output_path = os.path.join(get_tests_output_path(), 'output_extract_tts_spectrograms/')
        # load config
        c = load_config(config_path)
        # create model
        num_chars = len(phonemes if c.use_phonemes else symbols)
        model = setup_model(num_chars, 1, c, speaker_embedding_dim=None)
        # save model
        torch.save({"model": model.state_dict()}, checkpoint_path)
        # run test
        run_cli(f'CUDA_VISIBLE_DEVICES="" python TTS/bin/extract_tts_spectrograms.py --config_path "{config_path}" --checkpoint_path "{checkpoint_path}" --output_path "{output_path}"')
        run_cli(f'rm -rf "{output_path}" "{checkpoint_path}"')
    @staticmethod
    def test_Tacotron2():
        # set paths
        config_path = os.path.join(get_tests_input_path(), "test_tacotron2_config.json")
        checkpoint_path = os.path.join(get_tests_output_path(), 'checkpoint_test.pth.tar')
        output_path = os.path.join(get_tests_output_path(), 'output_extract_tts_spectrograms/')
        # load config
        c = load_config(config_path)
        # create model
        num_chars = len(phonemes if c.use_phonemes else symbols)
        model = setup_model(num_chars, 1, c, speaker_embedding_dim=None)
        # save model
        torch.save({"model": model.state_dict()}, checkpoint_path)
        # run test
        run_cli(f'CUDA_VISIBLE_DEVICES="" python TTS/bin/extract_tts_spectrograms.py --config_path "{config_path}" --checkpoint_path "{checkpoint_path}" --output_path "{output_path}"')
        run_cli(f'rm -rf "{output_path}" "{checkpoint_path}"')
    @staticmethod
    def test_Tacotron():
        # set paths
        config_path = os.path.join(get_tests_input_path(), "test_tacotron_config.json")
        checkpoint_path = os.path.join(get_tests_output_path(), 'checkpoint_test.pth.tar')
        output_path = os.path.join(get_tests_output_path(), 'output_extract_tts_spectrograms/')
        # load config
        c = load_config(config_path)
        # create model
        num_chars = len(phonemes if c.use_phonemes else symbols)
        model = setup_model(num_chars, 1, c, speaker_embedding_dim=None)
        # save model
        torch.save({"model": model.state_dict()}, checkpoint_path)
        # run test
        run_cli(f'CUDA_VISIBLE_DEVICES="" python TTS/bin/extract_tts_spectrograms.py --config_path "{config_path}" --checkpoint_path "{checkpoint_path}" --output_path "{output_path}"')
        run_cli(f'rm -rf "{output_path}" "{checkpoint_path}"')
