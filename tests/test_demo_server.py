import os
import unittest

import torch as T

from TTS.server.synthesizer import Synthesizer
from TTS.tests import get_tests_input_path, get_tests_output_path
from TTS.utils.text.symbols import make_symbols, phonemes, symbols
from TTS.utils.generic_utils import load_config, save_checkpoint, setup_model


class DemoServerTest(unittest.TestCase):
    # pylint: disable=R0201
    def _create_random_model(self):
        # pylint: disable=global-statement
        global symbols, phonemes
        config = load_config(os.path.join(get_tests_output_path(), 'dummy_model_config.json'))
        if 'characters' in config.keys():
            symbols, phonemes = make_symbols(**config.characters)

        num_chars = len(phonemes) if config.use_phonemes else len(symbols)
        model = setup_model(num_chars, 0, config)
        output_path = os.path.join(get_tests_output_path())
        save_checkpoint(model, None, None, None, output_path, 10, 10)

    def test_in_out(self):
        self._create_random_model()
        config = load_config(os.path.join(get_tests_input_path(), 'server_config.json'))
        tts_root_path = get_tests_output_path()
        config['tts_checkpoint'] = os.path.join(tts_root_path, config['tts_checkpoint'])
        config['tts_config'] = os.path.join(tts_root_path, config['tts_config'])
        synthesizer = Synthesizer(config)
        synthesizer.tts("Better this test works!!")
