import os
import unittest

import torch

from tests import get_tests_output_path, run_cli
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig

torch.manual_seed(1)

config_path = os.path.join(get_tests_output_path(), "test_model_config.json")

dataset_config_en = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="metadata.csv",
    meta_file_val="metadata.csv",
    path="tests/data/ljspeech",
    language="en",
)

"""
dataset_config_pt = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="metadata.csv",
    meta_file_val="metadata.csv",
    path="tests/data/ljspeech",
    language="pt-br",
)
"""


# pylint: disable=protected-access
class TestFindUniquePhonemes(unittest.TestCase):
    @staticmethod
    def test_espeak_phonemes():
        # prepare the config
        config = VitsConfig(
            batch_size=2,
            eval_batch_size=2,
            num_loader_workers=0,
            num_eval_loader_workers=0,
            text_cleaner="english_cleaners",
            use_phonemes=True,
            phoneme_language="en-us",
            phoneme_cache_path="tests/data/ljspeech/phoneme_cache/",
            run_eval=True,
            test_delay_epochs=-1,
            epochs=1,
            print_step=1,
            print_eval=True,
            datasets=[dataset_config_en],
        )
        config.save_json(config_path)

        # run test
        run_cli(f'CUDA_VISIBLE_DEVICES="" python TTS/bin/find_unique_phonemes.py --config_path "{config_path}"')

    @staticmethod
    def test_no_espeak_phonemes():
        # prepare the config
        config = VitsConfig(
            batch_size=2,
            eval_batch_size=2,
            num_loader_workers=0,
            num_eval_loader_workers=0,
            text_cleaner="english_cleaners",
            use_phonemes=True,
            phoneme_language="en-us",
            phoneme_cache_path="tests/data/ljspeech/phoneme_cache/",
            run_eval=True,
            test_delay_epochs=-1,
            epochs=1,
            print_step=1,
            print_eval=True,
            datasets=[dataset_config_en],
        )
        config.save_json(config_path)

        # run test
        run_cli(f'CUDA_VISIBLE_DEVICES="" python TTS/bin/find_unique_phonemes.py --config_path "{config_path}"')
