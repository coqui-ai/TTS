import os
import unittest

import torch

from tests import get_tests_input_path, get_tests_output_path, run_cli

torch.manual_seed(1)

# pylint: disable=protected-access
class TestRemoveSilenceVAD(unittest.TestCase):
    @staticmethod
    def test():
        # set paths
        wav_path = os.path.join(get_tests_input_path(), "../data/ljspeech/wavs")
        output_path = os.path.join(get_tests_output_path(), "output_wavs_removed_silence/")
        output_resample_path = os.path.join(get_tests_output_path(), "output_ljspeech_16khz/")

        # resample audios
        run_cli(
            f'CUDA_VISIBLE_DEVICES="" python TTS/bin/resample.py --input_dir "{wav_path}" --output_dir "{output_resample_path}" --output_sr 16000'
        )

        # run test
        run_cli(
            f'CUDA_VISIBLE_DEVICES="" python TTS/bin/remove_silence_using_vad.py --input_dir "{output_resample_path}" --output_dir "{output_path}"'
        )
        run_cli(f'rm -rf "{output_resample_path}"')
        run_cli(f'rm -rf "{output_path}"')
