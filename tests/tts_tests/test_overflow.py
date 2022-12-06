import os
import unittest

import torch

from tests import get_tests_output_path
from TTS.tts.configs.overflow_config import OverflowConfig
from TTS.tts.layers.overflow.common_layers import OverflowUtils
from TTS.tts.models.overflow import Overflow
from TTS.utils.audio import AudioProcessor

# pylint: disable=unused-variable

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config_global = OverflowConfig(num_chars=24)
ap = AudioProcessor.init_from_config(config_global)

config_path = os.path.join(get_tests_output_path(), "test_model_config.json")
output_path = os.path.join(get_tests_output_path(), "train_outputs")


class TestOverFlow(unittest.TestCase):
    @staticmethod
    def _create_inputs(batch_size=8):
        input_dummy = torch.randint(0, 24, (batch_size, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (batch_size,)).long().to(device).sort(descending=True)[0]
        input_lengths[-1] = 128
        mel_spec = torch.rand(batch_size, 30, config_global.audio["num_mels"]).to(device)
        mel_lengths = torch.randint(20, 30, (batch_size,)).long().to(device)
        return input_dummy, input_lengths, mel_spec, mel_lengths

    @staticmethod
    def get_model(config=None):
        if config is None:
            config = config_global
        model = Overflow(config)
        model = model.to(device)
        return model

    def test_inference(self):
        model = self.get_model()
        input_dummy, input_lengths, mel_spec, mel_lengths = self._create_inputs()
        output_dict = model.inference(input_dummy)
        self.assertEqual(output_dict["model_outputs"].shape[2], config_global.out_channels)


class TestOverFlowUtils(unittest.TestCase):
    def logsumexp_test(self):
        a = torch.randn(10)  # random numbers
        self.assertTrue(torch.eq(torch.logsumexp(a, dim=0), OverflowUtils.logsumexp(a, dim=0)).all())

        a = torch.zeros(10)  # all zeros
        self.assertTrue(torch.eq(torch.logsumexp(a, dim=0), OverflowUtils.logsumexp(a, dim=0)).all())

        a = torch.ones(10)  # all ones
        self.assertTrue(torch.eq(torch.logsumexp(a, dim=0), OverflowUtils.logsumexp(a, dim=0)).all())
