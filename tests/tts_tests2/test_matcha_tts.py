import unittest

import torch

from TTS.tts.configs.matcha_tts import MatchaTTSConfig
from TTS.tts.models.matcha_tts import MatchaTTS

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

c = MatchaTTSConfig()


class TestMatchTTS(unittest.TestCase):
    @staticmethod
    def _create_inputs(batch_size=8):
        input_dummy = torch.randint(0, 24, (batch_size, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (batch_size,)).long().to(device)
        input_lengths[-1] = 128
        mel_spec = torch.rand(batch_size, 30, c.audio["num_mels"]).to(device)
        mel_lengths = torch.randint(20, 30, (batch_size,)).long().to(device)
        speaker_ids = torch.randint(0, 5, (batch_size,)).long().to(device)
        return input_dummy, input_lengths, mel_spec, mel_lengths, speaker_ids

    def _test_forward(self, batch_size):
        input_dummy, input_lengths, mel_spec, mel_lengths, _ = self._create_inputs(batch_size)
        config = MatchaTTSConfig(num_chars=32)
        model = MatchaTTS(config).to(device)

        model.train()

        model.forward(input_dummy, input_lengths, mel_spec, mel_lengths)

    def test_forward(self):
        self._test_forward(1)
