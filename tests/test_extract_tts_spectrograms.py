import os
import unittest

import torch

from tests import get_tests_input_path

from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.models.glow_tts import GlowTTS

from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config

from TTS.bin.extract_tts_spectrograms import inference

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

c = load_config(os.path.join(get_tests_input_path(), "test_config.json"))
# set params from tacotron inference
c.bidirectional_decoder = False
c.double_decoder_consistency = False
ap = AudioProcessor(**c.audio)


# pylint: disable=protected-access
class TestExtractTTSSpectrograms(unittest.TestCase):
    @staticmethod
    def test_GlowTTS():
        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (8,)).long().to(device)
        input_lengths[-1] = 128
        mel_spec = torch.rand(8, c.audio["num_mels"], 30).to(device)
        mel_lengths = torch.randint(20, 30, (8,)).long().to(device)

        # create model
        model = GlowTTS(
            num_chars=32,
            hidden_channels_enc=48,
            hidden_channels_dec=48,
            hidden_channels_dp=32,
            out_channels=c.audio["num_mels"],
            encoder_type="rel_pos_transformer",
            encoder_params={
                "kernel_size": 3,
                "dropout_p": 0.1,
                "num_layers": 6,
                "num_heads": 2,
                "hidden_channels_ffn": 16,  # 4 times the hidden_channels
                "input_length": None,
            },
            use_encoder_prenet=True,
            num_flow_blocks_dec=12,
            kernel_size_dec=5,
            dilation_rate=1,
            num_block_layers=4,
            dropout_p_dec=0.0,
            num_speakers=0,
            c_in_channels=0,
            num_splits=4,
            num_squeeze=1,
            sigmoid_scale=False,
            mean_only=False,
        ).to(device)

        model.eval()
        _ = inference('glow_tts', model, c, ap, input_dummy, input_lengths, mel_spec.permute(0, 2, 1), mel_lengths)
        print("GlowTTS extract tts spectrograms ok !")

    @staticmethod
    def test_Tacotron():
        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 128, (8,)).long().to(device)
        input_lengths = torch.sort(input_lengths, descending=True)[0]
        mel_spec = torch.rand(8, 30, c.audio["num_mels"]).to(device)
        mel_lengths = torch.randint(20, 30, (8,)).long().to(device)
        mel_lengths[0] = 30

        # create model
        model = Tacotron2(num_chars=24, decoder_output_dim=c.audio["num_mels"], r=c.r, num_speakers=1).to(device)
        model.eval()

        _ = inference('tacotron2', model, c, ap, input_dummy, input_lengths, mel_spec, mel_lengths)
        print("Tacotron extract tts spectrograms ok !")
