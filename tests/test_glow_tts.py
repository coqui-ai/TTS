import copy
import os
import unittest

import torch
from tests import get_tests_input_path
from torch import optim

from TTS.tts.layers.losses import GlowTTSLoss
from TTS.tts.models.glow_tts import GlowTTS
from TTS.utils.io import load_config
from TTS.utils.audio import AudioProcessor

#pylint: disable=unused-variable

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

c = load_config(os.path.join(get_tests_input_path(), 'test_config.json'))

ap = AudioProcessor(**c.audio)
WAV_FILE = os.path.join(get_tests_input_path(), "example_1.wav")


def count_parameters(model):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GlowTTSTrainTest(unittest.TestCase):
    @staticmethod
    def test_train_step():
        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (8, )).long().to(device)
        input_lengths[-1] = 128
        mel_spec = torch.rand(8, c.audio['num_mels'], 30).to(device)
        mel_lengths = torch.randint(20, 30, (8, )).long().to(device)
        speaker_ids = torch.randint(0, 5, (8, )).long().to(device)

        criterion = GlowTTSLoss()

        # model to train
        model = GlowTTS(
            num_chars=32,
            hidden_channels_enc=48,
            hidden_channels_dec=48,
            hidden_channels_dp=32,
            out_channels=80,
            encoder_type='rel_pos_transformer',
            encoder_params={
                'kernel_size': 3,
                'dropout_p': 0.1,
                'num_layers': 6,
                'num_heads': 2,
                'hidden_channels_ffn': 16,  # 4 times the hidden_channels
                'input_length': None
            },
            use_encoder_prenet=True,
            num_flow_blocks_dec=12,
            kernel_size_dec=5,
            dilation_rate=1,
            num_block_layers=4,
            dropout_p_dec=0.,
            num_speakers=0,
            c_in_channels=0,
            num_splits=4,
            num_squeeze=1,
            sigmoid_scale=False,
            mean_only=False).to(device)

        # reference model to compare model weights
        model_ref = GlowTTS(
            num_chars=32,
            hidden_channels_enc=48,
            hidden_channels_dec=48,
            hidden_channels_dp=32,
            out_channels=80,
            encoder_type='rel_pos_transformer',
            encoder_params={
                'kernel_size': 3,
                'dropout_p': 0.1,
                'num_layers': 6,
                'num_heads': 2,
                'hidden_channels_ffn': 16,  # 4 times the hidden_channels
                'input_length': None
            },
            use_encoder_prenet=True,
            num_flow_blocks_dec=12,
            kernel_size_dec=5,
            dilation_rate=1,
            num_block_layers=4,
            dropout_p_dec=0.,
            num_speakers=0,
            c_in_channels=0,
            num_splits=4,
            num_squeeze=1,
            sigmoid_scale=False,
            mean_only=False).to(device)

        model.train()
        print(" > Num parameters for GlowTTS model:%s" %
              (count_parameters(model)))

        # pass the state to ref model
        model_ref.load_state_dict(copy.deepcopy(model.state_dict()))

        count = 0
        for param, param_ref in zip(model.parameters(),
                                    model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for _ in range(5):
            optimizer.zero_grad()
            z, logdet, y_mean, y_log_scale, alignments, o_dur_log, o_total_dur = model.forward(
                input_dummy, input_lengths, mel_spec, mel_lengths, None)
            loss_dict = criterion(z, y_mean, y_log_scale, logdet, mel_lengths,
                                  o_dur_log, o_total_dur, input_lengths)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()

        # check parameter changes
        count = 0
        for param, param_ref in zip(model.parameters(),
                                    model_ref.parameters()):
            assert (param != param_ref).any(
            ), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref)
            count += 1
