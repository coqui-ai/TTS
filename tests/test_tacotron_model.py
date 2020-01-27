import os
import copy
import torch
import unittest

from torch import optim
from torch import nn
from TTS.utils.generic_utils import load_config
from TTS.layers.losses import L1LossMasked
from TTS.models.tacotron import Tacotron

#pylint: disable=unused-variable

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file_path = os.path.dirname(os.path.realpath(__file__))
c = load_config(os.path.join(file_path, 'test_config.json'))


def count_parameters(model):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TacotronTrainTest(unittest.TestCase):
    @staticmethod
    def test_train_step():
        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (8, )).long().to(device)
        input_lengths[-1] = 128
        mel_spec = torch.rand(8, 30, c.audio['num_mels']).to(device)
        linear_spec = torch.rand(8, 30, c.audio['num_freq']).to(device)
        mel_lengths = torch.randint(20, 30, (8, )).long().to(device)
        stop_targets = torch.zeros(8, 30, 1).float().to(device)
        speaker_ids = torch.randint(0, 5, (8, )).long().to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()):, 0] = 1.0

        stop_targets = stop_targets.view(input_dummy.shape[0],
                                         stop_targets.size(1) // c.r, -1)
        stop_targets = (stop_targets.sum(2) >
                        0.0).unsqueeze(2).float().squeeze()

        criterion = L1LossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        model = Tacotron(
            num_chars=32,
            num_speakers=5,
            postnet_output_dim=c.audio['num_freq'],
            decoder_output_dim=c.audio['num_mels'],
            r=c.r,
            memory_size=c.memory_size
        ).to(device)  #FIXME: missing num_speakers parameter to Tacotron ctor
        model.train()
        print(" > Num parameters for Tacotron model:%s" %
              (count_parameters(model)))
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(),
                                    model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=c.lr)
        for _ in range(5):
            mel_out, linear_out, align, stop_tokens = model.forward(
                input_dummy, input_lengths, mel_spec, speaker_ids)
            optimizer.zero_grad()
            loss = criterion(mel_out, mel_spec, mel_lengths)
            stop_loss = criterion_st(stop_tokens, stop_targets)
            loss = loss + criterion(linear_out, linear_spec,
                                    mel_lengths) + stop_loss
            loss.backward()
            optimizer.step()
        # check parameter changes
        count = 0
        for param, param_ref in zip(model.parameters(),
                                    model_ref.parameters()):
            # ignore pre-higway layer since it works conditional
            # if count not in [145, 59]:
            assert (param != param_ref).any(
            ), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref)
            count += 1


class TacotronGSTTrainTest(unittest.TestCase):
    @staticmethod
    def test_train_step():
        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (8, )).long().to(device)
        input_lengths[-1] = 128
        mel_spec = torch.rand(8, 120, c.audio['num_mels']).to(device)
        linear_spec = torch.rand(8, 120, c.audio['num_freq']).to(device)
        mel_lengths = torch.randint(20, 120, (8, )).long().to(device)
        stop_targets = torch.zeros(8, 120, 1).float().to(device)
        speaker_ids = torch.randint(0, 5, (8, )).long().to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()):, 0] = 1.0

        stop_targets = stop_targets.view(input_dummy.shape[0],
                                         stop_targets.size(1) // c.r, -1)
        stop_targets = (stop_targets.sum(2) >
                        0.0).unsqueeze(2).float().squeeze()

        criterion = L1LossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        model = Tacotron(
            num_chars=32,
            num_speakers=5,
            gst=True,
            postnet_output_dim=c.audio['num_freq'],
            decoder_output_dim=c.audio['num_mels'],
            r=c.r,
            memory_size=c.memory_size
        ).to(device)  #FIXME: missing num_speakers parameter to Tacotron ctor
        model.train()
        print(model)
        print(" > Num parameters for Tacotron GST model:%s" %
              (count_parameters(model)))
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(),
                                    model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=c.lr)
        for _ in range(10):
            mel_out, linear_out, align, stop_tokens = model.forward(
                input_dummy, input_lengths, mel_spec, speaker_ids)
            optimizer.zero_grad()
            loss = criterion(mel_out, mel_spec, mel_lengths)
            stop_loss = criterion_st(stop_tokens, stop_targets)
            loss = loss + criterion(linear_out, linear_spec,
                                    mel_lengths) + stop_loss
            loss.backward()
            optimizer.step()
        # check parameter changes
        count = 0
        for param, param_ref in zip(model.parameters(),
                                    model_ref.parameters()):
            # ignore pre-higway layer since it works conditional
            assert (param != param_ref).any(
            ), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref)
            count += 1
