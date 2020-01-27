import os
import copy
import torch
import unittest
import numpy as np

from torch import optim
from torch import nn
from TTS.utils.generic_utils import load_config
from TTS.layers.losses import MSELossMasked
from TTS.models.tacotron2 import Tacotron2

#pylint: disable=unused-variable

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file_path = os.path.dirname(os.path.realpath(__file__))
c = load_config(os.path.join(file_path, 'test_config.json'))


class TacotronTrainTest(unittest.TestCase):
    def test_train_step(self):
        input = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 128, (8, )).long().to(device)
        input_lengths = torch.sort(input_lengths, descending=True)[0]
        mel_spec = torch.rand(8, 30, c.audio['num_mels']).to(device)
        mel_postnet_spec = torch.rand(8, 30, c.audio['num_mels']).to(device)
        mel_lengths = torch.randint(20, 30, (8, )).long().to(device)
        stop_targets = torch.zeros(8, 30, 1).float().to(device)
        speaker_ids = torch.randint(0, 5, (8, )).long().to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()):, 0] = 1.0

        stop_targets = stop_targets.view(input.shape[0],
                                         stop_targets.size(1) // c.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze()

        criterion = MSELossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        model = Tacotron2(num_chars=24, r=c.r, num_speakers=5).to(device)
        model.train()
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(),
                                    model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=c.lr)
        for i in range(5):
            mel_out, mel_postnet_out, align, stop_tokens = model.forward(
                input, input_lengths, mel_spec, speaker_ids)
            assert torch.sigmoid(stop_tokens).data.max() <= 1.0
            assert torch.sigmoid(stop_tokens).data.min() >= 0.0
            optimizer.zero_grad()
            loss = criterion(mel_out, mel_spec, mel_lengths)
            stop_loss = criterion_st(stop_tokens, stop_targets)
            loss = loss + criterion(mel_postnet_out, mel_postnet_spec, mel_lengths) + stop_loss
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
