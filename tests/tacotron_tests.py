import os
import copy
import torch
import unittest
import numpy as np

from torch import optim
from torch import nn
from TTS.utils.generic_utils import load_config
from TTS.layers.losses import L1LossMasked
from TTS.models.tacotron import Tacotron

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file_path = os.path.dirname(os.path.realpath(__file__))
c = load_config(os.path.join(file_path, 'test_config.json'))


class TacotronTrainTest(unittest.TestCase):
    def test_train_step(self):
        input = torch.randint(0, 24, (8, 128)).long().to(device)
        mel_spec = torch.rand(8, 30, c.num_mels).to(device)
        linear_spec = torch.rand(8, 30, c.num_freq).to(device)
        mel_lengths = torch.randint(20, 30, (8, )).long().to(device)
        stop_targets = torch.zeros(8, 30, 1).float().to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()):, 0] = 1.0

        stop_targets = stop_targets.view(input.shape[0],
                                         stop_targets.size(1) // c.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float()

        criterion = L1LossMasked().to(device)
        criterion_st = nn.BCELoss().to(device)
        model = Tacotron(c.embedding_size, c.num_freq, c.num_mels,
                         c.r).to(device)
        model.train()
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(),
                                    model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=c.lr)
        for i in range(5):
            mel_out, linear_out, align, stop_tokens = model.forward(
                input, mel_spec)
            assert stop_tokens.data.max() <= 1.0
            assert stop_tokens.data.min() >= 0.0
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