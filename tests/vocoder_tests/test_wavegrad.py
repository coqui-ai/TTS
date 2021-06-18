import unittest

import numpy as np
import torch
from torch import optim

from TTS.vocoder.configs import WavegradConfig
from TTS.vocoder.models.wavegrad import Wavegrad, WavegradArgs

# pylint: disable=unused-variable

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WavegradTrainTest(unittest.TestCase):
    def test_train_step(self):  # pylint: disable=no-self-use
        """Test if all layers are updated in a basic training cycle"""
        input_dummy = torch.rand(8, 1, 20 * 300).to(device)
        mel_spec = torch.rand(8, 80, 20).to(device)

        criterion = torch.nn.L1Loss().to(device)
        args = WavegradArgs(
            in_channels=80,
            out_channels=1,
            upsample_factors=[5, 5, 3, 2, 2],
            upsample_dilations=[[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]],
        )
        config = WavegradConfig(model_params=args)
        model = Wavegrad(config)

        model_ref = Wavegrad(config)
        model.train()
        model.to(device)
        betas = np.linspace(1e-6, 1e-2, 1000)
        model.compute_noise_level(betas)
        model_ref.load_state_dict(model.state_dict())
        model_ref.to(device)
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for i in range(5):
            y_hat = model.forward(input_dummy, mel_spec, torch.rand(8).to(device))
            optimizer.zero_grad()
            loss = criterion(y_hat, input_dummy)
            loss.backward()
            optimizer.step()
        # check parameter changes
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            # ignore pre-higway layer since it works conditional
            # if count not in [145, 59]:
            assert (param != param_ref).any(), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref
            )
            count += 1
