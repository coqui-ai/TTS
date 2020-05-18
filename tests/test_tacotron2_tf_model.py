import os
import copy
import torch
import unittest
import numpy as np
import tensorflow as tf

from torch import optim
from torch import nn
from TTS.utils.io import load_config
from TTS.layers.losses import MSELossMasked
from TTS.tf.models.tacotron2 import Tacotron2

#pylint: disable=unused-variable

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file_path = os.path.dirname(os.path.realpath(__file__))
c = load_config(os.path.join(file_path, 'test_config.json'))


class TacotronTFTrainTest(unittest.TestCase):
    def test_train_step(self):
        ''' test forward pass '''
        input = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 128, (8, )).long().to(device)
        input_lengths = torch.sort(input_lengths, descending=True)[0]
        mel_spec = torch.rand(8, 30, c.audio['num_mels']).to(device)
        mel_postnet_spec = torch.rand(8, 30, c.audio['num_mels']).to(device)
        mel_lengths = torch.randint(20, 30, (8, )).long().to(device)
        stop_targets = torch.zeros(8, 30, 1).float().to(device)
        speaker_ids = torch.randint(0, 5, (8, )).long().to(device)

        input = tf.convert_to_tensor(input.cpu().numpy())
        input_lengths = tf.convert_to_tensor(input_lengths.cpu().numpy())
        mel_spec = tf.convert_to_tensor(mel_spec.cpu().numpy())

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()):, 0] = 1.0

        stop_targets = stop_targets.view(input.shape[0],
                                         stop_targets.size(1) // c.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze()

        model = Tacotron2(num_chars=24, r=c.r, num_speakers=5)
        # training pass
        output = model(input, input_lengths, mel_spec, training=True)

        # check model output shapes
        assert np.all(output[0].shape == mel_spec.shape)
        assert np.all(output[1].shape == mel_spec.shape)
        assert output[2].shape[2] == input.shape[1]
        assert output[2].shape[1] == (mel_spec.shape[1] // model.decoder.r)
        assert output[3].shape[1] == (mel_spec.shape[1] // model.decoder.r)

        # inference pass
        output = model(input, training=False)
