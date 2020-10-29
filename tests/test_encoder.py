import os
import unittest

import torch as T
from tests import get_tests_input_path

from TTS.speaker_encoder.losses import GE2ELoss, AngleProtoLoss
from TTS.speaker_encoder.model import SpeakerEncoder
from TTS.utils.io import load_config

file_path = get_tests_input_path()
c = load_config(os.path.join(file_path, "test_config.json"))


class SpeakerEncoderTests(unittest.TestCase):
    # pylint: disable=R0201
    def test_in_out(self):
        dummy_input = T.rand(4, 20, 80)  # B x T x D
        dummy_hidden = [T.rand(2, 4, 128), T.rand(2, 4, 128)]
        model = SpeakerEncoder(
            input_dim=80, proj_dim=256, lstm_dim=768, num_lstm_layers=3
        )
        # computing d vectors
        output = model.forward(dummy_input)
        assert output.shape[0] == 4
        assert output.shape[1] == 256
        output = model.inference(dummy_input)
        assert output.shape[0] == 4
        assert output.shape[1] == 256
        # compute d vectors by passing LSTM hidden
        # output = model.forward(dummy_input, dummy_hidden)
        # assert output.shape[0] == 4
        # assert output.shape[1] == 20
        # assert output.shape[2] == 256
        # check normalization
        output_norm = T.nn.functional.normalize(output, dim=1, p=2)
        assert_diff = (output_norm - output).sum().item()
        assert output.type() == "torch.FloatTensor"
        assert (
            abs(assert_diff) < 1e-4
        ), f" [!] output_norm has wrong values - {assert_diff}"
        # compute d for a given batch
        dummy_input = T.rand(1, 240, 80)  # B x T x D
        output = model.compute_embedding(dummy_input, num_frames=160, overlap=0.5)
        assert output.shape[0] == 1
        assert output.shape[1] == 256
        assert len(output.shape) == 2


class GE2ELossTests(unittest.TestCase):
    # pylint: disable=R0201
    def test_in_out(self):
        # check random input
        dummy_input = T.rand(4, 5, 64)  # num_speaker x num_utterance x dim
        loss = GE2ELoss(loss_method="softmax")
        output = loss.forward(dummy_input)
        assert output.item() >= 0.0
        # check all zeros
        dummy_input = T.ones(4, 5, 64)  # num_speaker x num_utterance x dim
        loss = GE2ELoss(loss_method="softmax")
        output = loss.forward(dummy_input)
        assert output.item() >= 0.0
        # check speaker loss with orthogonal d-vectors
        dummy_input = T.empty(3, 64)
        dummy_input = T.nn.init.orthogonal_(dummy_input)
        dummy_input = T.cat(
            [
                dummy_input[0].repeat(5, 1, 1).transpose(0, 1),
                dummy_input[1].repeat(5, 1, 1).transpose(0, 1),
                dummy_input[2].repeat(5, 1, 1).transpose(0, 1),
            ]
        )  # num_speaker x num_utterance x dim
        loss = GE2ELoss(loss_method="softmax")
        output = loss.forward(dummy_input)
        assert output.item() < 0.005

class AngleProtoLossTests(unittest.TestCase):
    # pylint: disable=R0201
    def test_in_out(self):
        # check random input
        dummy_input = T.rand(4, 5, 64)  # num_speaker x num_utterance x dim
        loss = AngleProtoLoss()
        output = loss.forward(dummy_input)
        assert output.item() >= 0.0

        # check all zeros
        dummy_input = T.ones(4, 5, 64)  # num_speaker x num_utterance x dim
        loss = AngleProtoLoss()
        output = loss.forward(dummy_input)
        assert output.item() >= 0.0

        # check speaker loss with orthogonal d-vectors
        dummy_input = T.empty(3, 64)
        dummy_input = T.nn.init.orthogonal_(dummy_input)
        dummy_input = T.cat(
            [
                dummy_input[0].repeat(5, 1, 1).transpose(0, 1),
                dummy_input[1].repeat(5, 1, 1).transpose(0, 1),
                dummy_input[2].repeat(5, 1, 1).transpose(0, 1),
            ]
        )  # num_speaker x num_utterance x dim
        loss = AngleProtoLoss()
        output = loss.forward(dummy_input)
        assert output.item() < 0.005

# class LoaderTest(unittest.TestCase):
#     def test_output(self):
#         items = libri_tts("/home/erogol/Data/Libri-TTS/train-clean-360/")
#         ap = AudioProcessor(**c['audio'])
#         dataset = MyDataset(ap, items, 1.6, 64, 10)
#         loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
#         count = 0
#         for mel, spk in loader:
#             print(mel.shape)
#             if count == 4:
#                 break
#             count += 1
