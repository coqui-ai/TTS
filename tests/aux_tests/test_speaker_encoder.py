import unittest

import torch as T

from tests import get_tests_input_path
from TTS.encoder.losses import AngleProtoLoss, GE2ELoss, SoftmaxAngleProtoLoss
from TTS.encoder.models.lstm import LSTMSpeakerEncoder
from TTS.encoder.models.resnet import ResNetSpeakerEncoder

file_path = get_tests_input_path()


class LSTMSpeakerEncoderTests(unittest.TestCase):
    # pylint: disable=R0201
    def test_in_out(self):
        dummy_input = T.rand(4, 80, 20)  # B x D x T
        dummy_hidden = [T.rand(2, 4, 128), T.rand(2, 4, 128)]
        model = LSTMSpeakerEncoder(input_dim=80, proj_dim=256, lstm_dim=768, num_lstm_layers=3)
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
        assert abs(assert_diff) < 1e-4, f" [!] output_norm has wrong values - {assert_diff}"
        # compute d for a given batch
        dummy_input = T.rand(1, 80, 240)  # B x T x D
        output = model.compute_embedding(dummy_input, num_frames=160, num_eval=5)
        assert output.shape[0] == 1
        assert output.shape[1] == 256
        assert len(output.shape) == 2


class ResNetSpeakerEncoderTests(unittest.TestCase):
    # pylint: disable=R0201
    def test_in_out(self):
        dummy_input = T.rand(4, 80, 20)  # B x D x T
        dummy_hidden = [T.rand(2, 4, 128), T.rand(2, 4, 128)]
        model = ResNetSpeakerEncoder(input_dim=80, proj_dim=256)
        # computing d vectors
        output = model.forward(dummy_input)
        assert output.shape[0] == 4
        assert output.shape[1] == 256
        output = model.forward(dummy_input, l2_norm=True)
        assert output.shape[0] == 4
        assert output.shape[1] == 256

        # check normalization
        output_norm = T.nn.functional.normalize(output, dim=1, p=2)
        assert_diff = (output_norm - output).sum().item()
        assert output.type() == "torch.FloatTensor"
        assert abs(assert_diff) < 1e-4, f" [!] output_norm has wrong values - {assert_diff}"
        # compute d for a given batch
        dummy_input = T.rand(1, 80, 240)  # B x D x T
        output = model.compute_embedding(dummy_input, num_frames=160, num_eval=10)
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


class SoftmaxAngleProtoLossTests(unittest.TestCase):
    # pylint: disable=R0201
    def test_in_out(self):
        embedding_dim = 64
        num_speakers = 5
        batch_size = 4

        dummy_label = T.randint(low=0, high=num_speakers, size=(batch_size, num_speakers))
        # check random input
        dummy_input = T.rand(batch_size, num_speakers, embedding_dim)  # num_speaker x num_utterance x dim
        loss = SoftmaxAngleProtoLoss(embedding_dim=embedding_dim, n_speakers=num_speakers)
        output = loss.forward(dummy_input, dummy_label)
        assert output.item() >= 0.0

        # check all zeros
        dummy_input = T.ones(batch_size, num_speakers, embedding_dim)  # num_speaker x num_utterance x dim
        loss = SoftmaxAngleProtoLoss(embedding_dim=embedding_dim, n_speakers=num_speakers)
        output = loss.forward(dummy_input, dummy_label)
        assert output.item() >= 0.0
