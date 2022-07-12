import unittest
import torch as T

from TTS.tts.utils.helpers import sequence_mask
from TTS.tts.layers.losses import L1LossMasked, SSIMLoss, MSELossMasked


class L1LossMaskedTests(unittest.TestCase):
    def test_in_out(self):  # pylint: disable=no-self-use
        # test input == target
        layer = L1LossMasked(seq_len_norm=False)
        dummy_input = T.ones(4, 8, 128).float()
        dummy_target = T.ones(4, 8, 128).float()
        dummy_length = (T.ones(4) * 8).long()
        output = layer(dummy_input, dummy_target, dummy_length)
        assert output.item() == 0.0

        # test input != target
        dummy_input = T.ones(4, 8, 128).float()
        dummy_target = T.zeros(4, 8, 128).float()
        dummy_length = (T.ones(4) * 8).long()
        output = layer(dummy_input, dummy_target, dummy_length)
        assert output.item() == 1.0, "1.0 vs {}".format(output.item())

        # test if padded values of input makes any difference
        dummy_input = T.ones(4, 8, 128).float()
        dummy_target = T.zeros(4, 8, 128).float()
        dummy_length = (T.arange(5, 9)).long()
        mask = ((sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert output.item() == 1.0, "1.0 vs {}".format(output.item())

        dummy_input = T.rand(4, 8, 128).float()
        dummy_target = dummy_input.detach()
        dummy_length = (T.arange(5, 9)).long()
        mask = ((sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert output.item() == 0, "0 vs {}".format(output.item())

        # seq_len_norm = True
        # test input == target
        layer = L1LossMasked(seq_len_norm=True)
        dummy_input = T.ones(4, 8, 128).float()
        dummy_target = T.ones(4, 8, 128).float()
        dummy_length = (T.ones(4) * 8).long()
        output = layer(dummy_input, dummy_target, dummy_length)
        assert output.item() == 0.0

        # test input != target
        dummy_input = T.ones(4, 8, 128).float()
        dummy_target = T.zeros(4, 8, 128).float()
        dummy_length = (T.ones(4) * 8).long()
        output = layer(dummy_input, dummy_target, dummy_length)
        assert output.item() == 1.0, "1.0 vs {}".format(output.item())

        # test if padded values of input makes any difference
        dummy_input = T.ones(4, 8, 128).float()
        dummy_target = T.zeros(4, 8, 128).float()
        dummy_length = (T.arange(5, 9)).long()
        mask = ((sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert abs(output.item() - 1.0) < 1e-5, "1.0 vs {}".format(output.item())

        dummy_input = T.rand(4, 8, 128).float()
        dummy_target = dummy_input.detach()
        dummy_length = (T.arange(5, 9)).long()
        mask = ((sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert output.item() == 0, "0 vs {}".format(output.item())


class MSELossMaskedTests(unittest.TestCase):
    def test_in_out(self):  # pylint: disable=no-self-use
        # test input == target
        layer = MSELossMasked(seq_len_norm=False)
        dummy_input = T.ones(4, 8, 128).float()
        dummy_target = T.ones(4, 8, 128).float()
        dummy_length = (T.ones(4) * 8).long()
        output = layer(dummy_input, dummy_target, dummy_length)
        assert output.item() == 0.0

        # test input != target
        dummy_input = T.ones(4, 8, 128).float()
        dummy_target = T.zeros(4, 8, 128).float()
        dummy_length = (T.ones(4) * 8).long()
        output = layer(dummy_input, dummy_target, dummy_length)
        assert output.item() == 1.0, "1.0 vs {}".format(output.item())

        # test if padded values of input makes any difference
        dummy_input = T.ones(4, 8, 128).float()
        dummy_target = T.zeros(4, 8, 128).float()
        dummy_length = (T.arange(5, 9)).long()
        mask = ((sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert output.item() == 1.0, "1.0 vs {}".format(output.item())

        dummy_input = T.rand(4, 8, 128).float()
        dummy_target = dummy_input.detach()
        dummy_length = (T.arange(5, 9)).long()
        mask = ((sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert output.item() == 0, "0 vs {}".format(output.item())

        # seq_len_norm = True
        # test input == target
        layer = MSELossMasked(seq_len_norm=True)
        dummy_input = T.ones(4, 8, 128).float()
        dummy_target = T.ones(4, 8, 128).float()
        dummy_length = (T.ones(4) * 8).long()
        output = layer(dummy_input, dummy_target, dummy_length)
        assert output.item() == 0.0

        # test input != target
        dummy_input = T.ones(4, 8, 128).float()
        dummy_target = T.zeros(4, 8, 128).float()
        dummy_length = (T.ones(4) * 8).long()
        output = layer(dummy_input, dummy_target, dummy_length)
        assert output.item() == 1.0, "1.0 vs {}".format(output.item())

        # test if padded values of input makes any difference
        dummy_input = T.ones(4, 8, 128).float()
        dummy_target = T.zeros(4, 8, 128).float()
        dummy_length = (T.arange(5, 9)).long()
        mask = ((sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert abs(output.item() - 1.0) < 1e-5, "1.0 vs {}".format(output.item())

        dummy_input = T.rand(4, 8, 128).float()
        dummy_target = dummy_input.detach()
        dummy_length = (T.arange(5, 9)).long()
        mask = ((sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert output.item() == 0, "0 vs {}".format(output.item())



class SSIMLossTests(unittest.TestCase):
    def test_in_out(self):  # pylint: disable=no-self-use
        # test input == target
        layer = SSIMLoss()
        dummy_input = T.ones(4, 57, 128).float()
        dummy_target = T.ones(4, 57, 128).float()
        dummy_length = (T.ones(4) * 8).long()
        output = layer(dummy_input, dummy_target, dummy_length)
        assert output.item() == 0.0

        # test input != target
        dummy_input = T.arange(0, 4 * 57 * 128)
        dummy_input = dummy_input.reshape(4, 57, 128).float()
        dummy_target = T.arange(-4 * 57 * 128, 0)
        dummy_target = dummy_target.reshape(4, 57, 128).float()
        dummy_target = (-dummy_target)

        dummy_length = (T.ones(4) * 58).long()
        output = layer(dummy_input, dummy_target, dummy_length)
        assert output.item() >= 1.0, "0 vs {}".format(output.item())

        # test if padded values of input makes any difference
        dummy_input = T.ones(4, 57, 128).float()
        dummy_target = T.zeros(4, 57, 128).float()
        dummy_length = (T.arange(54, 58)).long()
        mask = ((sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert output.item() == 0.0

        dummy_input = T.rand(4, 57, 128).float()
        dummy_target = dummy_input.detach()
        dummy_length = (T.arange(54, 58)).long()
        mask = ((sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert output.item() == 0, "0 vs {}".format(output.item())

        # seq_len_norm = True
        # test input == target
        layer = L1LossMasked(seq_len_norm=True)
        dummy_input = T.ones(4, 57, 128).float()
        dummy_target = T.ones(4, 57, 128).float()
        dummy_length = (T.ones(4) * 8).long()
        output = layer(dummy_input, dummy_target, dummy_length)
        assert output.item() == 0.0

        # test input != target
        dummy_input = T.ones(4, 57, 128).float()
        dummy_target = T.zeros(4, 57, 128).float()
        dummy_length = (T.ones(4) * 8).long()
        output = layer(dummy_input, dummy_target, dummy_length)
        assert output.item() == 1.0, "1.0 vs {}".format(output.item())

        # test if padded values of input makes any difference
        dummy_input = T.ones(4, 57, 128).float()
        dummy_target = T.zeros(4, 57, 128).float()
        dummy_length = (T.arange(54, 58)).long()
        mask = ((sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert abs(output.item() - 1.0) < 1e-5, "1.0 vs {}".format(output.item())

        dummy_input = T.rand(4, 57, 128).float()
        dummy_target = dummy_input.detach()
        dummy_length = (T.arange(54, 58)).long()
        mask = ((sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert output.item() == 0, "0 vs {}".format(output.item())
