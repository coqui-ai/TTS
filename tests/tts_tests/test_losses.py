import unittest

import torch as T

from TTS.tts.layers.losses import BCELossMasked, L1LossMasked, MSELossMasked, SSIMLoss
from TTS.tts.utils.helpers import sequence_mask


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
        dummy_target = -dummy_target

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


class BCELossTest(unittest.TestCase):
    def test_in_out(self):  # pylint: disable=no-self-use
        layer = BCELossMasked(pos_weight=5.0)

        length = T.tensor([95])
        target = (
            1.0 - sequence_mask(length - 1, 100).float()
        )  # [0, 0, .... 1, 1] where the first 1 is the last mel frame
        true_x = target * 200 - 100  # creates logits of [-100, -100, ... 100, 100] corresponding to target
        zero_x = T.zeros(target.shape) - 100.0  # simulate logits if it never stops decoding
        early_x = -200.0 * sequence_mask(length - 3, 100).float() + 100.0  # simulate logits on early stopping
        late_x = -200.0 * sequence_mask(length + 1, 100).float() + 100.0  # simulate logits on late stopping

        loss = layer(true_x, target, length)
        self.assertEqual(loss.item(), 0.0)

        loss = layer(early_x, target, length)
        self.assertAlmostEqual(loss.item(), 2.1053, places=4)

        loss = layer(late_x, target, length)
        self.assertAlmostEqual(loss.item(), 5.2632, places=4)

        loss = layer(zero_x, target, length)
        self.assertAlmostEqual(loss.item(), 5.2632, places=4)

        # pos_weight should be < 1 to penalize early stopping
        layer = BCELossMasked(pos_weight=0.2)
        loss = layer(true_x, target, length)
        self.assertEqual(loss.item(), 0.0)

        # when pos_weight < 1 overweight the early stopping loss

        loss_early = layer(early_x, target, length)
        loss_late = layer(late_x, target, length)
        self.assertGreater(loss_early.item(), loss_late.item())
