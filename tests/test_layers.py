import unittest
import torch as T

from TTS.tts.layers.tacotron.tacotron import Prenet, CBHG, Decoder, Encoder
from TTS.tts.layers.losses import L1LossMasked, SSIMLoss
from TTS.tts.utils.generic_utils import sequence_mask

# pylint: disable=unused-variable


class PrenetTests(unittest.TestCase):
    def test_in_out(self):  #pylint: disable=no-self-use
        layer = Prenet(128, out_features=[256, 128])
        dummy_input = T.rand(4, 128)

        print(layer)
        output = layer(dummy_input)
        assert output.shape[0] == 4
        assert output.shape[1] == 128


class CBHGTests(unittest.TestCase):
    def test_in_out(self):
        #pylint: disable=attribute-defined-outside-init
        layer = self.cbhg = CBHG(
            128,
            K=8,
            conv_bank_features=80,
            conv_projections=[160, 128],
            highway_features=80,
            gru_features=80,
            num_highways=4)
        # B x D x T
        dummy_input = T.rand(4, 128, 8)

        print(layer)
        output = layer(dummy_input)
        assert output.shape[0] == 4
        assert output.shape[1] == 8
        assert output.shape[2] == 160


class DecoderTests(unittest.TestCase):
    @staticmethod
    def test_in_out():
        layer = Decoder(
            in_channels=256,
            frame_channels=80,
            r=2,
            memory_size=4,
            attn_windowing=False,
            attn_norm="sigmoid",
            attn_K=5,
            attn_type="original",
            prenet_type='original',
            prenet_dropout=True,
            forward_attn=True,
            trans_agent=True,
            forward_attn_mask=True,
            location_attn=True,
            separate_stopnet=True)
        dummy_input = T.rand(4, 8, 256)
        dummy_memory = T.rand(4, 2, 80)

        output, alignment, stop_tokens = layer(
            dummy_input, dummy_memory, mask=None)

        assert output.shape[0] == 4
        assert output.shape[1] == 80, "size not {}".format(output.shape[1])
        assert output.shape[2] == 2, "size not {}".format(output.shape[2])
        assert stop_tokens.shape[0] == 4

class EncoderTests(unittest.TestCase):
    def test_in_out(self):  #pylint: disable=no-self-use
        layer = Encoder(128)
        dummy_input = T.rand(4, 8, 128)

        print(layer)
        output = layer(dummy_input)
        print(output.shape)
        assert output.shape[0] == 4
        assert output.shape[1] == 8
        assert output.shape[2] == 256  # 128 * 2 BiRNN


class L1LossMaskedTests(unittest.TestCase):
    def test_in_out(self):  #pylint: disable=no-self-use
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
        mask = (
            (sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert output.item() == 1.0, "1.0 vs {}".format(output.item())

        dummy_input = T.rand(4, 8, 128).float()
        dummy_target = dummy_input.detach()
        dummy_length = (T.arange(5, 9)).long()
        mask = (
            (sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
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
        mask = (
            (sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert abs(output.item() - 1.0) < 1e-5, "1.0 vs {}".format(output.item())

        dummy_input = T.rand(4, 8, 128).float()
        dummy_target = dummy_input.detach()
        dummy_length = (T.arange(5, 9)).long()
        mask = (
            (sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert output.item() == 0, "0 vs {}".format(output.item())


class SSIMLossTests(unittest.TestCase):
    def test_in_out(self):  #pylint: disable=no-self-use
        # test input == target
        layer = SSIMLoss()
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
        assert abs(output.item() - 1.0) < 1e-4, "1.0 vs {}".format(output.item())

        # test if padded values of input makes any difference
        dummy_input = T.ones(4, 8, 128).float()
        dummy_target = T.zeros(4, 8, 128).float()
        dummy_length = (T.arange(5, 9)).long()
        mask = (
            (sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert abs(output.item() - 1.0) < 1e-4, "1.0 vs {}".format(output.item())

        dummy_input = T.rand(4, 8, 128).float()
        dummy_target = dummy_input.detach()
        dummy_length = (T.arange(5, 9)).long()
        mask = (
            (sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
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
        mask = (
            (sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert abs(output.item() - 1.0) < 1e-5, "1.0 vs {}".format(output.item())

        dummy_input = T.rand(4, 8, 128).float()
        dummy_target = dummy_input.detach()
        dummy_length = (T.arange(5, 9)).long()
        mask = (
            (sequence_mask(dummy_length).float() - 1.0) * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert output.item() == 0, "0 vs {}".format(output.item())
