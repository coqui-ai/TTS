import unittest

import torch as T

from TTS.tts.layers.tacotron.tacotron import CBHG, Decoder, Encoder, Prenet

# pylint: disable=unused-variable


class PrenetTests(unittest.TestCase):
    def test_in_out(self):  # pylint: disable=no-self-use
        layer = Prenet(128, out_features=[256, 128])
        dummy_input = T.rand(4, 128)

        print(layer)
        output = layer(dummy_input)
        assert output.shape[0] == 4
        assert output.shape[1] == 128


class CBHGTests(unittest.TestCase):
    def test_in_out(self):
        # pylint: disable=attribute-defined-outside-init
        layer = self.cbhg = CBHG(
            128,
            K=8,
            conv_bank_features=80,
            conv_projections=[160, 128],
            highway_features=80,
            gru_features=80,
            num_highways=4,
        )
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
            prenet_type="original",
            prenet_dropout=True,
            forward_attn=True,
            trans_agent=True,
            forward_attn_mask=True,
            location_attn=True,
            separate_stopnet=True,
            max_decoder_steps=50,
        )
        dummy_input = T.rand(4, 8, 256)
        dummy_memory = T.rand(4, 2, 80)

        output, alignment, stop_tokens = layer(dummy_input, dummy_memory, mask=None)

        assert output.shape[0] == 4
        assert output.shape[1] == 80, "size not {}".format(output.shape[1])
        assert output.shape[2] == 2, "size not {}".format(output.shape[2])
        assert stop_tokens.shape[0] == 4


class EncoderTests(unittest.TestCase):
    def test_in_out(self):  # pylint: disable=no-self-use
        layer = Encoder(128)
        dummy_input = T.rand(4, 8, 128)

        print(layer)
        output = layer(dummy_input)
        print(output.shape)
        assert output.shape[0] == 4
        assert output.shape[1] == 8
        assert output.shape[2] == 256  # 128 * 2 BiRNN
