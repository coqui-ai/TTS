import unittest
import torch as T

from TTS.layers.tacotron import Prenet, CBHG, Decoder, Encoder


class PrenetTests(unittest.TestCase):

    def test_in_out(self):
        layer = Prenet(128, out_features=[256, 128])
        dummy_input = T.autograd.Variable(T.rand(4, 128))

        print(layer)
        output = layer(dummy_input)
        assert output.shape[0] == 4
        assert output.shape[1] == 128


class CBHGTests(unittest.TestCase):

    def test_in_out(self):
        layer = CBHG(128, K= 6, projections=[128, 128], num_highways=2)
        dummy_input = T.autograd.Variable(T.rand(4, 8, 128))

        print(layer)
        output = layer(dummy_input)
        assert output.shape[0] == 4
        assert output.shape[1] == 8
        assert output.shape[2] == 256


class DecoderTests(unittest.TestCase):

    def test_in_out(self):
        layer = Decoder(in_features=128, memory_dim=32, r=5)
        dummy_input = T.autograd.Variable(T.rand(4, 8, 128))
        dummy_memory = T.autograd.Variable(T.rand(4, 120, 32))

        print(layer)
        output, alignment = layer(dummy_input, dummy_memory)
        print(output.shape)
        
        assert output.shape[0] == 4
        assert output.shape[1] == 120 / 5
        assert output.shape[2] == 32 * 5
        

class EncoderTests(unittest.TestCase):

    def test_in_out(self):
        layer = Encoder(128)
        dummy_input = T.autograd.Variable(T.rand(4, 8, 128))

        print(layer)
        output = layer(dummy_input)
        print(output.shape)
        assert output.shape[0] == 4
        assert output.shape[1] == 8
        assert output.shape[2] == 256  # 128 * 2 BiRNN

