import os
import unittest

import numpy as np
import tensorflow as tf
import torch
from tests import get_tests_input_path
from TTS.tts.tf.models.tacotron2 import Tacotron2
from TTS.tts.tf.utils.tflite import (convert_tacotron2_to_tflite,
                                     load_tflite_model)
from TTS.utils.io import load_config

tf.get_logger().setLevel('INFO')



#pylint: disable=unused-variable

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

c = load_config(os.path.join(get_tests_input_path(), 'test_config.json'))


class TacotronTFTrainTest(unittest.TestCase):

    @staticmethod
    def generate_dummy_inputs():
        chars_seq = torch.randint(0, 24, (8, 128)).long().to(device)
        chars_seq_lengths = torch.randint(100, 128, (8, )).long().to(device)
        chars_seq_lengths = torch.sort(chars_seq_lengths, descending=True)[0]
        mel_spec = torch.rand(8, 30, c.audio['num_mels']).to(device)
        mel_postnet_spec = torch.rand(8, 30, c.audio['num_mels']).to(device)
        mel_lengths = torch.randint(20, 30, (8, )).long().to(device)
        stop_targets = torch.zeros(8, 30, 1).float().to(device)
        speaker_ids = torch.randint(0, 5, (8, )).long().to(device)

        chars_seq = tf.convert_to_tensor(chars_seq.cpu().numpy())
        chars_seq_lengths = tf.convert_to_tensor(chars_seq_lengths.cpu().numpy())
        mel_spec = tf.convert_to_tensor(mel_spec.cpu().numpy())
        return chars_seq, chars_seq_lengths, mel_spec, mel_postnet_spec, mel_lengths,\
            stop_targets, speaker_ids

    def test_train_step(self):
        ''' test forward pass '''
        chars_seq, chars_seq_lengths, mel_spec, mel_postnet_spec, mel_lengths,\
            stop_targets, speaker_ids = self.generate_dummy_inputs()

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()):, 0] = 1.0

        stop_targets = stop_targets.view(chars_seq.shape[0],
                                         stop_targets.size(1) // c.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze()

        model = Tacotron2(num_chars=24, r=c.r, num_speakers=5)
        # training pass
        output = model(chars_seq, chars_seq_lengths, mel_spec, training=True)

        # check model output shapes
        assert np.all(output[0].shape == mel_spec.shape)
        assert np.all(output[1].shape == mel_spec.shape)
        assert output[2].shape[2] == chars_seq.shape[1]
        assert output[2].shape[1] == (mel_spec.shape[1] // model.decoder.r)
        assert output[3].shape[1] == (mel_spec.shape[1] // model.decoder.r)

        # inference pass
        output = model(chars_seq, training=False)

    def test_forward_attention(self,):
        chars_seq, chars_seq_lengths, mel_spec, mel_postnet_spec, mel_lengths,\
            stop_targets, speaker_ids = self.generate_dummy_inputs()

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()):, 0] = 1.0

        stop_targets = stop_targets.view(chars_seq.shape[0],
                                         stop_targets.size(1) // c.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze()

        model = Tacotron2(num_chars=24, r=c.r, num_speakers=5, forward_attn=True)
        # training pass
        output = model(chars_seq, chars_seq_lengths, mel_spec, training=True)

        # check model output shapes
        assert np.all(output[0].shape == mel_spec.shape)
        assert np.all(output[1].shape == mel_spec.shape)
        assert output[2].shape[2] == chars_seq.shape[1]
        assert output[2].shape[1] == (mel_spec.shape[1] // model.decoder.r)
        assert output[3].shape[1] == (mel_spec.shape[1] // model.decoder.r)

        # inference pass
        output = model(chars_seq, training=False)

    def test_tflite_conversion(self, ):  #pylint:disable=no-self-use
        model = Tacotron2(num_chars=24,
                          num_speakers=0,
                          r=3,
                          postnet_output_dim=80,
                          decoder_output_dim=80,
                          attn_type='original',
                          attn_win=False,
                          attn_norm='sigmoid',
                          prenet_type='original',
                          prenet_dropout=True,
                          forward_attn=False,
                          trans_agent=False,
                          forward_attn_mask=False,
                          location_attn=True,
                          attn_K=0,
                          separate_stopnet=True,
                          bidirectional_decoder=False,
                          enable_tflite=True)
        model.build_inference()
        convert_tacotron2_to_tflite(model, output_path='test_tacotron2.tflite', experimental_converter=True)
        # init tflite model
        tflite_model = load_tflite_model('test_tacotron2.tflite')
        # fake input
        inputs = tf.random.uniform([1, 4], maxval=10, dtype=tf.int32)  #pylint:disable=unexpected-keyword-arg
        # run inference
        # get input and output details
        input_details = tflite_model.get_input_details()
        output_details = tflite_model.get_output_details()
        # reshape input tensor for the new input shape
        tflite_model.resize_tensor_input(input_details[0]['index'], inputs.shape)  #pylint:disable=unexpected-keyword-arg
        tflite_model.allocate_tensors()
        detail = input_details[0]
        input_shape = detail['shape']
        tflite_model.set_tensor(detail['index'], inputs)
        # run the tflite_model
        tflite_model.invoke()
        # collect outputs
        decoder_output = tflite_model.get_tensor(output_details[0]['index'])
        postnet_output = tflite_model.get_tensor(output_details[1]['index'])
        # remove tflite binary
        os.remove('test_tacotron2.tflite')
