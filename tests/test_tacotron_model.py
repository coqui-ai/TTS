import copy
import os
import unittest

import torch
from tests import get_tests_input_path
from torch import nn, optim

from TTS.tts.layers.losses import L1LossMasked
from TTS.tts.models.tacotron import Tacotron
from TTS.utils.io import load_config
from TTS.utils.audio import AudioProcessor

#pylint: disable=unused-variable

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

c = load_config(os.path.join(get_tests_input_path(), 'test_config.json'))

ap = AudioProcessor(**c.audio)
WAV_FILE = os.path.join(get_tests_input_path(), "example_1.wav")


def count_parameters(model):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TacotronTrainTest(unittest.TestCase):
    @staticmethod
    def test_train_step():
        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (8, )).long().to(device)
        input_lengths[-1] = 128
        mel_spec = torch.rand(8, 30, c.audio['num_mels']).to(device)
        linear_spec = torch.rand(8, 30, c.audio['fft_size']).to(device)
        mel_lengths = torch.randint(20, 30, (8, )).long().to(device)
        stop_targets = torch.zeros(8, 30, 1).float().to(device)
        speaker_ids = torch.randint(0, 5, (8, )).long().to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()):, 0] = 1.0

        stop_targets = stop_targets.view(input_dummy.shape[0],
                                         stop_targets.size(1) // c.r, -1)
        stop_targets = (stop_targets.sum(2) >
                        0.0).unsqueeze(2).float().squeeze()

        criterion = L1LossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        model = Tacotron(
            num_chars=32,
            num_speakers=5,
            postnet_output_dim=c.audio['fft_size'],
            decoder_output_dim=c.audio['num_mels'],
            r=c.r,
            memory_size=c.memory_size
        ).to(device)  #FIXME: missing num_speakers parameter to Tacotron ctor
        model.train()
        print(" > Num parameters for Tacotron model:%s" %
              (count_parameters(model)))
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(),
                                    model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=c.lr)
        for _ in range(5):
            mel_out, linear_out, align, stop_tokens = model.forward(
                input_dummy, input_lengths, mel_spec, mel_lengths, speaker_ids)
            optimizer.zero_grad()
            loss = criterion(mel_out, mel_spec, mel_lengths)
            stop_loss = criterion_st(stop_tokens, stop_targets)
            loss = loss + criterion(linear_out, linear_spec,
                                    mel_lengths) + stop_loss
            loss.backward()
            optimizer.step()
        # check parameter changes
        count = 0
        for param, param_ref in zip(model.parameters(),
                                    model_ref.parameters()):
            # ignore pre-higway layer since it works conditional
            # if count not in [145, 59]:
            assert (param != param_ref).any(
            ), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref)
            count += 1

class MultiSpeakeTacotronTrainTest(unittest.TestCase):
    @staticmethod
    def test_train_step():
        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (8, )).long().to(device)
        input_lengths[-1] = 128
        mel_spec = torch.rand(8, 30, c.audio['num_mels']).to(device)
        linear_spec = torch.rand(8, 30, c.audio['fft_size']).to(device)
        mel_lengths = torch.randint(20, 30, (8, )).long().to(device)
        stop_targets = torch.zeros(8, 30, 1).float().to(device)
        speaker_embeddings = torch.rand(8, 55).to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()):, 0] = 1.0

        stop_targets = stop_targets.view(input_dummy.shape[0],
                                         stop_targets.size(1) // c.r, -1)
        stop_targets = (stop_targets.sum(2) >
                        0.0).unsqueeze(2).float().squeeze()

        criterion = L1LossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        model = Tacotron(
            num_chars=32,
            num_speakers=5,
            postnet_output_dim=c.audio['fft_size'],
            decoder_output_dim=c.audio['num_mels'],
            r=c.r,
            memory_size=c.memory_size,
            speaker_embedding_dim=55,
        ).to(device)  #FIXME: missing num_speakers parameter to Tacotron ctor
        model.train()
        print(" > Num parameters for Tacotron model:%s" %
              (count_parameters(model)))
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(),
                                    model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=c.lr)
        for _ in range(5):
            mel_out, linear_out, align, stop_tokens = model.forward(
                input_dummy, input_lengths, mel_spec, mel_lengths,
                speaker_embeddings=speaker_embeddings)
            optimizer.zero_grad()
            loss = criterion(mel_out, mel_spec, mel_lengths)
            stop_loss = criterion_st(stop_tokens, stop_targets)
            loss = loss + criterion(linear_out, linear_spec,
                                    mel_lengths) + stop_loss
            loss.backward()
            optimizer.step()
        # check parameter changes
        count = 0
        for param, param_ref in zip(model.parameters(),
                                    model_ref.parameters()):
            # ignore pre-higway layer since it works conditional
            # if count not in [145, 59]:
            assert (param != param_ref).any(
            ), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref)
            count += 1

class TacotronGSTTrainTest(unittest.TestCase):
    @staticmethod
    def test_train_step():
        # with random gst mel style
        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (8, )).long().to(device)
        input_lengths[-1] = 128
        mel_spec = torch.rand(8, 120, c.audio['num_mels']).to(device)
        linear_spec = torch.rand(8, 120, c.audio['fft_size']).to(device)
        mel_lengths = torch.randint(20, 120, (8, )).long().to(device)
        mel_lengths[-1] = 120
        stop_targets = torch.zeros(8, 120, 1).float().to(device)
        speaker_ids = torch.randint(0, 5, (8, )).long().to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()):, 0] = 1.0

        stop_targets = stop_targets.view(input_dummy.shape[0],
                                         stop_targets.size(1) // c.r, -1)
        stop_targets = (stop_targets.sum(2) >
                        0.0).unsqueeze(2).float().squeeze()

        criterion = L1LossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        model = Tacotron(
            num_chars=32,
            num_speakers=5,
            gst=True,
            gst_embedding_dim=c.gst['gst_embedding_dim'],
            gst_num_heads=c.gst['gst_num_heads'],
            gst_style_tokens=c.gst['gst_style_tokens'],
            postnet_output_dim=c.audio['fft_size'],
            decoder_output_dim=c.audio['num_mels'],
            r=c.r,
            memory_size=c.memory_size
        ).to(device)  #FIXME: missing num_speakers parameter to Tacotron ctor
        model.train()
        # print(model)
        print(" > Num parameters for Tacotron GST model:%s" %
              (count_parameters(model)))
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(),
                                    model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=c.lr)
        for _ in range(10):
            mel_out, linear_out, align, stop_tokens = model.forward(
                input_dummy, input_lengths, mel_spec, mel_lengths, speaker_ids)
            optimizer.zero_grad()
            loss = criterion(mel_out, mel_spec, mel_lengths)
            stop_loss = criterion_st(stop_tokens, stop_targets)
            loss = loss + criterion(linear_out, linear_spec,
                                    mel_lengths) + stop_loss
            loss.backward()
            optimizer.step()
        # check parameter changes
        count = 0
        for param, param_ref in zip(model.parameters(),
                                    model_ref.parameters()):
            # ignore pre-higway layer since it works conditional
            assert (param != param_ref).any(
            ), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref)
            count += 1

        # with file gst style
        mel_spec = torch.FloatTensor(ap.melspectrogram(ap.load_wav(WAV_FILE)))[:, :120].unsqueeze(0).transpose(1, 2).to(device)
        mel_spec = mel_spec.repeat(8, 1, 1)

        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (8, )).long().to(device)
        input_lengths[-1] = 128
        linear_spec = torch.rand(8, mel_spec.size(1), c.audio['fft_size']).to(device)
        mel_lengths = torch.randint(20, mel_spec.size(1), (8, )).long().to(device)
        mel_lengths[-1] = mel_spec.size(1)
        stop_targets = torch.zeros(8, mel_spec.size(1), 1).float().to(device)
        speaker_ids = torch.randint(0, 5, (8, )).long().to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()):, 0] = 1.0

        stop_targets = stop_targets.view(input_dummy.shape[0],
                                         stop_targets.size(1) // c.r, -1)
        stop_targets = (stop_targets.sum(2) >
                        0.0).unsqueeze(2).float().squeeze()

        criterion = L1LossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        model = Tacotron(
            num_chars=32,
            num_speakers=5,
            gst=True,
            gst_embedding_dim=c.gst['gst_embedding_dim'],
            gst_num_heads=c.gst['gst_num_heads'],
            gst_style_tokens=c.gst['gst_style_tokens'],
            postnet_output_dim=c.audio['fft_size'],
            decoder_output_dim=c.audio['num_mels'],
            r=c.r,
            memory_size=c.memory_size
        ).to(device)  #FIXME: missing num_speakers parameter to Tacotron ctor
        model.train()
        # print(model)
        print(" > Num parameters for Tacotron GST model:%s" %
              (count_parameters(model)))
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(),
                                    model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=c.lr)
        for _ in range(10):
            mel_out, linear_out, align, stop_tokens = model.forward(
                input_dummy, input_lengths, mel_spec, mel_lengths, speaker_ids)
            optimizer.zero_grad()
            loss = criterion(mel_out, mel_spec, mel_lengths)
            stop_loss = criterion_st(stop_tokens, stop_targets)
            loss = loss + criterion(linear_out, linear_spec,
                                    mel_lengths) + stop_loss
            loss.backward()
            optimizer.step()
        # check parameter changes
        count = 0
        for param, param_ref in zip(model.parameters(),
                                    model_ref.parameters()):
            # ignore pre-higway layer since it works conditional
            assert (param != param_ref).any(
            ), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref)
            count += 1

class SCGSTMultiSpeakeTacotronTrainTest(unittest.TestCase):
    @staticmethod
    def test_train_step():
        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (8, )).long().to(device)
        input_lengths[-1] = 128
        mel_spec = torch.rand(8, 30, c.audio['num_mels']).to(device)
        linear_spec = torch.rand(8, 30, c.audio['fft_size']).to(device)
        mel_lengths = torch.randint(20, 30, (8, )).long().to(device)
        mel_lengths[-1] = mel_spec.size(1)
        stop_targets = torch.zeros(8, 30, 1).float().to(device)
        speaker_embeddings = torch.rand(8, 55).to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()):, 0] = 1.0

        stop_targets = stop_targets.view(input_dummy.shape[0],
                                         stop_targets.size(1) // c.r, -1)
        stop_targets = (stop_targets.sum(2) >
                        0.0).unsqueeze(2).float().squeeze()

        criterion = L1LossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        model = Tacotron(
            num_chars=32,
            num_speakers=5,
            postnet_output_dim=c.audio['fft_size'],
            decoder_output_dim=c.audio['num_mels'],
            gst=True,
            gst_embedding_dim=c.gst['gst_embedding_dim'],
            gst_num_heads=c.gst['gst_num_heads'],
            gst_style_tokens=c.gst['gst_style_tokens'],
            gst_use_speaker_embedding=c.gst['gst_use_speaker_embedding'],
            r=c.r,
            memory_size=c.memory_size,
            speaker_embedding_dim=55,
        ).to(device)  #FIXME: missing num_speakers parameter to Tacotron ctor
        model.train()
        print(" > Num parameters for Tacotron model:%s" %
              (count_parameters(model)))
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(),
                                    model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=c.lr)
        for _ in range(5):
            mel_out, linear_out, align, stop_tokens = model.forward(
                input_dummy, input_lengths, mel_spec, mel_lengths,
                speaker_embeddings=speaker_embeddings)
            optimizer.zero_grad()
            loss = criterion(mel_out, mel_spec, mel_lengths)
            stop_loss = criterion_st(stop_tokens, stop_targets)
            loss = loss + criterion(linear_out, linear_spec,
                                    mel_lengths) + stop_loss
            loss.backward()
            optimizer.step()
        # check parameter changes
        count = 0
        for name_param, param_ref in zip(model.named_parameters(),
                                         model_ref.parameters()):
            # ignore pre-higway layer since it works conditional
            # if count not in [145, 59]:
            name, param = name_param
            if name == 'gst_layer.encoder.recurrence.weight_hh_l0':
                continue
            assert (param != param_ref).any(
            ), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref)
            count += 1
