import copy
import os
import unittest

import torch
from torch import nn, optim

from tests import get_tests_input_path
from TTS.tts.configs.shared_configs import CapacitronVAEConfig, GSTConfig
from TTS.tts.configs.tacotron_config import TacotronConfig
from TTS.tts.layers.losses import L1LossMasked
from TTS.tts.models.tacotron import Tacotron
from TTS.utils.audio import AudioProcessor

# pylint: disable=unused-variable

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

config_global = TacotronConfig(num_chars=32, num_speakers=5, out_channels=513, decoder_output_dim=80)

ap = AudioProcessor(**config_global.audio)
WAV_FILE = os.path.join(get_tests_input_path(), "example_1.wav")


def count_parameters(model):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TacotronTrainTest(unittest.TestCase):
    @staticmethod
    def test_train_step():
        config = config_global.copy()
        config.use_speaker_embedding = False
        config.num_speakers = 1

        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (8,)).long().to(device)
        input_lengths[-1] = 128
        mel_spec = torch.rand(8, 30, config.audio["num_mels"]).to(device)
        linear_spec = torch.rand(8, 30, config.audio["fft_size"] // 2 + 1).to(device)
        mel_lengths = torch.randint(20, 30, (8,)).long().to(device)
        mel_lengths[-1] = mel_spec.size(1)
        stop_targets = torch.zeros(8, 30, 1).float().to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()) :, 0] = 1.0

        stop_targets = stop_targets.view(input_dummy.shape[0], stop_targets.size(1) // config.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze()

        criterion = L1LossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        model = Tacotron(config).to(device)  # FIXME: missing num_speakers parameter to Tacotron ctor
        model.train()
        print(" > Num parameters for Tacotron model:%s" % (count_parameters(model)))
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        for _ in range(5):
            outputs = model.forward(input_dummy, input_lengths, mel_spec, mel_lengths)
            optimizer.zero_grad()
            loss = criterion(outputs["decoder_outputs"], mel_spec, mel_lengths)
            stop_loss = criterion_st(outputs["stop_tokens"], stop_targets)
            loss = loss + criterion(outputs["model_outputs"], linear_spec, mel_lengths) + stop_loss
            loss.backward()
            optimizer.step()
        # check parameter changes
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            # ignore pre-higway layer since it works conditional
            # if count not in [145, 59]:
            assert (param != param_ref).any(), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref
            )
            count += 1


class MultiSpeakeTacotronTrainTest(unittest.TestCase):
    @staticmethod
    def test_train_step():
        config = config_global.copy()
        config.use_speaker_embedding = True
        config.num_speakers = 5

        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (8,)).long().to(device)
        input_lengths[-1] = 128
        mel_spec = torch.rand(8, 30, config.audio["num_mels"]).to(device)
        linear_spec = torch.rand(8, 30, config.audio["fft_size"] // 2 + 1).to(device)
        mel_lengths = torch.randint(20, 30, (8,)).long().to(device)
        mel_lengths[-1] = mel_spec.size(1)
        stop_targets = torch.zeros(8, 30, 1).float().to(device)
        speaker_ids = torch.randint(0, 5, (8,)).long().to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()) :, 0] = 1.0

        stop_targets = stop_targets.view(input_dummy.shape[0], stop_targets.size(1) // config.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze()

        criterion = L1LossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        config.d_vector_dim = 55
        model = Tacotron(config).to(device)  # FIXME: missing num_speakers parameter to Tacotron ctor
        model.train()
        print(" > Num parameters for Tacotron model:%s" % (count_parameters(model)))
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        for _ in range(5):
            outputs = model.forward(
                input_dummy, input_lengths, mel_spec, mel_lengths, aux_input={"speaker_ids": speaker_ids}
            )
            optimizer.zero_grad()
            loss = criterion(outputs["decoder_outputs"], mel_spec, mel_lengths)
            stop_loss = criterion_st(outputs["stop_tokens"], stop_targets)
            loss = loss + criterion(outputs["model_outputs"], linear_spec, mel_lengths) + stop_loss
            loss.backward()
            optimizer.step()
        # check parameter changes
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            # ignore pre-higway layer since it works conditional
            # if count not in [145, 59]:
            assert (param != param_ref).any(), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref
            )
            count += 1


class TacotronGSTTrainTest(unittest.TestCase):
    @staticmethod
    def test_train_step():
        config = config_global.copy()
        config.use_speaker_embedding = True
        config.num_speakers = 10
        config.use_gst = True
        config.gst = GSTConfig()
        # with random gst mel style
        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (8,)).long().to(device)
        input_lengths[-1] = 128
        mel_spec = torch.rand(8, 120, config.audio["num_mels"]).to(device)
        linear_spec = torch.rand(8, 120, config.audio["fft_size"] // 2 + 1).to(device)
        mel_lengths = torch.randint(20, 120, (8,)).long().to(device)
        mel_lengths[-1] = 120
        stop_targets = torch.zeros(8, 120, 1).float().to(device)
        speaker_ids = torch.randint(0, 5, (8,)).long().to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()) :, 0] = 1.0

        stop_targets = stop_targets.view(input_dummy.shape[0], stop_targets.size(1) // config.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze()

        criterion = L1LossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        config.use_gst = True
        config.gst = GSTConfig()
        model = Tacotron(config).to(device)  # FIXME: missing num_speakers parameter to Tacotron ctor
        model.train()
        # print(model)
        print(" > Num parameters for Tacotron GST model:%s" % (count_parameters(model)))
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        for _ in range(10):
            outputs = model.forward(
                input_dummy, input_lengths, mel_spec, mel_lengths, aux_input={"speaker_ids": speaker_ids}
            )
            optimizer.zero_grad()
            loss = criterion(outputs["decoder_outputs"], mel_spec, mel_lengths)
            stop_loss = criterion_st(outputs["stop_tokens"], stop_targets)
            loss = loss + criterion(outputs["model_outputs"], linear_spec, mel_lengths) + stop_loss
            loss.backward()
            optimizer.step()
        # check parameter changes
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            # ignore pre-higway layer since it works conditional
            assert (param != param_ref).any(), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref
            )
            count += 1

        # with file gst style
        mel_spec = (
            torch.FloatTensor(ap.melspectrogram(ap.load_wav(WAV_FILE)))[:, :120].unsqueeze(0).transpose(1, 2).to(device)
        )
        mel_spec = mel_spec.repeat(8, 1, 1)

        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (8,)).long().to(device)
        input_lengths[-1] = 128
        linear_spec = torch.rand(8, mel_spec.size(1), config.audio["fft_size"] // 2 + 1).to(device)
        mel_lengths = torch.randint(20, mel_spec.size(1), (8,)).long().to(device)
        mel_lengths[-1] = mel_spec.size(1)
        stop_targets = torch.zeros(8, mel_spec.size(1), 1).float().to(device)
        speaker_ids = torch.randint(0, 5, (8,)).long().to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()) :, 0] = 1.0

        stop_targets = stop_targets.view(input_dummy.shape[0], stop_targets.size(1) // config.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze()

        criterion = L1LossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        model = Tacotron(config).to(device)  # FIXME: missing num_speakers parameter to Tacotron ctor
        model.train()
        # print(model)
        print(" > Num parameters for Tacotron GST model:%s" % (count_parameters(model)))
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        for _ in range(10):
            outputs = model.forward(
                input_dummy, input_lengths, mel_spec, mel_lengths, aux_input={"speaker_ids": speaker_ids}
            )
            optimizer.zero_grad()
            loss = criterion(outputs["decoder_outputs"], mel_spec, mel_lengths)
            stop_loss = criterion_st(outputs["stop_tokens"], stop_targets)
            loss = loss + criterion(outputs["model_outputs"], linear_spec, mel_lengths) + stop_loss
            loss.backward()
            optimizer.step()
        # check parameter changes
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            # ignore pre-higway layer since it works conditional
            assert (param != param_ref).any(), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref
            )
            count += 1


class TacotronCapacitronTrainTest(unittest.TestCase):
    @staticmethod
    def test_train_step():
        config = TacotronConfig(
            num_chars=32,
            num_speakers=10,
            use_speaker_embedding=True,
            out_channels=513,
            decoder_output_dim=80,
            use_capacitron_vae=True,
            capacitron_vae=CapacitronVAEConfig(),
            optimizer="CapacitronOptimizer",
            optimizer_params={
                "RAdam": {"betas": [0.9, 0.998], "weight_decay": 1e-6},
                "SGD": {"lr": 1e-5, "momentum": 0.9},
            },
        )

        batch = dict({})
        batch["text_input"] = torch.randint(0, 24, (8, 128)).long().to(device)
        batch["text_lengths"] = torch.randint(100, 129, (8,)).long().to(device)
        batch["text_lengths"] = torch.sort(batch["text_lengths"], descending=True)[0]
        batch["text_lengths"][0] = 128
        batch["linear_input"] = torch.rand(8, 120, config.audio["fft_size"] // 2 + 1).to(device)
        batch["mel_input"] = torch.rand(8, 120, config.audio["num_mels"]).to(device)
        batch["mel_lengths"] = torch.randint(20, 120, (8,)).long().to(device)
        batch["mel_lengths"] = torch.sort(batch["mel_lengths"], descending=True)[0]
        batch["mel_lengths"][0] = 120
        batch["stop_targets"] = torch.zeros(8, 120, 1).float().to(device)
        batch["stop_target_lengths"] = torch.randint(0, 120, (8,)).to(device)
        batch["speaker_ids"] = torch.randint(0, 5, (8,)).long().to(device)
        batch["d_vectors"] = None

        for idx in batch["mel_lengths"]:
            batch["stop_targets"][:, int(idx.item()) :, 0] = 1.0

        batch["stop_targets"] = batch["stop_targets"].view(
            batch["text_input"].shape[0], batch["stop_targets"].size(1) // config.r, -1
        )
        batch["stop_targets"] = (batch["stop_targets"].sum(2) > 0.0).unsqueeze(2).float().squeeze()
        model = Tacotron(config).to(device)
        criterion = model.get_criterion()
        optimizer = model.get_optimizer()
        model.train()
        print(" > Num parameters for Tacotron with Capacitron VAE model:%s" % (count_parameters(model)))
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        for _ in range(10):
            _, loss_dict = model.train_step(batch, criterion)
            optimizer.zero_grad()
            loss_dict["capacitron_vae_beta_loss"].backward()
            optimizer.first_step()
            loss_dict["loss"].backward()
            optimizer.step()
        # check parameter changes
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            # ignore pre-higway layer since it works conditional
            assert (param != param_ref).any(), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref
            )
            count += 1


class SCGSTMultiSpeakeTacotronTrainTest(unittest.TestCase):
    @staticmethod
    def test_train_step():
        config = config_global.copy()
        config.use_d_vector_file = True

        config.use_gst = True
        config.gst = GSTConfig()

        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (8,)).long().to(device)
        input_lengths[-1] = 128
        mel_spec = torch.rand(8, 30, config.audio["num_mels"]).to(device)
        linear_spec = torch.rand(8, 30, config.audio["fft_size"] // 2 + 1).to(device)
        mel_lengths = torch.randint(20, 30, (8,)).long().to(device)
        mel_lengths[-1] = mel_spec.size(1)
        stop_targets = torch.zeros(8, 30, 1).float().to(device)
        speaker_embeddings = torch.rand(8, 55).to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()) :, 0] = 1.0

        stop_targets = stop_targets.view(input_dummy.shape[0], stop_targets.size(1) // config.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze()

        criterion = L1LossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        config.d_vector_dim = 55
        model = Tacotron(config).to(device)  # FIXME: missing num_speakers parameter to Tacotron ctor
        model.train()
        print(" > Num parameters for Tacotron model:%s" % (count_parameters(model)))
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        for _ in range(5):
            outputs = model.forward(
                input_dummy, input_lengths, mel_spec, mel_lengths, aux_input={"d_vectors": speaker_embeddings}
            )
            optimizer.zero_grad()
            loss = criterion(outputs["decoder_outputs"], mel_spec, mel_lengths)
            stop_loss = criterion_st(outputs["stop_tokens"], stop_targets)
            loss = loss + criterion(outputs["model_outputs"], linear_spec, mel_lengths) + stop_loss
            loss.backward()
            optimizer.step()
        # check parameter changes
        count = 0
        for name_param, param_ref in zip(model.named_parameters(), model_ref.parameters()):
            # ignore pre-higway layer since it works conditional
            # if count not in [145, 59]:
            name, param = name_param
            if name == "gst_layer.encoder.recurrence.weight_hh_l0":
                continue
            assert (param != param_ref).any(), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref
            )
            count += 1
