import copy
import os
import unittest

import torch
from torch import nn, optim

from tests import get_tests_input_path
from TTS.tts.configs.shared_configs import CapacitronVAEConfig, GSTConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.layers.losses import MSELossMasked
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.utils.audio import AudioProcessor

# pylint: disable=unused-variable

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config_global = Tacotron2Config(num_chars=32, num_speakers=5, out_channels=80, decoder_output_dim=80)

ap = AudioProcessor(**config_global.audio)
WAV_FILE = os.path.join(get_tests_input_path(), "example_1.wav")


class TacotronTrainTest(unittest.TestCase):
    """Test vanilla Tacotron2 model."""

    def test_train_step(self):  # pylint: disable=no-self-use
        config = config_global.copy()
        config.use_speaker_embedding = False
        config.num_speakers = 1

        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 128, (8,)).long().to(device)
        input_lengths = torch.sort(input_lengths, descending=True)[0]
        mel_spec = torch.rand(8, 30, config.audio["num_mels"]).to(device)
        mel_postnet_spec = torch.rand(8, 30, config.audio["num_mels"]).to(device)
        mel_lengths = torch.randint(20, 30, (8,)).long().to(device)
        mel_lengths[0] = 30
        stop_targets = torch.zeros(8, 30, 1).float().to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()) :, 0] = 1.0

        stop_targets = stop_targets.view(input_dummy.shape[0], stop_targets.size(1) // config.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze()

        criterion = MSELossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        model = Tacotron2(config).to(device)
        model.train()
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        for i in range(5):
            outputs = model.forward(input_dummy, input_lengths, mel_spec, mel_lengths)
            assert torch.sigmoid(outputs["stop_tokens"]).data.max() <= 1.0
            assert torch.sigmoid(outputs["stop_tokens"]).data.min() >= 0.0
            optimizer.zero_grad()
            loss = criterion(outputs["decoder_outputs"], mel_spec, mel_lengths)
            stop_loss = criterion_st(outputs["stop_tokens"], stop_targets)
            loss = loss + criterion(outputs["model_outputs"], mel_postnet_spec, mel_lengths) + stop_loss
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


class MultiSpeakerTacotronTrainTest(unittest.TestCase):
    """Test multi-speaker Tacotron2 with speaker embedding layer"""

    @staticmethod
    def test_train_step():
        config = config_global.copy()
        config.use_speaker_embedding = True
        config.num_speakers = 5

        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 128, (8,)).long().to(device)
        input_lengths = torch.sort(input_lengths, descending=True)[0]
        mel_spec = torch.rand(8, 30, config.audio["num_mels"]).to(device)
        mel_postnet_spec = torch.rand(8, 30, config.audio["num_mels"]).to(device)
        mel_lengths = torch.randint(20, 30, (8,)).long().to(device)
        mel_lengths[0] = 30
        stop_targets = torch.zeros(8, 30, 1).float().to(device)
        speaker_ids = torch.randint(0, 5, (8,)).long().to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()) :, 0] = 1.0

        stop_targets = stop_targets.view(input_dummy.shape[0], stop_targets.size(1) // config.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze()

        criterion = MSELossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        config.d_vector_dim = 55
        model = Tacotron2(config).to(device)
        model.train()
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
            assert torch.sigmoid(outputs["stop_tokens"]).data.max() <= 1.0
            assert torch.sigmoid(outputs["stop_tokens"]).data.min() >= 0.0
            optimizer.zero_grad()
            loss = criterion(outputs["decoder_outputs"], mel_spec, mel_lengths)
            stop_loss = criterion_st(outputs["stop_tokens"], stop_targets)
            loss = loss + criterion(outputs["model_outputs"], mel_postnet_spec, mel_lengths) + stop_loss
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
    """Test multi-speaker Tacotron2 with Global Style Token and Speaker Embedding"""

    # pylint: disable=no-self-use
    def test_train_step(self):
        # with random gst mel style
        config = config_global.copy()
        config.use_speaker_embedding = True
        config.num_speakers = 10
        config.use_gst = True
        config.gst = GSTConfig()

        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 128, (8,)).long().to(device)
        input_lengths = torch.sort(input_lengths, descending=True)[0]
        mel_spec = torch.rand(8, 30, config.audio["num_mels"]).to(device)
        mel_postnet_spec = torch.rand(8, 30, config.audio["num_mels"]).to(device)
        mel_lengths = torch.randint(20, 30, (8,)).long().to(device)
        mel_lengths[0] = 30
        stop_targets = torch.zeros(8, 30, 1).float().to(device)
        speaker_ids = torch.randint(0, 5, (8,)).long().to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()) :, 0] = 1.0

        stop_targets = stop_targets.view(input_dummy.shape[0], stop_targets.size(1) // config.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze()

        criterion = MSELossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        config.use_gst = True
        config.gst = GSTConfig()
        model = Tacotron2(config).to(device)
        model.train()
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        for i in range(10):
            outputs = model.forward(
                input_dummy, input_lengths, mel_spec, mel_lengths, aux_input={"speaker_ids": speaker_ids}
            )
            assert torch.sigmoid(outputs["stop_tokens"]).data.max() <= 1.0
            assert torch.sigmoid(outputs["stop_tokens"]).data.min() >= 0.0
            optimizer.zero_grad()
            loss = criterion(outputs["decoder_outputs"], mel_spec, mel_lengths)
            stop_loss = criterion_st(outputs["stop_tokens"], stop_targets)
            loss = loss + criterion(outputs["model_outputs"], mel_postnet_spec, mel_lengths) + stop_loss
            loss.backward()
            optimizer.step()
        # check parameter changes
        count = 0
        for name_param, param_ref in zip(model.named_parameters(), model_ref.parameters()):
            # ignore pre-higway layer since it works conditional
            # if count not in [145, 59]:
            name, param = name_param
            if name == "gst_layer.encoder.recurrence.weight_hh_l0":
                # print(param.grad)
                continue
            assert (param != param_ref).any(), "param {} {} with shape {} not updated!! \n{}\n{}".format(
                name, count, param.shape, param, param_ref
            )
            count += 1

        # with file gst style
        mel_spec = (
            torch.FloatTensor(ap.melspectrogram(ap.load_wav(WAV_FILE)))[:, :30].unsqueeze(0).transpose(1, 2).to(device)
        )
        mel_spec = mel_spec.repeat(8, 1, 1)
        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 128, (8,)).long().to(device)
        input_lengths = torch.sort(input_lengths, descending=True)[0]
        mel_postnet_spec = torch.rand(8, 30, config.audio["num_mels"]).to(device)
        mel_lengths = torch.randint(20, 30, (8,)).long().to(device)
        mel_lengths[0] = 30
        stop_targets = torch.zeros(8, 30, 1).float().to(device)
        speaker_ids = torch.randint(0, 5, (8,)).long().to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()) :, 0] = 1.0

        stop_targets = stop_targets.view(input_dummy.shape[0], stop_targets.size(1) // config.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze()

        criterion = MSELossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        model = Tacotron2(config).to(device)
        model.train()
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        for i in range(10):
            outputs = model.forward(
                input_dummy, input_lengths, mel_spec, mel_lengths, aux_input={"speaker_ids": speaker_ids}
            )
            assert torch.sigmoid(outputs["stop_tokens"]).data.max() <= 1.0
            assert torch.sigmoid(outputs["stop_tokens"]).data.min() >= 0.0
            optimizer.zero_grad()
            loss = criterion(outputs["decoder_outputs"], mel_spec, mel_lengths)
            stop_loss = criterion_st(outputs["stop_tokens"], stop_targets)
            loss = loss + criterion(outputs["model_outputs"], mel_postnet_spec, mel_lengths) + stop_loss
            loss.backward()
            optimizer.step()
        # check parameter changes
        count = 0
        for name_param, param_ref in zip(model.named_parameters(), model_ref.parameters()):
            # ignore pre-higway layer since it works conditional
            # if count not in [145, 59]:
            name, param = name_param
            if name == "gst_layer.encoder.recurrence.weight_hh_l0":
                # print(param.grad)
                continue
            assert (param != param_ref).any(), "param {} {} with shape {} not updated!! \n{}\n{}".format(
                name, count, param.shape, param, param_ref
            )
            count += 1


class TacotronCapacitronTrainTest(unittest.TestCase):
    @staticmethod
    def test_train_step():
        config = Tacotron2Config(
            num_chars=32,
            num_speakers=10,
            use_speaker_embedding=True,
            out_channels=80,
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

        model = Tacotron2(config).to(device)
        criterion = model.get_criterion().to(device)
        optimizer = model.get_optimizer()

        model.train()
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
    """Test multi-speaker Tacotron2 with Global Style Tokens and d-vector inputs."""

    @staticmethod
    def test_train_step():
        config = config_global.copy()
        config.use_d_vector_file = True

        config.use_gst = True
        config.gst = GSTConfig()

        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 128, (8,)).long().to(device)
        input_lengths = torch.sort(input_lengths, descending=True)[0]
        mel_spec = torch.rand(8, 30, config.audio["num_mels"]).to(device)
        mel_postnet_spec = torch.rand(8, 30, config.audio["num_mels"]).to(device)
        mel_lengths = torch.randint(20, 30, (8,)).long().to(device)
        mel_lengths[0] = 30
        stop_targets = torch.zeros(8, 30, 1).float().to(device)
        speaker_embeddings = torch.rand(8, 55).to(device)

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()) :, 0] = 1.0

        stop_targets = stop_targets.view(input_dummy.shape[0], stop_targets.size(1) // config.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze()
        criterion = MSELossMasked(seq_len_norm=False).to(device)
        criterion_st = nn.BCEWithLogitsLoss().to(device)
        config.d_vector_dim = 55
        model = Tacotron2(config).to(device)
        model.train()
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        for i in range(5):
            outputs = model.forward(
                input_dummy, input_lengths, mel_spec, mel_lengths, aux_input={"d_vectors": speaker_embeddings}
            )
            assert torch.sigmoid(outputs["stop_tokens"]).data.max() <= 1.0
            assert torch.sigmoid(outputs["stop_tokens"]).data.min() >= 0.0
            optimizer.zero_grad()
            loss = criterion(outputs["decoder_outputs"], mel_spec, mel_lengths)
            stop_loss = criterion_st(outputs["stop_tokens"], stop_targets)
            loss = loss + criterion(outputs["model_outputs"], mel_postnet_spec, mel_lengths) + stop_loss
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
