import copy
import os
import unittest

import torch
from torch import optim

from tests import get_tests_input_path
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.layers.losses import GlowTTSLoss
from TTS.tts.models.glow_tts import GlowTTS
from TTS.utils.audio import AudioProcessor

# pylint: disable=unused-variable

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

c = GlowTTSConfig()

ap = AudioProcessor(**c.audio)
WAV_FILE = os.path.join(get_tests_input_path(), "example_1.wav")


def count_parameters(model):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GlowTTSTrainTest(unittest.TestCase):
    @staticmethod
    def test_train_step():
        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (8,)).long().to(device)
        input_lengths[-1] = 128
        mel_spec = torch.rand(8, 30, c.audio["num_mels"]).to(device)
        mel_lengths = torch.randint(20, 30, (8,)).long().to(device)
        speaker_ids = torch.randint(0, 5, (8,)).long().to(device)

        criterion = GlowTTSLoss()

        # model to train
        config = GlowTTSConfig(num_chars=32)
        model = GlowTTS(config).to(device)

        # reference model to compare model weights
        model_ref = GlowTTS(config).to(device)

        model.train()
        print(" > Num parameters for GlowTTS model:%s" % (count_parameters(model)))

        # pass the state to ref model
        model_ref.load_state_dict(copy.deepcopy(model.state_dict()))

        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for _ in range(5):
            optimizer.zero_grad()
            outputs = model.forward(input_dummy, input_lengths, mel_spec, mel_lengths, None)
            loss_dict = criterion(
                outputs["z"],
                outputs["y_mean"],
                outputs["y_log_scale"],
                outputs["logdet"],
                mel_lengths,
                outputs["durations_log"],
                outputs["total_durations_log"],
                input_lengths,
            )
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

        # check parameter changes
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param != param_ref).any(), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref
            )
            count += 1


class GlowTTSInferenceTest(unittest.TestCase):
    @staticmethod
    def test_inference():
        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (8,)).long().to(device)
        input_lengths[-1] = 128
        mel_spec = torch.rand(8, 30, c.audio["num_mels"]).to(device)
        mel_lengths = torch.randint(20, 30, (8,)).long().to(device)
        speaker_ids = torch.randint(0, 5, (8,)).long().to(device)

        # create model
        config = GlowTTSConfig(num_chars=32)
        model = GlowTTS(config).to(device)

        model.eval()
        print(" > Num parameters for GlowTTS model:%s" % (count_parameters(model)))

        # inference encoder and decoder with MAS
        y = model.inference_with_MAS(input_dummy, input_lengths, mel_spec, mel_lengths)

        y2 = model.decoder_inference(mel_spec, mel_lengths)

        assert (
            y2["model_outputs"].shape == y["model_outputs"].shape
        ), "Difference between the shapes of the glowTTS inference with MAS ({}) and the inference using only the decoder ({}) !!".format(
            y["model_outputs"].shape, y2["model_outputs"].shape
        )
