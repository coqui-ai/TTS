import os
import unittest

import torch

from tests import get_tests_input_path
from TTS.vc.configs.freevc_config import FreeVCConfig
from TTS.vc.models.freevc import FreeVC

# pylint: disable=unused-variable
# pylint: disable=no-self-use

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

c = FreeVCConfig()

WAV_FILE = os.path.join(get_tests_input_path(), "example_1.wav")
BATCH_SIZE = 3


def count_parameters(model):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TestFreeVC(unittest.TestCase):
    def _create_inputs(self, config, batch_size=2):
        input_dummy = torch.rand(batch_size, 30 * config.audio["hop_length"]).to(device)
        input_lengths = torch.randint(100, 30 * config.audio["hop_length"], (batch_size,)).long().to(device)
        input_lengths[-1] = 30 * config.audio["hop_length"]
        spec = torch.rand(batch_size, 30, config.audio["filter_length"] // 2 + 1).to(device)
        mel = torch.rand(batch_size, 30, config.audio["n_mel_channels"]).to(device)
        spec_lengths = torch.randint(20, 30, (batch_size,)).long().to(device)
        spec_lengths[-1] = spec.size(2)
        waveform = torch.rand(batch_size, spec.size(2) * config.audio["hop_length"]).to(device)
        return input_dummy, input_lengths, mel, spec, spec_lengths, waveform

    @staticmethod
    def _create_inputs_inference():
        source_wav = torch.rand(16000)
        target_wav = torch.rand(16000)
        return source_wav, target_wav

    @staticmethod
    def _check_parameter_changes(model, model_ref):
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param != param_ref).any(), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref
            )
            count += 1

    def test_methods(self):
        config = FreeVCConfig()
        model = FreeVC(config).to(device)
        model.load_pretrained_speaker_encoder()
        model.init_multispeaker(config)
        wavlm_feats = model.extract_wavlm_features(torch.rand(1, 16000))
        assert wavlm_feats.shape == (1, 1024, 49), wavlm_feats.shape

    def test_load_audio(self):
        config = FreeVCConfig()
        model = FreeVC(config).to(device)
        wav = model.load_audio(WAV_FILE)
        wav2 = model.load_audio(wav)
        assert all(torch.isclose(wav, wav2))

    def _test_forward(self, batch_size):
        # create model
        config = FreeVCConfig()
        model = FreeVC(config).to(device)
        model.train()
        print(" > Num parameters for FreeVC model:%s" % (count_parameters(model)))

        _, _, mel, spec, spec_lengths, waveform = self._create_inputs(config, batch_size)

        wavlm_vec = model.extract_wavlm_features(waveform)
        wavlm_vec_lengths = torch.ones(batch_size, dtype=torch.long)

        y = model.forward(wavlm_vec, spec, None, mel, spec_lengths, wavlm_vec_lengths)
        # TODO: assert with training implementation

    def test_forward(self):
        self._test_forward(1)
        self._test_forward(3)

    def _test_inference(self, batch_size):
        config = FreeVCConfig()
        model = FreeVC(config).to(device)
        model.eval()

        _, _, mel, _, _, waveform = self._create_inputs(config, batch_size)

        wavlm_vec = model.extract_wavlm_features(waveform)
        wavlm_vec_lengths = torch.ones(batch_size, dtype=torch.long)

        output_wav = model.inference(wavlm_vec, None, mel, wavlm_vec_lengths)
        assert (
            output_wav.shape[-1] // config.audio.hop_length == wavlm_vec.shape[-1]
        ), f"{output_wav.shape[-1] // config.audio.hop_length} != {wavlm_vec.shape}"

    def test_inference(self):
        self._test_inference(1)
        self._test_inference(3)

    def test_voice_conversion(self):
        config = FreeVCConfig()
        model = FreeVC(config).to(device)
        model.eval()

        source_wav, target_wav = self._create_inputs_inference()
        output_wav = model.voice_conversion(source_wav, target_wav)
        assert (
            output_wav.shape[0] + config.audio.hop_length == source_wav.shape[0]
        ), f"{output_wav.shape} != {source_wav.shape}"

    def test_train_step(self):
        ...

    def test_train_eval_log(self):
        ...

    def test_test_run(self):
        ...

    def test_load_checkpoint(self):
        ...

    def test_get_criterion(self):
        ...

    def test_init_from_config(self):
        ...
