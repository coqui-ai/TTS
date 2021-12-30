import os
import unittest

import torch

from tests import assertHasAttr, assertHasNotAttr, get_tests_input_path
from TTS.config import load_config
from TTS.speaker_encoder.utils.generic_utils import setup_speaker_encoder_model
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, VitsArgs
from TTS.tts.utils.speakers import SpeakerManager

LANG_FILE = os.path.join(get_tests_input_path(), "language_ids.json")
SPEAKER_ENCODER_CONFIG = os.path.join(get_tests_input_path(), "test_speaker_encoder_config.json")


torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# pylint: disable=no-self-use
class TestVits(unittest.TestCase):
    def test_init_multispeaker(self):
        num_speakers = 10
        args = VitsArgs(num_speakers=num_speakers, use_speaker_embedding=True)
        model = Vits(args)
        assertHasAttr(self, model, "emb_g")

        args = VitsArgs(num_speakers=0, use_speaker_embedding=True)
        model = Vits(args)
        assertHasNotAttr(self, model, "emb_g")

        args = VitsArgs(num_speakers=10, use_speaker_embedding=False)
        model = Vits(args)
        assertHasNotAttr(self, model, "emb_g")

        args = VitsArgs(d_vector_dim=101, use_d_vector_file=True)
        model = Vits(args)
        self.assertEqual(model.embedded_speaker_dim, 101)

    def test_init_multilingual(self):
        args = VitsArgs(language_ids_file=None, use_language_embedding=False)
        model = Vits(args)
        self.assertEqual(model.language_manager, None)
        self.assertEqual(model.embedded_language_dim, 0)
        self.assertEqual(model.emb_l, None)

        args = VitsArgs(language_ids_file=LANG_FILE)
        model = Vits(args)
        self.assertNotEqual(model.language_manager, None)
        self.assertEqual(model.embedded_language_dim, 0)
        self.assertEqual(model.emb_l, None)

        args = VitsArgs(language_ids_file=LANG_FILE, use_language_embedding=True)
        model = Vits(args)
        self.assertNotEqual(model.language_manager, None)
        self.assertEqual(model.embedded_language_dim, args.embedded_language_dim)
        self.assertNotEqual(model.emb_l, None)

        args = VitsArgs(language_ids_file=LANG_FILE, use_language_embedding=True, embedded_language_dim=102)
        model = Vits(args)
        self.assertNotEqual(model.language_manager, None)
        self.assertEqual(model.embedded_language_dim, args.embedded_language_dim)
        self.assertNotEqual(model.emb_l, None)

    def test_get_aux_input(self):
        aux_input = {"speaker_ids": None, "style_wav": None, "d_vectors": None, "language_ids": None}
        args = VitsArgs()
        model = Vits(args)
        aux_out = model.get_aux_input(aux_input)

        speaker_id = torch.randint(10, (1,))
        language_id = torch.randint(10, (1,))
        d_vector = torch.rand(1, 128)
        aux_input = {"speaker_ids": speaker_id, "style_wav": None, "d_vectors": d_vector, "language_ids": language_id}
        aux_out = model.get_aux_input(aux_input)
        self.assertEqual(aux_out["speaker_ids"].shape, speaker_id.shape)
        self.assertEqual(aux_out["language_ids"].shape, language_id.shape)
        self.assertEqual(aux_out["d_vectors"].shape, d_vector.unsqueeze(0).transpose(2, 1).shape)

    def test_voice_conversion(self):
        num_speakers = 10
        spec_len = 101
        spec_effective_len = 50

        args = VitsArgs(num_speakers=num_speakers, use_speaker_embedding=True)
        model = Vits(args)

        ref_inp = torch.randn(1, spec_len, 513)
        ref_inp_len = torch.randint(1, spec_effective_len, (1,))
        ref_spk_id = torch.randint(1, num_speakers, (1,))
        tgt_spk_id = torch.randint(1, num_speakers, (1,))
        o_hat, y_mask, (z, z_p, z_hat) = model.voice_conversion(ref_inp, ref_inp_len, ref_spk_id, tgt_spk_id)

        self.assertEqual(o_hat.shape, (1, 1, spec_len * 256))
        self.assertEqual(y_mask.shape, (1, 1, spec_len))
        self.assertEqual(y_mask.sum(), ref_inp_len[0])
        self.assertEqual(z.shape, (1, args.hidden_channels, spec_len))
        self.assertEqual(z_p.shape, (1, args.hidden_channels, spec_len))
        self.assertEqual(z_hat.shape, (1, args.hidden_channels, spec_len))

    def _init_inputs(self, config):
        input_dummy = torch.randint(0, 24, (8, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (8,)).long().to(device)
        input_lengths[-1] = 128
        spec = torch.rand(8, config.audio["fft_size"] // 2 + 1, 30).to(device)
        spec_lengths = torch.randint(20, 30, (8,)).long().to(device)
        spec_lengths[-1] = spec.size(2)
        waveform = torch.rand(8, 1, spec.size(2) * config.audio["hop_length"]).to(device)
        return input_dummy, input_lengths, spec, spec_lengths, waveform

    def _check_forward_outputs(self, config, output_dict, encoder_config=None):
        self.assertEqual(
            output_dict["model_outputs"].shape[2], config.model_args.spec_segment_size * config.audio["hop_length"]
        )
        self.assertEqual(output_dict["alignments"].shape, (8, 128, 30))
        self.assertEqual(output_dict["alignments"].max(), 1)
        self.assertEqual(output_dict["alignments"].min(), 0)
        self.assertEqual(output_dict["z"].shape, (8, config.model_args.hidden_channels, 30))
        self.assertEqual(output_dict["z_p"].shape, (8, config.model_args.hidden_channels, 30))
        self.assertEqual(output_dict["m_p"].shape, (8, config.model_args.hidden_channels, 30))
        self.assertEqual(output_dict["logs_p"].shape, (8, config.model_args.hidden_channels, 30))
        self.assertEqual(output_dict["m_q"].shape, (8, config.model_args.hidden_channels, 30))
        self.assertEqual(output_dict["logs_q"].shape, (8, config.model_args.hidden_channels, 30))
        self.assertEqual(
            output_dict["waveform_seg"].shape[2], config.model_args.spec_segment_size * config.audio["hop_length"]
        )
        if encoder_config:
            self.assertEqual(output_dict["gt_spk_emb"].shape, (8, encoder_config.model_params["proj_dim"]))
            self.assertEqual(output_dict["syn_spk_emb"].shape, (8, encoder_config.model_params["proj_dim"]))
        else:
            self.assertEqual(output_dict["gt_spk_emb"], None)
            self.assertEqual(output_dict["syn_spk_emb"], None)

    def test_forward(self):
        num_speakers = 0
        config = VitsConfig(num_speakers=num_speakers, use_speaker_embedding=True)
        config.model_args.spec_segment_size = 10
        input_dummy, input_lengths, spec, spec_lengths, waveform = self._init_inputs(config)
        model = Vits(config).to(device)
        output_dict = model.forward(input_dummy, input_lengths, spec, spec_lengths, waveform)
        self._check_forward_outputs(config, output_dict)

    def test_multispeaker_forward(self):
        num_speakers = 10

        config = VitsConfig(num_speakers=num_speakers, use_speaker_embedding=True)
        config.model_args.spec_segment_size = 10

        input_dummy, input_lengths, spec, spec_lengths, waveform = self._init_inputs(config)
        speaker_ids = torch.randint(0, num_speakers, (8,)).long().to(device)

        model = Vits(config).to(device)
        output_dict = model.forward(
            input_dummy, input_lengths, spec, spec_lengths, waveform, aux_input={"speaker_ids": speaker_ids}
        )
        self._check_forward_outputs(config, output_dict)

    def test_multilingual_forward(self):
        num_speakers = 10
        num_langs = 3

        args = VitsArgs(language_ids_file=LANG_FILE, use_language_embedding=True, spec_segment_size=10)
        config = VitsConfig(num_speakers=num_speakers, use_speaker_embedding=True, model_args=args)

        input_dummy, input_lengths, spec, spec_lengths, waveform = self._init_inputs(config)
        speaker_ids = torch.randint(0, num_speakers, (8,)).long().to(device)
        lang_ids = torch.randint(0, num_langs, (8,)).long().to(device)

        model = Vits(config).to(device)
        output_dict = model.forward(
            input_dummy,
            input_lengths,
            spec,
            spec_lengths,
            waveform,
            aux_input={"speaker_ids": speaker_ids, "language_ids": lang_ids},
        )
        self._check_forward_outputs(config, output_dict)

    def test_secl_forward(self):
        num_speakers = 10
        num_langs = 3

        speaker_encoder_config = load_config(SPEAKER_ENCODER_CONFIG)
        speaker_encoder_config.model_params["use_torch_spec"] = True
        speaker_encoder = setup_speaker_encoder_model(speaker_encoder_config).to(device)
        speaker_manager = SpeakerManager()
        speaker_manager.speaker_encoder = speaker_encoder

        args = VitsArgs(
            language_ids_file=LANG_FILE,
            use_language_embedding=True,
            spec_segment_size=10,
            use_speaker_encoder_as_loss=True,
        )
        config = VitsConfig(num_speakers=num_speakers, use_speaker_embedding=True, model_args=args)
        config.audio.sample_rate = 16000

        input_dummy, input_lengths, spec, spec_lengths, waveform = self._init_inputs(config)
        speaker_ids = torch.randint(0, num_speakers, (8,)).long().to(device)
        lang_ids = torch.randint(0, num_langs, (8,)).long().to(device)

        model = Vits(config, speaker_manager=speaker_manager).to(device)
        output_dict = model.forward(
            input_dummy,
            input_lengths,
            spec,
            spec_lengths,
            waveform,
            aux_input={"speaker_ids": speaker_ids, "language_ids": lang_ids},
        )
        self._check_forward_outputs(config, output_dict, speaker_encoder_config)

    def test_inference(self):
        num_speakers = 0
        config = VitsConfig(num_speakers=num_speakers, use_speaker_embedding=True)
        input_dummy = torch.randint(0, 24, (1, 128)).long().to(device)
        model = Vits(config).to(device)
        _ = model.inference(input_dummy)

    def test_multispeaker_inference(self):
        num_speakers = 10
        config = VitsConfig(num_speakers=num_speakers, use_speaker_embedding=True)
        input_dummy = torch.randint(0, 24, (1, 128)).long().to(device)
        speaker_ids = torch.randint(0, num_speakers, (1,)).long().to(device)
        model = Vits(config).to(device)
        _ = model.inference(input_dummy, {"speaker_ids": speaker_ids})

    def test_multilingual_inference(self):
        num_speakers = 10
        num_langs = 3
        args = VitsArgs(language_ids_file=LANG_FILE, use_language_embedding=True, spec_segment_size=10)
        config = VitsConfig(num_speakers=num_speakers, use_speaker_embedding=True, model_args=args)
        input_dummy = torch.randint(0, 24, (1, 128)).long().to(device)
        speaker_ids = torch.randint(0, num_speakers, (1,)).long().to(device)
        lang_ids = torch.randint(0, num_langs, (1,)).long().to(device)
        model = Vits(config).to(device)
        _ = model.inference(input_dummy, {"speaker_ids": speaker_ids, "language_ids": lang_ids})
