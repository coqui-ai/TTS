import copy
import os
import unittest

import torch
from trainer.logging.tensorboard_logger import TensorboardLogger

from tests import assertHasAttr, assertHasNotAttr, get_tests_data_path, get_tests_input_path, get_tests_output_path
from TTS.config import load_config
from TTS.encoder.utils.generic_utils import setup_encoder_model
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import (
    Vits,
    VitsArgs,
    VitsAudioConfig,
    amp_to_db,
    db_to_amp,
    load_audio,
    spec_to_mel,
    wav_to_mel,
    wav_to_spec,
)
from TTS.tts.utils.speakers import SpeakerManager

LANG_FILE = os.path.join(get_tests_input_path(), "language_ids.json")
SPEAKER_ENCODER_CONFIG = os.path.join(get_tests_input_path(), "test_speaker_encoder_config.json")
WAV_FILE = os.path.join(get_tests_input_path(), "example_1.wav")


torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# pylint: disable=no-self-use
class TestVits(unittest.TestCase):
    def test_load_audio(self):
        wav, sr = load_audio(WAV_FILE)
        self.assertEqual(wav.shape, (1, 41885))
        self.assertEqual(sr, 22050)

        spec = wav_to_spec(wav, n_fft=1024, hop_length=512, win_length=1024, center=False)
        mel = wav_to_mel(
            wav,
            n_fft=1024,
            num_mels=80,
            sample_rate=sr,
            hop_length=512,
            win_length=1024,
            fmin=0,
            fmax=8000,
            center=False,
        )
        mel2 = spec_to_mel(spec, n_fft=1024, num_mels=80, sample_rate=sr, fmin=0, fmax=8000)

        self.assertEqual((mel - mel2).abs().max(), 0)
        self.assertEqual(spec.shape[0], mel.shape[0])
        self.assertEqual(spec.shape[2], mel.shape[2])

        spec_db = amp_to_db(spec)
        spec_amp = db_to_amp(spec_db)

        self.assertAlmostEqual((spec - spec_amp).abs().max(), 0, delta=1e-4)

    def test_dataset(self):
        """TODO:"""
        ...

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
        assertHasNotAttr(self, model, "emb_l")

        args = VitsArgs(language_ids_file=LANG_FILE)
        model = Vits(args)
        self.assertNotEqual(model.language_manager, None)
        self.assertEqual(model.embedded_language_dim, 0)
        assertHasNotAttr(self, model, "emb_l")

        args = VitsArgs(language_ids_file=LANG_FILE, use_language_embedding=True)
        model = Vits(args)
        self.assertNotEqual(model.language_manager, None)
        self.assertEqual(model.embedded_language_dim, args.embedded_language_dim)
        assertHasAttr(self, model, "emb_l")

        args = VitsArgs(language_ids_file=LANG_FILE, use_language_embedding=True, embedded_language_dim=102)
        model = Vits(args)
        self.assertNotEqual(model.language_manager, None)
        self.assertEqual(model.embedded_language_dim, args.embedded_language_dim)
        assertHasAttr(self, model, "emb_l")

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

        ref_inp = torch.randn(1, 513, spec_len)
        ref_inp_len = torch.randint(1, spec_effective_len, (1,))
        ref_spk_id = torch.randint(1, num_speakers, (1,)).item()
        tgt_spk_id = torch.randint(1, num_speakers, (1,)).item()
        o_hat, y_mask, (z, z_p, z_hat) = model.voice_conversion(ref_inp, ref_inp_len, ref_spk_id, tgt_spk_id)

        self.assertEqual(o_hat.shape, (1, 1, spec_len * 256))
        self.assertEqual(y_mask.shape, (1, 1, spec_len))
        self.assertEqual(y_mask.sum(), ref_inp_len[0])
        self.assertEqual(z.shape, (1, args.hidden_channels, spec_len))
        self.assertEqual(z_p.shape, (1, args.hidden_channels, spec_len))
        self.assertEqual(z_hat.shape, (1, args.hidden_channels, spec_len))

    def _create_inputs(self, config, batch_size=2):
        input_dummy = torch.randint(0, 24, (batch_size, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (batch_size,)).long().to(device)
        input_lengths[-1] = 128
        spec = torch.rand(batch_size, config.audio["fft_size"] // 2 + 1, 30).to(device)
        mel = torch.rand(batch_size, config.audio["num_mels"], 30).to(device)
        spec_lengths = torch.randint(20, 30, (batch_size,)).long().to(device)
        spec_lengths[-1] = spec.size(2)
        waveform = torch.rand(batch_size, 1, spec.size(2) * config.audio["hop_length"]).to(device)
        return input_dummy, input_lengths, mel, spec, spec_lengths, waveform

    def _check_forward_outputs(self, config, output_dict, encoder_config=None, batch_size=2):
        self.assertEqual(
            output_dict["model_outputs"].shape[2], config.model_args.spec_segment_size * config.audio["hop_length"]
        )
        self.assertEqual(output_dict["alignments"].shape, (batch_size, 128, 30))
        self.assertEqual(output_dict["alignments"].max(), 1)
        self.assertEqual(output_dict["alignments"].min(), 0)
        self.assertEqual(output_dict["z"].shape, (batch_size, config.model_args.hidden_channels, 30))
        self.assertEqual(output_dict["z_p"].shape, (batch_size, config.model_args.hidden_channels, 30))
        self.assertEqual(output_dict["m_p"].shape, (batch_size, config.model_args.hidden_channels, 30))
        self.assertEqual(output_dict["logs_p"].shape, (batch_size, config.model_args.hidden_channels, 30))
        self.assertEqual(output_dict["m_q"].shape, (batch_size, config.model_args.hidden_channels, 30))
        self.assertEqual(output_dict["logs_q"].shape, (batch_size, config.model_args.hidden_channels, 30))
        self.assertEqual(
            output_dict["waveform_seg"].shape[2], config.model_args.spec_segment_size * config.audio["hop_length"]
        )
        if encoder_config:
            self.assertEqual(output_dict["gt_spk_emb"].shape, (batch_size, encoder_config.model_params["proj_dim"]))
            self.assertEqual(output_dict["syn_spk_emb"].shape, (batch_size, encoder_config.model_params["proj_dim"]))
        else:
            self.assertEqual(output_dict["gt_spk_emb"], None)
            self.assertEqual(output_dict["syn_spk_emb"], None)

    def test_forward(self):
        num_speakers = 0
        config = VitsConfig(num_speakers=num_speakers, use_speaker_embedding=True)
        config.model_args.spec_segment_size = 10
        input_dummy, input_lengths, _, spec, spec_lengths, waveform = self._create_inputs(config)
        model = Vits(config).to(device)
        output_dict = model.forward(input_dummy, input_lengths, spec, spec_lengths, waveform)
        self._check_forward_outputs(config, output_dict)

    def test_multispeaker_forward(self):
        num_speakers = 10

        config = VitsConfig(num_speakers=num_speakers, use_speaker_embedding=True)
        config.model_args.spec_segment_size = 10

        input_dummy, input_lengths, _, spec, spec_lengths, waveform = self._create_inputs(config)
        speaker_ids = torch.randint(0, num_speakers, (8,)).long().to(device)

        model = Vits(config).to(device)
        output_dict = model.forward(
            input_dummy, input_lengths, spec, spec_lengths, waveform, aux_input={"speaker_ids": speaker_ids}
        )
        self._check_forward_outputs(config, output_dict)

    def test_d_vector_forward(self):
        batch_size = 2
        args = VitsArgs(
            spec_segment_size=10,
            num_chars=32,
            use_d_vector_file=True,
            d_vector_dim=256,
            d_vector_file=[os.path.join(get_tests_data_path(), "dummy_speakers.json")],
        )
        config = VitsConfig(model_args=args)
        model = Vits.init_from_config(config, verbose=False).to(device)
        model.train()
        input_dummy, input_lengths, _, spec, spec_lengths, waveform = self._create_inputs(config, batch_size=batch_size)
        d_vectors = torch.randn(batch_size, 256).to(device)
        output_dict = model.forward(
            input_dummy, input_lengths, spec, spec_lengths, waveform, aux_input={"d_vectors": d_vectors}
        )
        self._check_forward_outputs(config, output_dict)

    def test_multilingual_forward(self):
        num_speakers = 10
        num_langs = 3
        batch_size = 2

        args = VitsArgs(language_ids_file=LANG_FILE, use_language_embedding=True, spec_segment_size=10)
        config = VitsConfig(num_speakers=num_speakers, use_speaker_embedding=True, model_args=args)

        input_dummy, input_lengths, _, spec, spec_lengths, waveform = self._create_inputs(config, batch_size=batch_size)
        speaker_ids = torch.randint(0, num_speakers, (batch_size,)).long().to(device)
        lang_ids = torch.randint(0, num_langs, (batch_size,)).long().to(device)

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
        batch_size = 2

        speaker_encoder_config = load_config(SPEAKER_ENCODER_CONFIG)
        speaker_encoder_config.model_params["use_torch_spec"] = True
        speaker_encoder = setup_encoder_model(speaker_encoder_config).to(device)
        speaker_manager = SpeakerManager()
        speaker_manager.encoder = speaker_encoder

        args = VitsArgs(
            language_ids_file=LANG_FILE,
            use_language_embedding=True,
            spec_segment_size=10,
            use_speaker_encoder_as_loss=True,
        )
        config = VitsConfig(num_speakers=num_speakers, use_speaker_embedding=True, model_args=args)
        config.audio.sample_rate = 16000

        input_dummy, input_lengths, _, spec, spec_lengths, waveform = self._create_inputs(config, batch_size=batch_size)
        speaker_ids = torch.randint(0, num_speakers, (batch_size,)).long().to(device)
        lang_ids = torch.randint(0, num_langs, (batch_size,)).long().to(device)

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

    def _check_inference_outputs(self, config, outputs, input_dummy, batch_size=1):
        feat_len = outputs["z"].shape[2]
        self.assertEqual(outputs["model_outputs"].shape[:2], (batch_size, 1))  # we don't know the channel dimension
        self.assertEqual(outputs["alignments"].shape, (batch_size, input_dummy.shape[1], feat_len))
        self.assertEqual(outputs["z"].shape, (batch_size, config.model_args.hidden_channels, feat_len))
        self.assertEqual(outputs["z_p"].shape, (batch_size, config.model_args.hidden_channels, feat_len))
        self.assertEqual(outputs["m_p"].shape, (batch_size, config.model_args.hidden_channels, feat_len))
        self.assertEqual(outputs["logs_p"].shape, (batch_size, config.model_args.hidden_channels, feat_len))

    def test_inference(self):
        num_speakers = 0
        config = VitsConfig(num_speakers=num_speakers, use_speaker_embedding=True)
        model = Vits(config).to(device)

        batch_size = 1
        input_dummy, *_ = self._create_inputs(config, batch_size=batch_size)
        outputs = model.inference(input_dummy)
        self._check_inference_outputs(config, outputs, input_dummy, batch_size=batch_size)

        batch_size = 2
        input_dummy, input_lengths, *_ = self._create_inputs(config, batch_size=batch_size)
        outputs = model.inference(input_dummy, aux_input={"x_lengths": input_lengths})
        self._check_inference_outputs(config, outputs, input_dummy, batch_size=batch_size)

    def test_multispeaker_inference(self):
        num_speakers = 10
        config = VitsConfig(num_speakers=num_speakers, use_speaker_embedding=True)
        model = Vits(config).to(device)

        batch_size = 1
        input_dummy, *_ = self._create_inputs(config, batch_size=batch_size)
        speaker_ids = torch.randint(0, num_speakers, (batch_size,)).long().to(device)
        outputs = model.inference(input_dummy, {"speaker_ids": speaker_ids})
        self._check_inference_outputs(config, outputs, input_dummy, batch_size=batch_size)

        batch_size = 2
        input_dummy, input_lengths, *_ = self._create_inputs(config, batch_size=batch_size)
        speaker_ids = torch.randint(0, num_speakers, (batch_size,)).long().to(device)
        outputs = model.inference(input_dummy, {"x_lengths": input_lengths, "speaker_ids": speaker_ids})
        self._check_inference_outputs(config, outputs, input_dummy, batch_size=batch_size)

    def test_multilingual_inference(self):
        num_speakers = 10
        num_langs = 3
        args = VitsArgs(language_ids_file=LANG_FILE, use_language_embedding=True, spec_segment_size=10)
        config = VitsConfig(num_speakers=num_speakers, use_speaker_embedding=True, model_args=args)
        model = Vits(config).to(device)

        input_dummy = torch.randint(0, 24, (1, 128)).long().to(device)
        speaker_ids = torch.randint(0, num_speakers, (1,)).long().to(device)
        lang_ids = torch.randint(0, num_langs, (1,)).long().to(device)
        _ = model.inference(input_dummy, {"speaker_ids": speaker_ids, "language_ids": lang_ids})

        batch_size = 1
        input_dummy, *_ = self._create_inputs(config, batch_size=batch_size)
        speaker_ids = torch.randint(0, num_speakers, (batch_size,)).long().to(device)
        lang_ids = torch.randint(0, num_langs, (batch_size,)).long().to(device)
        outputs = model.inference(input_dummy, {"speaker_ids": speaker_ids, "language_ids": lang_ids})
        self._check_inference_outputs(config, outputs, input_dummy, batch_size=batch_size)

        batch_size = 2
        input_dummy, input_lengths, *_ = self._create_inputs(config, batch_size=batch_size)
        speaker_ids = torch.randint(0, num_speakers, (batch_size,)).long().to(device)
        lang_ids = torch.randint(0, num_langs, (batch_size,)).long().to(device)
        outputs = model.inference(
            input_dummy, {"x_lengths": input_lengths, "speaker_ids": speaker_ids, "language_ids": lang_ids}
        )
        self._check_inference_outputs(config, outputs, input_dummy, batch_size=batch_size)

    def test_d_vector_inference(self):
        args = VitsArgs(
            spec_segment_size=10,
            num_chars=32,
            use_d_vector_file=True,
            d_vector_dim=256,
            d_vector_file=[os.path.join(get_tests_data_path(), "dummy_speakers.json")],
        )
        config = VitsConfig(model_args=args)
        model = Vits.init_from_config(config, verbose=False).to(device)
        model.eval()
        # batch size = 1
        input_dummy = torch.randint(0, 24, (1, 128)).long().to(device)
        d_vectors = torch.randn(1, 256).to(device)
        outputs = model.inference(input_dummy, aux_input={"d_vectors": d_vectors})
        self._check_inference_outputs(config, outputs, input_dummy)
        # batch size = 2
        input_dummy, input_lengths, *_ = self._create_inputs(config)
        d_vectors = torch.randn(2, 256).to(device)
        outputs = model.inference(input_dummy, aux_input={"x_lengths": input_lengths, "d_vectors": d_vectors})
        self._check_inference_outputs(config, outputs, input_dummy, batch_size=2)

    @staticmethod
    def _check_parameter_changes(model, model_ref):
        count = 0
        for item1, item2 in zip(model.named_parameters(), model_ref.named_parameters()):
            name = item1[0]
            param = item1[1]
            param_ref = item2[1]
            assert (param != param_ref).any(), "param {} with shape {} not updated!! \n{}\n{}".format(
                name, param.shape, param, param_ref
            )
            count = count + 1

    def _create_batch(self, config, batch_size):
        input_dummy, input_lengths, mel, spec, mel_lengths, _ = self._create_inputs(config, batch_size)
        batch = {}
        batch["tokens"] = input_dummy
        batch["token_lens"] = input_lengths
        batch["spec_lens"] = mel_lengths
        batch["mel_lens"] = mel_lengths
        batch["spec"] = spec
        batch["mel"] = mel
        batch["waveform"] = torch.rand(batch_size, 1, config.audio["sample_rate"] * 10).to(device)
        batch["d_vectors"] = None
        batch["speaker_ids"] = None
        batch["language_ids"] = None
        return batch

    def test_train_step(self):
        # setup the model
        with torch.autograd.set_detect_anomaly(True):
            config = VitsConfig(model_args=VitsArgs(num_chars=32, spec_segment_size=10))
            model = Vits(config).to(device)
            model.train()
            # model to train
            optimizers = model.get_optimizer()
            criterions = model.get_criterion()
            criterions = [criterions[0].to(device), criterions[1].to(device)]
            # reference model to compare model weights
            model_ref = Vits(config).to(device)
            # # pass the state to ref model
            model_ref.load_state_dict(copy.deepcopy(model.state_dict()))
            count = 0
            for param, param_ref in zip(model.parameters(), model_ref.parameters()):
                assert (param - param_ref).sum() == 0, param
                count = count + 1
            for _ in range(5):
                batch = self._create_batch(config, 2)
                for idx in [0, 1]:
                    outputs, loss_dict = model.train_step(batch, criterions, idx)
                    self.assertFalse(not outputs)
                    self.assertFalse(not loss_dict)
                    loss_dict["loss"].backward()
                    optimizers[idx].step()
                    optimizers[idx].zero_grad()

        # check parameter changes
        self._check_parameter_changes(model, model_ref)

    def test_train_step_upsampling(self):
        """Upsampling by the decoder upsampling layers"""
        # setup the model
        with torch.autograd.set_detect_anomaly(True):
            audio_config = VitsAudioConfig(sample_rate=22050)
            model_args = VitsArgs(
                num_chars=32,
                spec_segment_size=10,
                encoder_sample_rate=11025,
                interpolate_z=False,
                upsample_rates_decoder=[8, 8, 4, 2],
            )
            config = VitsConfig(model_args=model_args, audio=audio_config)
            model = Vits(config).to(device)
            model.train()
            # model to train
            optimizers = model.get_optimizer()
            criterions = model.get_criterion()
            criterions = [criterions[0].to(device), criterions[1].to(device)]
            # reference model to compare model weights
            model_ref = Vits(config).to(device)
            # # pass the state to ref model
            model_ref.load_state_dict(copy.deepcopy(model.state_dict()))
            count = 0
            for param, param_ref in zip(model.parameters(), model_ref.parameters()):
                assert (param - param_ref).sum() == 0, param
                count = count + 1
            for _ in range(5):
                batch = self._create_batch(config, 2)
                for idx in [0, 1]:
                    outputs, loss_dict = model.train_step(batch, criterions, idx)
                    self.assertFalse(not outputs)
                    self.assertFalse(not loss_dict)
                    loss_dict["loss"].backward()
                    optimizers[idx].step()
                    optimizers[idx].zero_grad()

        # check parameter changes
        self._check_parameter_changes(model, model_ref)

    def test_train_step_upsampling_interpolation(self):
        """Upsampling by interpolation"""
        # setup the model
        with torch.autograd.set_detect_anomaly(True):
            audio_config = VitsAudioConfig(sample_rate=22050)
            model_args = VitsArgs(
                num_chars=32,
                spec_segment_size=10,
                encoder_sample_rate=11025,
                interpolate_z=True,
                upsample_rates_decoder=[8, 8, 2, 2],
            )
            config = VitsConfig(model_args=model_args, audio=audio_config)
            model = Vits(config).to(device)
            model.train()
            # model to train
            optimizers = model.get_optimizer()
            criterions = model.get_criterion()
            criterions = [criterions[0].to(device), criterions[1].to(device)]
            # reference model to compare model weights
            model_ref = Vits(config).to(device)
            # # pass the state to ref model
            model_ref.load_state_dict(copy.deepcopy(model.state_dict()))
            count = 0
            for param, param_ref in zip(model.parameters(), model_ref.parameters()):
                assert (param - param_ref).sum() == 0, param
                count = count + 1
            for _ in range(5):
                batch = self._create_batch(config, 2)
                for idx in [0, 1]:
                    outputs, loss_dict = model.train_step(batch, criterions, idx)
                    self.assertFalse(not outputs)
                    self.assertFalse(not loss_dict)
                    loss_dict["loss"].backward()
                    optimizers[idx].step()
                    optimizers[idx].zero_grad()

        # check parameter changes
        self._check_parameter_changes(model, model_ref)

    def test_train_eval_log(self):
        batch_size = 2
        config = VitsConfig(model_args=VitsArgs(num_chars=32, spec_segment_size=10))
        model = Vits.init_from_config(config, verbose=False).to(device)
        model.run_data_dep_init = False
        model.train()
        batch = self._create_batch(config, batch_size)
        logger = TensorboardLogger(
            log_dir=os.path.join(get_tests_output_path(), "dummy_vits_logs"), model_name="vits_test_train_log"
        )
        criterion = model.get_criterion()
        criterion = [criterion[0].to(device), criterion[1].to(device)]
        outputs = [None] * 2
        outputs[0], _ = model.train_step(batch, criterion, 0)
        outputs[1], _ = model.train_step(batch, criterion, 1)
        model.train_log(batch, outputs, logger, None, 1)

        model.eval_log(batch, outputs, logger, None, 1)
        logger.finish()

    def test_test_run(self):
        config = VitsConfig(model_args=VitsArgs(num_chars=32))
        model = Vits.init_from_config(config, verbose=False).to(device)
        model.run_data_dep_init = False
        model.eval()
        test_figures, test_audios = model.test_run(None)
        self.assertTrue(test_figures is not None)
        self.assertTrue(test_audios is not None)

    def test_load_checkpoint(self):
        chkp_path = os.path.join(get_tests_output_path(), "dummy_glow_tts_checkpoint.pth")
        config = VitsConfig(VitsArgs(num_chars=32))
        model = Vits.init_from_config(config, verbose=False).to(device)
        chkp = {}
        chkp["model"] = model.state_dict()
        torch.save(chkp, chkp_path)
        model.load_checkpoint(config, chkp_path)
        self.assertTrue(model.training)
        model.load_checkpoint(config, chkp_path, eval=True)
        self.assertFalse(model.training)

    def test_get_criterion(self):
        config = VitsConfig(VitsArgs(num_chars=32))
        model = Vits.init_from_config(config, verbose=False).to(device)
        criterion = model.get_criterion()
        self.assertTrue(criterion is not None)

    def test_init_from_config(self):
        config = VitsConfig(model_args=VitsArgs(num_chars=32))
        model = Vits.init_from_config(config, verbose=False).to(device)

        config = VitsConfig(model_args=VitsArgs(num_chars=32, num_speakers=2))
        model = Vits.init_from_config(config, verbose=False).to(device)
        self.assertTrue(not hasattr(model, "emb_g"))

        config = VitsConfig(model_args=VitsArgs(num_chars=32, num_speakers=2, use_speaker_embedding=True))
        model = Vits.init_from_config(config, verbose=False).to(device)
        self.assertEqual(model.num_speakers, 2)
        self.assertTrue(hasattr(model, "emb_g"))

        config = VitsConfig(
            model_args=VitsArgs(
                num_chars=32,
                num_speakers=2,
                use_speaker_embedding=True,
                speakers_file=os.path.join(get_tests_data_path(), "ljspeech", "speakers.json"),
            )
        )
        model = Vits.init_from_config(config, verbose=False).to(device)
        self.assertEqual(model.num_speakers, 10)
        self.assertTrue(hasattr(model, "emb_g"))

        config = VitsConfig(
            model_args=VitsArgs(
                num_chars=32,
                use_d_vector_file=True,
                d_vector_dim=256,
                d_vector_file=[os.path.join(get_tests_data_path(), "dummy_speakers.json")],
            )
        )
        model = Vits.init_from_config(config, verbose=False).to(device)
        self.assertTrue(model.num_speakers == 1)
        self.assertTrue(not hasattr(model, "emb_g"))
        self.assertTrue(model.embedded_speaker_dim == config.d_vector_dim)
