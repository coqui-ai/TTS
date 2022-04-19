import copy
import os
import unittest

import torch
from trainer.logging.tensorboard_logger import TensorboardLogger

from tests import assertHasAttr, assertHasNotAttr, get_tests_data_path, get_tests_input_path, get_tests_output_path
from TTS.tts.configs.fast_pitch_e2e_config import FastPitchE2eConfig
from TTS.tts.models.forward_tts_e2e import ForwardTTSE2e, ForwardTTSE2eArgs

LANG_FILE = os.path.join(get_tests_input_path(), "language_ids.json")
SPEAKER_ENCODER_CONFIG = os.path.join(get_tests_input_path(), "test_speaker_encoder_config.json")
WAV_FILE = os.path.join(get_tests_input_path(), "example_1.wav")


torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_INPUT_LEN = 57
MAX_SPEC_LEN = 33


# pylint: disable=no-self-use
class TestFastPitchE2E(unittest.TestCase):
    def _create_inputs(self, config, batch_size=2):

        input_dummy = torch.randint(0, 24, (batch_size, MAX_INPUT_LEN)).long().to(device)
        input_lengths = torch.randint(10, MAX_INPUT_LEN, (batch_size,)).long().to(device)
        input_lengths[-1] = MAX_INPUT_LEN
        spec = torch.rand(batch_size, MAX_SPEC_LEN, config.audio["num_mels"]).to(device)
        spec_lengths = torch.randint(20, MAX_SPEC_LEN, (batch_size,)).long().to(device)
        spec_lengths[-1] = MAX_SPEC_LEN
        waveform = torch.rand(batch_size, 1, spec.size(1) * config.audio["hop_length"]).to(device)
        pitch = torch.rand(batch_size, 1, spec.size(1)).to(device)
        return input_dummy, input_lengths, spec, spec_lengths, waveform, pitch

    def _check_forward_outputs(self, config, output_dict, batch_size=2):
        self.assertEqual(
            output_dict["model_outputs"].shape[2], config.model_args.spec_segment_size * config.audio["hop_length"]
        )
        self.assertEqual(output_dict["alignments"].shape, (batch_size, MAX_SPEC_LEN, MAX_INPUT_LEN))
        self.assertEqual(output_dict["alignments"].max(), 1)
        self.assertEqual(output_dict["alignments"].min(), 0)
        self.assertEqual(
            output_dict["waveform_seg"].shape[2], config.model_args.spec_segment_size * config.audio["hop_length"]
        )

    def _check_inference_outputs(self, outputs, input_dummy, batch_size=1):
        feat_dim = 256  # hard-coded based on model architecture
        feat_len = outputs["o_en_ex"].shape[2]
        self.assertEqual(outputs["o_en_ex"].shape, (batch_size, feat_dim, feat_len))
        self.assertEqual(outputs["model_outputs"].shape[:2], (batch_size, 1))  # we don't know the channel dimension
        self.assertEqual(outputs["alignments"].shape, (batch_size, input_dummy.shape[1], feat_len))

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
        input_dummy, input_lengths, spec, spec_lengths, waveform, pitch = self._create_inputs(config, batch_size)
        batch = {}
        batch["text_input"] = input_dummy
        batch["text_lengths"] = input_lengths
        batch["mel_lengths"] = spec_lengths
        batch["mel_input"] = spec
        batch["waveform"] = waveform  # B x C X T
        batch["d_vectors"] = None
        batch["speaker_ids"] = None
        batch["language_ids"] = None
        batch["pitch"] = pitch
        return batch

    def test_init_multispeaker(self):

        num_speakers = 10
        model_args = ForwardTTSE2eArgs()
        model_args.num_speakers = num_speakers
        model_args.use_speaker_embedding = True
        model = ForwardTTSE2e(model_args)
        assertHasAttr(self, model.encoder_model, "emb_g")

        model_args = ForwardTTSE2eArgs()
        model_args.num_speakers = 0
        model_args.use_speaker_embedding = True
        model = ForwardTTSE2e(model_args)
        assertHasNotAttr(self, model.encoder_model, "emb_g")

        model_args = ForwardTTSE2eArgs()
        model_args.num_speakers = 10
        model_args.use_speaker_embedding = False
        model = ForwardTTSE2e(model_args)
        assertHasNotAttr(self, model.encoder_model, "emb_g")

        model_args = ForwardTTSE2eArgs(d_vector_dim=101, use_d_vector_file=True)
        model = ForwardTTSE2e(model_args)
        self.assertEqual(model.encoder_model.embedded_speaker_dim, 101)

    def test_init_multilingual(self):
        """TODO"""

    def test_get_aux_input(self):
        aux_input = {"speaker_ids": None, "style_wav": None, "d_vectors": None, "language_ids": None}
        model_args = ForwardTTSE2eArgs()
        model = ForwardTTSE2e(model_args)
        aux_out = model.get_aux_input(aux_input)

        speaker_id = torch.randint(10, (1,))
        language_id = torch.randint(10, (1,))
        d_vector = torch.rand(1, 128)
        aux_input = {"speaker_ids": speaker_id, "style_wav": None, "d_vectors": d_vector, "language_ids": language_id}
        aux_out = model.get_aux_input(aux_input)
        self.assertEqual(aux_out["speaker_ids"].shape, speaker_id.shape)
        self.assertEqual(aux_out["language_ids"].shape, language_id.shape)
        self.assertEqual(aux_out["d_vectors"].shape, d_vector.unsqueeze(0).transpose(2, 1).shape)

    def test_forward(self):
        model_args = ForwardTTSE2eArgs(spec_segment_size=10)
        config = FastPitchE2eConfig(model_args=model_args)
        input_dummy, input_lengths, spec, spec_lengths, waveform, pitch = self._create_inputs(config)
        model = ForwardTTSE2e(config).to(device)
        output_dict = model.forward(
            x=input_dummy, x_lengths=input_lengths, spec=spec, spec_lengths=spec_lengths, waveform=waveform, pitch=pitch
        )
        self._check_forward_outputs(config, output_dict)

    def test_multispeaker_forward(self):
        batch_size = 2
        num_speakers = 10
        model_args = ForwardTTSE2eArgs(spec_segment_size=10, num_speakers=num_speakers, use_speaker_embedding=True)
        config = FastPitchE2eConfig(model_args=model_args)
        config.model_args.spec_segment_size = 10

        input_dummy, input_lengths, spec, spec_lengths, waveform, pitch = self._create_inputs(
            config, batch_size=batch_size
        )
        speaker_ids = torch.randint(0, num_speakers, (batch_size,)).long().to(device)

        model = ForwardTTSE2e(config).to(device)
        output_dict = model.forward(
            x=input_dummy,
            x_lengths=input_lengths,
            spec=spec,
            spec_lengths=spec_lengths,
            waveform=waveform,
            pitch=pitch,
            aux_input={"speaker_ids": speaker_ids},
        )
        self._check_forward_outputs(config, output_dict)

    def test_d_vector_forward(self):
        batch_size = 2
        model_args = ForwardTTSE2eArgs(spec_segment_size=10, use_d_vector_file=True, d_vector_dim=256)
        config = FastPitchE2eConfig(model_args=model_args)
        config.model_args.spec_segment_size = 10
        model = ForwardTTSE2e(config).to(device)
        model.train()
        input_dummy, input_lengths, spec, spec_lengths, waveform, pitch = self._create_inputs(
            config, batch_size=batch_size
        )
        d_vectors = torch.randn(batch_size, 256).to(device)
        output_dict = model.forward(
            x=input_dummy,
            x_lengths=input_lengths,
            spec=spec,
            spec_lengths=spec_lengths,
            waveform=waveform,
            pitch=pitch,
            aux_input={"d_vectors": d_vectors},
        )
        self._check_forward_outputs(config, output_dict)

    # def test_multilingual_forward(self):
    #     """TODO"""

    def test_inference(self):
        model_args = ForwardTTSE2eArgs(spec_segment_size=10)
        config = FastPitchE2eConfig(model_args=model_args)
        model = ForwardTTSE2e(config).to(device)
        model.eval()

        batch_size = 1
        input_dummy, *_ = self._create_inputs(config, batch_size=batch_size)
        outputs = model.inference(input_dummy.to(device))
        self._check_inference_outputs(outputs, input_dummy, batch_size=batch_size)

        # TODO implemented batched inferenece
        # batch_size = 2
        # input_dummy, input_lengths, *_ = self._create_inputs(config, batch_size=batch_size)
        # outputs = model.inference(input_dummy, aux_input={"x_lengths": input_lengths})
        # self._check_inference_outputs(outputs, input_dummy, batch_size=batch_size)

    def test_multispeaker_inference(self):
        num_speakers = 10
        model_args = ForwardTTSE2eArgs(spec_segment_size=10, num_speakers=num_speakers, use_speaker_embedding=True)
        config = FastPitchE2eConfig(model_args=model_args)
        model = ForwardTTSE2e(config).to(device)

        batch_size = 1
        input_dummy, *_ = self._create_inputs(config, batch_size=batch_size)
        speaker_ids = torch.randint(0, num_speakers, (batch_size,)).long().to(device)
        outputs = model.inference(input_dummy, {"speaker_ids": speaker_ids})
        self._check_inference_outputs(outputs, input_dummy, batch_size=batch_size)

        # batch_size = 2
        # input_dummy, input_lengths, *_ = self._create_inputs(config, batch_size=batch_size)
        # speaker_ids = torch.randint(0, num_speakers, (batch_size,)).long().to(device)
        # outputs = model.inference(input_dummy, {"x_lengths": input_lengths, "speaker_ids": speaker_ids})
        # self._check_inference_outputs(outputs, input_dummy, batch_size=batch_size)

    # def test_multilingual_inference(self):
    #     """TODO"""

    def test_d_vector_inference(self):
        model_args = ForwardTTSE2eArgs(
            spec_segment_size=10,
            num_chars=32,
            use_d_vector_file=True,
            d_vector_dim=256,
            d_vector_file=os.path.join(get_tests_data_path(), "dummy_speakers.json"),
        )
        config = FastPitchE2eConfig(model_args=model_args)
        model = ForwardTTSE2e(config).to(device)
        model.eval()
        # batch size = 1
        input_dummy, *_ = self._create_inputs(config, batch_size=1)
        d_vectors = torch.randn(1, 256).to(device)
        outputs = model.inference(input_dummy, aux_input={"d_vectors": d_vectors})
        self._check_inference_outputs(outputs, input_dummy)
        # batch size = 2
        # input_dummy, input_lengths, *_ = self._create_inputs(config)
        # d_vectors = torch.randn(2, 256).to(device)
        # outputs = model.inference(input_dummy, aux_input={"x_lengths": input_lengths, "d_vectors": d_vectors})
        # self._check_inference_outputs(outputs, input_dummy, batch_size=2)

    def test_train_step(self):
        # setup the model
        with torch.autograd.set_detect_anomaly(True):
            model_args = ForwardTTSE2eArgs(spec_segment_size=10)
            config = FastPitchE2eConfig(model_args=model_args)
            model = ForwardTTSE2e(config).to(device)
            model.train()
            # model to train
            optimizers = model.get_optimizer()
            criterions = model.get_criterion()
            criterions = [criterions[0].to(device), criterions[1].to(device)]
            # reference model to compare model weights
            model_ref = ForwardTTSE2e(config).to(device)
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
        model_args = ForwardTTSE2eArgs(spec_segment_size=10)
        config = FastPitchE2eConfig(model_args=model_args)
        model = ForwardTTSE2e.init_from_config(config, verbose=False).to(device)
        model.train()
        model.on_init_start(trainer=None)  # create mel_basis
        batch = self._create_batch(config, batch_size)
        logger = TensorboardLogger(
            log_dir=os.path.join(get_tests_output_path(), "dummy_fast_pitch_e2e_logs"),
            model_name="fast_pitch_e2e_test_train_log",
        )
        criterion = model.get_criterion()
        criterion = [criterion[0].to(device), criterion[1].to(device)]
        outputs = [None] * 2
        outputs[0], _ = model.train_step(batch, criterion, 0)
        outputs[1], _ = model.train_step(batch, criterion, 1)
        model.train_log(batch=batch, outputs=outputs, logger=logger, assets=None, steps=1)
        model.eval_log(batch, outputs, logger, None, 1)
        logger.finish()

    def test_test_run(self):
        model_args = ForwardTTSE2eArgs(spec_segment_size=10)
        config = FastPitchE2eConfig(model_args=model_args)
        model = ForwardTTSE2e.init_from_config(config, verbose=False).to(device)
        model.eval()
        model.on_init_start(trainer=None)  # create mel_basis
        test_figures, test_audios = model.test_run(None)
        self.assertTrue(test_figures is not None)
        self.assertTrue(test_audios is not None)

    def test_load_checkpoint(self):
        chkp_path = os.path.join(get_tests_output_path(), "dummy_fast_pitch_e2e_tts_checkpoint.pth")
        model_args = ForwardTTSE2eArgs(spec_segment_size=10)
        config = FastPitchE2eConfig(model_args=model_args)
        model = ForwardTTSE2e.init_from_config(config, verbose=False).to(device)
        chkp = {}
        chkp["model"] = model.state_dict()
        torch.save(chkp, chkp_path)
        model.load_checkpoint(config, chkp_path)
        self.assertTrue(model.training)
        model.load_checkpoint(config, chkp_path, eval=True)
        self.assertFalse(model.training)

    def test_get_criterion(self):
        model_args = ForwardTTSE2eArgs(spec_segment_size=10)
        config = FastPitchE2eConfig(model_args=model_args)
        model = ForwardTTSE2e.init_from_config(config, verbose=False).to(device)
        criterion = model.get_criterion()
        self.assertTrue(criterion is not None)

    def test_init_from_config(self):
        model_args = ForwardTTSE2eArgs(spec_segment_size=10)
        config = FastPitchE2eConfig(model_args=model_args)
        model = ForwardTTSE2e.init_from_config(config, verbose=False).to(device)

        model_args = ForwardTTSE2eArgs(spec_segment_size=10, num_speakers=2)
        config = FastPitchE2eConfig(model_args=model_args)
        model = ForwardTTSE2e.init_from_config(config, verbose=False).to(device)
        self.assertTrue(not hasattr(model, "emb_g"))

        model_args = ForwardTTSE2eArgs(spec_segment_size=10, num_speakers=2, use_speaker_embedding=True)
        config = FastPitchE2eConfig(model_args=model_args)
        model = ForwardTTSE2e.init_from_config(config, verbose=False).to(device)
        self.assertEqual(model.num_speakers, 2)
        self.assertTrue(hasattr(model, "emb_g"))

        model_args = ForwardTTSE2eArgs(
            spec_segment_size=10,
            num_speakers=2,
            use_speaker_embedding=True,
            speakers_file=os.path.join(get_tests_data_path(), "ljspeech", "speakers.json"),
        )
        config = FastPitchE2eConfig(model_args=model_args)
        model = ForwardTTSE2e.init_from_config(config, verbose=False).to(device)
        self.assertEqual(model.num_speakers, 10)
        self.assertTrue(hasattr(model, "emb_g"))

        model_args = ForwardTTSE2eArgs(
            spec_segment_size=10,
            use_d_vector_file=True,
            d_vector_dim=256,
            d_vector_file=os.path.join(get_tests_data_path(), "ljspeech", "speakers.json"),
        )
        config = FastPitchE2eConfig(model_args=model_args)
        model = ForwardTTSE2e.init_from_config(config, verbose=False).to(device)
        self.assertTrue(model.num_speakers == 10)
        self.assertTrue(not hasattr(model, "emb_g"))
        self.assertTrue(model.embedded_speaker_dim == config.model_args.d_vector_dim)
