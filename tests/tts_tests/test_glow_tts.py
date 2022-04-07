import copy
import os
import unittest

import torch
from torch import optim
from trainer.logging.tensorboard_logger import TensorboardLogger

from tests import get_tests_data_path, get_tests_input_path, get_tests_output_path
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.layers.losses import GlowTTSLoss
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor

# pylint: disable=unused-variable

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

c = GlowTTSConfig()

ap = AudioProcessor(**c.audio)
WAV_FILE = os.path.join(get_tests_input_path(), "example_1.wav")
BATCH_SIZE = 3


def count_parameters(model):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TestGlowTTS(unittest.TestCase):
    @staticmethod
    def _create_inputs(batch_size=8):
        input_dummy = torch.randint(0, 24, (batch_size, 128)).long().to(device)
        input_lengths = torch.randint(100, 129, (batch_size,)).long().to(device)
        input_lengths[-1] = 128
        mel_spec = torch.rand(batch_size, 30, c.audio["num_mels"]).to(device)
        mel_lengths = torch.randint(20, 30, (batch_size,)).long().to(device)
        speaker_ids = torch.randint(0, 5, (batch_size,)).long().to(device)
        return input_dummy, input_lengths, mel_spec, mel_lengths, speaker_ids

    @staticmethod
    def _check_parameter_changes(model, model_ref):
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param != param_ref).any(), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref
            )
            count += 1

    def test_init_multispeaker(self):
        config = GlowTTSConfig(num_chars=32)
        model = GlowTTS(config)
        # speaker embedding with default speaker_embedding_dim
        config.use_speaker_embedding = True
        config.num_speakers = 5
        config.d_vector_dim = None
        model.init_multispeaker(config)
        self.assertEqual(model.c_in_channels, model.hidden_channels_enc)
        # use external speaker embeddings with speaker_embedding_dim = 301
        config = GlowTTSConfig(num_chars=32)
        config.use_d_vector_file = True
        config.d_vector_dim = 301
        model = GlowTTS(config)
        model.init_multispeaker(config)
        self.assertEqual(model.c_in_channels, 301)
        # use speaker embedddings by the provided speaker_manager
        config = GlowTTSConfig(num_chars=32)
        config.use_speaker_embedding = True
        config.speakers_file = os.path.join(get_tests_data_path(), "ljspeech", "speakers.json")
        speaker_manager = SpeakerManager.init_from_config(config)
        model = GlowTTS(config)
        model.speaker_manager = speaker_manager
        model.init_multispeaker(config)
        self.assertEqual(model.c_in_channels, model.hidden_channels_enc)
        self.assertEqual(model.num_speakers, speaker_manager.num_speakers)
        # use external speaker embeddings by the provided speaker_manager
        config = GlowTTSConfig(num_chars=32)
        config.use_d_vector_file = True
        config.d_vector_dim = 256
        config.d_vector_file = os.path.join(get_tests_data_path(), "dummy_speakers.json")
        speaker_manager = SpeakerManager.init_from_config(config)
        model = GlowTTS(config)
        model.speaker_manager = speaker_manager
        model.init_multispeaker(config)
        self.assertEqual(model.c_in_channels, speaker_manager.embedding_dim)
        self.assertEqual(model.num_speakers, speaker_manager.num_speakers)

    def test_unlock_act_norm_layers(self):
        config = GlowTTSConfig(num_chars=32)
        model = GlowTTS(config).to(device)
        model.unlock_act_norm_layers()
        for f in model.decoder.flows:
            if getattr(f, "set_ddi", False):
                self.assertFalse(f.initialized)

    def test_lock_act_norm_layers(self):
        config = GlowTTSConfig(num_chars=32)
        model = GlowTTS(config).to(device)
        model.lock_act_norm_layers()
        for f in model.decoder.flows:
            if getattr(f, "set_ddi", False):
                self.assertTrue(f.initialized)

    def _test_forward(self, batch_size):
        input_dummy, input_lengths, mel_spec, mel_lengths, speaker_ids = self._create_inputs(batch_size)
        # create model
        config = GlowTTSConfig(num_chars=32)
        model = GlowTTS(config).to(device)
        model.train()
        print(" > Num parameters for GlowTTS model:%s" % (count_parameters(model)))
        # inference encoder and decoder with MAS
        y = model.forward(input_dummy, input_lengths, mel_spec, mel_lengths)
        self.assertEqual(y["z"].shape, mel_spec.shape)
        self.assertEqual(y["logdet"].shape, torch.Size([batch_size]))
        self.assertEqual(y["y_mean"].shape, mel_spec.shape)
        self.assertEqual(y["y_log_scale"].shape, mel_spec.shape)
        self.assertEqual(y["alignments"].shape, mel_spec.shape[:2] + (input_dummy.shape[1],))
        self.assertEqual(y["durations_log"].shape, input_dummy.shape + (1,))
        self.assertEqual(y["total_durations_log"].shape, input_dummy.shape + (1,))

    def test_forward(self):
        self._test_forward(1)
        self._test_forward(3)

    def _test_forward_with_d_vector(self, batch_size):
        input_dummy, input_lengths, mel_spec, mel_lengths, speaker_ids = self._create_inputs(batch_size)
        d_vector = torch.rand(batch_size, 256).to(device)
        # create model
        config = GlowTTSConfig(
            num_chars=32,
            use_d_vector_file=True,
            d_vector_dim=256,
            d_vector_file=os.path.join(get_tests_data_path(), "dummy_speakers.json"),
        )
        model = GlowTTS.init_from_config(config, verbose=False).to(device)
        model.train()
        print(" > Num parameters for GlowTTS model:%s" % (count_parameters(model)))
        # inference encoder and decoder with MAS
        y = model.forward(input_dummy, input_lengths, mel_spec, mel_lengths, {"d_vectors": d_vector})
        self.assertEqual(y["z"].shape, mel_spec.shape)
        self.assertEqual(y["logdet"].shape, torch.Size([batch_size]))
        self.assertEqual(y["y_mean"].shape, mel_spec.shape)
        self.assertEqual(y["y_log_scale"].shape, mel_spec.shape)
        self.assertEqual(y["alignments"].shape, mel_spec.shape[:2] + (input_dummy.shape[1],))
        self.assertEqual(y["durations_log"].shape, input_dummy.shape + (1,))
        self.assertEqual(y["total_durations_log"].shape, input_dummy.shape + (1,))

    def test_forward_with_d_vector(self):
        self._test_forward_with_d_vector(1)
        self._test_forward_with_d_vector(3)

    def _test_forward_with_speaker_id(self, batch_size):
        input_dummy, input_lengths, mel_spec, mel_lengths, speaker_ids = self._create_inputs(batch_size)
        speaker_ids = torch.randint(0, 24, (batch_size,)).long().to(device)
        # create model
        config = GlowTTSConfig(
            num_chars=32,
            use_speaker_embedding=True,
            num_speakers=24,
        )
        model = GlowTTS.init_from_config(config, verbose=False).to(device)
        model.train()
        print(" > Num parameters for GlowTTS model:%s" % (count_parameters(model)))
        # inference encoder and decoder with MAS
        y = model.forward(input_dummy, input_lengths, mel_spec, mel_lengths, {"speaker_ids": speaker_ids})
        self.assertEqual(y["z"].shape, mel_spec.shape)
        self.assertEqual(y["logdet"].shape, torch.Size([batch_size]))
        self.assertEqual(y["y_mean"].shape, mel_spec.shape)
        self.assertEqual(y["y_log_scale"].shape, mel_spec.shape)
        self.assertEqual(y["alignments"].shape, mel_spec.shape[:2] + (input_dummy.shape[1],))
        self.assertEqual(y["durations_log"].shape, input_dummy.shape + (1,))
        self.assertEqual(y["total_durations_log"].shape, input_dummy.shape + (1,))

    def test_forward_with_speaker_id(self):
        self._test_forward_with_speaker_id(1)
        self._test_forward_with_speaker_id(3)

    def _assert_inference_outputs(self, outputs, input_dummy, mel_spec):
        output_shape = outputs["model_outputs"].shape
        self.assertEqual(outputs["model_outputs"].shape[::2], mel_spec.shape[::2])
        self.assertEqual(outputs["logdet"], None)
        self.assertEqual(outputs["y_mean"].shape, output_shape)
        self.assertEqual(outputs["y_log_scale"].shape, output_shape)
        self.assertEqual(outputs["alignments"].shape, output_shape[:2] + (input_dummy.shape[1],))
        self.assertEqual(outputs["durations_log"].shape, input_dummy.shape + (1,))
        self.assertEqual(outputs["total_durations_log"].shape, input_dummy.shape + (1,))

    def _test_inference(self, batch_size):
        input_dummy, input_lengths, mel_spec, mel_lengths, speaker_ids = self._create_inputs(batch_size)
        config = GlowTTSConfig(num_chars=32)
        model = GlowTTS(config).to(device)
        model.eval()
        outputs = model.inference(input_dummy, {"x_lengths": input_lengths})
        self._assert_inference_outputs(outputs, input_dummy, mel_spec)

    def test_inference(self):
        self._test_inference(1)
        self._test_inference(3)

    def _test_inference_with_d_vector(self, batch_size):
        input_dummy, input_lengths, mel_spec, mel_lengths, speaker_ids = self._create_inputs(batch_size)
        d_vector = torch.rand(batch_size, 256).to(device)
        config = GlowTTSConfig(
            num_chars=32,
            use_d_vector_file=True,
            d_vector_dim=256,
            d_vector_file=os.path.join(get_tests_data_path(), "dummy_speakers.json"),
        )
        model = GlowTTS.init_from_config(config, verbose=False).to(device)
        model.eval()
        outputs = model.inference(input_dummy, {"x_lengths": input_lengths, "d_vectors": d_vector})
        self._assert_inference_outputs(outputs, input_dummy, mel_spec)

    def test_inference_with_d_vector(self):
        self._test_inference_with_d_vector(1)
        self._test_inference_with_d_vector(3)

    def _test_inference_with_speaker_ids(self, batch_size):
        input_dummy, input_lengths, mel_spec, mel_lengths, speaker_ids = self._create_inputs(batch_size)
        speaker_ids = torch.randint(0, 24, (batch_size,)).long().to(device)
        # create model
        config = GlowTTSConfig(
            num_chars=32,
            use_speaker_embedding=True,
            num_speakers=24,
        )
        model = GlowTTS.init_from_config(config, verbose=False).to(device)
        outputs = model.inference(input_dummy, {"x_lengths": input_lengths, "speaker_ids": speaker_ids})
        self._assert_inference_outputs(outputs, input_dummy, mel_spec)

    def test_inference_with_speaker_ids(self):
        self._test_inference_with_speaker_ids(1)
        self._test_inference_with_speaker_ids(3)

    def _test_inference_with_MAS(self, batch_size):
        input_dummy, input_lengths, mel_spec, mel_lengths, speaker_ids = self._create_inputs(batch_size)
        # create model
        config = GlowTTSConfig(num_chars=32)
        model = GlowTTS(config).to(device)
        model.eval()
        # inference encoder and decoder with MAS
        y = model.inference_with_MAS(input_dummy, input_lengths, mel_spec, mel_lengths)
        y2 = model.decoder_inference(mel_spec, mel_lengths)
        assert (
            y2["model_outputs"].shape == y["model_outputs"].shape
        ), "Difference between the shapes of the glowTTS inference with MAS ({}) and the inference using only the decoder ({}) !!".format(
            y["model_outputs"].shape, y2["model_outputs"].shape
        )

    def test_inference_with_MAS(self):
        self._test_inference_with_MAS(1)
        self._test_inference_with_MAS(3)

    def test_train_step(self):
        batch_size = BATCH_SIZE
        input_dummy, input_lengths, mel_spec, mel_lengths, speaker_ids = self._create_inputs(batch_size)
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
        self._check_parameter_changes(model, model_ref)

    def test_train_eval_log(self):
        batch_size = BATCH_SIZE
        input_dummy, input_lengths, mel_spec, mel_lengths, _ = self._create_inputs(batch_size)
        batch = {}
        batch["text_input"] = input_dummy
        batch["text_lengths"] = input_lengths
        batch["mel_lengths"] = mel_lengths
        batch["mel_input"] = mel_spec
        batch["d_vectors"] = None
        batch["speaker_ids"] = None
        config = GlowTTSConfig(num_chars=32)
        model = GlowTTS.init_from_config(config, verbose=False).to(device)
        model.run_data_dep_init = False
        model.train()
        logger = TensorboardLogger(
            log_dir=os.path.join(get_tests_output_path(), "dummy_glow_tts_logs"), model_name="glow_tts_test_train_log"
        )
        criterion = model.get_criterion()
        outputs, _ = model.train_step(batch, criterion)
        model.train_log(batch, outputs, logger, None, 1)
        model.eval_log(batch, outputs, logger, None, 1)
        logger.finish()

    def test_test_run(self):
        config = GlowTTSConfig(num_chars=32)
        model = GlowTTS.init_from_config(config, verbose=False).to(device)
        model.run_data_dep_init = False
        model.eval()
        test_figures, test_audios = model.test_run(None)
        self.assertTrue(test_figures is not None)
        self.assertTrue(test_audios is not None)

    def test_load_checkpoint(self):
        chkp_path = os.path.join(get_tests_output_path(), "dummy_glow_tts_checkpoint.pth")
        config = GlowTTSConfig(num_chars=32)
        model = GlowTTS.init_from_config(config, verbose=False).to(device)
        chkp = {}
        chkp["model"] = model.state_dict()
        torch.save(chkp, chkp_path)
        model.load_checkpoint(config, chkp_path)
        self.assertTrue(model.training)
        model.load_checkpoint(config, chkp_path, eval=True)
        self.assertFalse(model.training)

    def test_get_criterion(self):
        config = GlowTTSConfig(num_chars=32)
        model = GlowTTS.init_from_config(config, verbose=False).to(device)
        criterion = model.get_criterion()
        self.assertTrue(criterion is not None)

    def test_init_from_config(self):
        config = GlowTTSConfig(num_chars=32)
        model = GlowTTS.init_from_config(config, verbose=False).to(device)

        config = GlowTTSConfig(num_chars=32, num_speakers=2)
        model = GlowTTS.init_from_config(config, verbose=False).to(device)
        self.assertTrue(model.num_speakers == 2)
        self.assertTrue(not hasattr(model, "emb_g"))

        config = GlowTTSConfig(num_chars=32, num_speakers=2, use_speaker_embedding=True)
        model = GlowTTS.init_from_config(config, verbose=False).to(device)
        self.assertTrue(model.num_speakers == 2)
        self.assertTrue(hasattr(model, "emb_g"))

        config = GlowTTSConfig(
            num_chars=32,
            num_speakers=2,
            use_speaker_embedding=True,
            speakers_file=os.path.join(get_tests_data_path(), "ljspeech", "speakers.json"),
        )
        model = GlowTTS.init_from_config(config, verbose=False).to(device)
        self.assertTrue(model.num_speakers == 10)
        self.assertTrue(hasattr(model, "emb_g"))

        config = GlowTTSConfig(
            num_chars=32,
            use_d_vector_file=True,
            d_vector_dim=256,
            d_vector_file=os.path.join(get_tests_data_path(), "dummy_speakers.json"),
        )
        model = GlowTTS.init_from_config(config, verbose=False).to(device)
        self.assertTrue(model.num_speakers == 1)
        self.assertTrue(not hasattr(model, "emb_g"))
        self.assertTrue(model.c_in_channels == config.d_vector_dim)
