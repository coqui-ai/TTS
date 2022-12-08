import os
import random
import unittest
from copy import deepcopy

import torch

from tests import get_tests_output_path
from TTS.tts.configs.overflow_config import OverflowConfig
from TTS.tts.layers.overflow.common_layers import Encoder, OverflowUtils
from TTS.tts.layers.overflow.decoder import Decoder
from TTS.tts.layers.overflow.neural_hmm import NeuralHMM
from TTS.tts.models.overflow import Overflow
from TTS.tts.utils.helpers import sequence_mask
from TTS.utils.audio import AudioProcessor

# pylint: disable=unused-variable

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config_global = OverflowConfig(num_chars=24)
ap = AudioProcessor.init_from_config(config_global)

config_path = os.path.join(get_tests_output_path(), "test_model_config.json")
output_path = os.path.join(get_tests_output_path(), "train_outputs")
parameter_path = os.path.join(get_tests_output_path(), "lj_parameters.pt")

torch.save({"mean": -5.5138, "std": 2.0636, "init_transition_prob": 0.3212}, parameter_path)


def _create_inputs(batch_size=8):
    max_len_t, max_len_m = random.randint(25, 50), random.randint(50, 80)
    input_dummy = torch.randint(0, 24, (batch_size, max_len_t)).long().to(device)
    input_lengths = torch.randint(20, max_len_t, (batch_size,)).long().to(device).sort(descending=True)[0]
    input_lengths[0] = max_len_t
    input_dummy = input_dummy * sequence_mask(input_lengths)
    mel_spec = torch.randn(batch_size, max_len_m, config_global.audio["num_mels"]).to(device)
    mel_lengths = torch.randint(40, max_len_m, (batch_size,)).long().to(device).sort(descending=True)[0]
    mel_lengths[0] = max_len_m
    mel_spec = mel_spec * sequence_mask(mel_lengths).unsqueeze(2)
    return input_dummy, input_lengths, mel_spec, mel_lengths


def get_model(config=None):
    if config is None:
        config = config_global
    config.mel_statistics_parameter_path = parameter_path
    model = Overflow(config)
    model = model.to(device)
    return model


def reset_all_weights(model):
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)


class TestOverFlow(unittest.TestCase):
    def test_forward(self):
        model = get_model()
        input_dummy, input_lengths, mel_spec, mel_lengths = _create_inputs()
        outputs = model(input_dummy, input_lengths, mel_spec, mel_lengths)
        self.assertEqual(outputs["log_probs"].shape, (input_dummy.shape[0],))
        self.assertEqual(model.state_per_phone * max(input_lengths), outputs["alignments"].shape[2])

    def test_inference(self):
        model = get_model()
        input_dummy, input_lengths, mel_spec, mel_lengths = _create_inputs()
        output_dict = model.inference(input_dummy)
        self.assertEqual(output_dict["model_outputs"].shape[2], config_global.out_channels)

    def test_init_from_config(self):
        config = deepcopy(config_global)
        config.mel_statistics_parameter_path = parameter_path
        config.prenet_dim = 256
        model = Overflow.init_from_config(config_global)
        self.assertEqual(model.prenet_dim, config.prenet_dim)


class TestOverFlowPrenet(unittest.TestCase):
    @staticmethod
    def get_encoder(state_per_phone):
        config = deepcopy(config_global)
        config.state_per_phone = state_per_phone
        config.num_chars = 24
        return Encoder(config.num_chars, config.state_per_phone, config.prenet_dim, config.encoder_n_convolutions).to(
            device
        )

    def test_forward_with_state_per_phone_multiplication(self):
        for s_p_p in [1, 2, 3]:
            input_dummy, input_lengths, _, _ = _create_inputs()
            model = self.get_encoder(s_p_p)
            x, x_len = model(input_dummy, input_lengths)
            self.assertEqual(x.shape[1], input_dummy.shape[1] * s_p_p)

    def test_inference_with_state_per_phone_multiplication(self):
        for s_p_p in [1, 2, 3]:
            input_dummy, input_lengths, _, _ = _create_inputs()
            model = self.get_encoder(s_p_p)
            x, x_len = model.inference(input_dummy, input_lengths)
            self.assertEqual(x.shape[1], input_dummy.shape[1] * s_p_p)


class TestOverFlowUtils(unittest.TestCase):
    def test_logsumexp(self):
        a = torch.randn(10)  # random numbers
        self.assertTrue(torch.eq(torch.logsumexp(a, dim=0), OverflowUtils.logsumexp(a, dim=0)).all())

        a = torch.zeros(10)  # all zeros
        self.assertTrue(torch.eq(torch.logsumexp(a, dim=0), OverflowUtils.logsumexp(a, dim=0)).all())

        a = torch.ones(10)  # all ones
        self.assertTrue(torch.eq(torch.logsumexp(a, dim=0), OverflowUtils.logsumexp(a, dim=0)).all())


class TestOverFlowDecoder(unittest.TestCase):
    @staticmethod
    def _get_decoder(num_flow_blocks_dec=None, hidden_channels_dec=None, reset_weights=True):
        config = deepcopy(config_global)
        config.num_flow_blocks_dec = (
            num_flow_blocks_dec if num_flow_blocks_dec is not None else config.num_flow_blocks_dec
        )
        config.hidden_channels_dec = (
            hidden_channels_dec if hidden_channels_dec is not None else config.hidden_channels_dec
        )
        config.dropout_p_dec = 0.0  # turn off dropout to check invertibility
        decoder = Decoder(
            config.out_channels,
            config.hidden_channels_dec,
            config.kernel_size_dec,
            config.dilation_rate,
            config.num_flow_blocks_dec,
            config.num_block_layers,
            config.dropout_p_dec,
            config.num_splits,
            config.num_squeeze,
            config.sigmoid_scale,
            config.c_in_channels,
        ).to(device)
        if reset_weights:
            reset_all_weights(decoder)
        return decoder

    def test_decoder_forward_backward(self):
        for num_flow_blocks_dec in [8, None]:
            for hidden_channels_dec in [100, None]:
                decoder = self._get_decoder(num_flow_blocks_dec, hidden_channels_dec)
                _, _, mel_spec, mel_lengths = _create_inputs()
                z, z_len, _ = decoder(mel_spec.transpose(1, 2), mel_lengths)
                mel_spec_, mel_lengths_, _ = decoder(z, z_len, reverse=True)
                mask = sequence_mask(z_len).unsqueeze(1)
                print(mel_spec.shape, mask.shape)
                mel_spec = mel_spec[:, : z.shape[2], :].transpose(1, 2) * mask
                z = z * mask
                print(mel_spec[0], mel_spec_[0])
                self.assertTrue(
                    torch.isclose(mel_spec, mel_spec_, atol=1e-3).all(),
                    f"num_flow_blocks_dec={num_flow_blocks_dec}, hidden_channels_dec={hidden_channels_dec}",
                )


class TestNeuralHMM(unittest.TestCase):
    @staticmethod
    def _get_neural_hmm(deterministic_transition=None):
        config = deepcopy(config_global)
        neural_hmm = NeuralHMM(
            config.out_channels,
            config.ar_order,
            config.deterministic_transition if deterministic_transition is None else deterministic_transition,
            config.encoder_in_out_features,
            config.prenet_type,
            config.prenet_dim,
            config.prenet_n_layers,
            config.prenet_dropout,
            config.prenet_dropout_at_inference,
            config.memory_rnn_dim,
            config.outputnet_size,
            config.flat_start_params,
            config.std_floor,
        ).to(device)
        return neural_hmm

    @staticmethod
    def _get_embedded_input():
        input_dummy, input_lengths, mel_spec, mel_lengths = _create_inputs()
        input_dummy = torch.nn.Embedding(config_global.num_chars, config_global.encoder_in_out_features).to(device)(
            input_dummy
        )
        return input_dummy, input_lengths, mel_spec, mel_lengths

    def test_neural_hmm_forward(self):
        input_dummy, input_lengths, mel_spec, mel_lengths = self._get_embedded_input()
        neural_hmm = self._get_neural_hmm()
        log_prob, log_alpha_scaled, transition_matrix, means = neural_hmm(
            input_dummy, input_lengths, mel_spec.transpose(1, 2), mel_lengths
        )
        self.assertEqual(log_prob.shape, (input_dummy.shape[0],))
        self.assertEqual(log_alpha_scaled.shape, transition_matrix.shape)

    def test_mask_lengths(self):
        input_dummy, input_lengths, mel_spec, mel_lengths = self._get_embedded_input()
        neural_hmm = self._get_neural_hmm()
        log_prob, log_alpha_scaled, transition_matrix, means = neural_hmm(
            input_dummy, input_lengths, mel_spec.transpose(1, 2), mel_lengths
        )
        log_c = torch.randn(mel_spec.shape[0], mel_spec.shape[1], device=device)
        log_c, log_alpha_scaled = neural_hmm._mask_lengths(  # pylint: disable=protected-access
            mel_lengths, log_c, log_alpha_scaled
        )
        assertions = []
        for i in range(mel_spec.shape[0]):
            assertions.append(log_c[i, mel_lengths[i] :].sum() == 0.0)
        self.assertTrue(all(assertions), "Incorrect masking")
        assertions = []
        for i in range(mel_spec.shape[0]):
            assertions.append(log_alpha_scaled[i, mel_lengths[i] :, : input_lengths[i]].sum() == 0.0)
        self.assertTrue(all(assertions), "Incorrect masking")

    def test_process_ar_timestep(self):
        model = self._get_neural_hmm()
        input_dummy, input_lengths, mel_spec, mel_lengths = self._get_embedded_input()

        h_post_prenet, c_post_prenet = model._init_lstm_states(  # pylint: disable=protected-access
            input_dummy.shape[0], config_global.memory_rnn_dim, mel_spec
        )
        h_post_prenet, c_post_prenet = model._process_ar_timestep(  # pylint: disable=protected-access
            1,
            mel_spec,
            h_post_prenet,
            c_post_prenet,
        )

        assert h_post_prenet.shape == (input_dummy.shape[0], config_global.memory_rnn_dim)
        assert c_post_prenet.shape == (input_dummy.shape[0], config_global.memory_rnn_dim)
