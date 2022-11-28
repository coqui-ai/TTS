from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from TTS.tts.layers.tacotron.common_layers import Linear
from TTS.tts.layers.tacotron.tacotron2 import ConvBNBlock
from TTS.tts.utils.helpers import inverse_sigmod, inverse_softplus


class Encoder(nn.Module):
    r"""Neural HMM Encoder

    Same as Tacotron 2 encoder but increases the states per phone

    Args:
        in_out_channels (int): number of input and output channels.

    Shapes:
        - input: (B, C_in, T)
        - output: (B, C_in, T)
    """

    def __init__(self, state_per_phone, in_out_channels=512):
        super().__init__()

        self.state_per_phone = state_per_phone
        self.in_out_channels = in_out_channels

        self.convolutions = nn.ModuleList()
        for _ in range(3):
            self.convolutions.append(ConvBNBlock(in_out_channels, in_out_channels, 5, "relu"))
        self.lstm = nn.LSTM(
            in_out_channels,
            int(in_out_channels / 2) * state_per_phone,
            num_layers=1,
            batch_first=True,
            bias=True,
            bidirectional=True,
        )
        self.rnn_state = None

    def forward(self, x, input_lengths):
        b, _, T = x.shape
        o = x
        for layer in self.convolutions:
            o = layer(o)
        o = o.transpose(1, 2)
        o = nn.utils.rnn.pack_padded_sequence(o, input_lengths.cpu(), batch_first=True)
        self.lstm.flatten_parameters()
        o, _ = self.lstm(o)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        o = o.reshape(b, T * self.state_per_phone, self.in_out_channels)
        T = input_lengths * self.state_per_phone
        return o, T


class ParameterModel(nn.Module):
    r"""Main neural network of the outputnet

    Note: Do not put dropout layers here, the model will not converge.

    Args:
            parameternetwork (List[int]): the architecture of the parameter model
            input_size (int): size of input for the first layer
            output_size (int): size of output i.e size of the feature dim
            frame_channels (int): feature dim to set the flat start bias
            init_transition_probability (float): flat start transition probability
            init_mean (float): flat start mean
            init_std (float): flat start std
    """

    def __init__(
        self,
        parameternetwork: List[int],
        input_size: int,
        output_size: int,
        flat_start_params: dict,
        frame_channels: int,
    ):
        super().__init__()
        self.flat_start_params = flat_start_params

        self.layers = nn.ModuleList(
            [Linear(inp, out) for inp, out in zip([input_size] + parameternetwork[:-1], parameternetwork)]
        )
        last_layer = self._flat_start_output_layer(parameternetwork[-1], output_size, frame_channels)
        self.layers.append(last_layer)

    def _flat_start_output_layer(self, input_size, output_size, frame_channels):
        last_layer = nn.Linear(input_size, output_size)
        last_layer.weight.data.zero_()
        last_layer.bias.data[0:frame_channels] = self.flat_start_params["mean"]
        last_layer.bias.data[frame_channels : 2 * frame_channels] = inverse_softplus(self.flat_start_params["std"])
        last_layer.bias.data[2 * frame_channels :] = inverse_sigmod(self.flat_start_params["transition_p"])
        return last_layer

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x


class Outputnet(nn.Module):
    r"""
    This network takes current state and previous observed values as input
    and returns its parameters, mean, standard deviation and probability
    of transition to the next state
    """

    def __init__(
        self,
        encoder_dim: int,
        memory_rnn_dim: int,
        frame_channels: int,
        parameternetwork: List[int],
        flat_start_params: dict,
        std_floor: float = 1e-2,
    ):
        super().__init__()

        self.frame_channels = frame_channels
        self.flat_start_params = flat_start_params
        self.std_floor = std_floor

        input_size = memory_rnn_dim + encoder_dim
        output_size = 2 * frame_channels + 1

        self._validate_parameters()

        self.parametermodel = ParameterModel(
            parameternetwork=parameternetwork,
            input_size=input_size,
            output_size=output_size,
            flat_start_params=flat_start_params,
            frame_channels=frame_channels,
        )

    def _validate_parameters(self):
        """Validate the hyperparameters.

        Raises:
            AssertionError: when the parameters network is not defined
            AssertionError: transition probability is not between 0 and 1
        """
        assert (
            self.parameternetwork >= 1
        ), f"Parameter Network must have atleast one layer check the config file for parameter network. Provided: {self.parameternetwork}"
        assert (
            0 < self.flat_start_params["transition_p"] < 1
        ), f"Transition probability must be between 0 and 1. Provided: {self.flat_start_params['transition_p']}"

    def forward(self, ar_mels, inputs):
        r"""Inputs observation and returns the means, stds and transition probability for the current state

        Args:
            ar_mel_inputs (torch.FloatTensor): shape (batch, prenet_dim)
            states (torch.FloatTensor):  (hidden_states, hidden_state_dim)

        Returns:
            means: means for the emission observation for each feature
                - shape: (B, hidden_states, feature_size)
            stds: standard deviations for the emission observation for each feature
                - shape: (batch, hidden_states, feature_size)
            transition_vectors: transition vector for the current hidden state
                - shape: (batch, hidden_states)
        """
        batch_size, prenet_dim = ar_mels.shape[0], ar_mels.shape[1]
        N = inputs.shape[1]

        ar_mels = ar_mels.unsqueeze(1).expand(batch_size, N, prenet_dim)
        ar_mels = torch.cat((ar_mels, inputs), dim=2)
        ar_mels = self.parametermodel(ar_mels)

        mean, std, transition_vector = (
            ar_mels[:, :, 0 : self.frame_channels],
            ar_mels[:, :, self.frame_channels : 2 * self.frame_channels],
            ar_mels[:, :, 2 * self.frame_channels :].squeeze(2),
        )
        std = F.softplus(std)
        std = self._floor_std(std)
        return mean, std, transition_vector

    def _floor_std(self, std):
        r"""
        It clamps the standard deviation to not to go below some level
        This removes the problem when the model tries to cheat for higher likelihoods by converting
        one of the gaussians to a point mass.

        Args:
            std (float Tensor): tensor containing the standard deviation to be
        """
        original_tensor = std.clone().detach()
        std = torch.clamp(std, min=self.std_floor)
        if torch.any(original_tensor != std):
            print(
                "[*] Standard deviation was floored! The model is preventing overfitting, nothing serious to worry about"
            )
        return std
