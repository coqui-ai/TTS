from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from TTS.tts.layers.tacotron.common_layers import Linear
from TTS.tts.layers.tacotron.tacotron2 import ConvBNBlock


class Encoder(nn.Module):
    r"""Neural HMM Encoder

    Same as Tacotron 2 encoder but increases the input length by states per phone

    Args:
        num_chars (int): Number of characters in the input.
        state_per_phone (int): Number of states per phone.
        in_out_channels (int): number of input and output channels.
        n_convolutions (int): number of convolutional layers.
    """

    def __init__(self, num_chars, state_per_phone, in_out_channels=512, n_convolutions=3):
        super().__init__()

        self.state_per_phone = state_per_phone
        self.in_out_channels = in_out_channels

        self.emb = nn.Embedding(num_chars, in_out_channels)
        self.convolutions = nn.ModuleList()
        for _ in range(n_convolutions):
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

    def forward(self, x: torch.FloatTensor, x_len: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Forward pass to the encoder.

        Args:
            x (torch.FloatTensor): input text indices.
                - shape: :math:`(b, T_{in})`
            x_len (torch.LongTensor): input text lengths.
                - shape: :math:`(b,)`

        Returns:
            Tuple[torch.FloatTensor, torch.LongTensor]: encoder outputs and output lengths.
                -shape: :math:`((b, T_{in} * states_per_phone, in_out_channels), (b,))`
        """
        b, T = x.shape
        o = self.emb(x).transpose(1, 2)
        for layer in self.convolutions:
            o = layer(o)
        o = o.transpose(1, 2)
        o = nn.utils.rnn.pack_padded_sequence(o, x_len.cpu(), batch_first=True)
        self.lstm.flatten_parameters()
        o, _ = self.lstm(o)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        o = o.reshape(b, T * self.state_per_phone, self.in_out_channels)
        x_len = x_len * self.state_per_phone
        return o, x_len

    def inference(self, x, x_len):
        """Inference to the encoder.

        Args:
            x (torch.FloatTensor): input text indices.
                - shape: :math:`(b, T_{in})`
            x_len (torch.LongTensor): input text lengths.
                - shape: :math:`(b,)`

        Returns:
            Tuple[torch.FloatTensor, torch.LongTensor]: encoder outputs and output lengths.
                -shape: :math:`((b, T_{in} * states_per_phone, in_out_channels), (b,))`
        """
        b, T = x.shape
        o = self.emb(x).transpose(1, 2)
        for layer in self.convolutions:
            o = layer(o)
        o = o.transpose(1, 2)
        # self.lstm.flatten_parameters()
        o, _ = self.lstm(o)
        o = o.reshape(b, T * self.state_per_phone, self.in_out_channels)
        x_len = x_len * self.state_per_phone
        return o, x_len


class ParameterModel(nn.Module):
    r"""Main neural network of the outputnet

    Note: Do not put dropout layers here, the model will not converge.

    Args:
            outputnet_size (List[int]): the architecture of the parameter model
            input_size (int): size of input for the first layer
            output_size (int): size of output i.e size of the feature dim
            frame_channels (int): feature dim to set the flat start bias
            flat_start_params (dict): flat start parameters to set the bias
    """

    def __init__(
        self,
        outputnet_size: List[int],
        input_size: int,
        output_size: int,
        frame_channels: int,
        flat_start_params: dict,
    ):
        super().__init__()
        self.frame_channels = frame_channels

        self.layers = nn.ModuleList(
            [Linear(inp, out) for inp, out in zip([input_size] + outputnet_size[:-1], outputnet_size)]
        )
        self.last_layer = nn.Linear(outputnet_size[-1], output_size)
        self.flat_start_output_layer(
            flat_start_params["mean"], flat_start_params["std"], flat_start_params["transition_p"]
        )

    def flat_start_output_layer(self, mean, std, transition_p):
        self.last_layer.weight.data.zero_()
        self.last_layer.bias.data[0 : self.frame_channels] = mean
        self.last_layer.bias.data[self.frame_channels : 2 * self.frame_channels] = OverflowUtils.inverse_softplus(std)
        self.last_layer.bias.data[2 * self.frame_channels :] = OverflowUtils.inverse_sigmod(transition_p)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.last_layer(x)
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
        outputnet_size: List[int],
        flat_start_params: dict,
        std_floor: float = 1e-2,
    ):
        super().__init__()

        self.frame_channels = frame_channels
        self.flat_start_params = flat_start_params
        self.std_floor = std_floor

        input_size = memory_rnn_dim + encoder_dim
        output_size = 2 * frame_channels + 1

        self.parametermodel = ParameterModel(
            outputnet_size=outputnet_size,
            input_size=input_size,
            output_size=output_size,
            flat_start_params=flat_start_params,
            frame_channels=frame_channels,
        )

    def forward(self, ar_mels, inputs):
        r"""Inputs observation and returns the means, stds and transition probability for the current state

        Args:
            ar_mel_inputs (torch.FloatTensor): shape (batch, prenet_dim)
            states (torch.FloatTensor):  (batch, hidden_states, hidden_state_dim)

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


class OverflowUtils:
    @staticmethod
    def get_data_parameters_for_flat_start(
        data_loader: torch.utils.data.DataLoader, out_channels: int, states_per_phone: int
    ):
        """Generates data parameters for flat starting the HMM.

        Args:
            data_loader (torch.utils.data.Dataloader): _description_
            out_channels (int): mel spectrogram channels
            states_per_phone (_type_): HMM states per phone
        """

        # State related information for transition_p
        total_state_len = 0
        total_mel_len = 0

        # Useful for data mean an std
        total_mel_sum = 0
        total_mel_sq_sum = 0

        for batch in tqdm(data_loader, leave=False):
            text_lengths = batch["token_id_lengths"]
            mels = batch["mel"]
            mel_lengths = batch["mel_lengths"]

            total_state_len += torch.sum(text_lengths)
            total_mel_len += torch.sum(mel_lengths)
            total_mel_sum += torch.sum(mels)
            total_mel_sq_sum += torch.sum(torch.pow(mels, 2))

        data_mean = total_mel_sum / (total_mel_len * out_channels)
        data_std = torch.sqrt((total_mel_sq_sum / (total_mel_len * out_channels)) - torch.pow(data_mean, 2))
        average_num_states = total_state_len / len(data_loader.dataset)
        average_mel_len = total_mel_len / len(data_loader.dataset)
        average_duration_each_state = average_mel_len / average_num_states
        init_transition_prob = 1 / average_duration_each_state

        return data_mean, data_std, (init_transition_prob * states_per_phone)

    @staticmethod
    @torch.no_grad()
    def update_flat_start_transition(model, transition_p):
        model.neural_hmm.output_net.parametermodel.flat_start_output_layer(0.0, 1.0, transition_p)

    @staticmethod
    def log_clamped(x, eps=1e-04):
        """
        Avoids the log(0) problem

        Args:
            x (torch.tensor): input tensor
            eps (float, optional): lower bound. Defaults to 1e-04.

        Returns:
            torch.tensor: :math:`log(x)`
        """
        clamped_x = torch.clamp(x, min=eps)
        return torch.log(clamped_x)

    @staticmethod
    def inverse_sigmod(x):
        r"""
        Inverse of the sigmoid function
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return OverflowUtils.log_clamped(x / (1.0 - x))

    @staticmethod
    def inverse_softplus(x):
        r"""
        Inverse of the softplus function
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return OverflowUtils.log_clamped(torch.exp(x) - 1.0)

    @staticmethod
    def logsumexp(x, dim):
        r"""
        Differentiable LogSumExp: Does not creates nan gradients
            when all the inputs are -inf yeilds 0 gradients.
        Args:
            x : torch.Tensor -  The input tensor
            dim: int - The dimension on which the log sum exp has to be applied
        """

        m, _ = x.max(dim=dim)
        mask = m == -float("inf")
        s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
        return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float("inf"))

    @staticmethod
    def double_pad(list_of_different_shape_tensors):
        r"""
        Pads the list of tensors in 2 dimensions
        """
        second_dim_lens = [len(a) for a in [i[0] for i in list_of_different_shape_tensors]]
        second_dim_max = max(second_dim_lens)
        padded_x = [F.pad(x, (0, second_dim_max - len(x[0]))) for x in list_of_different_shape_tensors]
        return nn.utils.rnn.pad_sequence(padded_x, batch_first=True)
