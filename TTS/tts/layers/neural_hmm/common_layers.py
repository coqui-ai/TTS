import torch.nn as nn


class ConvBNBlock(nn.Module):
    r"""Convolutions with Batch Normalization and non-linear activation.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        kernel_size (int): convolution kernel size.
        activation (str): 'relu', 'tanh', None (linear).

    Shapes:
        - input: (B, C_in, T)
        - output: (B, C_out, T)
    """

    def __init__(self, in_channels, out_channels, kernel_size, activation=None):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        self.convolution1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.batch_normalization = nn.BatchNorm1d(out_channels, momentum=0.1, eps=1e-5)
        self.dropout = nn.Dropout(p=0.5)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        o = self.convolution1d(x)
        o = self.batch_normalization(o)
        o = self.activation(o)
        o = self.dropout(o)
        return o


class Encoder(nn.Module):
    r"""Neural HMM Encoder

        - Three 1-d convolution banks
        - Bidirectional LSTM

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
            in_out_channels, int(in_out_channels / 2), num_layers=1, batch_first=True, bias=True, bidirectional=True
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
