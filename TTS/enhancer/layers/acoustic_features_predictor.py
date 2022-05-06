import torch
from torch import nn
from torch.nn import functional as F


class ConvNormLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activattion="relu"):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activattion = activattion

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activattion == "relu":
            x = F.relu(x)
        elif self.activattion == "tanh":
            x = torch.tanh(x)
        elif self.activattion == "none":
            pass
        else:
            raise ValueError("Activation function not supported")
        return x


class AcousticFeaturesPredictor(nn.Module):
    def __init__(self, hidden=512, input_dim=80, output_dim=18):
        super().__init__()
        self.hidden = hidden
        self.prenet = nn.Sequential(
            ConvNormLayer(input_dim, hidden, 5),
            ConvNormLayer(hidden, hidden, 5),
            ConvNormLayer(hidden, hidden, 5),
        )
        self.rnn = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=3,
            bidirectional=True,
        )
        self.h0 = nn.Parameter(torch.rand(6, 1, hidden))
        self.c0 = nn.Parameter(torch.rand(6, 1, hidden))
        self.linear = nn.Conv1d(hidden, output_dim, 1)
        self.postnet = nn.Sequential(
            ConvNormLayer(output_dim, hidden, 5, activattion="tanh"),
            ConvNormLayer(hidden, hidden, 5, activattion="tanh"),
            ConvNormLayer(hidden, hidden, 5, activattion="tanh"),
            ConvNormLayer(hidden, hidden, 5, activattion="tanh"),
            ConvNormLayer(hidden, output_dim, 1, activattion="none"),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.prenet(x)
        x = x.permute(2, 0, 1)
        h0 = self.h0.expand(6, batch_size, self.hidden).contiguous()
        c0 = self.c0.expand(6, batch_size, self.hidden).contiguous()
        x, (h, c) = self.rnn(x, (h0, c0))
        x = x[:, :, self.hidden :] + x[:, :, : self.hidden]
        x = x.permute(1, 2, 0)
        x = self.linear(x)
        x = self.postnet(x)
        return x
