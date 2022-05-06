import torch
from torch import nn
from torch.nn import functional as F

class ConvNormLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activattion='relu'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activattion = activattion

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activattion == 'relu':
            x = F.relu(x)
        elif self.activattion == 'tanh':
            x = F.tanh(x)
        elif self.activattion == 'none':
            pass
        else:
            raise ValueError('Activation function not supported')
        return x

class AcousticFeaturesPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.prenet = nn.Sequential([
            ConvNormLayer(80, 512, 5),
            ConvNormLayer(512, 512, 5),
            ConvNormLayer(512, 512, 5),
        ])
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=3,
            bidirectional=True,
        )
        self.h0 = nn.Parameter(torch.rand(6, 1, 512))
        self.c0 = nn.Parameter(torch.rand(6, 1, 512))
        self.linear = nn.Conv1d(512, 18, 1)
        self.postnet = nn.Sequential([
            ConvNormLayer(18, 512, 5, activattion='tanh'),
            ConvNormLayer(512, 512, 5, activattion='tanh'),
            ConvNormLayer(512, 512, 5, activattion='tanh'),
            ConvNormLayer(512, 512, 5, activattion='tanh'),
            ConvNormLayer(512, 18, 1, activattion='none'),
        ])

    def forward(self, x):
        batch_size = x.size(0)
        x = self.prenet(x)
        x = x.transpose(0, 1)
        h0 = self.h0.expand(6, batch_size, 512).contiguous()
        c0 = self.c0.expand(6, batch_size, 512).contiguous()
        x, (h, c) = self.rnn(x, (h0, c0))
        x = x.transpose(0, 1)
        x = self.linear(x)
        x = self.postnet(x)
        return x