import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from math import log as ln


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.orthogonal_(self.weight)
        nn.init.zeros_(self.bias)


# class PositionalEncoding(nn.Module):
#     def __init__(self, n_channels):
#         super().__init__()
#         self.n_channels = n_channels
#         self.length = n_channels // 2
#         assert n_channels % 2 == 0

#     def forward(self, x, noise_level):
#         """
#         Shapes:
#             x: B x C x T
#             noise_level: B
#         """
#         return (x + self.encoding(noise_level)[:, :, None])

#     def encoding(self, noise_level):
#         step = torch.arange(
#             self.length, dtype=noise_level.dtype, device=noise_level.device) / self.length
#         encoding = noise_level.unsqueeze(1) * torch.exp(
#             -ln(1e4) * step.unsqueeze(0))
#         encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
#         return encoding


class PositionalEncoding(nn.Module):
    def __init__(self, n_channels, max_len=10000):
        super().__init__()
        self.n_channels = n_channels
        self.max_len = max_len
        self.C = 5000
        self.pe = torch.zeros(0, 0)

    def forward(self, x, noise_level):
        if x.shape[2] > self.pe.shape[1]:
            self.init_pe_matrix(x.shape[1] ,x.shape[2], x)
        return x + noise_level[..., None, None] + self.pe[:, :x.size(2)].repeat(x.shape[0], 1, 1) / self.C

    def init_pe_matrix(self, n_channels, max_len, x):
        pe = torch.zeros(max_len, n_channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, n_channels, 2).float() / n_channels)

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        self.pe = pe.transpose(0, 1).to(x)


class FiLM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.encoding = PositionalEncoding(input_size)
        self.input_conv = weight_norm(nn.Conv1d(input_size, input_size, 3, padding=1))
        self.output_conv = weight_norm(nn.Conv1d(input_size, output_size * 2, 3, padding=1))
        self.ini_parameters()

    def ini_parameters(self):
        nn.init.xavier_uniform_(self.input_conv.weight)
        nn.init.xavier_uniform_(self.output_conv.weight)
        nn.init.zeros_(self.input_conv.bias)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x, noise_scale):
        x = self.input_conv(x)
        x = F.leaky_relu(x, 0.2)
        x = self.encoding(x, noise_scale)
        shift, scale = torch.chunk(self.output_conv(x), 2, dim=1)
        return shift, scale


@torch.jit.script
def shif_and_scale(x, scale, shift):
    o = shift + scale * x
    return o


class UBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor, dilation):
        super().__init__()
        assert isinstance(dilation, (list, tuple))
        assert len(dilation) == 4

        self.factor = factor
        self.block1 = weight_norm(Conv1d(input_size, hidden_size, 1))
        self.block2 = nn.ModuleList([
            weight_norm(Conv1d(input_size,
                   hidden_size,
                   3,
                   dilation=dilation[0],
                   padding=dilation[0])),
            weight_norm(Conv1d(hidden_size,
                   hidden_size,
                   3,
                   dilation=dilation[1],
                   padding=dilation[1]))
        ])
        self.block3 = nn.ModuleList([
            weight_norm(Conv1d(hidden_size,
                   hidden_size,
                   3,
                   dilation=dilation[2],
                   padding=dilation[2])),
            weight_norm(Conv1d(hidden_size,
                   hidden_size,
                   3,
                   dilation=dilation[3],
                   padding=dilation[3]))
        ])

    def forward(self, x, shift, scale):
        block1 = F.interpolate(x, size=x.shape[-1] * self.factor)
        block1 = self.block1(block1)

        block2 = F.leaky_relu(x, 0.2)
        block2 = F.interpolate(block2, size=x.shape[-1] * self.factor)
        block2 = self.block2[0](block2)
        # block2 = film_shift + film_scale * block2
        block2 = shif_and_scale(block2, scale, shift)
        block2 = F.leaky_relu(block2, 0.2)
        block2 = self.block2[1](block2)

        x = block1 + block2

        # block3 = film_shift + film_scale * x
        block3 = shif_and_scale(x, scale, shift)
        block3 = F.leaky_relu(block3, 0.2)
        block3 = self.block3[0](block3)
        # block3 = film_shift + film_scale * block3
        block3 = shif_and_scale(block3, scale, shift)
        block3 = F.leaky_relu(block3, 0.2)
        block3 = self.block3[1](block3)

        x = x + block3
        return x


class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor):
        super().__init__()
        self.factor = factor
        self.residual_dense = weight_norm(Conv1d(input_size, hidden_size, 1))
        self.conv = nn.ModuleList([
            weight_norm(Conv1d(input_size, hidden_size, 3, dilation=1, padding=1)),
            weight_norm(Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2)),
            weight_norm(Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4)),
        ])

    def forward(self, x):
        size = x.shape[-1] // self.factor

        residual = self.residual_dense(x)
        residual = F.interpolate(residual, size=size)

        x = F.interpolate(x, size=size)
        for layer in self.conv:
            x = F.leaky_relu(x, 0.2)
            x = layer(x)

        return x + residual


