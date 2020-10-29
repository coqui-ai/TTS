import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.orthogonal_(self.weight)
        nn.init.zeros_(self.bias)


class PositionalEncoding(nn.Module):
    """Positional encoding with noise level conditioning"""
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
        o1 = F.interpolate(x, size=x.shape[-1] * self.factor)
        o1 = self.block1(o1)

        o2 = F.leaky_relu(x, 0.2)
        o2 = F.interpolate(o2, size=x.shape[-1] * self.factor)
        o2 = self.block2[0](o2)
        o2 = shif_and_scale(o2, scale, shift)
        o2 = F.leaky_relu(o2, 0.2)
        o2 = self.block2[1](o2)

        x = o1 + o2

        o3 = shif_and_scale(x, scale, shift)
        o3 = F.leaky_relu(o3, 0.2)
        o3 = self.block3[0](o3)

        o3 = shif_and_scale(o3, scale, shift)
        o3 = F.leaky_relu(o3, 0.2)
        o3 = self.block3[1](o3)

        o = x + o3
        return o


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


