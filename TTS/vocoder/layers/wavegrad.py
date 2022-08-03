import torch
import torch.nn.functional as F
from torch import nn
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
            self.init_pe_matrix(x.shape[1], x.shape[2], x)
        return x + noise_level[..., None, None] + self.pe[:, : x.size(2)].repeat(x.shape[0], 1, 1) / self.C

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
        self.input_conv = nn.Conv1d(input_size, input_size, 3, padding=1)
        self.output_conv = nn.Conv1d(input_size, output_size * 2, 3, padding=1)

        nn.init.xavier_uniform_(self.input_conv.weight)
        nn.init.xavier_uniform_(self.output_conv.weight)
        nn.init.zeros_(self.input_conv.bias)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x, noise_scale):
        o = self.input_conv(x)
        o = F.leaky_relu(o, 0.2)
        o = self.encoding(o, noise_scale)
        shift, scale = torch.chunk(self.output_conv(o), 2, dim=1)
        return shift, scale

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.input_conv)
        nn.utils.remove_weight_norm(self.output_conv)

    def apply_weight_norm(self):
        self.input_conv = weight_norm(self.input_conv)
        self.output_conv = weight_norm(self.output_conv)


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
        self.res_block = Conv1d(input_size, hidden_size, 1)
        self.main_block = nn.ModuleList(
            [
                Conv1d(input_size, hidden_size, 3, dilation=dilation[0], padding=dilation[0]),
                Conv1d(hidden_size, hidden_size, 3, dilation=dilation[1], padding=dilation[1]),
            ]
        )
        self.out_block = nn.ModuleList(
            [
                Conv1d(hidden_size, hidden_size, 3, dilation=dilation[2], padding=dilation[2]),
                Conv1d(hidden_size, hidden_size, 3, dilation=dilation[3], padding=dilation[3]),
            ]
        )

    def forward(self, x, shift, scale):
        x_inter = F.interpolate(x, size=x.shape[-1] * self.factor)
        res = self.res_block(x_inter)
        o = F.leaky_relu(x_inter, 0.2)
        o = F.interpolate(o, size=x.shape[-1] * self.factor)
        o = self.main_block[0](o)
        o = shif_and_scale(o, scale, shift)
        o = F.leaky_relu(o, 0.2)
        o = self.main_block[1](o)
        res2 = res + o
        o = shif_and_scale(res2, scale, shift)
        o = F.leaky_relu(o, 0.2)
        o = self.out_block[0](o)
        o = shif_and_scale(o, scale, shift)
        o = F.leaky_relu(o, 0.2)
        o = self.out_block[1](o)
        o = o + res2
        return o

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.res_block)
        for _, layer in enumerate(self.main_block):
            if len(layer.state_dict()) != 0:
                nn.utils.remove_weight_norm(layer)
        for _, layer in enumerate(self.out_block):
            if len(layer.state_dict()) != 0:
                nn.utils.remove_weight_norm(layer)

    def apply_weight_norm(self):
        self.res_block = weight_norm(self.res_block)
        for idx, layer in enumerate(self.main_block):
            if len(layer.state_dict()) != 0:
                self.main_block[idx] = weight_norm(layer)
        for idx, layer in enumerate(self.out_block):
            if len(layer.state_dict()) != 0:
                self.out_block[idx] = weight_norm(layer)


class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor):
        super().__init__()
        self.factor = factor
        self.res_block = Conv1d(input_size, hidden_size, 1)
        self.main_block = nn.ModuleList(
            [
                Conv1d(input_size, hidden_size, 3, dilation=1, padding=1),
                Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2),
                Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4),
            ]
        )

    def forward(self, x):
        size = x.shape[-1] // self.factor
        res = self.res_block(x)
        res = F.interpolate(res, size=size)
        o = F.interpolate(x, size=size)
        for layer in self.main_block:
            o = F.leaky_relu(o, 0.2)
            o = layer(o)
        return o + res

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.res_block)
        for _, layer in enumerate(self.main_block):
            if len(layer.state_dict()) != 0:
                nn.utils.remove_weight_norm(layer)

    def apply_weight_norm(self):
        self.res_block = weight_norm(self.res_block)
        for idx, layer in enumerate(self.main_block):
            if len(layer.state_dict()) != 0:
                self.main_block[idx] = weight_norm(layer)
