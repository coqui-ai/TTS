import math
import torch
from torch import nn
from torch.nn import functional as F


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        nn.init.zeros_(self.bias)


class NoiseLevelEncoding(nn.Module):
    """Noise level encoding applying same
    encoding vector to all time steps. It is
    different than the original implementation."""
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.length = n_channels // 2
        assert n_channels % 2 == 0

        enc = self.init_encoding(self.length)
        self.register_buffer('enc', enc)

    def forward(self, x, noise_level):
        """
        Shapes:
            x: B x C x T
            noise_level: B
        """
        return (x + self.encoding(noise_level)[:, :, None])

    @staticmethod
    def init_encoding(length):
        div_by = torch.arange(length) / length
        enc = torch.exp(-math.log(1e4) * div_by.unsqueeze(0))
        return enc

    def encoding(self, noise_level):
        encoding = noise_level.unsqueeze(1) * self.enc
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FiLM(nn.Module):
    """Feature-wise Linear Modulation. It combines information from
    both noisy waveform and input mel-spectrogram. The FiLM module
    produces both scale and bias vectors given inputs, which are
    used in a UBlock for feature-wise affine transformation."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoding = NoiseLevelEncoding(in_channels)
        self.conv_in = Conv1d(in_channels, in_channels, 3, padding=1)
        self.conv_out = Conv1d(in_channels, out_channels * 2, 3, padding=1)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.orthogonal_(self.conv_in.weight)
        nn.init.orthogonal_(self.conv_out.weight)

    def forward(self, x, noise_scale):
        x = self.conv_in(x)
        x = F.leaky_relu(x, 0.2)
        x = self.encoding(x, noise_scale)
        shift, scale = torch.chunk(self.conv_out(x), 2, dim=1)
        return shift, scale


@torch.jit.script
def shif_and_scale(x, scale, shift):
    o = shift + scale * x
    return o


class UBlock(nn.Module):
    def __init__(self, in_channels, hid_channels, upsample_factor, dilations):
        super().__init__()
        assert len(dilations) == 4

        self.upsample_factor = upsample_factor
        self.shortcut_conv = Conv1d(in_channels, hid_channels, 1)
        self.main_block1 = nn.ModuleList([
            Conv1d(in_channels,
                   hid_channels,
                   3,
                   dilation=dilations[0],
                   padding=dilations[0]),
            Conv1d(hid_channels,
                   hid_channels,
                   3,
                   dilation=dilations[1],
                   padding=dilations[1])
        ])
        self.main_block2 = nn.ModuleList([
            Conv1d(hid_channels,
                   hid_channels,
                   3,
                   dilation=dilations[2],
                   padding=dilations[2]),
            Conv1d(hid_channels,
                   hid_channels,
                   3,
                   dilation=dilations[3],
                   padding=dilations[3])
        ])

    def forward(self, x, shift, scale):
        upsample_size = x.shape[-1] * self.upsample_factor
        x = F.interpolate(x, size=upsample_size)
        res = self.shortcut_conv(x)

        o = F.leaky_relu(x, 0.2)
        o = self.main_block1[0](o)
        o = shif_and_scale(o, scale, shift)
        o = F.leaky_relu(o, 0.2)
        o = self.main_block1[1](o)

        o = o + res
        res = o

        o = shif_and_scale(o, scale, shift)
        o = F.leaky_relu(o, 0.2)
        o = self.main_block2[0](o)
        o = shif_and_scale(o, scale, shift)
        o = F.leaky_relu(o, 0.2)
        o = self.main_block2[1](o)

        o = o + res
        return o


class DBlock(nn.Module):
    def __init__(self, in_channels, hid_channels, downsample_factor):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.res_conv = Conv1d(in_channels, hid_channels, 1)
        self.main_convs = nn.ModuleList([
            Conv1d(in_channels, hid_channels, 3, dilation=1, padding=1),
            Conv1d(hid_channels, hid_channels, 3, dilation=2, padding=2),
            Conv1d(hid_channels, hid_channels, 3, dilation=4, padding=4),
        ])

    def forward(self, x):
        size = x.shape[-1] // self.downsample_factor

        res = self.res_conv(x)
        res = F.interpolate(res, size=size)

        o = F.interpolate(x, size=size)
        for layer in self.main_convs:
            o = F.leaky_relu(o, 0.2)
            o = layer(o)

        return o + res
