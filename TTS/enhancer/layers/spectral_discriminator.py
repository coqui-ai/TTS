import torch
from torch import nn
from torch.nn import functional as F

from TTS.utils.audio import TorchSTFT


class ConvLayerSpecDisc(nn.Module):
    def __init__(self, kernel_size, stride, first_layer):
        super().__init__()
        get_padding = lambda k, d: int((k * d - d) / 2)
        self.conv = nn.Conv2d(
            1 if first_layer else 32,
            32,
            kernel_size,
            stride,
            padding=(get_padding(kernel_size[0], 1), get_padding(kernel_size[1], 1)),
        )
        self.bn = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x)  # Change for GLU
        x = self.bn(x)
        return x


class SpectralDiscriminator(nn.Module):
    def __init__(self, sample_rate=48000, n_fft=1024, hop_length=256, n_mels=128):
        super().__init__()
        self.mel_spec = TorchSTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            sample_rate=sample_rate,
            n_mels=n_mels,
            use_mel=True,
            device="cuda",
            do_amp_to_db=True,
        )
        self.kernel_sizes = [(7, 7), (5, 5), (5, 5), (5, 5)]
        self.layers = nn.ModuleList(
            [ConvLayerSpecDisc(self.kernel_sizes[i], (1, 2), i == 0) for i in range(len(self.kernel_sizes))]
        )
        self.last_conv = nn.Conv2d(32, 1, (15, 5), (15, 1))

    def forward(self, x):
        feats = []
        x = self.mel_spec(x).transpose(1, 2).unsqueeze(1)
        for disc in self.layers:
            x = disc(x)
            feats.append(x)
        x = self.last_conv(x)
        feats.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feats
