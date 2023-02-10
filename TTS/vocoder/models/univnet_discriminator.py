import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm, weight_norm

from TTS.utils.audio.torch_transforms import TorchSTFT
from TTS.vocoder.models.hifigan_discriminator import MultiPeriodDiscriminator

LRELU_SLOPE = 0.1


class SpecDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(self, fft_size=1024, hop_length=120, win_length=600, use_spectral_norm=False):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.stft = TorchSTFT(fft_size, hop_length, win_length)
        self.discriminators = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ]
        )

        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):
        fmap = []
        with torch.no_grad():
            y = y.squeeze(1)
            y = self.stft(y)
        y = y.unsqueeze(1)
        for _, d in enumerate(self.discriminators):
            y = d(y)
            y = F.leaky_relu(y, LRELU_SLOPE)
            fmap.append(y)

        y = self.out(y)
        fmap.append(y)

        return torch.flatten(y, 1, -1), fmap


class MultiResSpecDiscriminator(torch.nn.Module):
    def __init__(  # pylint: disable=dangerous-default-value
        self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window="hann_window"
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                SpecDiscriminator(fft_sizes[0], hop_sizes[0], win_lengths[0], window),
                SpecDiscriminator(fft_sizes[1], hop_sizes[1], win_lengths[1], window),
                SpecDiscriminator(fft_sizes[2], hop_sizes[2], win_lengths[2], window),
            ]
        )

    def forward(self, x):
        scores = []
        feats = []
        for d in self.discriminators:
            score, feat = d(x)
            scores.append(score)
            feats.append(feat)

        return scores, feats


class UnivnetDiscriminator(nn.Module):
    """Univnet discriminator wrapping MPD and MSD."""

    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiResSpecDiscriminator()

    def forward(self, x):
        """
        Args:
            x (Tensor): input waveform.

        Returns:
            List[Tensor]: discriminator scores.
            List[List[Tensor]]: list of list of features from each layers of each discriminator.
        """
        scores, feats = self.mpd(x)
        scores_, feats_ = self.msd(x)
        return scores + scores_, feats + feats_
