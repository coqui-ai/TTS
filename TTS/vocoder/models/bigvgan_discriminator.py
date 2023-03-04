import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d
from torch.nn.utils import spectral_norm, weight_norm

from TTS.vocoder.utils.generic_utils import get_padding

LRELU_SLOPE = 0.1


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, discriminator_channel_mult=1, use_spectral_norm=False):
        super().__init__()
        self.period = period
        self.d_mult = discriminator_channel_mult
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv2d(1, int(32 * self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(
                    Conv2d(
                        int(32 * self.d_mult),
                        int(128 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        int(128 * self.d_mult),
                        int(512 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        int(512 * self.d_mult),
                        int(1024 * self.d_mult),
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(Conv2d(int(1024 * self.d_mult), int(1024 * self.d_mult), (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(int(1024 * self.d_mult), 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, mpd_reshapes, discriminator_channel_mult=1, use_spectral_norm=False):
        super().__init__()
        self.mpd_reshapes = mpd_reshapes
        print("mpd_reshapes: {}".format(self.mpd_reshapes))
        discriminators = [
            DiscriminatorP(
                rs, discriminator_channel_mult=discriminator_channel_mult, use_spectral_norm=use_spectral_norm
            )
            for rs in self.mpd_reshapes
        ]

        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for _, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, fmap_rs, y_d_gs, fmap_gs


class DiscriminatorR(nn.Module):
    def __init__(
        self,
        discriminator_channel_mult=1,
        use_spectral_norm=False,
        mrd_use_spectral_norm=False,
        mrd_channel_mult=1,
        resolution=None,
    ):
        super().__init__()

        self.resolution = resolution
        assert len(self.resolution) == 3, "MRD layer requires list with len=3, got {}".format(self.resolution)
        self.lrelu_slope = LRELU_SLOPE

        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        if mrd_use_spectral_norm:
            print("INFO: overriding MRD use_spectral_norm as {}".format(mrd_use_spectral_norm))
            norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.d_mult = discriminator_channel_mult
        if mrd_channel_mult != discriminator_channel_mult:
            print("INFO: overriding mrd channel multiplier as {}".format(mrd_channel_mult))
            self.d_mult = mrd_channel_mult

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, int(32 * self.d_mult), (3, 9), padding=(1, 4))),
                norm_f(nn.Conv2d(int(32 * self.d_mult), int(32 * self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(int(32 * self.d_mult), int(32 * self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(int(32 * self.d_mult), int(32 * self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(int(32 * self.d_mult), int(32 * self.d_mult), (3, 3), padding=(1, 1))),
            ]
        )
        self.conv_post = norm_f(nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode="reflect")
        x = x.squeeze(1)
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False, return_complex=True)
        x = torch.view_as_real(x)  # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag


class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        discriminator_channel_mult=1,
        use_spectral_norm=False,
        mrd_use_spectral_norm=False,
        mrd_channel_mult=1,
        resolutions=None,
    ):
        super().__init__()
        self.resolutions = resolutions
        assert (
            len(self.resolutions) == 3
        ), "MRD requires list of list with len=3, each element having a list with len=3. got {}".format(
            self.resolutions
        )
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorR(
                    discriminator_channel_mult, use_spectral_norm, mrd_use_spectral_norm, mrd_channel_mult, resolution
                )
                for resolution in self.resolutions
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for _, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, fmap_rs, y_d_gs, fmap_gs
