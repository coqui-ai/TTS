import torch
from torch import nn
from torch.nn.modules.conv import Conv1d

from TTS.vocoder.models.hifigan_discriminator import DiscriminatorP, MultiPeriodDiscriminator


class DiscriminatorS(torch.nn.Module):
    """HiFiGAN Scale Discriminator. Channel sizes are different from the original HiFiGAN.

    Args:
        use_spectral_norm (bool): if `True` swith to spectral norm instead of weight norm.
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        """
        Args:
            x (Tensor): input waveform.

        Returns:
            Tensor: discriminator scores.
            List[Tensor]: list of features from the convolutiona layers.
        """
        feat = []
        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, 0.1)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feat


class VitsDiscriminator(nn.Module):
    """VITS discriminator wrapping one Scale Discriminator and a stack of Period Discriminator.

    ::
        waveform -> ScaleDiscriminator() -> scores_sd, feats_sd --> append() -> scores, feats
               |--> MultiPeriodDiscriminator() -> scores_mpd, feats_mpd ^

    Args:
        use_spectral_norm (bool): if `True` swith to spectral norm instead of weight norm.
    """

    def __init__(self, periods=(2, 3, 5, 7, 11), use_spectral_norm=False, use_latent_disc=False, hidden_channels=None):
        super().__init__()
        self.nets = nn.ModuleList()
        self.nets.append(DiscriminatorS(use_spectral_norm=use_spectral_norm))
        self.nets.extend([DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods])
        self.disc_latent = None
        if use_latent_disc:
            self.disc_latent = LatentDiscriminator(use_spectral_norm=use_spectral_norm, hidden_channels=hidden_channels)

    def forward(self, x, x_hat=None, m_p=None, z_p=None):
        """
        Args:
            x (Tensor): ground truth waveform.
            x_hat (Tensor): predicted waveform.

        Returns:
            List[Tensor]: discriminator scores.
            List[List[Tensor]]: list of list of features from each layers of each discriminator.
        """
        x_scores = []
        x_hat_scores = [] if x_hat is not None else None
        x_feats = []
        x_hat_feats = [] if x_hat is not None else None
        for net in self.nets:
            x_score, x_feat = net(x)
            x_scores.append(x_score)
            x_feats.append(x_feat)
            if x_hat is not None:
                x_hat_score, x_hat_feat = net(x_hat)
                x_hat_scores.append(x_hat_score)
                x_hat_feats.append(x_hat_feat)

        # variables latent disc
        mp_scores, zp_scores, mp_feats, zp_feats = None, None, None, None
        if self.disc_latent is not None:
            if m_p is not None:
                mp_scores, mp_feats = self.disc_latent(m_p.unsqueeze(-1))
            if z_p is not None:
                zp_scores, zp_feats = self.disc_latent(z_p.unsqueeze(-1))

        return x_scores, x_feats, x_hat_scores, x_hat_feats, mp_scores, mp_feats, zp_scores, zp_feats


class LatentDiscriminator(nn.Module):
    """Discriminator with the same architecture as the Univnet SpecDiscriminator"""

    def __init__(self, use_spectral_norm=False, hidden_channels=None):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.discriminators = nn.ModuleList(
            [
                norm_f(
                    nn.Conv2d(1 if hidden_channels is None else hidden_channels, 32, kernel_size=(3, 9), padding=(1, 4))
                ),
                norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ]
        )

        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):
        fmap = []
        for _, d in enumerate(self.discriminators):
            y = d(y)
            y = torch.nn.functional.leaky_relu(y, 0.1)
            fmap.append(y)

        y = self.out(y)
        fmap.append(y)

        return torch.flatten(y, 1, -1), fmap
