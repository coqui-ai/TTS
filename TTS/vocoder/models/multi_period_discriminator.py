from torch import nn
import torch.nn.functional as F
from TTS.vocoder.models.melgan_multiscale_discriminator import MelganMultiscaleDiscriminator


class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super(PeriodDiscriminator, self).__init__()
        layer = []
        self.period = period
        inp = 1
        for l in range(4):
            out = int(2**(5 + l + 1))
            layer += [
                nn.utils.weight_norm(
                    nn.Conv2d(inp, out, kernel_size=(5, 1), stride=(3, 1))),
                nn.LeakyReLU(0.2)
            ]
            inp = out
        self.layer = nn.Sequential(*layer)
        self.output = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(out, 1024, kernel_size=(5, 1))),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1))))

    def forward(self, x):
        batch_size = x.shape[0]
        pad = self.period - (x.shape[-1] % self.period)
        x = F.pad(x, (0, pad))
        y = x.view(batch_size, -1, self.period).contiguous()
        y = y.unsqueeze(1)
        out1 = self.layer(y)
        return self.output(out1)


class HifiDiscriminator(nn.Module):
    def __init__(self,
                 periods=[2, 3, 5, 7, 11],
                 in_channels=1,
                 out_channels=1,
                 num_scales=3,
                 kernel_sizes=(5, 3),
                 base_channels=64,
                 max_channels=1024,
                 downsample_factors=(2, 2, 4, 4),
                 pooling_kernel_size=4,
                 pooling_stride=2,
                 pooling_padding=1):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(periods[0]),
            PeriodDiscriminator(periods[1]),
            PeriodDiscriminator(periods[2]),
            PeriodDiscriminator(periods[3]),
            PeriodDiscriminator(periods[4])
        ])

        self.msd = MelganMultiscaleDiscriminator(
            in_channels=in_channels,
            out_channels=out_channels,
            num_scales=num_scales,
            kernel_sizes=kernel_sizes,
            base_channels=base_channels,
            max_channels=max_channels,
            downsample_factors=downsample_factors,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_padding=pooling_padding,
            groups_denominator=32,
            max_groups=16)

    def forward(self, x):
        scores, feats = self.msd(x)
        for key, disc in enumerate(self.discriminators):
            score = disc(x)
            scores.append(score)
        return scores, feats
