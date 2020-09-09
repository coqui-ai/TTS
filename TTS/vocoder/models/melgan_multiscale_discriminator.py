from torch import nn

from TTS.vocoder.models.melgan_discriminator import MelganDiscriminator


class MelganMultiscaleDiscriminator(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 num_scales=3,
                 kernel_sizes=(5, 3),
                 base_channels=16,
                 max_channels=1024,
                 downsample_factors=(4, 4, 4),
                 pooling_kernel_size=4,
                 pooling_stride=2,
                 pooling_padding=1):
        super(MelganMultiscaleDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList([
            MelganDiscriminator(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_sizes=kernel_sizes,
                                base_channels=base_channels,
                                max_channels=max_channels,
                                downsample_factors=downsample_factors)
            for _ in range(num_scales)
        ])

        self.pooling = nn.AvgPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride, padding=pooling_padding, count_include_pad=False)


    def forward(self, x):
        scores = list()
        feats = list()
        for disc in self.discriminators:
            score, feat = disc(x)
            scores.append(score)
            feats.append(feat)
            x = self.pooling(x)
        return scores, feats
