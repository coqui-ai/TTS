import numpy as np
from torch import nn


class GBlock(nn.Module):
    def __init__(self, in_channels, cond_channels, downsample_factor):
        super().__init__()

        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.downsample_factor = downsample_factor

        self.start = nn.Sequential(
            nn.AvgPool1d(downsample_factor, stride=downsample_factor),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels * 2, kernel_size=3, padding=1),
        )
        self.lc_conv1d = nn.Conv1d(cond_channels, in_channels * 2, kernel_size=1)
        self.end = nn.Sequential(
            nn.ReLU(), nn.Conv1d(in_channels * 2, in_channels * 2, kernel_size=3, dilation=2, padding=2)
        )
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * 2, kernel_size=1),
            nn.AvgPool1d(downsample_factor, stride=downsample_factor),
        )

    def forward(self, inputs, conditions):
        outputs = self.start(inputs) + self.lc_conv1d(conditions)
        outputs = self.end(outputs)
        residual_outputs = self.residual(inputs)
        outputs = outputs + residual_outputs

        return outputs


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_factor):
        super().__init__()

        self.in_channels = in_channels
        self.downsample_factor = downsample_factor
        self.out_channels = out_channels

        self.donwsample_layer = nn.AvgPool1d(downsample_factor, stride=downsample_factor)
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2),
        )
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, inputs):
        if self.downsample_factor > 1:
            outputs = self.layers(self.donwsample_layer(inputs)) + self.donwsample_layer(self.residual(inputs))
        else:
            outputs = self.layers(inputs) + self.residual(inputs)
        return outputs


class ConditionalDiscriminator(nn.Module):
    def __init__(self, in_channels, cond_channels, downsample_factors=(2, 2, 2), out_channels=(128, 256)):
        super().__init__()

        assert len(downsample_factors) == len(out_channels) + 1

        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.downsample_factors = downsample_factors
        self.out_channels = out_channels

        self.pre_cond_layers = nn.ModuleList()
        self.post_cond_layers = nn.ModuleList()

        # layers before condition features
        self.pre_cond_layers += [DBlock(in_channels, 64, 1)]
        in_channels = 64
        for i, channel in enumerate(out_channels):
            self.pre_cond_layers.append(DBlock(in_channels, channel, downsample_factors[i]))
            in_channels = channel

        # condition block
        self.cond_block = GBlock(in_channels, cond_channels, downsample_factors[-1])

        # layers after condition block
        self.post_cond_layers += [
            DBlock(in_channels * 2, in_channels * 2, 1),
            DBlock(in_channels * 2, in_channels * 2, 1),
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels * 2, 1, kernel_size=1),
        ]

    def forward(self, inputs, conditions):
        batch_size = inputs.size()[0]
        outputs = inputs.view(batch_size, self.in_channels, -1)
        for layer in self.pre_cond_layers:
            outputs = layer(outputs)
        outputs = self.cond_block(outputs, conditions)
        for layer in self.post_cond_layers:
            outputs = layer(outputs)

        return outputs


class UnconditionalDiscriminator(nn.Module):
    def __init__(self, in_channels, base_channels=64, downsample_factors=(8, 4), out_channels=(128, 256)):
        super().__init__()

        self.downsample_factors = downsample_factors
        self.in_channels = in_channels
        self.downsample_factors = downsample_factors
        self.out_channels = out_channels

        self.layers = nn.ModuleList()
        self.layers += [DBlock(self.in_channels, base_channels, 1)]
        in_channels = base_channels
        for i, factor in enumerate(downsample_factors):
            self.layers.append(DBlock(in_channels, out_channels[i], factor))
            in_channels *= 2
        self.layers += [
            DBlock(in_channels, in_channels, 1),
            DBlock(in_channels, in_channels, 1),
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, 1, kernel_size=1),
        ]

    def forward(self, inputs):
        batch_size = inputs.size()[0]
        outputs = inputs.view(batch_size, self.in_channels, -1)
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs


class RandomWindowDiscriminator(nn.Module):
    """Random Window Discriminator as described in
    http://arxiv.org/abs/1909.11646"""

    def __init__(
        self,
        cond_channels,
        hop_length,
        uncond_disc_donwsample_factors=(8, 4),
        cond_disc_downsample_factors=((8, 4, 2, 2, 2), (8, 4, 2, 2), (8, 4, 2), (8, 4), (4, 2, 2)),
        cond_disc_out_channels=((128, 128, 256, 256), (128, 256, 256), (128, 256), (256,), (128, 256)),
        window_sizes=(512, 1024, 2048, 4096, 8192),
    ):
        super().__init__()
        self.cond_channels = cond_channels
        self.window_sizes = window_sizes
        self.hop_length = hop_length
        self.base_window_size = self.hop_length * 2
        self.ks = [ws // self.base_window_size for ws in window_sizes]

        # check arguments
        assert len(cond_disc_downsample_factors) == len(cond_disc_out_channels) == len(window_sizes)
        for ws in window_sizes:
            assert ws % hop_length == 0

        for idx, cf in enumerate(cond_disc_downsample_factors):
            assert np.prod(cf) == hop_length // self.ks[idx]

        # define layers
        self.unconditional_discriminators = nn.ModuleList([])
        for k in self.ks:
            layer = UnconditionalDiscriminator(
                in_channels=k, base_channels=64, downsample_factors=uncond_disc_donwsample_factors
            )
            self.unconditional_discriminators.append(layer)

        self.conditional_discriminators = nn.ModuleList([])
        for idx, k in enumerate(self.ks):
            layer = ConditionalDiscriminator(
                in_channels=k,
                cond_channels=cond_channels,
                downsample_factors=cond_disc_downsample_factors[idx],
                out_channels=cond_disc_out_channels[idx],
            )
            self.conditional_discriminators.append(layer)

    def forward(self, x, c):
        scores = []
        feats = []
        # unconditional pass
        for window_size, layer in zip(self.window_sizes, self.unconditional_discriminators):
            index = np.random.randint(x.shape[-1] - window_size)

            score = layer(x[:, :, index : index + window_size])
            scores.append(score)

        # conditional pass
        for window_size, layer in zip(self.window_sizes, self.conditional_discriminators):
            frame_size = window_size // self.hop_length
            lc_index = np.random.randint(c.shape[-1] - frame_size)
            sample_index = lc_index * self.hop_length
            x_sub = x[:, :, sample_index : (lc_index + frame_size) * self.hop_length]
            c_sub = c[:, :, lc_index : lc_index + frame_size]

            score = layer(x_sub, c_sub)
            scores.append(score)
        return scores, feats
