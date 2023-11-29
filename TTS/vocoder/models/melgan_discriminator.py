import numpy as np
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class MelganDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_sizes=(5, 3),
        base_channels=16,
        max_channels=1024,
        downsample_factors=(4, 4, 4, 4),
        groups_denominator=4,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        layer_kernel_size = np.prod(kernel_sizes)
        layer_padding = (layer_kernel_size - 1) // 2

        # initial layer
        self.layers += [
            nn.Sequential(
                nn.ReflectionPad1d(layer_padding),
                weight_norm(nn.Conv1d(in_channels, base_channels, layer_kernel_size, stride=1)),
                nn.LeakyReLU(0.2, inplace=True),
            )
        ]

        # downsampling layers
        layer_in_channels = base_channels
        for downsample_factor in downsample_factors:
            layer_out_channels = min(layer_in_channels * downsample_factor, max_channels)
            layer_kernel_size = downsample_factor * 10 + 1
            layer_padding = (layer_kernel_size - 1) // 2
            layer_groups = layer_in_channels // groups_denominator
            self.layers += [
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(
                            layer_in_channels,
                            layer_out_channels,
                            kernel_size=layer_kernel_size,
                            stride=downsample_factor,
                            padding=layer_padding,
                            groups=layer_groups,
                        )
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            ]
            layer_in_channels = layer_out_channels

        # last 2 layers
        layer_padding1 = (kernel_sizes[0] - 1) // 2
        layer_padding2 = (kernel_sizes[1] - 1) // 2
        self.layers += [
            nn.Sequential(
                weight_norm(
                    nn.Conv1d(
                        layer_out_channels,
                        layer_out_channels,
                        kernel_size=kernel_sizes[0],
                        stride=1,
                        padding=layer_padding1,
                    )
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            weight_norm(
                nn.Conv1d(
                    layer_out_channels, out_channels, kernel_size=kernel_sizes[1], stride=1, padding=layer_padding2
                )
            ),
        ]

    def forward(self, x):
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return x, feats
