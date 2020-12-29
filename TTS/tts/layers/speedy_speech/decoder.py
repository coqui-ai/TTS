from torch import nn
from TTS.tts.layers.generic.res_conv_bn import ConvBNBlock, ResidualConvBNBlock


class Decoder(nn.Module):
    """Decodes the expanded phoneme encoding into spectrograms
    Shapes:
        - input: (B, C, T)
    """
    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        out_channels,
        hidden_channels,
        residual_conv_bn_params={
            "kernel_size": 4,
            "dilations": 4 * [1, 2, 4, 8] + [1],
            "num_conv_blocks": 2,
            "num_res_blocks": 17
        }):
        super().__init__()

        self.decoder = ResidualConvBNBlock(hidden_channels,
                                           **residual_conv_bn_params)

        self.post_conv = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.post_net = nn.Sequential(
            ConvBNBlock(hidden_channels, residual_conv_bn_params['kernel_size'], 1, num_conv_blocks=2),
            nn.Conv1d(hidden_channels, out_channels, 1),
        )

    def forward(self, x, x_mask, g=None):  # pylint: disable=unused-argument
        # TODO: implement multi-speaker
        o = self.decoder(x, x_mask)
        o = self.post_conv(o) + x
        return self.post_net(o) * x_mask
