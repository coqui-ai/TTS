from torch import nn

from .normalization import LayerNorm


class GatedConvBlock(nn.Module):
    """Gated convolutional block as in https://arxiv.org/pdf/1612.08083.pdf
    Args:
        in_out_channels (int): number of input/output channels.
        kernel_size (int): convolution kernel size.
        dropout_p (float): dropout rate.
    """
    def __init__(self, in_out_channels, kernel_size, dropout_p, num_layers):
        super().__init__()
        # class arguments
        self.dropout_p = dropout_p
        self.num_layers = num_layers
        # define layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers += [
                nn.Conv1d(in_out_channels,
                          2 * in_out_channels,
                          kernel_size,
                          padding=kernel_size // 2)
            ]
            self.norm_layers += [LayerNorm(2 * in_out_channels)]

    def forward(self, x, x_mask):
        o = x
        res = x
        for idx in range(self.num_layers):
            o = nn.functional.dropout(o,
                                      p=self.dropout_p,
                                      training=self.training)
            o = self.conv_layers[idx](o * x_mask)
            o = self.norm_layers[idx](o)
            o = nn.functional.glu(o, dim=1)
            o = res + o
            res = o
        return o