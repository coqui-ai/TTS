from torch import nn


class ZeroTemporalPad(nn.Module):
    """Pad sequences to equal lentgh in the temporal dimension"""
    def __init__(self, kernel_size, dilation):
        super().__init__()
        total_pad = (dilation * (kernel_size - 1))
        begin = total_pad // 2
        end = total_pad - begin
        self.pad_layer = nn.ZeroPad2d((0, 0, begin, end))

    def forward(self, x):
        return self.pad_layer(x)


class ConvBN(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        padding = (dilation * (kernel_size - 1))
        pad_s = padding // 2
        pad_e = padding - pad_s
        self.conv1d = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.pad = nn.ZeroPad2d((pad_s, pad_e, 0, 0))  # uneven left and right padding
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        o = self.conv1d(x)
        o = self.pad(o)
        o = self.norm(o)
        o = nn.functional.relu(o)
        return o


class ConvBNBlock(nn.Module):
    """Implements conv->PReLU->norm n-times"""

    def __init__(self, channels, kernel_size, dilation, num_conv_blocks=2):
        super().__init__()
        self.conv_bn_blocks = nn.Sequential(*[
            ConvBN(channels, kernel_size, dilation)
            for _ in range(num_conv_blocks)
        ])

    def forward(self, x):
        """
        Shapes:
            x: (B, D, T)
        """
        return self.conv_bn_blocks(x)


class ResidualConvBNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations, num_res_blocks=13, num_conv_blocks=2):
        super().__init__()
        assert len(dilations) == num_res_blocks
        self.res_blocks = nn.ModuleList()
        for dilation in dilations:
            block = ConvBNBlock(channels, kernel_size, dilation, num_conv_blocks)
            self.res_blocks.append(block)

    def forward(self, x, x_mask=None):
        o = x
        for block in self.res_blocks:
            res = o
            o = block(o * x_mask if x_mask is not None else o)
            o = o + res
        return o
