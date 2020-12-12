import torch
from torch import nn


class TimeDepthSeparableConv(nn.Module):
    """Time depth separable convolution as in https://arxiv.org/pdf/1904.02619.pdf
    It shows competative results with less computation and memory footprint."""
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 kernel_size,
                 bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels
        self.kernel_size = kernel_size

        self.time_conv = nn.Conv1d(
            in_channels,
            2 * hid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.norm1 = nn.BatchNorm1d(2 * hid_channels)
        self.depth_conv = nn.Conv1d(
            hid_channels,
            hid_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=hid_channels,
            bias=bias,
        )
        self.norm2 = nn.BatchNorm1d(hid_channels)
        self.time_conv2 = nn.Conv1d(
            hid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.norm3 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x_res = x
        x = self.time_conv(x)
        x = self.norm1(x)
        x = nn.functional.glu(x, dim=1)
        x = self.depth_conv(x)
        x = self.norm2(x)
        x = x * torch.sigmoid(x)
        x = self.time_conv2(x)
        x = self.norm3(x)
        x = x_res + x
        return x


class TimeDepthSeparableConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_layers,
                 kernel_size,
                 bias=True):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        assert num_layers > 1

        self.layers = nn.ModuleList()
        layer = TimeDepthSeparableConv(
            in_channels, hid_channels,
            out_channels if num_layers == 1 else hid_channels, kernel_size,
            bias)
        self.layers.append(layer)
        for idx in range(num_layers - 1):
            layer = TimeDepthSeparableConv(
                hid_channels, hid_channels, out_channels if
                (idx + 1) == (num_layers - 1) else hid_channels, kernel_size,
                bias)
            self.layers.append(layer)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x * mask)
        return x
