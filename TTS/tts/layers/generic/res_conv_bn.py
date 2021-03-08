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


class Conv1dBN(nn.Module):
    """1d convolutional with batch norm.
    conv1d -> relu -> BN blocks.

    Note:
        Batch normalization is applied after ReLU regarding the original implementation.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        kernel_size (int): kernel size for convolutional filters.
        dilation (int): dilation for convolution layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (dilation * (kernel_size - 1))
        pad_s = padding // 2
        pad_e = padding - pad_s
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.pad = nn.ZeroPad2d((pad_s, pad_e, 0, 0))  # uneven left and right padding
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        o = self.conv1d(x)
        o = self.pad(o)
        o = nn.functional.relu(o)
        o = self.norm(o)
        return o


class Conv1dBNBlock(nn.Module):
    """1d convolutional block with batch norm. It is a set of conv1d -> relu -> BN blocks.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        hidden_channels (int): number of inner convolution channels.
        kernel_size (int): kernel size for convolutional filters.
        dilation (int): dilation for convolution layers.
        num_conv_blocks (int, optional): number of convolutional blocks. Defaults to 2.
    """
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation, num_conv_blocks=2):
        super().__init__()
        self.conv_bn_blocks = []
        for idx in range(num_conv_blocks):
            layer = Conv1dBN(in_channels if idx == 0 else hidden_channels,
                             out_channels if idx == (num_conv_blocks - 1) else hidden_channels,
                             kernel_size,
                             dilation)
            self.conv_bn_blocks.append(layer)
        self.conv_bn_blocks = nn.Sequential(*self.conv_bn_blocks)

    def forward(self, x):
        """
        Shapes:
            x: (B, D, T)
        """
        return self.conv_bn_blocks(x)


class ResidualConv1dBNBlock(nn.Module):
    """Residual Convolutional Blocks with BN
    Each block has 'num_conv_block' conv layers and 'num_res_blocks' such blocks are connected
    with residual connections.

    conv_block = (conv1d -> relu -> bn) x 'num_conv_blocks'
    residuak_conv_block =  (x -> conv_block ->  + ->) x 'num_res_blocks'
                            ' - - - - - - - - - ^
    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        hidden_channels (int): number of inner convolution channels.
        kernel_size (int): kernel size for convolutional filters.
        dilations (list): dilations for each convolution layer.
        num_res_blocks (int, optional): number of residual blocks. Defaults to 13.
        num_conv_blocks (int, optional): number of convolutional blocks in each residual block. Defaults to 2.
    """
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilations, num_res_blocks=13, num_conv_blocks=2):

        super().__init__()
        assert len(dilations) == num_res_blocks
        self.res_blocks = nn.ModuleList()
        for idx, dilation in enumerate(dilations):
            block = Conv1dBNBlock(in_channels if idx == 0 else hidden_channels,
                                  out_channels if (idx + 1) == len(dilations) else hidden_channels,
                                  hidden_channels,
                                  kernel_size,
                                  dilation,
                                  num_conv_blocks)
            self.res_blocks.append(block)

    def forward(self, x, x_mask=None):
        if x_mask is None:
            x_mask = 1.0
        o = x * x_mask
        for block in self.res_blocks:
            res = o
            o = block(o)
            o = o + res
            if x_mask is not None:
                o = o * x_mask
        return o
