import torch
from torch.nn import functional as F


class ResidualBlock(torch.nn.Module):
    """Residual block module in WaveNet."""

    def __init__(
        self,
        kernel_size=3,
        res_channels=64,
        gate_channels=128,
        skip_channels=64,
        aux_channels=80,
        dropout=0.0,
        dilation=1,
        bias=True,
        use_causal_conv=False,
    ):
        super().__init__()
        self.dropout = dropout
        # no future time stamps available
        if use_causal_conv:
            padding = (kernel_size - 1) * dilation
        else:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
            padding = (kernel_size - 1) // 2 * dilation
        self.use_causal_conv = use_causal_conv

        # dilation conv
        self.conv = torch.nn.Conv1d(
            res_channels, gate_channels, kernel_size, padding=padding, dilation=dilation, bias=bias
        )

        # local conditioning
        if aux_channels > 0:
            self.conv1x1_aux = torch.nn.Conv1d(aux_channels, gate_channels, 1, bias=False)
        else:
            self.conv1x1_aux = None

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = torch.nn.Conv1d(gate_out_channels, res_channels, 1, bias=bias)
        self.conv1x1_skip = torch.nn.Conv1d(gate_out_channels, skip_channels, 1, bias=bias)

    def forward(self, x, c):
        """
        x: B x D_res x T
        c: B x D_aux x T
        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x)

        # remove future time steps if use_causal_conv conv
        x = x[:, :, : residual.size(-1)] if self.use_causal_conv else x

        # split into two part for gated activation
        splitdim = 1
        xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)

        # local conditioning
        if c is not None:
            assert self.conv1x1_aux is not None
            c = self.conv1x1_aux(c)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ca, xb + cb

        x = torch.tanh(xa) * torch.sigmoid(xb)

        # for skip connection
        s = self.conv1x1_skip(x)

        # for residual connection
        x = (self.conv1x1_out(x) + residual) * (0.5**2)

        return x, s
