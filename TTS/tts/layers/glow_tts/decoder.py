import torch
from torch import nn

from TTS.tts.layers.glow_tts.glow import InvConvNear, CouplingBlock
from TTS.tts.layers.glow_tts.normalization import ActNorm


def squeeze(x, x_mask=None, num_sqz=2):
    b, c, t = x.size()

    t = (t // num_sqz) * num_sqz
    x = x[:, :, :t]
    x_sqz = x.view(b, c, t // num_sqz, num_sqz)
    x_sqz = x_sqz.permute(0, 3, 1,
                          2).contiguous().view(b, c * num_sqz, t // num_sqz)

    if x_mask is not None:
        x_mask = x_mask[:, :, num_sqz - 1::num_sqz]
    else:
        x_mask = torch.ones(b, 1, t // num_sqz).to(device=x.device,
                                                   dtype=x.dtype)
    return x_sqz * x_mask, x_mask


def unsqueeze(x, x_mask=None, num_sqz=2):
    b, c, t = x.size()

    x_unsqz = x.view(b, num_sqz, c // num_sqz, t)
    x_unsqz = x_unsqz.permute(0, 2, 3,
                              1).contiguous().view(b, c // num_sqz,
                                                   t * num_sqz)

    if x_mask is not None:
        x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1,
                                             num_sqz).view(b, 1, t * num_sqz)
    else:
        x_mask = torch.ones(b, 1, t * num_sqz).to(device=x.device,
                                                  dtype=x.dtype)
    return x_unsqz * x_mask, x_mask


class Decoder(nn.Module):
    """Stack of Glow Modules"""
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 num_flow_blocks,
                 num_coupling_layers,
                 dropout_p=0.,
                 num_splits=4,
                 num_sqz=2,
                 sigmoid_scale=False,
                 c_in_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_flow_blocks = num_flow_blocks
        self.num_coupling_layers = num_coupling_layers
        self.dropout_p = dropout_p
        self.num_splits = num_splits
        self.num_sqz = num_sqz
        self.sigmoid_scale = sigmoid_scale
        self.c_in_channels = c_in_channels

        self.flows = nn.ModuleList()
        for _ in range(num_flow_blocks):
            self.flows.append(ActNorm(channels=in_channels * num_sqz))
            self.flows.append(
                InvConvNear(channels=in_channels * num_sqz,
                            num_splits=num_splits))
            self.flows.append(
                CouplingBlock(in_channels * num_sqz,
                              hidden_channels,
                              kernel_size=kernel_size,
                              dilation_rate=dilation_rate,
                              num_layers=num_coupling_layers,
                              c_in_channels=c_in_channels,
                              dropout_p=dropout_p,
                              sigmoid_scale=sigmoid_scale))

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        if self.num_sqz > 1:
            x, x_mask = squeeze(x, x_mask, self.num_sqz)
        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
        if self.num_sqz > 1:
            x, x_mask = unsqueeze(x, x_mask, self.num_sqz)
        return x, logdet_tot

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()
