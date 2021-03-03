import torch
from torch import nn
from ..generic.normalization import LayerNorm


class MDNBlock(nn.Module):
    """Mixture of Density Network implementation
    https://arxiv.org/pdf/2003.01950.pdf
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.mdn = nn.Sequential(nn.Conv1d(in_channels, in_channels, 1),
                                 LayerNorm(in_channels),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Conv1d(in_channels, out_channels, 1))

    def forward(self, x):
        mu_sigma = self.mdn(x)
        # TODO: check this sigmoid
        # mu = torch.sigmoid(mu_sigma[:, :self.out_channels//2, :])
        mu = mu_sigma[:, :self.out_channels//2, :]
        log_sigma = mu_sigma[:, self.out_channels//2:, :]
        return mu, log_sigma
