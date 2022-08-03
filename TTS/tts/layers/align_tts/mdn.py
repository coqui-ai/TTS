from torch import nn


class MDNBlock(nn.Module):
    """Mixture of Density Network implementation
    https://arxiv.org/pdf/2003.01950.pdf
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, in_channels, 1)
        self.norm = nn.LayerNorm(in_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        o = self.conv1(x)
        o = o.transpose(1, 2)
        o = self.norm(o)
        o = o.transpose(1, 2)
        o = self.relu(o)
        o = self.dropout(o)
        mu_sigma = self.conv2(o)
        # TODO: check this sigmoid
        # mu = torch.sigmoid(mu_sigma[:, :self.out_channels//2, :])
        mu = mu_sigma[:, : self.out_channels // 2, :]
        log_sigma = mu_sigma[:, self.out_channels // 2 :, :]
        return mu, log_sigma
