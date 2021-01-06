import torch
from torch import nn

from ..generic.normalization import LayerNorm


class DurationPredictor(nn.Module):
    """Glow-TTS duration prediction model.
    [2 x (conv1d_kxk -> relu -> layer_norm -> dropout)] -> conv1d_1x1 -> durs

        Args:
            in_channels ([type]): [description]
            hidden_channels ([type]): [description]
            kernel_size ([type]): [description]
            dropout_p ([type]): [description]
    """
    def __init__(self, in_channels, hidden_channels, kernel_size, dropout_p):
        super().__init__()
        # class arguments
        self.in_channels = in_channels
        self.filter_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        # layers
        self.drop = nn.Dropout(dropout_p)
        self.conv_1 = nn.Conv1d(in_channels,
                                hidden_channels,
                                kernel_size,
                                padding=kernel_size // 2)
        self.norm_1 = LayerNorm(hidden_channels)
        self.conv_2 = nn.Conv1d(hidden_channels,
                                hidden_channels,
                                kernel_size,
                                padding=kernel_size // 2)
        self.norm_2 = LayerNorm(hidden_channels)
        # output layer
        self.proj = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, x, x_mask):
        """
        Shapes:
            x: [B, C, T]
            x_mask: [B, 1, T]

        Returns:
            [type]: [description]
        """
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask
