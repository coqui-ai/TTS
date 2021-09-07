import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"

    Args:
       channels (int): embedding size
       dropout_p (float): dropout rate applied to the output.
       max_len (int): maximum sequence length.
       use_scale (bool): whether to use a learnable scaling coefficient.
    """

    def __init__(self, channels, dropout_p=0.0, max_len=5000, use_scale=False):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with " "odd channels (got channels={:d})".format(channels)
            )
        self.use_scale = use_scale
        if use_scale:
            self.scale = torch.nn.Parameter(torch.ones(1))
        pe = torch.zeros(max_len, channels)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, channels, 2).float() / channels)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)
        self.register_buffer("pe", pe)
        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)
        self.channels = channels

    def forward(self, x, mask=None, first_idx=None, last_idx=None):
        """
        Shapes:
            x: [B, C, T]
            mask: [B, 1, T]
            first_idx: int
            last_idx: int
        """

        x = x * math.sqrt(self.channels)
        if first_idx is None:
            if self.pe.size(2) < x.size(2):
                raise RuntimeError(
                    f"Sequence is {x.size(2)} but PositionalEncoding is"
                    f" limited to {self.pe.size(2)}. See max_len argument."
                )
            if mask is not None:
                pos_enc = self.pe[:, :, : x.size(2)] * mask
            else:
                pos_enc = self.pe[:, :, : x.size(2)]
            if self.use_scale:
                x = x + self.scale * pos_enc
            else:
                x = x + pos_enc
        else:
            if self.use_scale:
                x = x + self.scale * self.pe[:, :, first_idx:last_idx]
            else:
                x = x + self.pe[:, :, first_idx:last_idx]
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        return x
