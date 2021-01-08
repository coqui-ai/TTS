import math
import torch
from torch import nn
from torch.nn import functional as F

from TTS.tts.layers.glow_tts.transformer import RelativePositionTransformer
from TTS.tts.layers.glow_tts.glow import ConvLayerNorm
from TTS.tts.layers.generic.res_conv_bn import ResidualConvBNBlock


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    Args:
       channels (int): embedding size
       dropout (float): dropout parameter
    """
    def __init__(self, channels, dropout=0.0, max_len=5000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd channels (got channels={:d})".format(channels))
        pe = torch.zeros(max_len, channels)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, channels, 2, dtype=torch.float) *
                             -(math.log(10000.0) / channels)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)
        self.register_buffer('pe', pe)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
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
                    f" limited to {self.pe.size(2)}. See max_len argument.")
            if mask is not None:
                pos_enc = (self.pe[:, :, :x.size(2)] * mask)
            else:
                pos_enc = self.pe[:, :, :x.size(2)]
            x = x + pos_enc
        else:
            x = x + self.pe[:, :, first_idx:last_idx]
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return x


class Encoder(nn.Module):
    # pylint: disable=dangerous-default-value
    """Speedy-Speech encoder using Transformers or Residual BN Convs internally.

    Args:
        num_chars (int): number of characters.
        out_channels (int): number of output channels.
        in_hidden_channels (int): input and hidden channels. Model keeps the input channels for the intermediate layers.
        encoder_type (str): encoder layer types. 'transformers' or 'residual_conv_bn'. Default 'residual_conv_bn'.
        encoder_params (dict): model parameters for specified encoder type.
        c_in_channels (int): number of channels for conditional input.

    Note:
        Default encoder_params...

        for 'transformer'
            encoder_params={
                'hidden_channels_ffn': 128,
                'num_heads': 2,
                "kernel_size": 3,
                "dropout_p": 0.1,
                "num_layers": 6,
                "rel_attn_window_size": 4,
                "input_length": None
            },

        for 'residual_conv_bn'
            encoder_params = {
                "kernel_size": 4,
                "dilations": 4 * [1, 2, 4] + [1],
                "num_conv_blocks": 2,
                "num_res_blocks": 13
            }
    """
    def __init__(
        self,
        in_hidden_channels,
        out_channels,
        encoder_type='residual_conv_bn',
        encoder_params={
            "kernel_size": 4,
            "dilations": 4 * [1, 2, 4] + [1],
            "num_conv_blocks": 2,
            "num_res_blocks": 13
        },
        c_in_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_hidden_channels
        self.hidden_channels = in_hidden_channels
        self.encoder_type = encoder_type
        self.c_in_channels = c_in_channels

        # init encoder
        if encoder_type.lower() == "transformer":
            # optional convolutional prenet
            self.pre = ConvLayerNorm(self.in_channels,
                                     self.hidden_channels,
                                     self.hidden_channels,
                                     kernel_size=5,
                                     num_layers=3,
                                     dropout_p=0.5)
            # text encoder
            self.encoder = RelativePositionTransformer(self.hidden_channels, **encoder_params)  # pylint: disable=unexpected-keyword-arg
        elif encoder_type.lower() == 'residual_conv_bn':
            self.pre = nn.Sequential(
                nn.Conv1d(self.in_channels, self.hidden_channels, 1),
                nn.ReLU())
            self.encoder = ResidualConvBNBlock(self.hidden_channels,
                                               **encoder_params)
        else:
            raise NotImplementedError(' [!] encoder type not implemented.')

        # final projection layers
        self.post_conv = nn.Conv1d(self.hidden_channels, self.hidden_channels,
                                   1)
        self.post_bn = nn.BatchNorm1d(self.hidden_channels)
        self.post_conv2 = nn.Conv1d(self.hidden_channels, self.out_channels, 1)

    def forward(self, x, x_mask, g=None):  # pylint: disable=unused-argument
        """
        Shapes:
            x: [B, C, T]
            x_mask: [B, 1, T]
            g: [B, C, 1]
        """
        # TODO: implement multi-speaker
        if self.encoder_type == 'transformer':
            o = self.pre(x, x_mask)
        else:
            o = self.pre(x) * x_mask
        o = self.encoder(o, x_mask)
        o = self.post_conv(o + x)
        o = F.relu(o)
        o = self.post_bn(o)
        o = self.post_conv2(o)
        # [B, C, T]
        return o * x_mask
