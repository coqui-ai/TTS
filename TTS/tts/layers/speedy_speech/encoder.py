import math
import torch
from torch import nn
from torch.nn import functional as F

from TTS.tts.layers.glow_tts.transformer import Transformer
from TTS.tts.layers.glow_tts.glow import ConvLayerNorm
from TTS.tts.layers.generic.res_conv_bn import ResidualConvBNBlock


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dim, dropout=0.0, max_len=5000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)
        self.register_buffer('pe', pe)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, x, mask=None, first_idx=None, last_idx=None):
        """Embed inputs.
        Args:
            x (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            mask (FloatTensor): Sequence mask.
            first_idx (int or NoneType): starting index for taking a
                certain part of the embeddings.
            last_idx (int or NoneType): ending index for taking a
                certain part of the embeddings.

        Shapes:
            x: B x C x T
        """

        x = x * math.sqrt(self.dim)
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

            Shapes:
                - input: (B, C, T)
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.encoder_type = encoder_type
        self.c_in_channels = c_in_channels

        # init encoder
        if encoder_type.lower() == "transformer":
            # optional convolutional prenet
            self.pre = ConvLayerNorm(hidden_channels,
                                     hidden_channels,
                                     hidden_channels,
                                     kernel_size=5,
                                     num_layers=3,
                                     dropout_p=0.5)
            # text encoder
            self.encoder = Transformer(hidden_channels, **encoder_params)  # pylint: disable=unexpected-keyword-arg
        elif encoder_type.lower() == 'residual_conv_bn':
            self.pre = nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, 1), nn.ReLU())
            self.encoder = ResidualConvBNBlock(hidden_channels,
                                               **encoder_params)
        else:
            raise NotImplementedError(' [!] encoder type not implemented.')

        # final projection layers
        self.post_conv = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.post_bn = nn.BatchNorm1d(hidden_channels)
        self.post_conv2 = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_mask, g=None):  # pylint: disable=unused-argument
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
