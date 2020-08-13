import math
import torch
from torch import nn
from torch.nn import functional as F

from mozilla_voice_tts.tts.utils.generic_utils import sequence_mask
from mozilla_voice_tts.tts.layers.glow_tts.glow import ConvLayerNorm
from mozilla_voice_tts.tts.layers.glow_tts.duration_predictor import DurationPredictor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', self._get_pe_matrix(d_model, max_len))

    def forward(self, x):
        return x + self.pe[:x.size(0)].unsqueeze(1)

    def _get_pe_matrix(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000,
                             torch.arange(0, d_model, 2).float() / d_model)

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        return pe


class Encoder(nn.Module):
    """Glow-TTS encoder module. We use Pytorch TransformerEncoder instead
    of the one with relative position embedding. We use positional encoding
    for capturing positiong information.

    Args:
        num_chars (int): number of characters.
        out_channels (int): number of output channels.
        hidden_channels (int): encoder's embedding size.
        filter_channels (int): transformer's feed-forward channels.
        num_head (int): number of attention heads in transformer.
        num_layers (int): number of transformer encoder stack.
        kernel_size (int): kernel size for conv layers and duration predictor.
        dropout_p (float): dropout rate for any dropout layer.
        mean_only (bool): if True, output only mean values and use constant std.
        use_prenet (bool): if True, use pre-convolutional layers before transformer layers.
        c_in_channels (int): number of channels in conditional input.

    Shapes:
        - input: (B, T, C)
    """

    def __init__(self,
                 num_chars,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 filter_channels_dp,
                 num_heads,
                 num_layers,
                 kernel_size,
                 dropout_p,
                 mean_only=False,
                 use_prenet=False,
                 c_in_channels=0):

        super().__init__()

        self.num_chars = num_chars
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        self.mean_only = mean_only
        self.use_prenet = use_prenet
        self.c_in_channels = c_in_channels

        self.emb = nn.Embedding(num_chars, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        self.register_buffer('pe', PositionalEncoding(hidden_channels).pe)

        if use_prenet:
            self.pre = ConvLayerNorm(hidden_channels,
                                     hidden_channels,
                                     hidden_channels,
                                     kernel_size=5,
                                     num_layers=3,
                                     dropout_p=0.5)

        encoder = nn.TransformerEncoderLayer(hidden_channels,
                                             num_heads,
                                             dim_feedforward=filter_channels,
                                             dropout=dropout_p)
        self.encoder = nn.TransformerEncoder(encoder, num_layers)

        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
        if not mean_only:
            self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)
        self.duration_predictor = DurationPredictor(
            hidden_channels + c_in_channels, filter_channels_dp, kernel_size,
            dropout_p)

    def forward(self, x, x_lengths, g=None):
        # pass embedding layer
        # [B ,T, D]
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x = x + self.pe[:x.shape[1]].unsqueeze(0)
        # [B, D, T]
        x = torch.transpose(x, 1, -1)
        # compute input sequence mask
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)),
                                 1).to(x.dtype)
        # pass pre-conv layers
        if self.use_prenet:
            x = self.pre(x, x_mask)
        # pass encoder
        x = self.encoder(x.permute(2, 0, 1),
                         src_key_padding_mask=~(x_mask.squeeze(1).to(bool)))
        x = x.permute(1, 2, 0)
        # set duration predictor input
        if g is not None:
            g_exp = g.expand(-1, -1, x.size(-1))
            x_dp = torch.cat([torch.detach(x), g_exp], 1)
        else:
            x_dp = torch.detach(x)
        # pass final projection layer
        x_m = self.proj_m(x) * x_mask
        if not self.mean_only:
            x_logs = self.proj_s(x) * x_mask
        else:
            x_logs = torch.zeros_like(x_m)
        # pass duration predictor
        logw = self.duration_predictor(x_dp, x_mask)
        return x_m, x_logs, logw, x_mask
