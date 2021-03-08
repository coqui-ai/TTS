import math
import torch
from torch import nn

from TTS.tts.layers.glow_tts.transformer import RelativePositionTransformer
from TTS.tts.layers.generic.gated_conv import GatedConvBlock
from TTS.tts.utils.generic_utils import sequence_mask
from TTS.tts.layers.glow_tts.glow import ResidualConv1dLayerNormBlock
from TTS.tts.layers.glow_tts.duration_predictor import DurationPredictor
from TTS.tts.layers.generic.time_depth_sep_conv import TimeDepthSeparableConvBlock
from TTS.tts.layers.generic.res_conv_bn import ResidualConv1dBNBlock


class Encoder(nn.Module):
    """Glow-TTS encoder module.

    embedding -> <prenet> -> encoder_module -> <postnet> --> proj_mean
                                                         |
                                                         |-> proj_var
                                                         |
                                                         |-> concat -> duration_predictor
                                                                ↑
                                                          speaker_embed
    Args:
        num_chars (int): number of characters.
        out_channels (int): number of output channels.
        hidden_channels (int): encoder's embedding size.
        hidden_channels_ffn (int): transformer's feed-forward channels.
        kernel_size (int): kernel size for conv layers and duration predictor.
        dropout_p (float): dropout rate for any dropout layer.
        mean_only (bool): if True, output only mean values and use constant std.
        use_prenet (bool): if True, use pre-convolutional layers before transformer layers.
        c_in_channels (int): number of channels in conditional input.

    Shapes:
        - input: (B, T, C)

    Notes:
        suggested encoder params...

        for encoder_type == 'rel_pos_transformer'
            encoder_params={
                'kernel_size':3,
                'dropout_p': 0.1,
                'num_layers': 6,
                'num_heads': 2,
                'hidden_channels_ffn': 768,  # 4 times the hidden_channels
                'input_length': None
            }

        for encoder_type == 'gated_conv'
            encoder_params={
                'kernel_size':5,
                'dropout_p': 0.1,
                'num_layers': 9,
            }

        for encoder_type == 'residual_conv_bn'
            encoder_params={
                "kernel_size": 4,
                "dilations": [1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1],
                "num_conv_blocks": 2,
                "num_res_blocks": 13
            }

         for encoder_type == 'time_depth_separable'
            encoder_params={
                "kernel_size": 5,
                'num_layers': 9,
            }
    """
    def __init__(self,
                 num_chars,
                 out_channels,
                 hidden_channels,
                 hidden_channels_dp,
                 encoder_type,
                 encoder_params,
                 dropout_p_dp=0.1,
                 mean_only=False,
                 use_prenet=True,
                 c_in_channels=0):
        super().__init__()
        # class arguments
        self.num_chars = num_chars
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.hidden_channels_dp = hidden_channels_dp
        self.dropout_p_dp = dropout_p_dp
        self.mean_only = mean_only
        self.use_prenet = use_prenet
        self.c_in_channels = c_in_channels
        self.encoder_type = encoder_type
        # embedding layer
        self.emb = nn.Embedding(num_chars, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        # init encoder module
        if encoder_type.lower() == "rel_pos_transformer":
            if use_prenet:
                self.prenet = ResidualConv1dLayerNormBlock(hidden_channels,
                                                           hidden_channels,
                                                           hidden_channels,
                                                           kernel_size=5,
                                                           num_layers=3,
                                                           dropout_p=0.5)
            self.encoder = RelativePositionTransformer(hidden_channels,
                                                       hidden_channels,
                                                       hidden_channels,
                                                       **encoder_params)
        elif encoder_type.lower() == 'gated_conv':
            self.encoder = GatedConvBlock(hidden_channels, **encoder_params)
        elif encoder_type.lower() == 'residual_conv_bn':
            if use_prenet:
                self.prenet = nn.Sequential(
                    nn.Conv1d(hidden_channels, hidden_channels, 1),
                    nn.ReLU()
                )
            self.encoder = ResidualConv1dBNBlock(hidden_channels,
                                                 hidden_channels,
                                                 hidden_channels,
                                                 **encoder_params)
            self.postnet = nn.Sequential(
                nn.Conv1d(self.hidden_channels, self.hidden_channels, 1),
                nn.BatchNorm1d(self.hidden_channels))
        elif encoder_type.lower() == 'time_depth_separable':
            if use_prenet:
                self.prenet = ResidualConv1dLayerNormBlock(hidden_channels,
                                                           hidden_channels,
                                                           hidden_channels,
                                                           kernel_size=5,
                                                           num_layers=3,
                                                           dropout_p=0.5)
            self.encoder = TimeDepthSeparableConvBlock(hidden_channels,
                                                       hidden_channels,
                                                       hidden_channels,
                                                       **encoder_params)
        else:
            raise ValueError(" [!] Unkown encoder type.")

        # final projection layers
        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
        if not mean_only:
            self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)
        # duration predictor
        self.duration_predictor = DurationPredictor(
            hidden_channels + c_in_channels, hidden_channels_dp, 3,
            dropout_p_dp)

    def forward(self, x, x_lengths, g=None):
        """
        Shapes:
            x: [B, C, T]
            x_lengths: [B]
            g (optional): [B, 1, T]
        """
        # embedding layer
        # [B ,T, D]
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        # [B, D, T]
        x = torch.transpose(x, 1, -1)
        # compute input sequence mask
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)),
                                 1).to(x.dtype)
        # prenet
        if hasattr(self, 'prenet') and self.use_prenet:
            x = self.prenet(x, x_mask)
        # encoder
        x = self.encoder(x, x_mask)
        # postnet
        if hasattr(self, 'postnet'):
            x = self.postnet(x) * x_mask
        # set duration predictor input
        if g is not None:
            g_exp = g.expand(-1, -1, x.size(-1))
            x_dp = torch.cat([torch.detach(x), g_exp], 1)
        else:
            x_dp = torch.detach(x)
        # final projection layer
        x_m = self.proj_m(x) * x_mask
        if not self.mean_only:
            x_logs = self.proj_s(x) * x_mask
        else:
            x_logs = torch.zeros_like(x_m)
        # duration predictor
        logw = self.duration_predictor(x_dp, x_mask)
        return x_m, x_logs, logw, x_mask
