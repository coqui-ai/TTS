import torch
from torch import nn
from TTS.tts.layers.generic.res_conv_bn import Conv1dBNBlock, ResidualConv1dBNBlock, Conv1dBN
from TTS.tts.layers.generic.wavenet import WNBlocks
from TTS.tts.layers.glow_tts.transformer import RelativePositionTransformer


class WaveNetDecoder(nn.Module):
    """WaveNet based decoder with a prenet and a postnet.

    prenet: conv1d_1x1
    postnet: 3 x [conv1d_1x1 -> relu] -> conv1d_1x1

    TODO: Integrate speaker conditioning vector.

    Note:
        default wavenet parameters;
            params = {
                "num_blocks": 12,
                "hidden_channels":192,
                "kernel_size": 5,
                "dilation_rate": 1,
                "num_layers": 4,
                "dropout_p": 0.05
            }

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        hidden_channels (int): number of hidden channels for prenet and postnet.
        params (dict): dictionary for residual convolutional blocks.
    """
    def __init__(self, in_channels, out_channels, hidden_channels, c_in_channels, params):
        super().__init__()
        # prenet
        self.prenet = torch.nn.Conv1d(in_channels, params['hidden_channels'], 1)
        # wavenet layers
        self.wn = WNBlocks(params['hidden_channels'], c_in_channels=c_in_channels, **params)
        # postnet
        self.postnet = [
            torch.nn.Conv1d(params['hidden_channels'], hidden_channels, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_channels, hidden_channels, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_channels, hidden_channels, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_channels, out_channels, 1),
        ]
        self.postnet = nn.Sequential(*self.postnet)

    def forward(self, x, x_mask=None, g=None):
        x = self.prenet(x) * x_mask
        x = self.wn(x, x_mask, g)
        o = self.postnet(x) * x_mask
        return o


class RelativePositionTransformerDecoder(nn.Module):
    """Decoder with Relative Positional Transformer.

    Note:
        Default params
            params={
                'hidden_channels_ffn': 128,
                'num_heads': 2,
                "kernel_size": 3,
                "dropout_p": 0.1,
                "num_layers": 8,
                "rel_attn_window_size": 4,
                "input_length": None
            }

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        hidden_channels (int): number of hidden channels including Transformer layers.
        params (dict): dictionary for residual convolutional blocks.
    """
    def __init__(self, in_channels, out_channels, hidden_channels, params):

        super().__init__()
        self.prenet = Conv1dBN(in_channels, hidden_channels, 1, 1)
        self.rel_pos_transformer = RelativePositionTransformer(
            in_channels, out_channels, hidden_channels, **params)

    def forward(self, x, x_mask=None, g=None):  # pylint: disable=unused-argument
        o = self.prenet(x) * x_mask
        o = self.rel_pos_transformer(o, x_mask)
        return o


class ResidualConv1dBNDecoder(nn.Module):
    """Residual Convolutional Decoder as in the original Speedy Speech paper

    TODO: Integrate speaker conditioning vector.

    Note:
        Default params
                params = {
                    "kernel_size": 4,
                    "dilations": 4 * [1, 2, 4, 8] + [1],
                    "num_conv_blocks": 2,
                    "num_res_blocks": 17
                }

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        hidden_channels (int): number of hidden channels including ResidualConv1dBNBlock layers.
        params (dict): dictionary for residual convolutional blocks.
    """
    def __init__(self, in_channels, out_channels, hidden_channels, params):
        super().__init__()
        self.res_conv_block = ResidualConv1dBNBlock(in_channels,
                                                    hidden_channels,
                                                    hidden_channels, **params)
        self.post_conv = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.postnet = nn.Sequential(
            Conv1dBNBlock(hidden_channels,
                          hidden_channels,
                          hidden_channels,
                          params['kernel_size'],
                          1,
                          num_conv_blocks=2),
            nn.Conv1d(hidden_channels, out_channels, 1),
        )

    def forward(self, x, x_mask=None, g=None):  # pylint: disable=unused-argument
        o = self.res_conv_block(x, x_mask)
        o = self.post_conv(o) + x
        return self.postnet(o) * x_mask


class Decoder(nn.Module):
    """Decodes the expanded phoneme encoding into spectrograms
    Args:
        out_channels (int): number of output channels.
        in_hidden_channels (int): input and hidden channels. Model keeps the input channels for the intermediate layers.
        decoder_type (str): decoder layer types. 'transformers' or 'residual_conv_bn'. Default 'residual_conv_bn'.
        decoder_params (dict): model parameters for specified decoder type.
        c_in_channels (int): number of channels for conditional input.

    Shapes:
        - input: (B, C, T)
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        out_channels,
        in_hidden_channels,
        decoder_type='residual_conv_bn',
        decoder_params={
            "kernel_size": 4,
            "dilations": 4 * [1, 2, 4, 8] + [1],
            "num_conv_blocks": 2,
            "num_res_blocks": 17
        },
        c_in_channels=0):
        super().__init__()

        if decoder_type == 'transformer':
            self.decoder = RelativePositionTransformerDecoder(
                in_channels=in_hidden_channels,
                out_channels=out_channels,
                hidden_channels=in_hidden_channels,
                params=decoder_params)
        elif decoder_type == 'residual_conv_bn':
            self.decoder = ResidualConv1dBNDecoder(
                in_channels=in_hidden_channels,
                out_channels=out_channels,
                hidden_channels=in_hidden_channels,
                params=decoder_params)
        elif decoder_type == 'wavenet':
            self.decoder = WaveNetDecoder(in_channels=in_hidden_channels,
                                          out_channels=out_channels,
                                          hidden_channels=in_hidden_channels,
                                          c_in_channels=c_in_channels,
                                          params=decoder_params)
        else:
            raise ValueError(f'[!] Unknown decoder type - {decoder_type}')

    def forward(self, x, x_mask, g=None):  # pylint: disable=unused-argument
        """
        Args:
            x: [B, C, T]
            x_mask: [B, 1, T]
            g: [B, C_g, 1]
        """
        # TODO: implement multi-speaker
        o = self.decoder(x, x_mask, g)
        return o