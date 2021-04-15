from torch import nn

from TTS.tts.layers.generic.res_conv_bn import ResidualConv1dBNBlock
from TTS.tts.layers.generic.transformer import FFTransformerBlock
from TTS.tts.layers.glow_tts.transformer import RelativePositionTransformer


class RelativePositionTransformerEncoder(nn.Module):
    """Speedy speech encoder built on Transformer with Relative Position encoding.

    TODO: Integrate speaker conditioning vector.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        hidden_channels (int): number of hidden channels
        params (dict): dictionary for residual convolutional blocks.
    """

    def __init__(self, in_channels, out_channels, hidden_channels, params):
        super().__init__()
        self.prenet = ResidualConv1dBNBlock(
            in_channels,
            hidden_channels,
            hidden_channels,
            kernel_size=5,
            num_res_blocks=3,
            num_conv_blocks=1,
            dilations=[1, 1, 1],
        )
        self.rel_pos_transformer = RelativePositionTransformer(hidden_channels, out_channels, hidden_channels, **params)

    def forward(self, x, x_mask=None, g=None):  # pylint: disable=unused-argument
        if x_mask is None:
            x_mask = 1
        o = self.prenet(x) * x_mask
        o = self.rel_pos_transformer(o, x_mask)
        return o


class ResidualConv1dBNEncoder(nn.Module):
    """Residual Convolutional Encoder as in the original Speedy Speech paper

    TODO: Integrate speaker conditioning vector.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        hidden_channels (int): number of hidden channels
        params (dict): dictionary for residual convolutional blocks.
    """

    def __init__(self, in_channels, out_channels, hidden_channels, params):
        super().__init__()
        self.prenet = nn.Sequential(nn.Conv1d(in_channels, hidden_channels, 1), nn.ReLU())
        self.res_conv_block = ResidualConv1dBNBlock(hidden_channels, hidden_channels, hidden_channels, **params)

        self.postnet = nn.Sequential(
            *[
                nn.Conv1d(hidden_channels, hidden_channels, 1),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_channels),
                nn.Conv1d(hidden_channels, out_channels, 1),
            ]
        )

    def forward(self, x, x_mask=None, g=None):  # pylint: disable=unused-argument
        if x_mask is None:
            x_mask = 1
        o = self.prenet(x) * x_mask
        o = self.res_conv_block(o, x_mask)
        o = self.postnet(o + x) * x_mask
        return o * x_mask


class Encoder(nn.Module):
    # pylint: disable=dangerous-default-value
    """Factory class for Speedy Speech encoder enables different encoder types internally.

    Args:
        num_chars (int): number of characters.
        out_channels (int): number of output channels.
        in_hidden_channels (int): input and hidden channels. Model keeps the input channels for the intermediate layers.
        encoder_type (str): encoder layer types. 'transformers' or 'residual_conv_bn'. Default 'residual_conv_bn'.
        encoder_params (dict): model parameters for specified encoder type.
        c_in_channels (int): number of channels for conditional input.

    Note:
        Default encoder_params to be set in config.json...

        ```python
        # for 'relative_position_transformer'
        encoder_params={
            'hidden_channels_ffn': 128,
            'num_heads': 2,
            "kernel_size": 3,
            "dropout_p": 0.1,
            "num_layers": 6,
            "rel_attn_window_size": 4,
            "input_length": None
        },

        # for 'residual_conv_bn'
        encoder_params = {
            "kernel_size": 4,
            "dilations": 4 * [1, 2, 4] + [1],
            "num_conv_blocks": 2,
            "num_res_blocks": 13
        }

        # for 'fftransformer'
        encoder_params = {
            "hidden_channels_ffn": 1024 ,
            "num_heads": 2,
            "num_layers": 6,
            "dropout_p": 0.1
        }
        ```
    """

    def __init__(
        self,
        in_hidden_channels,
        out_channels,
        encoder_type="residual_conv_bn",
        encoder_params={"kernel_size": 4, "dilations": 4 * [1, 2, 4] + [1], "num_conv_blocks": 2, "num_res_blocks": 13},
        c_in_channels=0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_hidden_channels
        self.hidden_channels = in_hidden_channels
        self.encoder_type = encoder_type
        self.c_in_channels = c_in_channels

        # init encoder
        if encoder_type.lower() == "relative_position_transformer":
            # text encoder
            # pylint: disable=unexpected-keyword-arg
            self.encoder = RelativePositionTransformerEncoder(
                in_hidden_channels, out_channels, in_hidden_channels, encoder_params
            )
        elif encoder_type.lower() == "residual_conv_bn":
            self.encoder = ResidualConv1dBNEncoder(in_hidden_channels, out_channels, in_hidden_channels, encoder_params)
        elif encoder_type.lower() == "fftransformer":
            assert (
                in_hidden_channels == out_channels
            ), "[!] must be `in_channels` == `out_channels` when encoder type is 'fftransformer'"
            # pylint: disable=unexpected-keyword-arg
            self.encoder = FFTransformerBlock(in_hidden_channels, **encoder_params)
        else:
            raise NotImplementedError(" [!] unknown encoder type.")

    def forward(self, x, x_mask, g=None):  # pylint: disable=unused-argument
        """
        Shapes:
            x: [B, C, T]
            x_mask: [B, 1, T]
            g: [B, C, 1]
        """
        o = self.encoder(x, x_mask)
        return o * x_mask
