from torch import nn
from TTS.tts.layers.generic.res_conv_bn import ConvBNBlock, ResidualConvBNBlock
from TTS.tts.layers.glow_tts.transformer import Transformer


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

    Note:
            Default decoder_params...

            for 'transformer'
                encoder_params={
                    'hidden_channels_ffn': 128,
                    'num_heads': 2,
                    "kernel_size": 3,
                    "dropout_p": 0.1,
                    "num_layers": 8,
                    "rel_attn_window_size": 4,
                    "input_length": None
                },

            for 'residual_conv_bn'
                encoder_params = {
                    "kernel_size": 4,
                    "dilations": 4 * [1, 2, 4, 8] + [1],
                    "num_conv_blocks": 2,
                    "num_res_blocks": 17
                }
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
        self.in_channels = in_hidden_channels
        self.hidden_channels = in_hidden_channels
        self.out_channels = out_channels

        if decoder_type == 'transformer':
            self.decoder = Transformer(self.hidden_channels, **decoder_params)
        elif decoder_type == 'residual_conv_bn':
            self.decoder = ResidualConvBNBlock(self.hidden_channels,
                                               **decoder_params)
        else:
            raise ValueError(f'[!] Unknown decoder type - {decoder_type}')

        self.post_conv = nn.Conv1d(self.hidden_channels, self.hidden_channels, 1)
        self.post_net = nn.Sequential(
            ConvBNBlock(self.hidden_channels, decoder_params['kernel_size'], 1, num_conv_blocks=2),
            nn.Conv1d(self.hidden_channels, out_channels, 1),
        )

    def forward(self, x, x_mask, g=None):  # pylint: disable=unused-argument
        # TODO: implement multi-speaker
        o = self.decoder(x, x_mask)
        o = self.post_conv(o) + x
        return self.post_net(o) * x_mask
