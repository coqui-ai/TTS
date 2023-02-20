import torch
from torch import nn

from TTS.tts.layers.glow_tts.decoder import Decoder as GlowDecoder
from TTS.tts.utils.helpers import sequence_mask


class Decoder(nn.Module):
    """Uses glow decoder with some modifications.
    ::

        Squeeze -> ActNorm -> InvertibleConv1x1 -> AffineCoupling -> Unsqueeze

    Args:
        in_channels (int): channels of input tensor.
        hidden_channels (int): hidden decoder channels.
        kernel_size (int): Coupling block kernel size. (Wavenet filter kernel size.)
        dilation_rate (int): rate to increase dilation by each layer in a decoder block.
        num_flow_blocks (int): number of decoder blocks.
        num_coupling_layers (int): number coupling layers. (number of wavenet layers.)
        dropout_p (float): wavenet dropout rate.
        sigmoid_scale (bool): enable/disable sigmoid scaling in coupling layer.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_flow_blocks,
        num_coupling_layers,
        dropout_p=0.0,
        num_splits=4,
        num_squeeze=2,
        sigmoid_scale=False,
        c_in_channels=0,
    ):
        super().__init__()

        self.glow_decoder = GlowDecoder(
            in_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            num_flow_blocks,
            num_coupling_layers,
            dropout_p,
            num_splits,
            num_squeeze,
            sigmoid_scale,
            c_in_channels,
        )
        self.n_sqz = num_squeeze

    def forward(self, x, x_len, g=None, reverse=False):
        """
        Input shapes:
            - x:  :math:`[B, C, T]`
            - x_len :math:`[B]`
            - g: :math:`[B, C]`

        Output shapes:
            - x:  :math:`[B, C, T]`
            - x_len :math:`[B]`
            - logget_tot :math:`[B]`
        """
        x, x_len, x_max_len = self.preprocess(x, x_len, x_len.max())
        x_mask = torch.unsqueeze(sequence_mask(x_len, x_max_len), 1).to(x.dtype)
        x, logdet_tot = self.glow_decoder(x, x_mask, g, reverse)
        return x, x_len, logdet_tot

    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = torch.div(y_max_length, self.n_sqz, rounding_mode="floor") * self.n_sqz
            y = y[:, :, :y_max_length]
        y_lengths = torch.div(y_lengths, self.n_sqz, rounding_mode="floor") * self.n_sqz
        return y, y_lengths, y_max_length

    def store_inverse(self):
        self.glow_decoder.store_inverse()
