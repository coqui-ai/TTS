from torch import nn

from TTS.tts.layers.generic.res_conv_bn import ConvBN


class DurationPredictor(nn.Module):
    """Predicts phoneme log durations based on the encoder outputs"""
    def __init__(self, hidden_channels):
        super().__init__()

        self.layers = nn.ModuleList([
            ConvBN(hidden_channels, 4, 1),
            ConvBN(hidden_channels, 3, 1),
            ConvBN(hidden_channels, 1, 1),
            nn.Conv1d(hidden_channels, 1, 1)
        ])

    def forward(self, x, x_mask):
        """Outputs interpreted as log(durations)
        To get actual durations, do exp transformation
        :param x:
        :return:
        """
        o = x
        for layer in self.layers:
            o = layer(o) * x_mask
        return o
