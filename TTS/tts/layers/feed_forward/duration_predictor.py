from torch import nn

from TTS.tts.layers.generic.res_conv_bn import Conv1dBN


class DurationPredictor(nn.Module):
    """Speedy Speech duration predictor model.
    Predicts phoneme durations from encoder outputs.

    Note:
        Outputs interpreted as log(durations)
        To get actual durations, do exp transformation

    conv_BN_4x1 -> conv_BN_3x1 -> conv_BN_1x1 -> conv_1x1

    Args:
        hidden_channels (int): number of channels in the inner layers.
    """
    def __init__(self, hidden_channels):

        super().__init__()

        self.layers = nn.ModuleList([
            Conv1dBN(hidden_channels, hidden_channels, 4, 1),
            Conv1dBN(hidden_channels, hidden_channels, 3, 1),
            Conv1dBN(hidden_channels, hidden_channels, 1, 1),
            nn.Conv1d(hidden_channels, 1, 1)
        ])

    def forward(self, x, x_mask):
        """
        Shapes:
            x: [B, C, T]
            x_mask: [B, 1, T]
        """
        o = x
        for layer in self.layers:
            o = layer(o) * x_mask
        return o
