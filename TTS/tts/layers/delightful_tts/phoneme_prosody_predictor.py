import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import

from TTS.tts.layers.delightful_tts.conv_layers import ConvTransposed


class PhonemeProsodyPredictor(nn.Module):
    """Non-parallel Prosody Predictor inspired by: https://arxiv.org/pdf/2102.00851.pdf
    It consists of 2 layers of  1D convolutions each followed by a relu activation, layer norm
    and dropout, then finally a linear layer.

    Args:
        hidden_size (int): Size of hidden channels.
        kernel_size (int): Kernel size for the conv layers.
        dropout: (float): Probability of dropout.
        bottleneck_size (int): bottleneck size for last linear layer.
        lrelu_slope (float): Slope of the leaky relu.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        dropout: float,
        bottleneck_size: int,
        lrelu_slope: float,
    ):
        super().__init__()
        self.d_model = hidden_size
        self.layers = nn.ModuleList(
            [
                ConvTransposed(
                    self.d_model,
                    self.d_model,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.LeakyReLU(lrelu_slope),
                nn.LayerNorm(self.d_model),
                nn.Dropout(dropout),
                ConvTransposed(
                    self.d_model,
                    self.d_model,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.LeakyReLU(lrelu_slope),
                nn.LayerNorm(self.d_model),
                nn.Dropout(dropout),
            ]
        )
        self.predictor_bottleneck = nn.Linear(self.d_model, bottleneck_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            x: :math: `[B, T, D]`
            mask: :math: `[B, T]`
        """
        mask = mask.unsqueeze(2)
        for layer in self.layers:
            x = layer(x)
        x = x.masked_fill(mask, 0.0)
        x = self.predictor_bottleneck(x)
        return x
