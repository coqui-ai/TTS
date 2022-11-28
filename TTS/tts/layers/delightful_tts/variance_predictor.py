import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import

from TTS.tts.layers.delightful_tts.networks import BottleneckLayer, ConvLSTMLinear, ConvTransposed


class VariancePredictorLSTM(nn.Module):
    def __init__(self, in_dim, out_dim=1, reduction_factor=4):
        super(VariancePredictorLSTM, self).__init__()  # pylint: disable=super-with-arguments
        self.bottleneck_layer = BottleneckLayer(
            in_dim, reduction_factor, norm="weightnorm", non_linearity="relu", kernel_size=3, use_partial_padding=False
        )
        self.feat_pred_fn = ConvLSTMLinear(
            self.bottleneck_layer.out_dim,
            out_dim,
            n_layers=2,
            n_channels=256,
            kernel_size=3,
            p_dropout=0.1,
            lstm_type="bilstm",
            use_linear=True,
        )

    def forward(self, txt_enc, lens):
        txt_enc = self.bottleneck_layer(txt_enc)
        x_hat = self.feat_pred_fn(txt_enc, lens)
        return x_hat


class VariancePredictor(nn.Module):
    """
    Network is 2-layer 1D convolutions with leaky relu activation and then
    followed by layer normalization then a dropout layer and finally an
    extra linear layer to project the hidden states into the output sequence.

    Args:
        channels_in (int): Number of in channels for conv layers.
        channels_out (int): Number of out channels for the last linear layer.
        kernel_size (int): Size the kernel for the conv layers.
        p_dropout (float): Probability of dropout.
        lrelu_slope (float): Slope for the leaky relu.

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, time): Tensor containing indices to be masked
    Returns:
        - **outputs** (batch, time): Tensor produced by last linear layer.
    """

    def __init__(
        self, channels_in: int, channels: int, channels_out: int, kernel_size: int, p_dropout: float, lrelu_slope: float
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                ConvTransposed(
                    channels_in,
                    channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.LeakyReLU(lrelu_slope),
                nn.LayerNorm(channels),
                nn.Dropout(p_dropout),
                ConvTransposed(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.LeakyReLU(lrelu_slope),
                nn.LayerNorm(channels),
                nn.Dropout(p_dropout),
            ]
        )

        self.linear_layer = nn.Linear(channels, channels_out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            x: :math: `[B, T_src, C]`
            mask: :math: `[B, T_src]`
        """
        for layer in self.layers:
            x = layer(x)
        x = self.linear_layer(x)
        x = x.squeeze(-1)
        x = x.masked_fill(mask, 0.0)
        return x
