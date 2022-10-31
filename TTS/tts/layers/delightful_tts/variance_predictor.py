import torch
import torch.nn as nn

from TTS.tts.layers.delightful_tts.networks import BottleneckLayer, ConvLSTMLinear, ConvTransposed

class VariancePredictorLSTM(nn.Module):
    def __init__(self, in_dim, out_dim=1, reduction_factor=4):
        super(VariancePredictorLSTM, self).__init__()
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
    """Duration and Pitch predictor"""

    def __init__(
        self,
        channels_in: int,
        channels: int,
        channels_out: int,
        kernel_size: int,
        p_dropout: float,
        lrelu_slope: float
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
        for layer in self.layers:
            x = layer(x)
        x = self.linear_layer(x)
        x = x.squeeze(-1)
        x = x.masked_fill(mask, 0.0)
        return x
