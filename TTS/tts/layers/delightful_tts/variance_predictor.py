import torch.nn as nn

from TTS.tts.layers.delightful_tts.delightful import BottleneckLayer, ConvLSTMLinear

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