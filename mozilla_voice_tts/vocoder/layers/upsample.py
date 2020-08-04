import torch
from torch.nn import functional as F


class Stretch2d(torch.nn.Module):
    def __init__(self, x_scale, y_scale, mode="nearest"):
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        """
            x (Tensor): Input tensor (B, C, F, T).
            Tensor: Interpolated tensor (B, C, F * y_scale, T * x_scale),
        """
        return F.interpolate(
            x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode)


class UpsampleNetwork(torch.nn.Module):
    # pylint: disable=dangerous-default-value
    def __init__(self,
                 upsample_factors,
                 nonlinear_activation=None,
                 nonlinear_activation_params={},
                 interpolate_mode="nearest",
                 freq_axis_kernel_size=1,
                 use_causal_conv=False,
                 ):
        super(UpsampleNetwork, self).__init__()
        self.use_causal_conv = use_causal_conv
        self.up_layers = torch.nn.ModuleList()
        for scale in upsample_factors:
            # interpolation layer
            stretch = Stretch2d(scale, 1, interpolate_mode)
            self.up_layers += [stretch]

            # conv layer
            assert (freq_axis_kernel_size - 1) % 2 == 0, "Not support even number freq axis kernel size."
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            kernel_size = (freq_axis_kernel_size, scale * 2 + 1)
            if use_causal_conv:
                padding = (freq_axis_padding, scale * 2)
            else:
                padding = (freq_axis_padding, scale)
            conv = torch.nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
            self.up_layers += [conv]

            # nonlinear
            if nonlinear_activation is not None:
                nonlinear = getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
                self.up_layers += [nonlinear]

    def forward(self, c):
        """
            c :  (B, C, T_in).
            Tensor: (B, C, T_upsample)
        """
        c = c.unsqueeze(1)  # (B, 1, C, T)
        for f in self.up_layers:
            c = f(c)
        return c.squeeze(1)  # (B, C, T')


class ConvUpsample(torch.nn.Module):
    # pylint: disable=dangerous-default-value
    def __init__(self,
                 upsample_factors,
                 nonlinear_activation=None,
                 nonlinear_activation_params={},
                 interpolate_mode="nearest",
                 freq_axis_kernel_size=1,
                 aux_channels=80,
                 aux_context_window=0,
                 use_causal_conv=False
                 ):
        super(ConvUpsample, self).__init__()
        self.aux_context_window = aux_context_window
        self.use_causal_conv = use_causal_conv and aux_context_window > 0
        # To capture wide-context information in conditional features
        kernel_size = aux_context_window + 1 if use_causal_conv else 2 * aux_context_window + 1
        # NOTE(kan-bayashi): Here do not use padding because the input is already padded
        self.conv_in = torch.nn.Conv1d(aux_channels, aux_channels, kernel_size=kernel_size, bias=False)
        self.upsample = UpsampleNetwork(
            upsample_factors=upsample_factors,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            interpolate_mode=interpolate_mode,
            freq_axis_kernel_size=freq_axis_kernel_size,
            use_causal_conv=use_causal_conv,
        )

    def forward(self, c):
        """
        c : (B, C, T_in).
        Tensor: (B, C, T_upsampled),
        """
        c_ = self.conv_in(c)
        c = c_[:, :, :-self.aux_context_window] if self.use_causal_conv else c_
        return self.upsample(c)
