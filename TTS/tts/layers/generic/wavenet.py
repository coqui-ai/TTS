import torch
from torch import nn


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WN(torch.nn.Module):
    """Wavenet layers with weight norm and no input conditioning.

         |-----------------------------------------------------------------------------|
         |                                    |-> tanh    -|                           |
    res -|- conv1d(dilation) -> dropout -> + -|            * -> conv1d1x1 -> split -|- + -> res
    g -------------------------------------|  |-> sigmoid -|                        |
    o --------------------------------------------------------------------------- + --------- o

    Args:
        in_channels (int): number of input channels.
        hidden_channes (int): number of hidden channels.
        kernel_size (int): filter kernel size for the first conv layer.
        dilation_rate (int): dilations rate to increase dilation per layer.
            If it is 2, dilations are 1, 2, 4, 8 for the next 4 layers.
        num_layers (int): number of wavenet layers.
        c_in_channels (int): number of channels of conditioning input.
        dropout_p (float): dropout rate.
        weight_norm (bool): enable/disable weight norm for convolution layers.
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 num_layers,
                 c_in_channels=0,
                 dropout_p=0,
                 weight_norm=True):
        super().__init__()
        assert kernel_size % 2 == 1
        assert hidden_channels % 2 == 0
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.c_in_channels = c_in_channels
        self.dropout_p = dropout_p

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.dropout = nn.Dropout(dropout_p)

        # init conditioning layer
        if c_in_channels > 0:
            cond_layer = torch.nn.Conv1d(c_in_channels,
                                         2 * hidden_channels * num_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer,
                                                         name='weight')
        # intermediate layers
        for i in range(num_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_channels,
                                       2 * hidden_channels,
                                       kernel_size,
                                       dilation=dilation,
                                       padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            if i < num_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels,
                                             res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer,
                                                        name='weight')
            self.res_skip_layers.append(res_skip_layer)
        # setup weight norm
        if not weight_norm:
            self.remove_weight_norm()

    def forward(self, x, x_mask=None, g=None, **kwargs):  # pylint: disable=unused-argument
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])
        if g is not None:
            g = self.cond_layer(g)
        for i in range(self.num_layers):
            x_in = self.in_layers[i](x)
            x_in = self.dropout(x_in)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l,
                                                   n_channels_tensor)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.num_layers - 1:
                x = (x + res_skip_acts[:, :self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.c_in_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)


class WNBlocks(nn.Module):
    """Wavenet blocks.

    Note: After each block dilation resets to 1 and it increases in each block
        along the dilation rate.

    Args:
        in_channels (int): number of input channels.
        hidden_channes (int): number of hidden channels.
        kernel_size (int): filter kernel size for the first conv layer.
        dilation_rate (int): dilations rate to increase dilation per layer.
            If it is 2, dilations are 1, 2, 4, 8 for the next 4 layers.
        num_blocks (int): number of wavenet blocks.
        num_layers (int): number of wavenet layers.
        c_in_channels (int): number of channels of conditioning input.
        dropout_p (float): dropout rate.
        weight_norm (bool): enable/disable weight norm for convolution layers.
    """

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 num_blocks,
                 num_layers,
                 c_in_channels=0,
                 dropout_p=0,
                 weight_norm=True):

        super().__init__()
        self.wn_blocks = nn.ModuleList()
        for idx in range(num_blocks):
            layer = WN(in_channels=in_channels if idx == 0 else hidden_channels,
                       hidden_channels=hidden_channels,
                       kernel_size=kernel_size,
                       dilation_rate=dilation_rate,
                       num_layers=num_layers,
                       c_in_channels=c_in_channels,
                       dropout_p=dropout_p,
                       weight_norm=weight_norm)
            self.wn_blocks.append(layer)

    def forward(self, x, x_mask, g=None):
        o = x
        for layer in self.wn_blocks:
            o = layer(o, x_mask, g)
        return o