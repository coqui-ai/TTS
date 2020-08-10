import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        """Layer norm for the 2nd dimension of the input.
        Args:
            channels (int): number of channels (2nd dimension) of the input.
            eps (float): to prevent 0 division

        Shapes:
            - input: (B, C, T)
            - output: (B, C, T)
        """
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, channels, 1) * 0.1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean)**2, 1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        x = x * self.gamma + self.beta
        return x


class ConvLayerNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size,
                 num_layers, dropout_p):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        assert num_layers > 1, " [!] number of layers should be > 0."
        assert kernel_size % 2 == 1, " [!] kernel size should be odd number."

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        self.conv_layers.append(
            nn.Conv1d(in_channels,
                      hidden_channels,
                      kernel_size,
                      padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))

        for _ in range(num_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(hidden_channels,
                          hidden_channels,
                          kernel_size,
                          padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))

        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.num_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x * x_mask)
            x = F.dropout(F.relu(x), self.dropout_p, training=self.training)
        x = x_org + self.proj(x)
        return x * x_mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 num_layers,
                 c_in_channels=0,
                 dropout_p=0):
        super(WN, self).__init__()
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

        if c_in_channels != 0:
            cond_layer = torch.nn.Conv1d(c_in_channels,
                                         2 * hidden_channels * num_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer,
                                                         name='weight')

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
                g_l = g[:,
                        cond_offset:cond_offset + 2 * self.hidden_channels, :]
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


class ActNorm(nn.Module):
    """Activation Normalization bijector as an alternative to Batch Norm. It computes
    mean and std from a sample data in advance and it uses these values
    for normalization at training.

    Args:
        channels (int): input channels.
        ddi (False): data depended initialization flag.

    Shapes:
        - inputs: (B, C, T)
        - outputs: (B, C, T)
    """

    def __init__(self, channels, ddi=False, **kwargs):  # pylint: disable=unused-argument
        super().__init__()
        self.channels = channels
        self.initialized = not ddi

        self.logs = nn.Parameter(torch.zeros(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x, x_mask=None, reverse=False, **kwargs):  # pylint: disable=unused-argument
        if x_mask is None:
            x_mask = torch.ones(x.size(0), 1, x.size(2)).to(device=x.device,
                                                            dtype=x.dtype)
        x_len = torch.sum(x_mask, [1, 2])
        if not self.initialized:
            self.initialize(x, x_mask)
            self.initialized = True

        if reverse:
            z = (x - self.bias) * torch.exp(-self.logs) * x_mask
            logdet = None
        else:
            z = (self.bias + torch.exp(self.logs) * x) * x_mask
            logdet = torch.sum(self.logs) * x_len  # [b]

        return z, logdet

    def store_inverse(self):
        pass

    def set_ddi(self, ddi):
        self.initialized = not ddi

    def initialize(self, x, x_mask):
        with torch.no_grad():
            denom = torch.sum(x_mask, [0, 2])
            m = torch.sum(x * x_mask, [0, 2]) / denom
            m_sq = torch.sum(x * x * x_mask, [0, 2]) / denom
            v = m_sq - (m**2)
            logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

            bias_init = (-m * torch.exp(-logs)).view(*self.bias.shape).to(
                dtype=self.bias.dtype)
            logs_init = (-logs).view(*self.logs.shape).to(
                dtype=self.logs.dtype)

            self.bias.data.copy_(bias_init)
            self.logs.data.copy_(logs_init)


class InvConvNear(nn.Module):
    def __init__(self, channels, num_splits=4, no_jacobian=False, **kwargs):  # pylint: disable=unused-argument
        super().__init__()
        assert num_splits % 2 == 0
        self.channels = channels
        self.num_splits = num_splits
        self.no_jacobian = no_jacobian
        self.weight_inv = None

        w_init = torch.qr(
            torch.FloatTensor(self.num_splits, self.num_splits).normal_())[0]
        if torch.det(w_init) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.weight = nn.Parameter(w_init)

    def forward(self, x, x_mask=None, reverse=False, **kwargs):  # pylint: disable=unused-argument
        """Split the input into groups of size self.num_splits and
        perform 1x1 convolution separately. Cast 1x1 conv operation
        to 2d by reshaping the input for efficienty.
        """

        b, c, t = x.size()
        assert c % self.num_splits == 0
        if x_mask is None:
            x_mask = 1
            x_len = torch.ones((b, ), dtype=x.dtype, device=x.device) * t
        else:
            x_len = torch.sum(x_mask, [1, 2])

        x = x.view(b, 2, c // self.num_splits, self.num_splits // 2, t)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.num_splits,
                                                       c // self.num_splits, t)

        if reverse:
            if self.weight_inv is not None:
                weight = self.weight_inv
            else:
                weight = torch.inverse(
                    self.weight.float()).to(dtype=self.weight.dtype)
            logdet = None
        else:
            weight = self.weight
            if self.no_jacobian:
                logdet = 0
            else:
                logdet = torch.logdet(
                    self.weight) * (c / self.num_splits) * x_len  # [b]

        weight = weight.view(self.num_splits, self.num_splits, 1, 1)
        z = F.conv2d(x, weight)

        z = z.view(b, 2, self.num_splits // 2, c // self.num_splits, t)
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
        return z, logdet

    def store_inverse(self):
        self.weight_inv = torch.inverse(
            self.weight.float()).to(dtype=self.weight.dtype)


class CouplingBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 num_layers,
                 c_in_channels=0,
                 dropout_p=0,
                 sigmoid_scale=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.c_in_channels = c_in_channels
        self.dropout_p = dropout_p
        self.sigmoid_scale = sigmoid_scale

        start = torch.nn.Conv1d(in_channels // 2, hidden_channels, 1)
        start = torch.nn.utils.weight_norm(start)
        self.start = start
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(hidden_channels, in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        self.wn = WN(in_channels, hidden_channels, kernel_size, dilation_rate,
                     num_layers, c_in_channels, dropout_p)

    def forward(self, x, x_mask=None, reverse=False, g=None, **kwargs):  # pylint: disable=unused-argument
        if x_mask is None:
            x_mask = 1
        x_0, x_1 = x[:, :self.in_channels // 2], x[:, self.in_channels // 2:]

        x = self.start(x_0) * x_mask
        x = self.wn(x, x_mask, g)
        out = self.end(x)

        z_0 = x_0
        m = out[:, :self.in_channels // 2, :]
        logs = out[:, self.in_channels // 2:, :]
        if self.sigmoid_scale:
            logs = torch.log(1e-6 + torch.sigmoid(logs + 2))

        if reverse:
            z_1 = (x_1 - m) * torch.exp(-logs) * x_mask
            logdet = None
        else:
            z_1 = (m + torch.exp(logs) * x_1) * x_mask
            logdet = torch.sum(logs * x_mask, [1, 2])

        z = torch.cat([z_0, z_1], 1)
        return z, logdet

    def store_inverse(self):
        self.wn.remove_weight_norm()
