import torch
from packaging.version import Version
from torch import nn
from torch.nn import functional as F

from TTS.tts.layers.generic.wavenet import WN

from ..generic.normalization import LayerNorm


class ResidualConv1dLayerNormBlock(nn.Module):
    """Conv1d with Layer Normalization and residual connection as in GlowTTS paper.
    https://arxiv.org/pdf/1811.00002.pdf

    ::

        x |-> conv1d -> layer_norm -> relu -> dropout -> + -> o
          |---------------> conv1d_1x1 ------------------|

    Args:
        in_channels (int): number of input tensor channels.
        hidden_channels (int): number of inner layer channels.
        out_channels (int): number of output tensor channels.
        kernel_size (int): kernel size of conv1d filter.
        num_layers (int): number of blocks.
        dropout_p (float): dropout rate for each block.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, num_layers, dropout_p):
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

        for idx in range(num_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels if idx == 0 else hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2
                )
            )
            self.norm_layers.append(LayerNorm(hidden_channels))

        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
        """
        x_res = x
        for i in range(self.num_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x * x_mask)
            x = F.dropout(F.relu(x), self.dropout_p, training=self.training)
        x = x_res + self.proj(x)
        return x * x_mask


class InvConvNear(nn.Module):
    """Invertible Convolution with input splitting as in GlowTTS paper.
    https://arxiv.org/pdf/1811.00002.pdf

    Args:
        channels (int): input and output channels.
        num_splits (int): number of splits, also H and W of conv layer.
        no_jacobian (bool): enable/disable jacobian computations.

    Note:
        Split the input into groups of size self.num_splits and
        perform 1x1 convolution separately. Cast 1x1 conv operation
        to 2d by reshaping the input for efficiency.
    """

    def __init__(self, channels, num_splits=4, no_jacobian=False, **kwargs):  # pylint: disable=unused-argument
        super().__init__()
        assert num_splits % 2 == 0
        self.channels = channels
        self.num_splits = num_splits
        self.no_jacobian = no_jacobian
        self.weight_inv = None

        if Version(torch.__version__) < Version("1.9"):
            w_init = torch.qr(torch.FloatTensor(self.num_splits, self.num_splits).normal_())[0]
        else:
            w_init = torch.linalg.qr(torch.FloatTensor(self.num_splits, self.num_splits).normal_(), "complete")[0]

        if torch.det(w_init) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.weight = nn.Parameter(w_init)

    def forward(self, x, x_mask=None, reverse=False, **kwargs):  # pylint: disable=unused-argument
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
        """
        b, c, t = x.size()
        assert c % self.num_splits == 0
        if x_mask is None:
            x_mask = 1
            x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
        else:
            x_len = torch.sum(x_mask, [1, 2])

        x = x.view(b, 2, c // self.num_splits, self.num_splits // 2, t)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.num_splits, c // self.num_splits, t)

        if reverse:
            if self.weight_inv is not None:
                weight = self.weight_inv
            else:
                weight = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)
            logdet = None
        else:
            weight = self.weight
            if self.no_jacobian:
                logdet = 0
            else:
                logdet = torch.logdet(self.weight) * (c / self.num_splits) * x_len  # [b]

        weight = weight.view(self.num_splits, self.num_splits, 1, 1)
        z = F.conv2d(x, weight)

        z = z.view(b, 2, self.num_splits // 2, c // self.num_splits, t)
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
        return z, logdet

    def store_inverse(self):
        weight_inv = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)
        self.weight_inv = nn.Parameter(weight_inv, requires_grad=False)


class CouplingBlock(nn.Module):
    """Glow Affine Coupling block as in GlowTTS paper.
    https://arxiv.org/pdf/1811.00002.pdf

    ::

        x --> x0 -> conv1d -> wavenet -> conv1d --> t, s -> concat(s*x1 + t, x0) -> o
        '-> x1 - - - - - - - - - - - - - - - - - - - - - - - - - ^

    Args:
         in_channels (int): number of input tensor channels.
         hidden_channels (int): number of hidden channels.
         kernel_size (int): WaveNet filter kernel size.
         dilation_rate (int): rate to increase dilation by each layer in a decoder block.
         num_layers (int): number of WaveNet layers.
         c_in_channels (int): number of conditioning input channels.
         dropout_p (int): wavenet dropout rate.
         sigmoid_scale (bool): enable/disable sigmoid scaling for output scale.

    Note:
         It does not use the conditional inputs differently from WaveGlow.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        c_in_channels=0,
        dropout_p=0,
        sigmoid_scale=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.c_in_channels = c_in_channels
        self.dropout_p = dropout_p
        self.sigmoid_scale = sigmoid_scale
        # input layer
        start = torch.nn.Conv1d(in_channels // 2, hidden_channels, 1)
        start = torch.nn.utils.parametrizations.weight_norm(start)
        self.start = start
        # output layer
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(hidden_channels, in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        # coupling layers
        self.wn = WN(hidden_channels, hidden_channels, kernel_size, dilation_rate, num_layers, c_in_channels, dropout_p)

    def forward(self, x, x_mask=None, reverse=False, g=None, **kwargs):  # pylint: disable=unused-argument
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        if x_mask is None:
            x_mask = 1
        x_0, x_1 = x[:, : self.in_channels // 2], x[:, self.in_channels // 2 :]

        x = self.start(x_0) * x_mask
        x = self.wn(x, x_mask, g)
        out = self.end(x)

        z_0 = x_0
        t = out[:, : self.in_channels // 2, :]
        s = out[:, self.in_channels // 2 :, :]
        if self.sigmoid_scale:
            s = torch.log(1e-6 + torch.sigmoid(s + 2))

        if reverse:
            z_1 = (x_1 - t) * torch.exp(-s) * x_mask
            logdet = None
        else:
            z_1 = (t + torch.exp(s) * x_1) * x_mask
            logdet = torch.sum(s * x_mask, [1, 2])

        z = torch.cat([z_0, z_1], 1)
        return z, logdet

    def store_inverse(self):
        self.wn.remove_weight_norm()
