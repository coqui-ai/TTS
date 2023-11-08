from typing import Tuple

import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import
import torch.nn.functional as F
from torch.nn.utils import parametrize

from TTS.tts.layers.delightful_tts.kernel_predictor import KernelPredictor


def calc_same_padding(kernel_size: int) -> Tuple[int, int]:
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class ConvNorm(nn.Module):
    """A 1-dimensional convolutional layer with optional weight normalization.

    This layer wraps a 1D convolutional layer from PyTorch and applies
    optional weight normalization. The layer can be used in a similar way to
    the convolutional layers in PyTorch's `torch.nn` module.

    Args:
        in_channels (int): The number of channels in the input signal.
        out_channels (int): The number of channels in the output signal.
        kernel_size (int, optional): The size of the convolving kernel.
            Defaults to 1.
        stride (int, optional): The stride of the convolution. Defaults to 1.
        padding (int, optional): Zero-padding added to both sides of the input.
            If `None`, the padding will be calculated so that the output has
            the same length as the input. Defaults to `None`.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, add bias after convolution. Defaults to `True`.
        w_init_gain (str, optional): The weight initialization function to use.
            Can be either 'linear' or 'relu'. Defaults to 'linear'.
        use_weight_norm (bool, optional): If `True`, apply weight normalization
            to the convolutional weights. Defaults to `False`.

    Shapes:
     - Input: :math:`[N, D, T]`

    - Output: :math:`[N, out_dim, T]` where `out_dim` is the number of output dimensions.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        use_weight_norm=False,
    ):
        super(ConvNorm, self).__init__()  # pylint: disable=super-with-arguments
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_weight_norm = use_weight_norm
        conv_fn = nn.Conv1d
        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))
        if self.use_weight_norm:
            self.conv = nn.utils.parametrizations.weight_norm(self.conv)

    def forward(self, signal, mask=None):
        conv_signal = self.conv(signal)
        if mask is not None:
            # always re-zero output if mask is
            # available to match zero-padding
            conv_signal = conv_signal * mask
        return conv_signal


class ConvLSTMLinear(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        n_layers=2,
        n_channels=256,
        kernel_size=3,
        p_dropout=0.1,
        lstm_type="bilstm",
        use_linear=True,
    ):
        super(ConvLSTMLinear, self).__init__()  # pylint: disable=super-with-arguments
        self.out_dim = out_dim
        self.lstm_type = lstm_type
        self.use_linear = use_linear
        self.dropout = nn.Dropout(p=p_dropout)

        convolutions = []
        for i in range(n_layers):
            conv_layer = ConvNorm(
                in_dim if i == 0 else n_channels,
                n_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            )
            conv_layer = nn.utils.parametrizations.weight_norm(conv_layer.conv, name="weight")
            convolutions.append(conv_layer)

        self.convolutions = nn.ModuleList(convolutions)

        if not self.use_linear:
            n_channels = out_dim

        if self.lstm_type != "":
            use_bilstm = False
            lstm_channels = n_channels
            if self.lstm_type == "bilstm":
                use_bilstm = True
                lstm_channels = int(n_channels // 2)

            self.bilstm = nn.LSTM(n_channels, lstm_channels, 1, batch_first=True, bidirectional=use_bilstm)
            lstm_norm_fn_pntr = nn.utils.spectral_norm
            self.bilstm = lstm_norm_fn_pntr(self.bilstm, "weight_hh_l0")
            if self.lstm_type == "bilstm":
                self.bilstm = lstm_norm_fn_pntr(self.bilstm, "weight_hh_l0_reverse")

        if self.use_linear:
            self.dense = nn.Linear(n_channels, out_dim)

    def run_padded_sequence(self, context, lens):
        context_embedded = []
        for b_ind in range(context.size()[0]):  # TODO: speed up
            curr_context = context[b_ind : b_ind + 1, :, : lens[b_ind]].clone()
            for conv in self.convolutions:
                curr_context = self.dropout(F.relu(conv(curr_context)))
            context_embedded.append(curr_context[0].transpose(0, 1))
        context = nn.utils.rnn.pad_sequence(context_embedded, batch_first=True)
        return context

    def run_unsorted_inputs(self, fn, context, lens):  # pylint: disable=no-self-use
        lens_sorted, ids_sorted = torch.sort(lens, descending=True)
        unsort_ids = [0] * lens.size(0)
        for i in range(len(ids_sorted)):  # pylint: disable=consider-using-enumerate
            unsort_ids[ids_sorted[i]] = i
        lens_sorted = lens_sorted.long().cpu()

        context = context[ids_sorted]
        context = nn.utils.rnn.pack_padded_sequence(context, lens_sorted, batch_first=True)
        context = fn(context)[0]
        context = nn.utils.rnn.pad_packed_sequence(context, batch_first=True)[0]

        # map back to original indices
        context = context[unsort_ids]
        return context

    def forward(self, context, lens):
        if context.size()[0] > 1:
            context = self.run_padded_sequence(context, lens)
            # to B, D, T
            context = context.transpose(1, 2)
        else:
            for conv in self.convolutions:
                context = self.dropout(F.relu(conv(context)))

        if self.lstm_type != "":
            context = context.transpose(1, 2)
            self.bilstm.flatten_parameters()
            if lens is not None:
                context = self.run_unsorted_inputs(self.bilstm, context, lens)
            else:
                context = self.bilstm(context)[0]
            context = context.transpose(1, 2)

        x_hat = context
        if self.use_linear:
            x_hat = self.dense(context.transpose(1, 2)).transpose(1, 2)

        return x_hat


class DepthWiseConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, groups=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PointwiseConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BSConv1d(nn.Module):
    """https://arxiv.org/pdf/2003.13549.pdf"""

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, padding: int):
        super().__init__()
        self.pointwise = nn.Conv1d(channels_in, channels_out, kernel_size=1)
        self.depthwise = nn.Conv1d(
            channels_out,
            channels_out,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels_out,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.pointwise(x)
        x2 = self.depthwise(x1)
        return x2


class BSConv2d(nn.Module):
    """https://arxiv.org/pdf/2003.13549.pdf"""

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, padding: int):
        super().__init__()
        self.pointwise = nn.Conv2d(channels_in, channels_out, kernel_size=1)
        self.depthwise = nn.Conv2d(
            channels_out,
            channels_out,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels_out,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.pointwise(x)
        x2 = self.depthwise(x1)
        return x2


class Conv1dGLU(nn.Module):
    """From DeepVoice 3"""

    def __init__(self, d_model: int, kernel_size: int, padding: int, embedding_dim: int):
        super().__init__()
        self.conv = BSConv1d(d_model, 2 * d_model, kernel_size=kernel_size, padding=padding)
        self.embedding_proj = nn.Linear(embedding_dim, d_model)
        self.register_buffer("sqrt", torch.sqrt(torch.FloatTensor([0.5])).squeeze(0))
        self.softsign = torch.nn.Softsign()

    def forward(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        x = x.permute((0, 2, 1))
        residual = x
        x = self.conv(x)
        splitdim = 1
        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
        embeddings = self.embedding_proj(embeddings).unsqueeze(2)
        softsign = self.softsign(embeddings)
        softsign = softsign.expand_as(a)
        a = a + softsign
        x = a * torch.sigmoid(b)
        x = x + residual
        x = x * self.sqrt
        x = x.permute((0, 2, 1))
        return x


class ConvTransposed(nn.Module):
    """
    A 1D convolutional transposed layer for PyTorch.
    This layer applies a 1D convolutional transpose operation to its input tensor,
    where the number of channels of the input tensor is the same as the number of channels of the output tensor.

    Attributes:
        in_channels (int): The number of channels in the input tensor.
        out_channels (int): The number of channels in the output tensor.
        kernel_size (int): The size of the convolutional kernel. Default: 1.
        padding (int): The number of padding elements to add to the input tensor. Default: 0.
        conv (BSConv1d): The 1D convolutional transpose layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = BSConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)
        return x


class DepthwiseConvModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7, expansion: int = 4, lrelu_slope: float = 0.3):
        super().__init__()
        padding = calc_same_padding(kernel_size)
        self.depthwise = nn.Conv1d(
            dim,
            dim * expansion,
            kernel_size=kernel_size,
            padding=padding[0],
            groups=dim,
        )
        self.act = nn.LeakyReLU(lrelu_slope)
        self.out = nn.Conv1d(dim * expansion, dim, 1, 1, 0)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = x.permute((0, 2, 1))
        x = self.depthwise(x)
        x = self.act(x)
        x = self.out(x)
        x = x.permute((0, 2, 1))
        return x


class AddCoords(nn.Module):
    def __init__(self, rank: int, with_r: bool = False):
        super().__init__()
        self.rank = rank
        self.with_r = with_r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = x.shape  # pylint: disable=unused-variable
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

            xx_channel = xx_channel.to(x.device)
            out = torch.cat([x, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = x.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            xx_channel = xx_channel.to(x.device)
            yy_channel = yy_channel.to(x.device)

            out = torch.cat([x, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = x.shape
            xx_ones = torch.ones([1, 1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, 1, dim_y], dtype=torch.int32)
            zz_ones = torch.ones([1, 1, 1, 1, dim_z], dtype=torch.int32)

            xy_range = torch.arange(dim_y, dtype=torch.int32)
            xy_range = xy_range[None, None, None, :, None]

            yz_range = torch.arange(dim_z, dtype=torch.int32)
            yz_range = yz_range[None, None, None, :, None]

            zx_range = torch.arange(dim_x, dtype=torch.int32)
            zx_range = zx_range[None, None, None, :, None]

            xy_channel = torch.matmul(xy_range, xx_ones)
            xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)

            yz_channel = torch.matmul(yz_range, yy_ones)
            yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
            yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)

            zx_channel = torch.matmul(zx_range, zz_ones)
            zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
            zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)

            xx_channel = xx_channel.to(x.device)
            yy_channel = yy_channel.to(x.device)
            zz_channel = zz_channel.to(x.device)
            out = torch.cat([x, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(
                    torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2) + torch.pow(zz_channel - 0.5, 2)
                )
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out


class CoordConv1d(nn.modules.conv.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        with_r: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.rank = 1
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv1d(
            in_channels + self.rank + int(with_r),
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.addcoords(x)
        x = self.conv(x)
        return x


class CoordConv2d(nn.modules.conv.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        with_r: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv2d(
            in_channels + self.rank + int(with_r),
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.addcoords(x)
        x = self.conv(x)
        return x


class LVCBlock(torch.nn.Module):
    """the location-variable convolutions"""

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        in_channels,
        cond_channels,
        stride,
        dilations=[1, 3, 9, 27],
        lReLU_slope=0.2,
        conv_kernel_size=3,
        cond_hop_length=256,
        kpnet_hidden_channels=64,
        kpnet_conv_size=3,
        kpnet_dropout=0.0,
    ):
        super().__init__()

        self.cond_hop_length = cond_hop_length
        self.conv_layers = len(dilations)
        self.conv_kernel_size = conv_kernel_size

        self.kernel_predictor = KernelPredictor(
            cond_channels=cond_channels,
            conv_in_channels=in_channels,
            conv_out_channels=2 * in_channels,
            conv_layers=len(dilations),
            conv_kernel_size=conv_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            kpnet_dropout=kpnet_dropout,
            kpnet_nonlinear_activation_params={"negative_slope": lReLU_slope},
        )

        self.convt_pre = nn.Sequential(
            nn.LeakyReLU(lReLU_slope),
            nn.utils.parametrizations.weight_norm(
                nn.ConvTranspose1d(
                    in_channels,
                    in_channels,
                    2 * stride,
                    stride=stride,
                    padding=stride // 2 + stride % 2,
                    output_padding=stride % 2,
                )
            ),
        )

        self.conv_blocks = nn.ModuleList()
        for dilation in dilations:
            self.conv_blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(lReLU_slope),
                    nn.utils.parametrizations.weight_norm(
                        nn.Conv1d(
                            in_channels,
                            in_channels,
                            conv_kernel_size,
                            padding=dilation * (conv_kernel_size - 1) // 2,
                            dilation=dilation,
                        )
                    ),
                    nn.LeakyReLU(lReLU_slope),
                )
            )

    def forward(self, x, c):
        """forward propagation of the location-variable convolutions.
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length)
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)

        Returns:
            Tensor: the output sequence (batch, in_channels, in_length)
        """
        _, in_channels, _ = x.shape  # (B, c_g, L')

        x = self.convt_pre(x)  # (B, c_g, stride * L')
        kernels, bias = self.kernel_predictor(c)

        for i, conv in enumerate(self.conv_blocks):
            output = conv(x)  # (B, c_g, stride * L')

            k = kernels[:, i, :, :, :, :]  # (B, 2 * c_g, c_g, kernel_size, cond_length)
            b = bias[:, i, :, :]  # (B, 2 * c_g, cond_length)

            output = self.location_variable_convolution(
                output, k, b, hop_size=self.cond_hop_length
            )  # (B, 2 * c_g, stride * L'): LVC
            x = x + torch.sigmoid(output[:, :in_channels, :]) * torch.tanh(
                output[:, in_channels:, :]
            )  # (B, c_g, stride * L'): GAU

        return x

    def location_variable_convolution(self, x, kernel, bias, dilation=1, hop_size=256):  # pylint: disable=no-self-use
        """perform location-variable convolution operation on the input sequence (x) using the local convolution kernl.
        Time: 414 μs ± 309 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), test on NVIDIA V100.
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length).
            kernel (Tensor): the local convolution kernel (batch, in_channel, out_channels, kernel_size, kernel_length)
            bias (Tensor): the bias for the local convolution (batch, out_channels, kernel_length)
            dilation (int): the dilation of convolution.
            hop_size (int): the hop_size of the conditioning sequence.
        Returns:
            (Tensor): the output sequence after performing local convolution. (batch, out_channels, in_length).
        """
        batch, _, in_length = x.shape
        batch, _, out_channels, kernel_size, kernel_length = kernel.shape
        assert in_length == (kernel_length * hop_size), "length of (x, kernel) is not matched"

        padding = dilation * int((kernel_size - 1) / 2)
        x = F.pad(x, (padding, padding), "constant", 0)  # (batch, in_channels, in_length + 2*padding)
        x = x.unfold(2, hop_size + 2 * padding, hop_size)  # (batch, in_channels, kernel_length, hop_size + 2*padding)

        if hop_size < dilation:
            x = F.pad(x, (0, dilation), "constant", 0)
        x = x.unfold(
            3, dilation, dilation
        )  # (batch, in_channels, kernel_length, (hop_size + 2*padding)/dilation, dilation)
        x = x[:, :, :, :, :hop_size]
        x = x.transpose(3, 4)  # (batch, in_channels, kernel_length, dilation, (hop_size + 2*padding)/dilation)
        x = x.unfold(4, kernel_size, 1)  # (batch, in_channels, kernel_length, dilation, _, kernel_size)

        o = torch.einsum("bildsk,biokl->bolsd", x, kernel)
        o = o.to(memory_format=torch.channels_last_3d)
        bias = bias.unsqueeze(-1).unsqueeze(-1).to(memory_format=torch.channels_last_3d)
        o = o + bias
        o = o.contiguous().view(batch, out_channels, -1)

        return o

    def remove_weight_norm(self):
        self.kernel_predictor.remove_weight_norm()
        parametrize.remove_parametrizations(self.convt_pre[1], "weight")
        for block in self.conv_blocks:
            parametrize.remove_parametrizations(block[1], "weight")
