import math

import torch
from torch import nn
from torch.nn import functional as F

from TTS.tts.layers.generic.normalization import LayerNorm2
from TTS.tts.layers.vits.transforms import piecewise_rational_quadratic_transform


class DilatedDepthSeparableConv(nn.Module):
    def __init__(self, channels, kernel_size, num_layers, dropout_p=0.0) -> torch.tensor:
        """Dilated Depth-wise Separable Convolution module.

        ::
            x |-> DDSConv(x) -> LayerNorm(x) -> GeLU(x) -> Conv1x1(x) -> LayerNorm(x) -> GeLU(x) -> + -> o
              |-------------------------------------------------------------------------------------^

        Args:
            channels ([type]): [description]
            kernel_size ([type]): [description]
            num_layers ([type]): [description]
            dropout_p (float, optional): [description]. Defaults to 0.0.

        Returns:
            torch.tensor: Network output masked by the input sequence mask.
        """
        super().__init__()
        self.num_layers = num_layers

        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(num_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(channels, channels, kernel_size, groups=channels, dilation=dilation, padding=padding)
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm2(channels))
            self.norms_2.append(LayerNorm2(channels))
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, x_mask, g=None):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
        """
        if g is not None:
            x = x + g
        for i in range(self.num_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.dropout(y)
            x = x + y
        return x * x_mask


class ElementwiseAffine(nn.Module):
    """Element-wise affine transform like no-population stats BatchNorm alternative.

    Args:
        channels (int): Number of input tensor channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.translation = nn.Parameter(torch.zeros(channels, 1))
        self.log_scale = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):  # pylint: disable=unused-argument
        if not reverse:
            y = (x * torch.exp(self.log_scale) + self.translation) * x_mask
            logdet = torch.sum(self.log_scale * x_mask, [1, 2])
            return y, logdet
        x = (x - self.translation) * torch.exp(-self.log_scale) * x_mask
        return x


class ConvFlow(nn.Module):
    """Dilated depth separable convolutional based spline flow.

    Args:
        in_channels (int): Number of input tensor channels.
        hidden_channels (int): Number of in network channels.
        kernel_size (int): Convolutional kernel size.
        num_layers (int): Number of convolutional layers.
        num_bins (int, optional): Number of spline bins. Defaults to 10.
        tail_bound (float, optional): Tail bound for PRQT. Defaults to 5.0.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        num_layers: int,
        num_bins=10,
        tail_bound=5.0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.hidden_channels = hidden_channels
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.convs = DilatedDepthSeparableConv(hidden_channels, kernel_size, num_layers, dropout_p=0.0)
        self.proj = nn.Conv1d(hidden_channels, self.half_channels * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, cx?, t] -> [b, c, t, ?]

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.hidden_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.hidden_channels)
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        return x


class StochasticDurationPredictor(nn.Module):
    """Stochastic duration predictor with Spline Flows.

    It applies Variational Dequantization and Variationsl Data Augmentation.

    Paper:
        SDP: https://arxiv.org/pdf/2106.06103.pdf
        Spline Flow: https://arxiv.org/abs/1906.04032

    ::
        ## Inference

        x -> TextCondEncoder() -> Flow() -> dr_hat
        noise ----------------------^

        ## Training
                                                                              |---------------------|
        x -> TextCondEncoder() -> + -> PosteriorEncoder() -> split() -> z_u, z_v -> (d - z_u) -> concat() -> Flow() -> noise
        d -> DurCondEncoder()  -> ^                                                    |
        |------------------------------------------------------------------------------|

    Args:
        in_channels (int): Number of input tensor channels.
        hidden_channels (int): Number of hidden channels.
        kernel_size (int): Kernel size of convolutional layers.
        dropout_p (float): Dropout rate.
        num_flows (int, optional): Number of flow blocks. Defaults to 4.
        cond_channels (int, optional): Number of channels of conditioning tensor. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dropout_p: float,
        num_flows=4,
        cond_channels=0,
        language_emb_dim=0,
    ):
        super().__init__()

        # add language embedding dim in the input
        if language_emb_dim:
            in_channels += language_emb_dim

        # condition encoder text
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.convs = DilatedDepthSeparableConv(hidden_channels, kernel_size, num_layers=3, dropout_p=dropout_p)
        self.proj = nn.Conv1d(hidden_channels, hidden_channels, 1)

        # posterior encoder
        self.flows = nn.ModuleList()
        self.flows.append(ElementwiseAffine(2))
        self.flows += [ConvFlow(2, hidden_channels, kernel_size, num_layers=3) for _ in range(num_flows)]

        # condition encoder duration
        self.post_pre = nn.Conv1d(1, hidden_channels, 1)
        self.post_convs = DilatedDepthSeparableConv(hidden_channels, kernel_size, num_layers=3, dropout_p=dropout_p)
        self.post_proj = nn.Conv1d(hidden_channels, hidden_channels, 1)

        # flow layers
        self.post_flows = nn.ModuleList()
        self.post_flows.append(ElementwiseAffine(2))
        self.post_flows += [ConvFlow(2, hidden_channels, kernel_size, num_layers=3) for _ in range(num_flows)]

        if cond_channels != 0 and cond_channels is not None:
            self.cond = nn.Conv1d(cond_channels, hidden_channels, 1)

        if language_emb_dim != 0 and language_emb_dim is not None:
            self.cond_lang = nn.Conv1d(language_emb_dim, hidden_channels, 1)

    def forward(self, x, x_mask, dr=None, g=None, lang_emb=None, reverse=False, noise_scale=1.0):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - dr: :math:`[B, 1, T]`
            - g: :math:`[B, C]`
        """
        # condition encoder text
        x = self.pre(x)
        if g is not None:
            x = x + self.cond(g)

        if lang_emb is not None:
            x = x + self.cond_lang(lang_emb)

        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert dr is not None

            # condition encoder duration
            h = self.post_pre(dr)
            h = self.post_convs(h, x_mask)
            h = self.post_proj(h) * x_mask
            noise = torch.randn(dr.size(0), 2, dr.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = noise

            # posterior encoder
            logdet_tot_q = 0.0
            for idx, flow in enumerate(self.post_flows):
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h))
                logdet_tot_q = logdet_tot_q + logdet_q
                if idx > 0:
                    z_q = torch.flip(z_q, [1])

            z_u, z_v = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (dr - u) * x_mask

            # posterior encoder - neg log likelihood
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
            nll_posterior_encoder = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (noise**2)) * x_mask, [1, 2]) - logdet_tot_q
            )

            z0 = torch.log(torch.clamp_min(z0, 1e-5)) * x_mask
            logdet_tot = torch.sum(-z0, [1, 2])
            z = torch.cat([z0, z_v], 1)

            # flow layers
            for idx, flow in enumerate(flows):
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
                if idx > 0:
                    z = torch.flip(z, [1])

            # flow layers - neg log likelihood
            nll_flow_layers = torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2]) - logdet_tot
            return nll_flow_layers + nll_posterior_encoder

        flows = list(reversed(self.flows))
        flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
        z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
        for flow in flows:
            z = torch.flip(z, [1])
            z = flow(z, x_mask, g=x, reverse=reverse)

        z0, _ = torch.split(z, [1, 1], 1)
        logw = z0
        return logw
