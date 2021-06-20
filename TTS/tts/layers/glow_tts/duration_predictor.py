import torch
import math
from torch import nn
from torch.nn import functional as F
from ..generic.normalization import LayerNorm
from .dp_utils import ElementwiseAffine, ConvFlow, DDSConv, Flip


class StochasticDurationPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, p_dropout=0.5, n_flows=4, g_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.g_channels = g_channels

        self.flows = nn.ModuleList()
        self.flows.append(ElementwiseAffine(2))

        for i in range(n_flows):
            self.flows.append(ConvFlow(2, self.hidden_channels, kernel_size, n_layers=3))
            self.flows.append(Flip())

        self.post_pre = nn.Conv1d(1, self.hidden_channels, 1)
        self.post_proj = nn.Conv1d(self.hidden_channels, self.hidden_channels, 1)

        self.post_convs = DDSConv(self.hidden_channels, kernel_size, n_layers=3, p_dropout=p_dropout)

        self.post_flows = nn.ModuleList()
        self.post_flows.append(ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(ConvFlow(2, self.hidden_channels, kernel_size, n_layers=3))
            self.post_flows.append(Flip())

        self.pre = nn.Conv1d(in_channels, self.hidden_channels, 1)
        self.proj = nn.Conv1d(self.hidden_channels, self.hidden_channels, 1)

        self.convs = DDSConv(self.hidden_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if g_channels != 0:
            self.cond = nn.Conv1d(g_channels, self.hidden_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)

        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0 
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1) 
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
            logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

            logdet_tot = 0
            z0 = torch.log(torch.clamp_min(z0, 1e-5)) * x_mask
            logdet = torch.sum(-z0, [1, 2])

            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
            return nll + logq # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]] # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw

class DeterministicDurationPredictor(nn.Module):
    """Glow-TTS duration prediction model.
    [2 x (conv1d_kxk -> relu -> layer_norm -> dropout)] -> conv1d_1x1 -> durs

        Args:
            in_channels ([type]): [description]
            hidden_channels ([type]): [description]
            kernel_size ([type]): [description]
            dropout_p ([type]): [description]
    """

    def __init__(self, in_channels, hidden_channels, kernel_size, dropout_p):
        super().__init__()
        # class arguments
        self.in_channels = in_channels
        self.self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        # layers
        self.drop = nn.Dropout(dropout_p)
        self.conv_1 = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(hidden_channels)
        self.conv_2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(hidden_channels)
        # output layer
        self.proj = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, x, x_mask):
        """
        Shapes:
            x: [B, C, T]
            x_mask: [B, 1, T]

        Returns:
            [type]: [description]
        """
        x = torch.detach(x)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask
