import numpy as np
import torch
from torch import nn

from ..layers.wavegrad import DBlock, FiLM, UBlock


class Wavegrad(nn.Module):
    # pylint: disable=dangerous-default-value
    def __init__(self,
                 in_channels=80,
                 out_channels=1,
                 x_conv_channels=32,
                 c_conv_channels=768,
                 dblock_out_channels=[128, 128, 256, 512],
                 ublock_out_channels=[512, 512, 256, 128, 128],
                 upsample_factors=[5, 5, 3, 2, 2],
                 upsample_dilations=[[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 4, 8],
                                     [1, 2, 4, 8], [1, 2, 4, 8]]):
        super().__init__()

        assert len(upsample_factors) == len(upsample_dilations)
        assert len(upsample_factors) == len(ublock_out_channels)

        # setup up-down sampling parameters
        self.hop_length = np.prod(upsample_factors)
        self.upsample_factors = upsample_factors
        self.downsample_factors = upsample_factors[::-1][:-1]

        ### define DBlocks, FiLM layers ###
        self.dblocks = nn.ModuleList([
            nn.Conv1d(out_channels, x_conv_channels, 5, padding=2),
        ])
        ic =  x_conv_channels
        self.films = nn.ModuleList([])
        for oc, df in zip(dblock_out_channels, self.downsample_factors):
            # print('dblock(', ic, ', ', oc, ', ', df, ")")
            layer = DBlock(ic, oc, df)
            self.dblocks.append(layer)

            # print('film(', ic, ', ', oc,")")
            layer = FiLM(ic, oc)
            self.films.append(layer)
            ic = oc
        # last FiLM block
        # print('film(', ic, ', ', dblock_out_channels[-1],")")
        self.films.append(FiLM(ic, dblock_out_channels[-1]))

        ### define UBlocks ###
        self.c_conv = nn.Conv1d(in_channels, c_conv_channels, 3, padding=1)
        self.ublocks = nn.ModuleList([])
        ic = c_conv_channels
        for idx, (oc, uf) in enumerate(zip(ublock_out_channels, self.upsample_factors)):
            # print('ublock(', ic, ', ', oc, ', ', uf, ")")
            layer = UBlock(ic, oc, uf, upsample_dilations[idx])
            self.ublocks.append(layer)
            ic = oc

        # define last layer
        # print(ic, 'last_conv--', out_channels)
        self.last_conv = nn.Conv1d(ic, out_channels, 3, padding=1)

        # inference time noise schedule params
        self.S = 1000
        self.init_noise_schedule(self.S)


    def init_noise_schedule(self, num_iter, min_val=1e-6, max_val=0.01):
        """compute noise schedule parameters"""
        device = self.last_conv.weight.device
        beta = torch.linspace(min_val, max_val, num_iter).to(device)
        alpha = 1 - beta
        alpha_cum = alpha.cumprod(dim=0)
        noise_level = torch.cat([torch.FloatTensor([1]).to(device), alpha_cum ** 0.5])

        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_cum', alpha_cum)
        self.register_buffer('noise_level', noise_level)

    def compute_noisy_x(self, x):
        B = x.shape[0]
        if len(x.shape) == 3:
            x = x.squeeze(1)
        s = torch.randint(1, self.S + 1, [B]).to(x).long()
        l_a, l_b = self.noise_level[s-1], self.noise_level[s]
        noise_scale = l_a + torch.rand(B).to(x) * (l_b - l_a)
        noise_scale = noise_scale.unsqueeze(1)
        noise = torch.randn_like(x)
        noisy_x = noise_scale * x + (1.0 - noise_scale**2)**0.5 * noise
        return noise.unsqueeze(1), noisy_x.unsqueeze(1), noise_scale[:, 0]

    def forward(self, x, c, noise_scale):
        assert len(c.shape) == 3  # B, C, T
        assert len(x.shape) == 3  # B, 1, T
        o = x
        shift_and_scales = []
        for film, dblock in zip(self.films, self.dblocks):
            o = dblock(o)
            shift_and_scales.append(film(o, noise_scale))

        o = self.c_conv(c)
        for ublock, (film_shift, film_scale) in zip(self.ublocks,
                                                    reversed(shift_and_scales)):
            o = ublock(o, film_shift, film_scale)
        o = self.last_conv(o)
        return o

    def inference(self, c):
        with torch.no_grad():
            x = torch.randn(c.shape[0], 1, self.hop_length * c.shape[-1]).to(c)
            noise_scale = (self.alpha_cum**0.5).unsqueeze(1).to(c)
            for n in range(len(self.alpha) - 1, -1, -1):
                c1 = 1 / self.alpha[n]**0.5
                c2 = (1 - self.alpha[n]) / (1 - self.alpha_cum[n])**0.5
                x = c1 * (x -
                          c2 * self.forward(x, c, noise_scale[n]).squeeze(1))
                if n > 0:
                    noise = torch.randn_like(x)
                    sigma = ((1.0 - self.alpha_cum[n - 1]) /
                             (1.0 - self.alpha_cum[n]) * self.beta[n])**0.5
                    x += sigma * noise
                x = torch.clamp(x, -1.0, 1.0)
        return x
