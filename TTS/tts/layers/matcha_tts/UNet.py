import math
from einops import pack
import torch
from torch import nn


class PositionalEncoding(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        emb = math.log(10000) / (self.channels // 2 - 1)
        emb = torch.exp(torch.arange(self.channels // 2, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.Mish()
        )

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        output = self.block(x)
        if mask is not None:
            output = output * mask
        return output
    

class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_channels, num_groups=8):
        super().__init__()
        self.block_1 = ConvBlock1D(in_channels, out_channels, num_groups=num_groups)
        self.mlp = nn.Sequential(
            nn.Mish(), 
            nn.Linear(time_embed_channels, out_channels)
        )
        self.block_2 = ConvBlock1D(in_channels, out_channels, num_groups=num_groups)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, mask, t):
        h = self.block_1(x, mask)
        h += self.mlp(t).unsqueeze(-1)
        h = self.block_2(h, mask)
        output = h + self.conv(x * mask)
        return output
    

class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_blocks: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_encoder = PositionalEncoding(in_channels)
        time_embed_channels = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(in_channels, time_embed_channels),
            nn.SiLU(),
            nn.Linear(time_embed_channels, time_embed_channels),
        )

        self.input_blocks = nn.ModuleList([])
        block_in_channels = in_channels * 2
        for _ in range(num_blocks):
            block = nn.ModuleList([])

            block.append(
                ResNetBlock1D(
                    in_channels=block_in_channels, 
                    out_channels=model_channels,
                    time_embed_channels=time_embed_channels
                )
            )

            self.input_blocks.append(block)

        self.middle_blocks = nn.ModuleList([])
        self.output_blocks = nn.ModuleList([])

        self.conv_block = ConvBlock1D(model_channels, model_channels)
        self.conv = nn.Conv1d(model_channels, self.out_channels, 1)

    def forward(self, x_t, mean, mask, t):
        t = self.time_encoder(t)
        t = self.time_embed(t)

        x_t = pack([x_t, mean], "b * t")[0]

        for block in self.input_blocks:
            res_net_block = block[0]
            x_t = res_net_block(x_t, mask, t)

        for _ in self.middle_blocks:
            pass

        for _ in self.output_blocks:
            pass

        output = self.conv_block(x_t)
        output = self.conv(x_t)

        return output * mask