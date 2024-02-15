import math
from einops import pack, rearrange
import torch
from torch import nn
import conformer


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
        self.block_2 = ConvBlock1D(in_channels=out_channels, out_channels=out_channels, num_groups=num_groups)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, mask, t):
        h = self.block_1(x, mask)
        h += self.mlp(t).unsqueeze(-1)
        h = self.block_2(h, mask)
        output = h + self.conv(x * mask)
        return output
    

class Downsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)
    

class Upsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels=channels, out_channels=channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)
    

class ConformerBlock(conformer.ConformerBlock):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        conv_dropout: float = 0.,
        conv_causal: bool = False,
    ):
        super().__init__(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
            conv_causal=conv_causal,
        )

    def forward(self, x, mask,):
        x = rearrange(x, "b c t -> b t c")
        mask = rearrange(mask, "b 1 t -> b t")
        output = super().forward(x=x, mask=mask.bool())
        return rearrange(output, "b t c -> b c t")


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_blocks: int,
        transformer_num_heads: int = 4,
        transformer_dim_head: int = 64,
        transformer_ff_mult: int = 1,
        transformer_conv_expansion_factor: int = 2,
        transformer_conv_kernel_size: int = 31,
        transformer_dropout: float = 0.05,
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
        block_out_channels = model_channels
        for level in range(num_blocks):
            block = nn.ModuleList([])

            block.append(
                ResNetBlock1D(
                    in_channels=block_in_channels, 
                    out_channels=block_out_channels,
                    time_embed_channels=time_embed_channels
                )
            )

            block.append(
                self._create_transformer_block(
                    block_out_channels,
                    dim_head=transformer_dim_head,
                    num_heads=transformer_num_heads,
                    ff_mult=transformer_ff_mult,
                    conv_expansion_factor=transformer_conv_expansion_factor,
                    conv_kernel_size=transformer_conv_kernel_size,
                    dropout=transformer_dropout,
                )
            )

            if level != num_blocks - 1:
                block.append(Downsample1D(block_out_channels))
            else:
                block.append(None)

            block_in_channels = block_out_channels
            self.input_blocks.append(block)

        self.middle_blocks = nn.ModuleList([])
        for i in range(2):
            block = nn.ModuleList([])

            block.append(
                ResNetBlock1D(
                    in_channels=block_out_channels,
                    out_channels=block_out_channels,
                    time_embed_channels=time_embed_channels
                )
            )

            block.append(
                self._create_transformer_block(
                    block_out_channels,
                    dim_head=transformer_dim_head,
                    num_heads=transformer_num_heads,
                    ff_mult=transformer_ff_mult,
                    conv_expansion_factor=transformer_conv_expansion_factor,
                    conv_kernel_size=transformer_conv_kernel_size,
                    dropout=transformer_dropout,
                )
            )

            self.middle_blocks.append(block)

        self.output_blocks = nn.ModuleList([])
        block_in_channels = block_out_channels * 2
        block_out_channels = model_channels
        for level in range(num_blocks):
            block = nn.ModuleList([])

            block.append(
                ResNetBlock1D(
                    in_channels=block_in_channels,
                    out_channels=block_out_channels,
                    time_embed_channels=time_embed_channels
                )
            )

            block.append(
                self._create_transformer_block(
                    block_out_channels,
                    dim_head=transformer_dim_head,
                    num_heads=transformer_num_heads,
                    ff_mult=transformer_ff_mult,
                    conv_expansion_factor=transformer_conv_expansion_factor,
                    conv_kernel_size=transformer_conv_kernel_size,
                    dropout=transformer_dropout,
                )
            )

            if level != num_blocks - 1:
                block.append(Upsample1D(block_out_channels))
            else:
                block.append(None)

            block_in_channels = block_out_channels * 2
            self.output_blocks.append(block)

        self.conv_block = ConvBlock1D(model_channels, model_channels)
        self.conv = nn.Conv1d(model_channels, self.out_channels, 1)

    def _create_transformer_block(
        self,
        dim,
        dim_head: int = 64,
        num_heads: int = 4,
        ff_mult: int = 1,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        dropout: float = 0.05,
    ):  
        return ConformerBlock(
            dim=dim,
            dim_head=dim_head,
            heads=num_heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=dropout,
            ff_dropout=dropout,
            conv_dropout=dropout,
            conv_causal=False,
        )

    def forward(self, x_t, mean, mask, t):
        t = self.time_encoder(t)
        t = self.time_embed(t)

        x_t = pack([x_t, mean], "b * t")[0]

        hidden_states = []
        mask_states = [mask]

        for block in self.input_blocks:
            res_net_block, transformer, downsample = block

            x_t = res_net_block(x_t, mask, t)
            x_t = transformer(x_t, mask)

            hidden_states.append(x_t)

            if downsample is not None:
                x_t = downsample(x_t * mask)
                mask = mask[:, :, ::2]
                mask_states.append(mask)

        for block in self.middle_blocks:
            res_net_block, transformer = block
            mask = mask_states[-1]
            x_t = res_net_block(x_t, mask, t)
            x_t = transformer(x_t, mask)

        for block in self.output_blocks:
            res_net_block, transformer, upsample = block

            x_t = pack([x_t, hidden_states.pop()], "b * t")[0]
            mask = mask_states.pop()
            x_t = res_net_block(x_t, mask, t)
            x_t = transformer(x_t, mask)

            if upsample is not None:
                x_t = upsample(x_t * mask)

        output = self.conv_block(x_t)
        output = self.conv(x_t)

        return output * mask