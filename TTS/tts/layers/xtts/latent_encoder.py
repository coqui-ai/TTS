# ported from: Originally ported from: https://github.com/neonbjb/tortoise-tts

import math

import torch
from torch import nn
from torch.nn import functional as F


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def normalization(channels):
    groups = 32
    if channels <= 16:
        groups = 8
    elif channels <= 64:
        groups = 16
    while channels % groups != 0:
        groups = int(groups / 2)
    assert groups > 2
    return GroupNorm32(groups, channels)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None, qk_bias=0):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = weight + qk_bias
        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)
            weight[mask.logical_not()] = -torch.inf
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """An attention block that allows spatial positions to attend to each other."""

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        out_channels=None,
        do_activation=False,
    ):
        super().__init__()
        self.channels = channels
        out_channels = channels if out_channels is None else out_channels
        self.do_activation = do_activation
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, out_channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        self.x_proj = nn.Identity() if out_channels == channels else conv_nd(1, channels, out_channels, 1)
        self.proj_out = zero_module(conv_nd(1, out_channels, out_channels, 1))

    def forward(self, x, mask=None, qk_bias=0):
        b, c, *spatial = x.shape
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0).repeat(x.shape[0], 1, 1)
            if mask.shape[1] != x.shape[-1]:
                mask = mask[:, : x.shape[-1], : x.shape[-1]]

        x = x.reshape(b, c, -1)
        x = self.norm(x)
        if self.do_activation:
            x = F.silu(x, inplace=True)
        qkv = self.qkv(x)
        h = self.attention(qkv, mask=mask, qk_bias=qk_bias)
        h = self.proj_out(h)
        xp = self.x_proj(x)
        return (xp + h).reshape(b, xp.shape[1], *spatial)


class ConditioningEncoder(nn.Module):
    def __init__(
        self,
        spec_dim,
        embedding_dim,
        attn_blocks=6,
        num_attn_heads=4,
    ):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim

    def forward(self, x):
        """
        x: (b, 80, s)
        """
        h = self.init(x)
        h = self.attn(h)
        return h
