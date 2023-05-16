import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2**0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim),
                negative_slope=negative_slope,
            )
            * scale
        )
    else:
        return F.leaky_relu(input, negative_slope=0.2) * scale


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale)
        out = fused_leaky_relu(out, self.bias * self.lr_mul)
        return out


class RandomLatentConverter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            *[EqualLinear(channels, channels, lr_mul=0.1) for _ in range(5)], nn.Linear(channels, channels)
        )
        self.channels = channels

    def forward(self, ref):
        r = torch.randn(ref.shape[0], self.channels, device=ref.device)
        y = self.layers(r)
        return y


if __name__ == "__main__":
    model = RandomLatentConverter(512)
    model(torch.randn(5, 512))
