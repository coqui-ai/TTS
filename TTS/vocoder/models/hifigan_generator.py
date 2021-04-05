import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

LRELU_SLOPE = 0.1


def get_padding(k, d):
    return int((k * d - d) / 2)


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(
                Conv1d(channels,
                       channels,
                       kernel_size,
                       1,
                       dilation=dilation[0],
                       padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(
                Conv1d(channels,
                       channels,
                       kernel_size,
                       1,
                       dilation=dilation[1],
                       padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(
                Conv1d(channels,
                       channels,
                       kernel_size,
                       1,
                       dilation=dilation[2],
                       padding=get_padding(kernel_size, dilation[2])))
        ])

        self.convs2 = nn.ModuleList([
            weight_norm(
                Conv1d(channels,
                       channels,
                       kernel_size,
                       1,
                       dilation=1,
                       padding=get_padding(kernel_size, 1))),
            weight_norm(
                Conv1d(channels,
                       channels,
                       kernel_size,
                       1,
                       dilation=1,
                       padding=get_padding(kernel_size, 1))),
            weight_norm(
                Conv1d(channels,
                       channels,
                       kernel_size,
                       1,
                       dilation=1,
                       padding=get_padding(kernel_size, 1)))
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(
                Conv1d(channels,
                       channels,
                       kernel_size,
                       1,
                       dilation=dilation[0],
                       padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(
                Conv1d(channels,
                       channels,
                       kernel_size,
                       1,
                       dilation=dilation[1],
                       padding=get_padding(kernel_size, dilation[1])))
        ])

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class HifiganGenerator(torch.nn.Module):
    def __init__(self, in_channels, out_channels, resblock_type, resblock_dilation_sizes,
                 resblock_kernel_sizes, upsample_kernel_sizes,
                 upsample_initial_channel, upsample_factors):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        self.conv_pre = weight_norm(
            Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if resblock_type == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_factors,
                                       upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(upsample_initial_channel // (2**i),
                                    upsample_initial_channel // (2**(i + 1)),
                                    k,
                                    u,
                                    padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i + 1))
            for j, (k, d) in enumerate(
                    zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, out_channels, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def inference(self, c):
        c = c.to(self.conv_pre.weight.device)
        c = torch.nn.functional.pad(
            c, (self.inference_padding, self.inference_padding), 'replicate')
        return self.forward(c)

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
