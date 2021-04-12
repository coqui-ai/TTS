from torch import nn


# pylint: disable=dangerous-default-value
class ResStack(nn.Module):
    def __init__(self, kernel, channel, padding, dilations=[1, 3, 5]):
        super().__init__()
        resstack = []
        for dilation in dilations:
            resstack += [
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(dilation),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=kernel, dilation=dilation)),
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(padding),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)),
            ]
        self.resstack = nn.Sequential(*resstack)

        self.shortcut = nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1))

    def forward(self, x):
        x1 = self.shortcut(x)
        x2 = self.resstack(x)
        return x1 + x2

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.shortcut)
        nn.utils.remove_weight_norm(self.resstack[2])
        nn.utils.remove_weight_norm(self.resstack[5])
        nn.utils.remove_weight_norm(self.resstack[8])
        nn.utils.remove_weight_norm(self.resstack[11])
        nn.utils.remove_weight_norm(self.resstack[14])
        nn.utils.remove_weight_norm(self.resstack[17])


class MRF(nn.Module):
    def __init__(self, kernels, channel, dilations=[1, 3, 5]):  # # pylint: disable=dangerous-default-value
        super().__init__()
        self.resblock1 = ResStack(kernels[0], channel, 0, dilations)
        self.resblock2 = ResStack(kernels[1], channel, 6, dilations)
        self.resblock3 = ResStack(kernels[2], channel, 12, dilations)

    def forward(self, x):
        x1 = self.resblock1(x)
        x2 = self.resblock2(x)
        x3 = self.resblock3(x)
        return x1 + x2 + x3

    def remove_weight_norm(self):
        self.resblock1.remove_weight_norm()
        self.resblock2.remove_weight_norm()
        self.resblock3.remove_weight_norm()
