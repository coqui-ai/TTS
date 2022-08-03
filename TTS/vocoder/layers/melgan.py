from torch import nn
from torch.nn.utils import weight_norm


class ResidualStack(nn.Module):
    def __init__(self, channels, num_res_blocks, kernel_size):
        super().__init__()

        assert (kernel_size - 1) % 2 == 0, " [!] kernel_size has to be odd."
        base_padding = (kernel_size - 1) // 2

        self.blocks = nn.ModuleList()
        for idx in range(num_res_blocks):
            layer_kernel_size = kernel_size
            layer_dilation = layer_kernel_size**idx
            layer_padding = base_padding * layer_dilation
            self.blocks += [
                nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.ReflectionPad1d(layer_padding),
                    weight_norm(
                        nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=layer_dilation, bias=True)
                    ),
                    nn.LeakyReLU(0.2),
                    weight_norm(nn.Conv1d(channels, channels, kernel_size=1, bias=True)),
                )
            ]

        self.shortcuts = nn.ModuleList(
            [weight_norm(nn.Conv1d(channels, channels, kernel_size=1, bias=True)) for i in range(num_res_blocks)]
        )

    def forward(self, x):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
        return x

    def remove_weight_norm(self):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            nn.utils.remove_weight_norm(block[2])
            nn.utils.remove_weight_norm(block[4])
            nn.utils.remove_weight_norm(shortcut)
