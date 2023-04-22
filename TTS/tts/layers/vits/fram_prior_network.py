from torch import nn
from TTS.tts.layers.generic.res_conv_bn import Conv1dBNBlock


class FramePriorNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, hidden_channels, kernel_size, num_res_blocks=13, num_conv_blocks=2
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for idx in range(num_res_blocks):
            block = Conv1dBNBlock(
                in_channels if idx == 0 else hidden_channels,
                out_channels if (idx + 1) == num_res_blocks else hidden_channels,
                hidden_channels,
                kernel_size,
                1,
                num_conv_blocks,
            )
            self.res_blocks.append(block)
    def forward(self, x, x_mask=None):
        if x_mask is None:
            x_mask = 1.0
        o = x * x_mask
        for block in self.res_blocks:
            res = o
            o = block(o)
            o = o + res
            if x_mask is not None:
                o = o * x_mask
        return o