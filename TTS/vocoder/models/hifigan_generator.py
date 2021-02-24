from torch import nn
from TTS.vocoder.layers.hifigan import MRF


class Generator(nn.Module):

    def __init__(self, in_channels=80, out_channels=1, base_channels=512, upsample_kernel=[16, 16, 4, 4],
                 resblock_kernel_sizes=[3, 7, 11], resblock_dilation_sizes=[1, 3, 5]):
        super(Generator, self).__init__()
        self.input = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(in_channels, base_channels, kernel_size=7))
        )

        generator = []

        for k in upsample_kernel:
            inp = base_channels
            out = int(inp / 2)
            generator += [
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(nn.ConvTranspose1d(inp, out, k, k//2)),
                MRF(resblock_kernel_sizes, out, resblock_dilation_sizes)
            ]
            base_channels = out
        self.generator = nn.Sequential(*generator)

        self.output = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(base_channels, out_channels, kernel_size=7, stride=1)),
            nn.Tanh()

        )

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.generator(x1)
        out = self.output(x2)
        return out

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.input[1])
        nn.utils.remove_weight_norm(self.output[2])

        for idx, layer in enumerate(self.generator):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()