import torch
from torch import nn
from TTS.vocoder.layers.hifigan import MRF


class HifiganGenerator(nn.Module):

    def __init__(self, in_channels=80, out_channels=1, base_channels=512, upsample_kernel=[16, 16, 4, 4],
                 resblock_kernel_sizes=[3, 7, 11], resblock_dilation_sizes=[1, 3, 5]):
        super(HifiganGenerator, self).__init__()

        self.inference_padding = 2

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

    def inference(self, c):
        c = c.to(self.layers[1].weight.device)
        c = torch.nn.functional.pad(
            c,
            (self.inference_padding, self.inference_padding),
            'replicate')
        return self.forward(c)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.input[1])
        nn.utils.remove_weight_norm(self.output[2])

        for idx, layer in enumerate(self.generator):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()

    def load_checkpoint(self, config, checkpoint_path, eval=False):  # pylint: disable=unused-argument, redefined-builtin
        state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.load_state_dict(state['model'])
        if eval:
            self.eval()
            assert not self.training
            self.remove_weight_norm()