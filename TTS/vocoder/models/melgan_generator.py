import torch
from torch import nn
from torch.nn.utils import weight_norm

from TTS.vocoder.layers.melgan import ResidualStack


class MelganGenerator(nn.Module):
    def __init__(self,
                 in_channels=80,
                 out_channels=1,
                 proj_kernel=7,
                 base_channels=512,
                 upsample_factors=(8, 8, 2, 2),
                 res_kernel=3,
                 num_res_blocks=3):
        super(MelganGenerator, self).__init__()

        # assert model parameters
        assert (proj_kernel -
                1) % 2 == 0, " [!] proj_kernel should be an odd number."

        # setup additional model parameters
        base_padding = (proj_kernel - 1) // 2
        act_slope = 0.2
        self.inference_padding = 2

        # initial layer
        layers = []
        layers += [
            nn.ReflectionPad1d(base_padding),
            weight_norm(
                nn.Conv1d(in_channels,
                          base_channels,
                          kernel_size=proj_kernel,
                          stride=1,
                          bias=True))
        ]

        # upsampling layers and residual stacks
        for idx, upsample_factor in enumerate(upsample_factors):
            layer_in_channels = base_channels // (2**idx)
            layer_out_channels = base_channels // (2**(idx + 1))
            layer_filter_size = upsample_factor * 2
            layer_stride = upsample_factor
            layer_output_padding = upsample_factor % 2
            layer_padding = upsample_factor // 2 + layer_output_padding
            layers += [
                nn.LeakyReLU(act_slope),
                weight_norm(
                    nn.ConvTranspose1d(layer_in_channels,
                                       layer_out_channels,
                                       layer_filter_size,
                                       stride=layer_stride,
                                       padding=layer_padding,
                                       output_padding=layer_output_padding,
                                       bias=True)),
                ResidualStack(
                    channels=layer_out_channels,
                    num_res_blocks=num_res_blocks,
                    kernel_size=res_kernel
                )
            ]

        layers += [nn.LeakyReLU(act_slope)]

        # final layer
        layers += [
            nn.ReflectionPad1d(base_padding),
            weight_norm(
                nn.Conv1d(layer_out_channels,
                          out_channels,
                          proj_kernel,
                          stride=1,
                          bias=True)),
            nn.Tanh()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, c):
        return self.layers(c)

    def inference(self, c):
        c = c.to(self.layers[1].weight.device)
        c = torch.nn.functional.pad(
            c,
            (self.inference_padding, self.inference_padding),
            'replicate')
        return self.layers(c)

    def remove_weight_norm(self):
        for _, layer in enumerate(self.layers):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except ValueError:
                    layer.remove_weight_norm()

    def load_checkpoint(self, config, checkpoint_path, eval=False):  # pylint: disable=unused-argument, redefined-builtin
        state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.load_state_dict(state['model'])
        if eval:
            self.eval()
            assert not self.training
            self.remove_weight_norm()
