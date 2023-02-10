import math

import numpy as np
import torch

from TTS.utils.io import load_fsspec
from TTS.vocoder.layers.parallel_wavegan import ResidualBlock
from TTS.vocoder.layers.upsample import ConvUpsample


class ParallelWaveganGenerator(torch.nn.Module):
    """PWGAN generator as in https://arxiv.org/pdf/1910.11480.pdf.
    It is similar to WaveNet with no causal convolution.
        It is conditioned on an aux feature (spectrogram) to generate
    an output waveform from an input noise.
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        num_res_blocks=30,
        stacks=3,
        res_channels=64,
        gate_channels=128,
        skip_channels=64,
        aux_channels=80,
        dropout=0.0,
        bias=True,
        use_weight_norm=True,
        upsample_factors=[4, 4, 4, 4],
        inference_padding=2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.num_res_blocks = num_res_blocks
        self.stacks = stacks
        self.kernel_size = kernel_size
        self.upsample_factors = upsample_factors
        self.upsample_scale = np.prod(upsample_factors)
        self.inference_padding = inference_padding
        self.use_weight_norm = use_weight_norm

        # check the number of layers and stacks
        assert num_res_blocks % stacks == 0
        layers_per_stack = num_res_blocks // stacks

        # define first convolution
        self.first_conv = torch.nn.Conv1d(in_channels, res_channels, kernel_size=1, bias=True)

        # define conv + upsampling network
        self.upsample_net = ConvUpsample(upsample_factors=upsample_factors)

        # define residual blocks
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(num_res_blocks):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                res_channels=res_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                dilation=dilation,
                dropout=dropout,
                bias=bias,
            )
            self.conv_layers += [conv]

        # define output layers
        self.last_conv_layers = torch.nn.ModuleList(
            [
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv1d(skip_channels, skip_channels, kernel_size=1, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv1d(skip_channels, out_channels, kernel_size=1, bias=True),
            ]
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, c):
        """
        c: (B, C ,T').
        o: Output tensor (B, out_channels, T)
        """
        # random noise
        x = torch.randn([c.shape[0], 1, c.shape[2] * self.upsample_scale])
        x = x.to(self.first_conv.bias.device)

        # perform upsampling
        if c is not None and self.upsample_net is not None:
            c = self.upsample_net(c)
            assert (
                c.shape[-1] == x.shape[-1]
            ), f" [!] Upsampling scale does not match the expected output. {c.shape} vs {x.shape}"

        # encode to hidden representation
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        # apply final layers
        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        return x

    @torch.no_grad()
    def inference(self, c):
        c = c.to(self.first_conv.weight.device)
        c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                # print(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)):
                torch.nn.utils.weight_norm(m)
                # print(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    @staticmethod
    def _get_receptive_field_size(layers, stacks, kernel_size, dilation=lambda x: 2**x):
        assert layers % stacks == 0
        layers_per_cycle = layers // stacks
        dilations = [dilation(i % layers_per_cycle) for i in range(layers)]
        return (kernel_size - 1) * sum(dilations) + 1

    @property
    def receptive_field_size(self):
        return self._get_receptive_field_size(self.layers, self.stacks, self.kernel_size)

    def load_checkpoint(
        self, config, checkpoint_path, eval=False, cache=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            assert not self.training
            if self.use_weight_norm:
                self.remove_weight_norm()
