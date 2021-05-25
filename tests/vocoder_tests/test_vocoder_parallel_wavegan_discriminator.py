import numpy as np
import torch

from TTS.vocoder.models.parallel_wavegan_discriminator import (
    ParallelWaveganDiscriminator,
    ResidualParallelWaveganDiscriminator,
)


def test_pwgan_disciminator():
    model = ParallelWaveganDiscriminator(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        num_layers=10,
        conv_channels=64,
        dilation_factor=1,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        bias=True,
    )
    dummy_x = torch.rand((4, 1, 64 * 256))
    output = model(dummy_x)
    assert np.all(output.shape == (4, 1, 64 * 256))
    model.remove_weight_norm()


def test_redisual_pwgan_disciminator():
    model = ResidualParallelWaveganDiscriminator(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        num_layers=30,
        stacks=3,
        res_channels=64,
        gate_channels=128,
        skip_channels=64,
        dropout=0.0,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
    )
    dummy_x = torch.rand((4, 1, 64 * 256))
    output = model(dummy_x)
    assert np.all(output.shape == (4, 1, 64 * 256))
    model.remove_weight_norm()
