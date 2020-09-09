import numpy as np
import torch

from TTS.vocoder.models.parallel_wavegan_generator import ParallelWaveganGenerator


def test_pwgan_generator():
    model = ParallelWaveganGenerator(
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
        upsample_factors=[4, 4, 4, 4])
    dummy_c = torch.rand((2, 80, 5))
    output = model(dummy_c)
    assert np.all(output.shape == (2, 1, 5 * 256)), output.shape
    model.remove_weight_norm()
    output = model.inference(dummy_c)
    assert np.all(output.shape == (2, 1, (5 + 4) * 256))
