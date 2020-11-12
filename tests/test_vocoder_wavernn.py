import numpy as np
import torch
import random
from TTS.vocoder.models.wavernn import WaveRNN


def test_wavernn():
    model = WaveRNN(
        rnn_dims=512,
        fc_dims=512,
        mode=10,
        mulaw=False,
        pad=2,
        use_aux_net=True,
        use_upsample_net=True,
        upsample_factors=[4, 8, 8],
        feat_dims=80,
        compute_dims=128,
        res_out_dims=128,
        num_res_blocks=10,
        hop_length=256,
        sample_rate=22050,
    )
    dummy_x = torch.rand((2, 1280))
    dummy_m = torch.rand((2, 80, 9))
    y_size = random.randrange(20, 60)
    dummy_y = torch.rand((80, y_size))
    output = model(dummy_x, dummy_m)
    assert np.all(output.shape == (2, 1280, 4 * 256)), output.shape
    output = model.inference(dummy_y, True, 5500, 550)
    assert np.all(output.shape == (256 * (y_size - 1),))
