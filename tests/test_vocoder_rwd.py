import torch
import numpy as np

from TTS.vocoder.models.random_window_discriminator import RandomWindowDiscriminator


def test_rwd():
    layer = RandomWindowDiscriminator(cond_channels=80,
                                      window_sizes=(512, 1024, 2048, 4096,
                                                    8192),
                                      cond_disc_downsample_factors=[
                                          (8, 4, 2, 2, 2), (8, 4, 2, 2),
                                          (8, 4, 2), (8, 4), (4, 2, 2)
                                      ],
                                      hop_length=256)
    x = torch.rand([4, 1, 22050])
    c = torch.rand([4, 80, 22050 // 256])

    scores, _ = layer(x, c)
    assert len(scores) == 10
    assert np.all(scores[0].shape == (4, 1, 1))
