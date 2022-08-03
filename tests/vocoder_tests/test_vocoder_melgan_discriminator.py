import numpy as np
import torch

from TTS.vocoder.models.melgan_discriminator import MelganDiscriminator
from TTS.vocoder.models.melgan_multiscale_discriminator import MelganMultiscaleDiscriminator


def test_melgan_discriminator():
    model = MelganDiscriminator()
    print(model)
    dummy_input = torch.rand((4, 1, 256 * 10))
    output, _ = model(dummy_input)
    assert np.all(output.shape == (4, 1, 10))


def test_melgan_multi_scale_discriminator():
    model = MelganMultiscaleDiscriminator()
    print(model)
    dummy_input = torch.rand((4, 1, 256 * 16))
    scores, feats = model(dummy_input)
    assert len(scores) == 3
    assert len(scores) == len(feats)
    assert np.all(scores[0].shape == (4, 1, 64))
    assert np.all(feats[0][0].shape == (4, 16, 4096))
    assert np.all(feats[0][1].shape == (4, 64, 1024))
    assert np.all(feats[0][2].shape == (4, 256, 256))
