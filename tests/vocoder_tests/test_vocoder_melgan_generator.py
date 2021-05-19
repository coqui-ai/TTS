import numpy as np
import torch

from TTS.vocoder.models.melgan_generator import MelganGenerator


def test_melgan_generator():
    model = MelganGenerator()
    print(model)
    dummy_input = torch.rand((4, 80, 64))
    output = model(dummy_input)
    assert np.all(output.shape == (4, 1, 64 * 256))
    output = model.inference(dummy_input)
    assert np.all(output.shape == (4, 1, (64 + 4) * 256))
